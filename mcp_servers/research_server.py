"""
mcp_servers/research_server.py — Nexus Research MCP Server

HTTP-backed research tool provider. Agents call these tools to gather
intelligence on leads before Verification and Scoring.

Tools:
    search_web          — DuckDuckGo search; returns titles, snippets, URLs
    scrape_url          — Jina Reader scrape; returns clean markdown for a URL
    find_company_info   — Combined: search for a company then scrape top result

Rate limits:
    DuckDuckGo: no API key, no account, no cost — used by default
    Jina Reader: no hard limit, no key required for basic use
                 Set JINA_API_KEY in .env to unlock higher rate limits

Optional upgrade path:
    SerpAPI (100 free searches/month) can replace DuckDuckGo for higher-volume
    or more structured results. See .env.example for config.

Run standalone (stdio transport, for ADK agent wiring):
    python mcp_servers/research_server.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv
from ddgs import DDGS
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

_JINA_BASE = "https://r.jina.ai"
_JINA_API_KEY = os.getenv("JINA_API_KEY", "")  # optional — increases rate limit

_HTTP_TIMEOUT = 20.0  # seconds

# ─── MCP Server ───────────────────────────────────────────────────────────────

mcp = FastMCP(
    "nexus-research",
    instructions=(
        "Research tool server for Nexus. "
        "Use search_web for web results, scrape_url for page content, "
        "and find_company_info for combined company intelligence."
    ),
)

# ─── HTTP Helpers ─────────────────────────────────────────────────────────────

def _jina_headers() -> dict[str, str]:
    headers = {"Accept": "text/markdown", "X-Return-Format": "markdown"}
    if _JINA_API_KEY:
        headers["Authorization"] = f"Bearer {_JINA_API_KEY}"
    return headers


def _ddg_search(query: str, max_results: int) -> list[dict[str, str]]:
    """Synchronous DuckDuckGo text search. Run via asyncio.to_thread."""
    results = []
    with DDGS() as ddgs:
        for i, r in enumerate(ddgs.text(query, max_results=max_results)):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
                "position": i + 1,
            })
    return results


# ─── Tools ────────────────────────────────────────────────────────────────────

@mcp.tool()
async def search_web(
    query: str,
    num_results: int = 5,
) -> dict[str, Any]:
    """
    Run a web search via DuckDuckGo and return structured results.

    No API key or account required. Results are returned in the same shape
    as a SerpAPI response so the tool surface is provider-agnostic.

    Args:
        query:       The search query string.
        num_results: Number of results to return (1–10, default 5).

    Returns:
        dict: {
            "query": str,
            "results": [
                {
                    "title": str,
                    "url": str,
                    "snippet": str,
                    "position": int,
                },
                ...
            ],
            "total_results_estimate": "N/A",  # DDG does not expose this
            "error": str | None,
        }
    """
    num_results = max(1, min(num_results, 10))

    results = await asyncio.to_thread(_ddg_search, query, num_results)

    return {
        "query": query,
        "results": results,
        "total_results_estimate": "N/A",
        "error": None,
    }


@mcp.tool()
async def scrape_url(
    url: str,
    max_chars: int = 4000,
) -> dict[str, Any]:
    """
    Fetch a URL via Jina Reader and return clean markdown content.

    Jina Reader strips nav, ads, and boilerplate — what you get back is
    the article/page body as readable markdown. Free, no key required
    for basic use (JINA_API_KEY in .env unlocks higher rate limits).

    Args:
        url:       The URL to scrape.
        max_chars: Truncate returned content to this many characters (default 4000).
                   Keeps token usage predictable when passing content to agents.

    Returns:
        dict: {
            "url": str,
            "content": str,    # markdown body, truncated to max_chars
            "truncated": bool, # True if content was cut
            "error": str | None,
        }
    """
    jina_url = f"{_JINA_BASE}/{url}"

    async with httpx.AsyncClient(
        timeout=_HTTP_TIMEOUT,
        follow_redirects=True,
    ) as client:
        response = await client.get(jina_url, headers=_jina_headers())
        response.raise_for_status()
        content = response.text

    truncated = len(content) > max_chars
    return {
        "url": url,
        "content": content[:max_chars],
        "truncated": truncated,
        "error": None,
    }


@mcp.tool()
async def find_company_info(
    company_name: str,
    lead_email_domain: Optional[str] = None,
    max_chars: int = 4000,
) -> dict[str, Any]:
    """
    Research a company: search for it, then scrape the top result.

    Combines search_web + scrape_url in one call. The Researcher agent
    should call this once per company rather than chaining the tools manually.

    Args:
        company_name:       Company name to research.
        lead_email_domain:  Optional email domain (e.g. "acme.com") — appended
                            to the search query to improve result targeting.
        max_chars:          Truncate scraped content to this many characters.

    Returns:
        dict: {
            "company_name": str,
            "query_used": str,
            "search_results": [...],     # same shape as search_web results
            "scraped_url": str | None,   # URL that was scraped (top result)
            "scraped_content": str,      # markdown body of top result
            "truncated": bool,
            "error": str | None,
        }
    """
    query_parts = [company_name]
    if lead_email_domain:
        query_parts.append(lead_email_domain)
    query_parts.append("company overview")
    query = " ".join(query_parts)

    # Step 1: search
    search_result = await search_web(query=query, num_results=3)

    if search_result.get("error") or not search_result["results"]:
        return {
            "company_name": company_name,
            "query_used": query,
            "search_results": search_result.get("results", []),
            "scraped_url": None,
            "scraped_content": "",
            "truncated": False,
            "error": search_result.get("error") or "No search results found.",
        }

    # Step 2: scrape top result
    top_url = search_result["results"][0]["url"]
    scrape_result = await scrape_url(url=top_url, max_chars=max_chars)

    return {
        "company_name": company_name,
        "query_used": query,
        "search_results": search_result["results"],
        "scraped_url": top_url,
        "scraped_content": scrape_result.get("content", ""),
        "truncated": scrape_result.get("truncated", False),
        "error": scrape_result.get("error"),
    }


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    mcp.run()  # stdio transport — ADK agents connect via subprocess
