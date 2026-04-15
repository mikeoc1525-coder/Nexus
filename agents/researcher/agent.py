"""
agents/researcher/agent.py — Nexus Researcher Agent

LlmAgent that calls the Research MCP tools to produce a ResearchBrief
and an AgentDecision for every lead it processes.

How it works:
  1. Caller sets up session state with lead context before the run.
  2. Agent calls find_company_info + search_web to gather intelligence.
  3. Agent calls submit_research_brief() — a FunctionTool that validates
     the Pydantic model and writes it to session state.
  4. Caller reads session.state['research_brief'] and
     session.state['researcher_decision'] after the run.

Why no output_schema:
  ADK disables all tool calls when output_schema is set (source: llm_agent.py:342).
  The submit_research_brief FunctionTool is used instead — the agent explicitly
  commits its output, which is also a better fit for the accountability contract.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.genai.types import Content, Part
from mcp import StdioServerParameters

from core.llm_factory import build_llm_with_fallback
from core.schemas import AgentDecision, ResearchBrief, hash_payload

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _coerce_str_list(v: list[str] | str | None) -> list[str]:
    """Normalise a list[str] parameter that the model may pass as a JSON string.

    Gemini occasionally serialises list arguments as a JSON-encoded string
    (e.g. '["a","b"]') rather than a native list. This coerces both forms to
    a plain Python list so Pydantic validation doesn't fail on correct data.
    """
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        stripped = v.strip()
        if stripped.startswith("["):
            try:
                import json as _json
                parsed = _json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
        return [stripped] if stripped else []
    return []

from .prompts import RESEARCHER_PROMPT

load_dotenv()

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_RESEARCH_SERVER = str(_REPO_ROOT / "mcp_servers" / "research_server.py")
_PYTHON = sys.executable
_LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")  # fallback if factory not used


# ─── MCP Toolset ──────────────────────────────────────────────────────────────

def _make_research_toolset() -> McpToolset:
    """Spawn the research MCP server as a subprocess, 30s tool timeout."""
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=_PYTHON,
                args=[_RESEARCH_SERVER],
            ),
            timeout=30,
        ),
    )


# ─── Submit Tool ──────────────────────────────────────────────────────────────

def submit_research_brief(
    summary: str,
    suggested_hook: str,
    sources: list[str],
    recent_news: Optional[list[str]] = None,
    pain_points: Optional[list[str]] = None,
    confidence_notes: Optional[str] = None,
    decision: str = "research_complete",
    reasoning: str = "",
    confidence: float = 0.8,
    tool_context: ToolContext = None,
) -> str:
    """
    Commit your completed ResearchBrief. Call this ONCE when research is done.

    Validates all fields against the ResearchBrief schema. Returns an error
    message if validation fails — fix your inputs and call again.

    Args:
        summary: 2-3 sentence factual company overview.
        suggested_hook: ONE specific, personalised opening line for a cold sales
            email. Must reference a concrete fact from your research.
        sources: List of URLs you retrieved data from. REQUIRED — cannot be empty.
        recent_news: Notable events from the past 6 months.
        pain_points: Inferred challenges based on company stage/industry.
        confidence_notes: Any gaps or uncertainty in your research.
        decision: 'research_complete' (default) or 'research_partial' if gaps remain.
        reasoning: Why you are confident in these findings.
        confidence: Research quality score, 0.0–1.0.
    """
    state = tool_context.state
    lead_id = state.get("lead_id")
    tenant_id = state.get("tenant_id")
    input_hash = state.get("researcher_input_hash")
    retries = state.get("researcher_retries", 0)

    try:
        brief = ResearchBrief(
            lead_id=lead_id,
            tenant_id=tenant_id,
            summary=summary,
            recent_news=_coerce_str_list(recent_news),
            pain_points=_coerce_str_list(pain_points),
            suggested_hook=suggested_hook,
            sources=_coerce_str_list(sources),
            confidence_notes=confidence_notes,
        )
    except Exception as exc:
        return (
            f"ERROR: ResearchBrief validation failed — {exc}. "
            "Fix your inputs and call submit_research_brief again."
        )

    try:
        decision_obj = AgentDecision(
            tenant_id=tenant_id,
            agent_name="Researcher",
            input_hash=input_hash,
            decision=decision,
            reasoning=reasoning,
            confidence=confidence,
            retries=retries,
            final=True,
        )
    except Exception as exc:
        return f"ERROR: AgentDecision validation failed — {exc}."

    state["research_brief"] = brief.model_dump(mode="json")
    state["researcher_decision"] = decision_obj.model_dump(mode="json")

    return (
        "ResearchBrief and AgentDecision committed to session state. "
        "Research complete — do not call this tool again."
    )


# ─── Agent Builder ────────────────────────────────────────────────────────────

def build_researcher_agent() -> LlmAgent:
    """Construct and return the Researcher LlmAgent.

    A new McpToolset (and thus a new subprocess connection to the research
    MCP server) is created on each call. For a long-lived pipeline, build
    the agent once and reuse it across runs.
    """
    return LlmAgent(
        name="Researcher",
        model=build_llm_with_fallback(),
        instruction=RESEARCHER_PROMPT,
        tools=[_make_research_toolset(), FunctionTool(func=submit_research_brief)],
        description=(
            "Gathers business intelligence on a lead using web search and "
            "page scraping, then commits a validated ResearchBrief to session state."
        ),
    )


# ─── Runner ───────────────────────────────────────────────────────────────────

async def run_researcher(
    lead_id: str,
    tenant_id: str,
    lead_email: str,
    lead_company: Optional[str] = None,
    lead_name: Optional[str] = None,
    retry_context: Optional[str] = None,
    retries: int = 0,
) -> tuple[ResearchBrief, AgentDecision]:
    """Run the Researcher agent for one lead.

    Args:
        lead_id:       UUID string of the lead record.
        tenant_id:     Tenant identifier (used for session namespacing).
        lead_email:    Lead's email address — domain is used for company research.
        lead_company:  Company name. Derived from email domain if not provided.
        lead_name:     Lead's full name (optional, improves personalisation).
        retry_context: Verifier critique from a previous attempt. Triggers
                       targeted re-research on challenged claims.
        retries:       How many retries have already occurred (tracked in the
                       AgentDecision for audit trail).

    Returns:
        (ResearchBrief, AgentDecision) — validated Pydantic models.

    Raises:
        RuntimeError: If the agent fails to call submit_research_brief.
    """
    email_domain = lead_email.split("@")[-1] if "@" in lead_email else ""
    company = lead_company or email_domain

    input_payload = {
        "lead_id": lead_id,
        "tenant_id": tenant_id,
        "email": lead_email,
        "company": company,
    }

    initial_state: dict = {
        "lead_id": lead_id,
        "tenant_id": tenant_id,
        "lead_email": lead_email,
        "lead_company": company,
        "lead_name": lead_name or "",
        "researcher_input_hash": hash_payload(input_payload),
        "researcher_retries": retries,
    }
    if retry_context:
        initial_state["retry_context"] = retry_context

    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="nexus",
        user_id=tenant_id,
        state=initial_state,
    )

    agent = build_researcher_agent()
    runner = Runner(
        agent=agent,
        app_name="nexus",
        session_service=session_service,
    )

    # Build trigger message with dynamic lead context
    message_lines = [
        f"Lead email: {lead_email}",
        f"Company: {company}",
    ]
    if lead_name:
        message_lines.append(f"Contact name: {lead_name}")
    if retry_context:
        message_lines.append(f"\nRETRY CONTEXT — address these challenged claims:\n{retry_context}")

    message = Content(
        role="user",
        parts=[Part(text="\n".join(message_lines))],
    )

    # Stream events — log errors, ignore the rest (results land in session state)
    async for event in runner.run_async(
        user_id=tenant_id,
        session_id=session.id,
        new_message=message,
    ):
        if event.error_code:
            logger.error(
                "Researcher event error: code=%s message=%s",
                event.error_code,
                event.error_message,
            )

    # Read results from session state
    final_session = await session_service.get_session(
        app_name="nexus",
        user_id=tenant_id,
        session_id=session.id,
    )

    brief_dict = final_session.state.get("research_brief")
    decision_dict = final_session.state.get("researcher_decision")

    if not brief_dict:
        raise RuntimeError(
            f"Researcher agent did not commit a ResearchBrief for lead {lead_id}. "
            "The agent may have failed to call submit_research_brief — "
            "check logs for tool errors or malformed responses."
        )

    return (
        ResearchBrief.model_validate(brief_dict),
        AgentDecision.model_validate(decision_dict),
    )
