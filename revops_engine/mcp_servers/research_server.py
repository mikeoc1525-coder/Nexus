"""Imperative web/research MCP server tools."""

from __future__ import annotations


class ResearchServer:
    def search_linkedin(self, persona: str) -> list[dict[str, str]]:
        return [
            {"name": "Alex Carter", "title": f"VP RevOps ({persona})", "company": "ForgeCloud"},
            {"name": "Mina Rao", "title": f"Head of GTM Ops ({persona})", "company": "SignalWorks"},
        ]

    def scrape_company_news(self, company: str) -> list[str]:
        return [
            f"{company} announced a new enterprise expansion initiative.",
            f"{company} highlighted pipeline efficiency as a strategic priority this quarter.",
        ]

    def find_decision_makers(self, company: str) -> list[dict[str, str]]:
        return [
            {"name": "Jordan Lee", "role": "CRO", "company": company},
            {"name": "Priya Singh", "role": "VP Sales", "company": company},
        ]
