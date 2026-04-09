"""Research pipeline agent with loop-based deepening."""

from __future__ import annotations

from core.schemas import Lead, LeadStatus, ResearchBrief
from mcp_servers.enrichment_server import EnrichmentServer
from mcp_servers.research_server import ResearchServer


class ResearchAgent:
    def __init__(self, research_server: ResearchServer, enrichment_server: EnrichmentServer):
        self.research_server = research_server
        self.enrichment_server = enrichment_server

    def run(self, lead: Lead, depth: int = 2) -> tuple[Lead, ResearchBrief]:
        domain = lead.email.split("@")[-1].lower()
        company_name = domain.split(".")[0].capitalize() if "." in domain else "Unknown"

        news: list[str] = []
        signals: list[str] = []
        for _ in range(max(1, depth)):
            news.extend(self.research_server.scrape_company_news(company_name))
            decision_makers = self.research_server.find_decision_makers(company_name)
            if decision_makers:
                signals.append(f"Decision maker identified: {decision_makers[0]['role']}")

        tech_stack = self.enrichment_server.get_technographics(domain)
        funding = self.enrichment_server.fetch_funding_data(company_name)

        lead.technographics = tech_stack
        lead.firmographics.industry = "SaaS"
        lead.firmographics.employees = 350
        lead.firmographics.revenue = "$50M-$100M"
        lead.status = LeadStatus.ENRICHED

        brief = ResearchBrief(
            summary=(
                f"{company_name} appears to be scaling GTM execution with a focus on pipeline "
                f"efficiency. Funding stage: {funding['stage']}."
            ),
            recent_news=list(dict.fromkeys(news))[:3],
            pain_points=[
                "Manual lead qualification slows response time.",
                "Inconsistent CRM hygiene reduces forecasting confidence.",
            ],
            suggested_hook=(
                "Noticed your recent growth push. Teams at your stage often unlock revenue by "
                "automating lead research and enforcing CRM data quality."
            ),
        )

        lead.scoring.signals.extend(signals)
        lead.scoring.signals.append(f"Tech stack fit: {', '.join(tech_stack[:2])}")
        return lead, brief
