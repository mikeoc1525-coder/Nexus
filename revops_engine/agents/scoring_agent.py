"""Scoring and qualification agent."""

from __future__ import annotations

from core.schemas import Lead, LeadStatus, ResearchBrief, Tier


class ScoringAgent:
    def qualify(self, lead: Lead, brief: ResearchBrief) -> tuple[Lead, str]:
        points = 0
        reasons: list[str] = []

        if lead.firmographics.employees >= 200:
            points += 30
            reasons.append("Employee count is in ICP range")
        if any(tool in lead.technographics for tool in ("Salesforce", "HubSpot")):
            points += 25
            reasons.append("Technographic fit with target stack")
        if "efficiency" in brief.summary.lower():
            points += 20
            reasons.append("Active pipeline efficiency initiative")
        if brief.recent_news:
            points += 15
            reasons.append("Recent activity indicates buying motion")
        if lead.email.endswith((".ai", ".com", ".io")):
            points += 10
            reasons.append("Reachable business domain email")

        if points >= 75:
            tier = Tier.A
        elif points >= 50:
            tier = Tier.B
        else:
            tier = Tier.C

        lead.scoring.points = points
        lead.scoring.tier = tier
        lead.status = LeadStatus.QUALIFIED if tier in (Tier.A, Tier.B) else LeadStatus.JUNK
        qualification_reason = "; ".join(reasons) if reasons else "Insufficient intent signals"
        return lead, qualification_reason
