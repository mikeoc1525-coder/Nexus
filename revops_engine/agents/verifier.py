"""Verifier agent to gate CRM writes."""

from __future__ import annotations

from core.schemas import Lead, ResearchBrief, Tier


class Verifier:
    def review(self, lead: Lead, brief: ResearchBrief, reason: str) -> tuple[bool, str]:
        if not lead.email or "@" not in lead.email:
            return False, "Invalid email format."
        if not brief.summary or len(brief.summary) < 25:
            return False, "Research summary lacks depth."
        if lead.scoring.tier == Tier.C:
            return False, "Lead tier below CRM threshold."
        if not reason:
            return False, "Qualification reason missing."
        return True, "Output passed quality control."
