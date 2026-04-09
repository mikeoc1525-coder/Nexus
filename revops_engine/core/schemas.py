"""Core schema definitions for the RevOps Autonomous Engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4


class LeadStatus(str, Enum):
    NEW = "new"
    ENRICHED = "enriched"
    QUALIFIED = "qualified"
    JUNK = "junk"


class Tier(str, Enum):
    A = "A"
    B = "B"
    C = "C"


@dataclass
class Firmographics:
    employees: int = 0
    revenue: str = "unknown"
    industry: str = "unknown"


@dataclass
class Scoring:
    points: int = 0
    tier: Tier = Tier.C
    signals: list[str] = field(default_factory=list)


@dataclass
class Lead:
    id: str = field(default_factory=lambda: str(uuid4()))
    email: str = ""
    status: LeadStatus = LeadStatus.NEW
    firmographics: Firmographics = field(default_factory=Firmographics)
    technographics: list[str] = field(default_factory=list)
    scoring: Scoring = field(default_factory=Scoring)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["scoring"]["tier"] = self.scoring.tier.value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Lead":
        firmographics = Firmographics(**data.get("firmographics", {}))
        scoring_payload = data.get("scoring", {})
        scoring = Scoring(
            points=int(scoring_payload.get("points", 0)),
            tier=Tier(scoring_payload.get("tier", Tier.C.value)),
            signals=list(scoring_payload.get("signals", [])),
        )
        return cls(
            id=data.get("id", str(uuid4())),
            email=data.get("email", ""),
            status=LeadStatus(data.get("status", LeadStatus.NEW.value)),
            firmographics=firmographics,
            technographics=list(data.get("technographics", [])),
            scoring=scoring,
        )


@dataclass
class ResearchBrief:
    summary: str
    recent_news: list[str]
    pain_points: list[str]
    suggested_hook: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualifiedLead:
    lead: Lead
    research: ResearchBrief
    qualification_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "lead": self.lead.to_dict(),
            "research": self.research.to_dict(),
            "qualification_reason": self.qualification_reason,
        }
