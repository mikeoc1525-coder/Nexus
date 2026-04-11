"""
core/schemas.py — Nexus data contracts

Every agent reads from and writes to these schemas.
No agent may pass raw dicts between pipeline stages.
tenant_id is required on every model — isolation is not optional.
AgentDecision must be emitted alongside every agent output — no silent decisions.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def hash_payload(payload: dict[str, Any]) -> str:
    """Deterministic 16-char SHA-256 prefix of a dict. Used for input tracing."""
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ─── Tenant ───────────────────────────────────────────────────────────────────

class TenantConfig(BaseModel):
    """
    One record per deployed customer instance.
    Stored in tenants/<tenant_id>.json — never hardcoded.
    """
    tenant_id: str
    display_name: str
    crm_backend: str = "sqlite"          # "sqlite" | "zoho" | "hubspot" | "salesforce"
    llm_provider: str = "gemini"         # "gemini" | "openai" | "anthropic"
    llm_model: str = "gemini-2.0-flash"
    max_agent_retries: int = 2
    require_hitl_above_tier: str = "A"   # escalate to human for Tier A writes


# ─── Lead ─────────────────────────────────────────────────────────────────────

class Firmographics(BaseModel):
    employees: Optional[int] = None
    revenue: Optional[str] = None
    industry: Optional[str] = None
    location: Optional[str] = None


class Lead(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    tenant_id: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company: Optional[str] = None
    title: Optional[str] = None
    phone: Optional[str] = None
    status: str = "new"
    # Valid statuses: new | researched | verified | scored | qualified | junk | pending_review
    firmographics: Firmographics = Field(default_factory=Firmographics)
    technographics: list[str] = Field(default_factory=list)
    source: Optional[str] = None          # e.g. "apollo", "form_submit", "manual"
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @field_validator("email")
    @classmethod
    def email_must_contain_at(cls, v: str) -> str:
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError(f"Invalid email address: {v!r}")
        return v.lower().strip()

    @field_validator("status")
    @classmethod
    def status_must_be_valid(cls, v: str) -> str:
        valid = {"new", "researched", "verified", "scored", "qualified", "junk", "pending_review"}
        if v not in valid:
            raise ValueError(f"Invalid status {v!r}. Must be one of: {valid}")
        return v


# ─── Research ─────────────────────────────────────────────────────────────────

class ResearchBrief(BaseModel):
    """
    Output of the Researcher agent.
    Every claim here is a candidate for Verifier challenge.
    sources must be populated — unsourced claims are unverifiable.
    """
    lead_id: UUID
    tenant_id: str
    summary: str
    recent_news: list[str] = Field(default_factory=list)
    pain_points: list[str] = Field(default_factory=list)
    suggested_hook: str
    sources: list[str] = Field(default_factory=list)
    confidence_notes: Optional[str] = None   # Researcher's own uncertainty flags
    created_at: datetime = Field(default_factory=_now)

    @field_validator("sources")
    @classmethod
    def sources_must_not_be_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("ResearchBrief.sources cannot be empty — all claims must be traceable.")
        return v


# ─── Verification ─────────────────────────────────────────────────────────────

class VerificationClaim(BaseModel):
    """
    One claim extracted from a ResearchBrief, checked by the Verifier.
    counter_evidence is required when verified=False.
    """
    claim: str
    verified: bool
    counter_evidence: Optional[str] = None
    source: Optional[str] = None

    @field_validator("counter_evidence")
    @classmethod
    def counter_evidence_required_on_failure(
        cls, v: Optional[str], info: Any
    ) -> Optional[str]:
        # info.data contains already-validated fields
        if info.data.get("verified") is False and not v:
            raise ValueError(
                "counter_evidence is required when verified=False. "
                "The Verifier must cite why the claim failed."
            )
        return v


class VerificationResult(BaseModel):
    """
    Output of the Verifier agent.
    retry_required=True triggers the Orchestrator to re-run the Researcher
    with this critique injected into its context.
    """
    lead_id: UUID
    tenant_id: str
    brief_hash: str                        # hash_payload() of the ResearchBrief
    status: str                            # "pass" | "fail" | "partial"
    claims_checked: list[VerificationClaim] = Field(default_factory=list)
    critique: Optional[str] = None        # overall critique sent to Researcher on retry
    retry_required: bool = False
    retry_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)

    @field_validator("status")
    @classmethod
    def status_must_be_valid(cls, v: str) -> str:
        valid = {"pass", "fail", "partial"}
        if v not in valid:
            raise ValueError(f"VerificationResult.status must be one of {valid}, got {v!r}")
        return v

    @field_validator("retry_reason")
    @classmethod
    def retry_reason_required_when_retry(
        cls, v: Optional[str], info: Any
    ) -> Optional[str]:
        if info.data.get("retry_required") and not v:
            raise ValueError("retry_reason is required when retry_required=True.")
        return v


# ─── Scoring ──────────────────────────────────────────────────────────────────

class ScoredLead(BaseModel):
    """
    Output of the Scorer agent.
    reasoning is non-optional — buyers need to audit why a lead was tiered.
    Scorer must only receive leads whose VerificationResult.status != "fail".
    """
    lead_id: UUID
    tenant_id: str
    points: int = Field(ge=0, le=100)
    tier: str                              # "A" | "B" | "C" | "disqualified"
    signals: list[str] = Field(default_factory=list)
    reasoning: str                         # required — no black-box scores
    disqualification_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)

    @field_validator("tier")
    @classmethod
    def tier_must_be_valid(cls, v: str) -> str:
        valid = {"A", "B", "C", "disqualified"}
        if v not in valid:
            raise ValueError(f"ScoredLead.tier must be one of {valid}, got {v!r}")
        return v

    @field_validator("disqualification_reason")
    @classmethod
    def disqualification_reason_required(
        cls, v: Optional[str], info: Any
    ) -> Optional[str]:
        if info.data.get("tier") == "disqualified" and not v:
            raise ValueError("disqualification_reason is required when tier='disqualified'.")
        return v

    @field_validator("signals")
    @classmethod
    def signals_must_not_be_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("ScoredLead.signals cannot be empty — scoring must cite evidence.")
        return v


# ─── CRM Record ───────────────────────────────────────────────────────────────

class CRMRecord(BaseModel):
    """
    Output of the CRM Writer agent.
    backend is the concrete adapter used — allows Auditor to correlate
    records across tenants using different CRM providers.
    """
    lead_id: UUID
    tenant_id: str
    external_id: Optional[str] = None     # record ID in Zoho, HubSpot, etc.
    backend: str                           # "sqlite" | "zoho" | "hubspot" | "salesforce"
    status: str = "draft"
    # Valid statuses: draft | active | rejected | pending_approval
    write_approved_by: Optional[str] = None   # agent_name or "human:<user>"
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    @field_validator("status")
    @classmethod
    def status_must_be_valid(cls, v: str) -> str:
        valid = {"draft", "active", "rejected", "pending_approval"}
        if v not in valid:
            raise ValueError(f"CRMRecord.status must be one of {valid}, got {v!r}")
        return v


# ─── Agent Accountability ─────────────────────────────────────────────────────

class AgentDecision(BaseModel):
    """
    Every agent emits one of these alongside its primary output.
    This is the audit trail. No silent decisions.

    Usage:
        decision = AgentDecision(
            tenant_id=tenant_id,
            agent_name="Researcher",
            input_hash=hash_payload(lead.model_dump()),
            decision="research_complete",
            reasoning="Found 3 recent news items and confirmed Series B funding via TechCrunch.",
            confidence=0.85,
        )
    """
    id: UUID = Field(default_factory=uuid4)
    tenant_id: str
    agent_name: str
    input_hash: str                        # hash_payload() of this agent's input
    decision: str                          # e.g. "research_complete", "verification_fail", "tier_A"
    reasoning: str                         # required — explains the decision
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=_now)

    # Challenge/retry fields — populated by the challenging agent
    challenged_by: Optional[str] = None         # agent_name of challenger
    challenge_reasoning: Optional[str] = None   # why the challenger rejected this decision

    # Override fields — populated by Auditor or human operator
    override_reason: Optional[str] = None       # explains any human/auditor override

    retries: int = Field(default=0, ge=0)
    final: bool = True   # False while a retry is in progress


class AuditReport(BaseModel):
    """
    Output of the Auditor agent after reviewing a full pipeline run.
    flags contains plain-English descriptions of anything suspicious.
    """
    tenant_id: str
    pipeline_run_id: UUID = Field(default_factory=uuid4)
    decisions_reviewed: int
    flags: list[str] = Field(default_factory=list)
    escalate_to_human: bool = False
    escalation_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)


# ─── Integration Layer ────────────────────────────────────────────────────────

# Event type constants — buyers use these strings in their webhook payloads.
# Not an enum so third-party tools can use plain strings without importing this module.
class IntegrationEventType:
    CALL_LOGGED = "call_logged"
    VOICEMAIL_TRANSCRIBED = "voicemail_transcribed"
    EMAIL_RECEIVED = "email_received"
    MEETING_SCHEDULED = "meeting_scheduled"
    FORM_SUBMITTED = "form_submitted"
    CUSTOM = "custom"

    ALL = {
        CALL_LOGGED, VOICEMAIL_TRANSCRIBED, EMAIL_RECEIVED,
        MEETING_SCHEDULED, FORM_SUBMITTED, CUSTOM,
    }


class IntegrationEvent(BaseModel):
    """
    Generic inbound event from any external tool.
    The webhook receiver validates payloads into this shape.
    source identifies the tool (e.g. "twilio", "openphone", "gmail", "calendly").
    payload holds the raw event data — agents extract what they need.
    """
    id: UUID = Field(default_factory=uuid4)
    tenant_id: str
    source: str                            # e.g. "twilio" | "openphone" | "gmail"
    event_type: str                        # use IntegrationEventType constants
    lead_email: Optional[str] = None       # used to route to the correct Lead record
    lead_id: Optional[UUID] = None         # populated after resolution
    payload: dict[str, Any]                # raw event body — schema varies by source
    processed: bool = False
    processed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=_now)

    @field_validator("event_type")
    @classmethod
    def event_type_must_be_known(cls, v: str) -> str:
        if v not in IntegrationEventType.ALL:
            # Allow unknown types — log them but don't block ingestion.
            # Buyers may send custom event types.
            pass
        return v

    @field_validator("lead_email")
    @classmethod
    def email_format_if_present(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and ("@" not in v or "." not in v.split("@")[-1]):
            raise ValueError(f"IntegrationEvent.lead_email is malformed: {v!r}")
        return v.lower().strip() if v else v
