"""Governance callbacks and policy helpers."""

from __future__ import annotations

from core.schemas import Tier


def validate_crm_write(request) -> tuple[bool, str | None]:
    """Callback to prevent messy CRM data."""
    lead_data = request.args.get("data") or {}
    email = str(lead_data.get("email", "")).strip()
    if not email or "@" not in email:
        return False, "Invalid Email: Write Blocked"
    return True, None


def hitl_required(lead_payload: dict) -> tuple[bool, str | None]:
    """Escalate risky writes for human approval."""
    tier = (
        lead_payload.get("scoring", {}).get("tier")
        if isinstance(lead_payload.get("scoring"), dict)
        else None
    )
    if tier == Tier.A.value and not lead_payload.get("approved_by_human"):
        return True, "Tier A lead requires approval before write."
    return False, None
