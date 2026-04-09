"""Declarative CRM MCP server tools."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


class CRMServer:
    """MCP-style CRM server with deterministic local behavior."""

    def __init__(self):
        self._records_by_email: dict[str, dict[str, Any]] = {}
        self._deal_stage_by_id: dict[str, str] = {}
        self._history_resource = [
            {"industry": "SaaS", "employees_band": "200-1000", "outcome": "won"},
            {"industry": "Fintech", "employees_band": "50-500", "outcome": "lost"},
            {"industry": "Healthcare", "employees_band": "100-2000", "outcome": "won"},
        ]

    def upsert_lead(self, data: dict[str, Any]) -> dict[str, Any]:
        email = str(data.get("email", "")).lower()
        if not email:
            raise ValueError("upsert_lead requires email")

        existing = self._records_by_email.get(email)
        if existing:
            existing.update(data)
            existing["updated_at"] = datetime.now(UTC).isoformat()
            return {"record_id": existing["record_id"], "status": "updated", "duplicate": True}

        record = {
            **data,
            "record_id": f"crm-{len(self._records_by_email) + 1}",
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self._records_by_email[email] = record
        self._deal_stage_by_id[record["record_id"]] = "Draft"
        return {"record_id": record["record_id"], "status": "created", "duplicate": False}

    def get_account_history(self, account_name: str) -> dict[str, Any]:
        return {
            "account_name": account_name,
            "recent_interactions": [
                "No recent opportunities",
                "Marketing engagement score: medium",
            ],
        }

    def update_deal_stage(self, record_id: str, stage: str) -> dict[str, str]:
        if record_id not in self._deal_stage_by_id:
            raise ValueError(f"Record id not found: {record_id}")
        self._deal_stage_by_id[record_id] = stage
        return {"record_id": record_id, "stage": stage}

    def historical_win_loss_resource(self) -> list[dict[str, str]]:
        """Read-only resource for lookalike analysis."""
        return list(self._history_resource)
