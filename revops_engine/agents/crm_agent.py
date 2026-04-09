"""CRM agent using a sequential (formatter -> pusher) pattern."""

from __future__ import annotations

from core.adk_compat import LLMAgent, SequentialAgent
from core.governance import validate_crm_write
from core.schemas import Lead
from mcp_servers.crm_server import CRMServer


class CRMAgent:
    def __init__(self, crm_server: CRMServer):
        self.crm_server = crm_server

        formatter = LLMAgent(
            model="gemini-1.5-flash",
            instructions="Clean lead data into CRM JSON schema.",
            handler=self._format_payload,
            name="formatter",
        )
        pusher = LLMAgent(
            model="gemini-1.5-flash",
            tools=["crm_mcp.upsert_lead"],
            instructions="Use the upsert_lead tool. Always check for duplicates first.",
            handler=self._push_payload,
            name="pusher",
        )
        self.agent = SequentialAgent(agents=[formatter, pusher], name="crm-sequential")
        self.agent.add_callback("before_agent", validate_crm_write)

    def _format_payload(self, payload: dict, context: dict | None = None) -> dict:
        formatted = dict(payload)
        formatted["email"] = str(formatted.get("email", "")).strip().lower()
        formatted["status"] = "qualified"
        return formatted

    def _push_payload(self, payload: dict, context: dict | None = None) -> dict:
        tool_registry = (context or {}).get("tool_registry", {})
        upsert = tool_registry.get("crm_mcp.upsert_lead")
        if not upsert:
            raise RuntimeError("crm_mcp.upsert_lead tool not registered")
        return upsert(payload)

    def run(self, lead: Lead) -> dict:
        context = {"tool_registry": {"crm_mcp.upsert_lead": self.crm_server.upsert_lead}}
        return self.agent.run(lead.to_dict(), context=context)
