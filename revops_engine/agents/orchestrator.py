"""Gemini-powered orchestrator using an A2A hub-and-spoke model."""

from __future__ import annotations

from core.adk_compat import Orchestrator, RemoteA2AAgent
from core.memory import MemoryLayer
from core.schemas import Lead


class RevOpsOrchestrator(Orchestrator):
    def __init__(self, research_agent, scoring_agent, verifier, crm_agent, memory: MemoryLayer):
        self.research_agent = research_agent
        self.scoring_agent = scoring_agent
        self.verifier = verifier
        self.crm_agent = crm_agent
        self.memory = memory

        self.researcher = RemoteA2AAgent(
            url="https://research-agent-service-url",
            handler=self._research_step,
            name="research-agent",
        )
        self.crm_manager = RemoteA2AAgent(
            url="https://crm-agent-service-url",
            handler=self._crm_step,
            name="crm-agent",
        )
        super().__init__(
            agents=[self.researcher, self.crm_manager],
            model="gemini-1.5-pro",
            instructions=(
                "You are the lead RevOps Architect. Route leads to research first, "
                "then score, verify, and sync to CRM."
            ),
            name="revops-orchestrator",
        )

    def _research_step(self, payload):
        lead = payload if isinstance(payload, Lead) else Lead.from_dict(payload)
        return self.research_agent.run(lead)

    def _crm_step(self, payload):
        lead = payload if isinstance(payload, Lead) else Lead.from_dict(payload)
        return self.crm_agent.run(lead)

    def run_objective(self, objective: dict) -> dict:
        try:
            lead = Lead.from_dict(objective)
            self.memory.upsert_lead(lead)

            lead, brief = self.researcher.run(lead)
            lead, reason = self.scoring_agent.qualify(lead, brief)
            approved, critique = self.verifier.review(lead, brief, reason)

            if not approved:
                self.memory.append_event(
                    {"type": "verification_failed", "lead_email": lead.email, "critique": critique}
                )
                return {
                    "status": "blocked",
                    "lead": lead.to_dict(),
                    "critique": critique,
                    "reason": reason,
                }

            crm_result = self.crm_manager.run(lead)
            self.memory.upsert_lead(lead)
            self.memory.append_event(
                {
                    "type": "crm_sync",
                    "lead_email": lead.email,
                    "record_id": crm_result.get("record_id"),
                }
            )
            return {
                "status": "synced",
                "lead": lead.to_dict(),
                "research_brief": brief.to_dict(),
                "reason": reason,
                "crm": crm_result,
                "verifier": critique,
            }
        except Exception as error:
            self.on_failure(error)
            return {"status": "error", "message": str(error)}

    def on_failure(self, error):
        print(f"Workflow Interrupted: {error}")
