"""
agents/verifier/agent.py — Nexus Verifier Agent

Adversarial LlmAgent that counter-searches claims from a ResearchBrief and
emits a VerificationResult + AgentDecision for every brief it checks.

How it works:
  1. Caller puts a serialised ResearchBrief dict into the trigger message.
  2. Agent calls search_web() to counter-search each key claim.
  3. Agent calls submit_verification_result() — a FunctionTool that validates
     the Pydantic models and writes them to session state.
  4. Caller reads session.state['verification_result'] and
     session.state['verifier_decision'] after the run.

Why standalone run_verifier():
  Used for isolated unit testing against fake briefs (no Researcher call needed).
  Production use goes through agents/pipeline/agent.py which wires the LoopAgent.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional
from uuid import UUID

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.genai.types import Content, Part
from mcp import StdioServerParameters

from core.schemas import AgentDecision, ResearchBrief, VerificationClaim, VerificationResult, hash_payload

from .prompts import VERIFIER_PROMPT

load_dotenv()

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_RESEARCH_SERVER = str(_REPO_ROOT / "mcp_servers" / "research_server.py")
_PYTHON = sys.executable
_LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")


# ─── MCP Toolset ──────────────────────────────────────────────────────────────

def _make_research_toolset() -> McpToolset:
    """Spawn the research MCP server as a subprocess, 30s tool timeout."""
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=_PYTHON,
                args=[_RESEARCH_SERVER],
            ),
            timeout=30,
        ),
    )


# ─── Coercion helpers ─────────────────────────────────────────────────────────

def _coerce_str_list(v: list[str] | str | None) -> list[str]:
    """Normalise a list[str] param that Gemini may pass as a JSON string."""
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    if isinstance(v, str):
        stripped = v.strip()
        if stripped.startswith("["):
            try:
                import json as _json
                parsed = _json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
        return [stripped] if stripped else []
    return []


def _coerce_claims(
    raw: list[dict] | str | None,
) -> list[VerificationClaim]:
    """
    Normalise the claims argument from the LLM into VerificationClaim objects.

    Gemini may pass the list as:
      - a native list of dicts          → parse each dict
      - a JSON-encoded string           → decode then parse
      - a list containing a JSON string → unwrap then parse
    """
    if raw is None:
        return []

    # Unwrap JSON string
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("["):
            try:
                raw = json.loads(raw)
            except Exception:
                return []
        else:
            return []

    if not isinstance(raw, list):
        return []

    claims = []
    for item in raw:
        # Each item may itself be a JSON string
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except Exception:
                continue
        if not isinstance(item, dict):
            continue
        try:
            claims.append(VerificationClaim(**item))
        except Exception:
            continue
    return claims


# ─── Submit Tool ──────────────────────────────────────────────────────────────

def submit_verification_result(
    status: str,
    retry_required: bool,
    claims: list[dict],
    critique: Optional[str] = None,
    retry_reason: Optional[str] = None,
    decision: str = "verification_complete",
    reasoning: str = "",
    confidence: float = 0.8,
    tool_context: ToolContext = None,
) -> str:
    """
    Commit your completed VerificationResult. Call this ONCE when verification is done.

    Args:
        status: 'pass' (all claims verified), 'fail' (retry warranted), or
                'partial' (minor gaps, no retry needed).
        retry_required: True if the Researcher should redo the brief.
        claims: List of dicts, each with keys: claim (str), verified (bool),
                counter_evidence (str, required if verified=False),
                source (str, optional).
        critique: Actionable critique for the Researcher. Required if retry_required=True.
        retry_reason: Short reason why retry is needed. Required if retry_required=True.
        decision: 'verification_complete' (default) or 'retry_required'.
        reasoning: Why you reached this verdict.
        confidence: Verification quality score, 0.0–1.0.
    """
    state = tool_context.state
    lead_id = state.get("lead_id")
    tenant_id = state.get("tenant_id")
    input_hash = state.get("verifier_input_hash")
    retries = state.get("verifier_retries", 0)

    parsed_claims = _coerce_claims(claims)

    try:
        result = VerificationResult(
            lead_id=lead_id,
            tenant_id=tenant_id,
            brief_hash=state.get("brief_hash", ""),
            status=status,
            claims_checked=parsed_claims,
            critique=critique,
            retry_required=retry_required,
            retry_reason=retry_reason,
        )
    except Exception as exc:
        return (
            f"ERROR: VerificationResult validation failed — {exc}. "
            "Fix your inputs and call submit_verification_result again."
        )

    # Decide final flag: final=False when retry is required (pipeline will loop)
    is_final = not retry_required
    actual_decision = "retry_required" if retry_required else decision

    try:
        decision_obj = AgentDecision(
            tenant_id=tenant_id,
            agent_name="Verifier",
            input_hash=input_hash,
            decision=actual_decision,
            reasoning=reasoning,
            confidence=confidence,
            retries=retries,
            final=is_final,
            challenged_by="Verifier" if retry_required else None,
            challenge_reasoning=critique if retry_required else None,
        )
    except Exception as exc:
        return f"ERROR: AgentDecision validation failed — {exc}."

    state["verification_result"] = result.model_dump(mode="json")
    state["verifier_decision"] = decision_obj.model_dump(mode="json")

    if retry_required:
        return (
            "VerificationResult committed — retry_required=True. "
            f"Critique for Researcher: {critique}. "
            "Verification complete — do not call this tool again."
        )
    return (
        "VerificationResult committed — verification_complete. "
        "Do not call this tool again."
    )


# ─── Agent Builder ────────────────────────────────────────────────────────────

def build_verifier_agent() -> LlmAgent:
    """Construct and return the Verifier LlmAgent.

    Reuses the same Research MCP server as the Researcher — both agents search
    the same information space so disagreement reflects real discrepancies, not
    "we looked at different stuff."
    """
    return LlmAgent(
        name="Verifier",
        model=_LLM_MODEL,
        instruction=VERIFIER_PROMPT,
        tools=[_make_research_toolset(), FunctionTool(func=submit_verification_result)],
        description=(
            "Adversarially counter-searches claims from a ResearchBrief. "
            "Marks claims verified=False with counter_evidence if independent "
            "confirmation cannot be found. Commits a VerificationResult to session state."
        ),
    )


# ─── Runner ───────────────────────────────────────────────────────────────────

async def run_verifier(
    lead_id: str,
    tenant_id: str,
    brief: ResearchBrief,
    retries: int = 0,
) -> tuple[VerificationResult, AgentDecision]:
    """Run the Verifier agent for one ResearchBrief.

    Used for isolated unit testing. Production use goes through run_pipeline()
    in agents/pipeline/agent.py which wires the LoopAgent retry loop.

    Args:
        lead_id:   UUID string of the lead record.
        tenant_id: Tenant identifier (used for session namespacing).
        brief:     The ResearchBrief to verify.
        retries:   How many retries have already occurred.

    Returns:
        (VerificationResult, AgentDecision) — validated Pydantic models.

    Raises:
        RuntimeError: If the agent fails to call submit_verification_result.
    """
    brief_dict = brief.model_dump(mode="json")
    brief_hash = hash_payload(brief_dict)

    input_payload = {
        "lead_id": lead_id,
        "tenant_id": tenant_id,
        "brief_hash": brief_hash,
    }

    initial_state: dict = {
        "lead_id": lead_id,
        "tenant_id": tenant_id,
        "brief_hash": brief_hash,
        "verifier_input_hash": hash_payload(input_payload),
        "verifier_retries": retries,
    }

    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="nexus",
        user_id=tenant_id,
        state=initial_state,
    )

    agent = build_verifier_agent()
    runner = Runner(
        agent=agent,
        app_name="nexus",
        session_service=session_service,
    )

    # Serialise the brief as JSON for the trigger message
    brief_json = json.dumps(brief_dict, indent=2, default=str)
    message = Content(
        role="user",
        parts=[Part(text=f"Verify the following ResearchBrief:\n\n{brief_json}")],
    )

    async for event in runner.run_async(
        user_id=tenant_id,
        session_id=session.id,
        new_message=message,
    ):
        if event.error_code:
            logger.error(
                "Verifier event error: code=%s message=%s",
                event.error_code,
                event.error_message,
            )

    final_session = await session_service.get_session(
        app_name="nexus",
        user_id=tenant_id,
        session_id=session.id,
    )

    result_dict = final_session.state.get("verification_result")
    decision_dict = final_session.state.get("verifier_decision")

    if not result_dict:
        raise RuntimeError(
            f"Verifier agent did not commit a VerificationResult for lead {lead_id}. "
            "The agent may have failed to call submit_verification_result — "
            "check logs for tool errors or malformed responses."
        )

    return (
        VerificationResult.model_validate(result_dict),
        AgentDecision.model_validate(decision_dict),
    )
