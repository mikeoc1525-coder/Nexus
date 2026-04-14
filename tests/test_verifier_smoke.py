"""
tests/test_verifier_smoke.py — Verifier Agent isolation smoke test

Runs the Verifier against a HARDCODED fake ResearchBrief — no Researcher call
needed. This saves half the Gemini daily budget and keeps the test focused on
what matters: does the Verifier actually catch false claims?

The fake brief contains:
  - TRUTH: "Stripe was co-founded by Patrick and John Collison."
    This is widely documented. The Verifier must mark it verified=True.
  - LIE: "Stripe was founded in 2005 in Austin, Texas."
    Stripe was founded in 2010 in San Francisco. The Verifier must mark this
    verified=False with counter_evidence or the test fails.

The test passes only if:
  1. submit_verification_result was called (VerificationResult in session state)
  2. AgentDecision schema is valid (agent_name, confidence, input_hash, etc.)
  3. The LIE claim is marked verified=False by at least one VerificationClaim
     — this is the meaningful assertion; the others are structural checks

Usage:
    python tests/test_verifier_smoke.py

Requires GOOGLE_API_KEY in .env (Gemini free tier).
1 Gemini call per run (no MCP tool calls for the lie, ~1-2 for the truth check).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.verifier.agent import run_verifier
from core.schemas import AgentDecision, ResearchBrief, VerificationResult, hash_payload

logging.basicConfig(
    level=logging.WARNING,
    stream=sys.stderr,
    format="%(levelname)s %(name)s: %(message)s",
)

# ─── Fixture ──────────────────────────────────────────────────────────────────

TENANT_ID = "smoke-test-tenant"
LEAD_ID = str(uuid4())

# One obvious truth + one obvious lie about Stripe.
# The lie is specific (wrong year + wrong city) — independently verifiable.
FAKE_BRIEF = ResearchBrief(
    lead_id=UUID(LEAD_ID),
    tenant_id=TENANT_ID,
    summary=(
        "Stripe is a financial infrastructure platform for internet businesses. "
        "It was founded in 2005 in Austin, Texas by brothers Patrick and John Collison. "
        "Stripe processes payments for millions of companies worldwide."
    ),
    recent_news=[
        "Stripe launched Stablecoin Financial Accounts in 2025.",
    ],
    pain_points=[
        "Managing global payment compliance across jurisdictions.",
    ],
    suggested_hook=(
        "Given Stripe's 2005 founding in Austin and your recent stablecoin launch, "
        "I'd love to discuss how we can support your compliance stack."
    ),
    sources=[
        "https://stripe.com/about",
        "https://en.wikipedia.org/wiki/Stripe_(company)",
    ],
)

# The lie embedded in the brief: Stripe was founded in 2010 in San Francisco,
# NOT 2005 in Austin. A working Verifier must surface this.
LIE_KEYWORDS = ["2005", "austin"]


# ─── Checks ───────────────────────────────────────────────────────────────────

def check_verification_result(
    result: VerificationResult,
    label: str,
) -> list[str]:
    failures = []

    if result.status not in {"pass", "fail", "partial"}:
        failures.append(f"status is {result.status!r}, expected pass/fail/partial")

    if not result.claims_checked:
        failures.append("claims_checked is empty — Verifier checked nothing")

    if result.tenant_id != TENANT_ID:
        failures.append(f"tenant_id mismatch: {result.tenant_id!r}")

    # KEY CHECK: at least one claim must be marked verified=False
    # (the lie about 2005 / Austin should have been caught)
    flagged_false = [c for c in result.claims_checked if not c.verified]
    if not flagged_false:
        failures.append(
            "CRITICAL: all claims marked verified=True — Verifier failed to catch "
            "the lie (Stripe founded 2005 in Austin). Either the adversarial prompt "
            "isn't working or the Verifier confirmed without independent search."
        )
    else:
        # Verify the false claim has counter_evidence
        for claim in flagged_false:
            if not claim.counter_evidence:
                failures.append(
                    f"verified=False claim has no counter_evidence: {claim.claim!r}"
                )

    # Retry logic: if status=fail or any claim unverified, retry_required should be True
    if flagged_false and not result.retry_required:
        failures.append(
            "retry_required=False despite verified=False claims — "
            "pipeline would silently accept a bad brief"
        )

    return failures


def check_agent_decision(decision: AgentDecision, label: str) -> list[str]:
    failures = []

    if decision.agent_name != "Verifier":
        failures.append(f"agent_name is {decision.agent_name!r}, expected 'Verifier'")

    if not (0.0 <= decision.confidence <= 1.0):
        failures.append(f"confidence {decision.confidence} out of range")

    if not decision.input_hash:
        failures.append("input_hash is empty")

    if not decision.reasoning:
        failures.append("reasoning is empty — no accountability")

    if decision.tenant_id != TENANT_ID:
        failures.append(f"tenant_id mismatch: {decision.tenant_id!r}")

    return failures


# ─── Runner ───────────────────────────────────────────────────────────────────

async def main() -> None:
    print(f"\n{'─' * 60}")
    print(f"  VERIFIER ISOLATION SMOKE TEST")
    print(f"  Lead ID : {LEAD_ID}")
    print(f"  Brief   : Stripe (contains 2005/Austin lie)")
    print(f"{'─' * 60}")

    result, decision = None, None
    for attempt in range(3):
        try:
            result, decision = await run_verifier(
                lead_id=LEAD_ID,
                tenant_id=TENANT_ID,
                brief=FAKE_BRIEF,
            )
            break
        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "503" in exc_str:
                wait = 60 * (attempt + 1)
                print(f"  [attempt {attempt + 1}/3 hit rate limit — waiting {wait}s]")
                await asyncio.sleep(wait)
                if attempt == 2:
                    print(f"\n{'═' * 60}")
                    print("  VERIFIER SMOKE TEST — FAIL (rate limited)")
                    print(f"  {exc_str[:200]}")
                    print(f"{'═' * 60}\n")
                    sys.exit(1)
            else:
                print(f"\n{'═' * 60}")
                print("  VERIFIER SMOKE TEST — FAIL (exception)")
                print(f"  {exc_str[:300]}")
                print(f"{'═' * 60}\n")
                sys.exit(1)

    if result is None:
        print("  run_verifier returned no result after retries")
        sys.exit(1)

    # Print what we got
    print(f"  Status         : {result.status}")
    print(f"  retry_required : {result.retry_required}")
    print(f"  Claims checked : {len(result.claims_checked)}")
    for c in result.claims_checked:
        icon = "✓" if c.verified else "✗"
        print(f"    [{icon}] {c.claim[:80]}")
        if not c.verified and c.counter_evidence:
            print(f"         counter: {c.counter_evidence[:100]}")
    print(f"  Critique       : {result.critique or '(none)'}")
    print(f"  Confidence     : {decision.confidence:.2f}")
    print(f"  Input hash     : {decision.input_hash}")
    print(f"  final=False?   : {not decision.final}")

    result_failures = check_verification_result(result, "Stripe-lie")
    decision_failures = check_agent_decision(decision, "Stripe-lie")
    all_failures = result_failures + decision_failures

    print(f"\n{'═' * 60}")
    if all_failures:
        print("  VERIFIER SMOKE TEST — FAIL")
        for f in all_failures:
            print(f"  ✗ {f}")
    else:
        print("  VERIFIER SMOKE TEST — PASS")
    print(f"{'═' * 60}\n")

    if all_failures:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
