"""
tests/test_researcher_smoke.py — Researcher Agent end-to-end smoke test

Runs the Researcher against three hardcoded companies and validates:
  - ResearchBrief schema (all required fields, sources non-empty)
  - AgentDecision schema (correct agent_name, confidence range, input_hash present)
  - Accountability contract (input_hash is deterministic and non-empty)

Usage:
    python tests/test_researcher_smoke.py

Requires GOOGLE_API_KEY in .env (Gemini free tier — no billing needed).
Runs one company at a time with a 10s pause between to avoid DDG rate limits.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path
from uuid import uuid4

# Add repo root to path so local packages resolve without install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.researcher.agent import run_researcher
from core.schemas import AgentDecision, ResearchBrief, hash_payload

logging.basicConfig(
    level=logging.WARNING,   # suppress ADK debug noise; set INFO to see tool calls
    stream=sys.stderr,
    format="%(levelname)s %(name)s: %(message)s",
)

# ─── Test Fixtures ────────────────────────────────────────────────────────────

LEADS = [
    {
        "company": "Stripe",
        "email": "finance@stripe.com",
        "name": "Patrick Collison",
    },
    {
        "company": "Notion",
        "email": "hello@notion.so",
        "name": "Ivan Zhao",
    },
    {
        "company": "Figma",
        "email": "contact@figma.com",
        "name": "Dylan Field",
    },
]

TENANT_ID = "smoke-test-tenant"

# ─── Checks ───────────────────────────────────────────────────────────────────

def check_research_brief(brief: ResearchBrief, label: str) -> list[str]:
    """Return a list of failure messages. Empty list = pass."""
    failures = []

    if not brief.summary or len(brief.summary.strip()) < 20:
        failures.append("summary is empty or too short")

    if not brief.suggested_hook or len(brief.suggested_hook.strip()) < 10:
        failures.append("suggested_hook is empty or too short")

    if not brief.sources:
        failures.append("sources is empty — schema validator should have caught this")

    for src in brief.sources:
        if not src.startswith("http"):
            failures.append(f"source is not a URL: {src!r}")

    if brief.tenant_id != TENANT_ID:
        failures.append(f"tenant_id mismatch: {brief.tenant_id!r}")

    return failures


def check_agent_decision(decision: AgentDecision, expected_hash: str, label: str) -> list[str]:
    failures = []

    if decision.agent_name != "Researcher":
        failures.append(f"agent_name is {decision.agent_name!r}, expected 'Researcher'")

    if not (0.0 <= decision.confidence <= 1.0):
        failures.append(f"confidence {decision.confidence} out of range 0.0–1.0")

    if not decision.input_hash:
        failures.append("input_hash is empty")

    if decision.input_hash != expected_hash:
        failures.append(
            f"input_hash mismatch: got {decision.input_hash!r}, "
            f"expected {expected_hash!r} — audit trail broken"
        )

    if not decision.reasoning:
        failures.append("reasoning is empty — no accountability")

    if decision.final is not True:
        failures.append("final should be True on a completed research run")

    if decision.retries != 0:
        failures.append(f"retries should be 0 on first run, got {decision.retries}")

    if decision.tenant_id != TENANT_ID:
        failures.append(f"tenant_id mismatch: {decision.tenant_id!r}")

    return failures


# ─── Runner ───────────────────────────────────────────────────────────────────

async def run_one(lead: dict, lead_id: str) -> tuple[bool, list[str]]:
    """Run Researcher for one lead, with retry on 429/503. Returns (passed, messages)."""
    messages = []
    company = lead["company"]
    email = lead["email"]

    # Compute expected input_hash the same way agent.py does
    expected_hash = hash_payload({
        "lead_id": lead_id,
        "tenant_id": TENANT_ID,
        "email": email,
        "company": company,
    })

    print(f"\n{'─' * 60}")
    print(f"  Company : {company}")
    print(f"  Email   : {email}")
    print(f"  Lead ID : {lead_id}")
    print(f"{'─' * 60}")

    brief, decision = None, None
    for attempt in range(3):
        try:
            brief, decision = await run_researcher(
                lead_id=lead_id,
                tenant_id=TENANT_ID,
                lead_email=email,
                lead_company=company,
                lead_name=lead.get("name"),
            )
            break  # success
        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "503" in exc_str:
                wait = 60 * (attempt + 1)
                print(f"  [attempt {attempt + 1}/3 hit rate limit — waiting {wait}s]")
                await asyncio.sleep(wait)
                if attempt == 2:
                    messages.append(f"run_researcher raised after 3 attempts: {exc_str[:200]}")
                    return False, messages
            else:
                messages.append(f"run_researcher raised: {exc_str[:300]}")
                return False, messages

    if brief is None:
        messages.append("run_researcher returned no result after retries")
        return False, messages

    # Print what we got
    print(f"  Summary        : {brief.summary[:120]}...")
    print(f"  Hook           : {brief.suggested_hook}")
    print(f"  Sources ({len(brief.sources)})     : {brief.sources[:2]}")
    print(f"  Recent news    : {len(brief.recent_news)} item(s)")
    print(f"  Pain points    : {len(brief.pain_points)} item(s)")
    print(f"  Confidence     : {decision.confidence:.2f}")
    print(f"  Input hash     : {decision.input_hash}")

    brief_failures = check_research_brief(brief, company)
    decision_failures = check_agent_decision(decision, expected_hash, company)
    all_failures = brief_failures + decision_failures

    if all_failures:
        for f in all_failures:
            messages.append(f"  FAIL: {f}")
        return False, messages

    messages.append("  PASS")
    return True, messages


async def main() -> None:
    passed = 0
    failed = 0
    results: list[tuple[str, bool, list[str]]] = []

    for i, lead in enumerate(LEADS):
        lead_id = str(uuid4())
        ok, messages = await run_one(lead, lead_id)

        if ok:
            passed += 1
        else:
            failed += 1
        results.append((lead["company"], ok, messages))

        # Pause between runs — DDG rate limiting + Gemini 2.5-flash free tier (5 RPM)
        if i < len(LEADS) - 1:
            print(f"\n  [pausing 90s before next company to respect rate limits]")
            await asyncio.sleep(90)

    # ─── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  RESEARCHER SMOKE TEST — {passed}/{len(LEADS)} passed")
    print(f"{'═' * 60}")

    for company, ok, messages in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {company}")
        for msg in messages:
            if not msg.strip().startswith("PASS"):
                print(f"         {msg}")

    print(f"{'═' * 60}\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
