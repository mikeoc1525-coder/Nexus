"""
tests/test_llm_fallback.py — FallbackLlm integration test

Verifies that FallbackLlm actually fires when the primary provider fails with
a transient error, and that the structured log line appears in output.

Test design:
  - Primary: LiteLlm pointed at http://127.0.0.1:9999 (connection refused)
    This raises litellm.APIConnectionError — a valid fallback trigger.
  - Fallback: Real Groq (GROQ_API_KEY from .env)
  - Asserts: (a) response is non-empty, (b) structured WARNING log line
    containing LLM_FALLBACK_TRIGGERED was emitted

Why connection-refused instead of bad API key:
  - AuthenticationError (401) is a hard error per our fallback policy —
    a bad key on production is a deployment bug, not a transient failure.
    Falling back on auth errors would hide misconfiguration.
  - Connection-refused is a genuine transient scenario (provider unreachable)
    and is fully deterministic in tests.

Budget: 0 primary calls (connection refused), 1 Groq fallback call.

Usage:
    python tests/test_llm_fallback.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()


# ─── Log capture ──────────────────────────────────────────────────────────────

class _FallbackLogCapture(logging.Handler):
    """Captures WARNING log records from core.llm_factory."""

    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord):
        self.records.append(record)

    def triggered_messages(self) -> list[str]:
        return [
            r.getMessage()
            for r in self.records
            if "LLM_FALLBACK_TRIGGERED" in r.getMessage()
        ]


# ─── Test ─────────────────────────────────────────────────────────────────────

async def test_fallback_fires_on_connection_refused():
    """Primary points at 127.0.0.1:9999 (refused). Groq fallback must serve the call."""
    import warnings
    from core.llm_factory import FallbackLlm
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.models.llm_request import LlmRequest
    from google.genai.types import Content, Part

    # Attach log capture before building the model
    capture = _FallbackLogCapture()
    factory_logger = logging.getLogger("core.llm_factory")
    factory_logger.addHandler(capture)
    factory_logger.setLevel(logging.WARNING)

    # Broken primary: openai provider, bad base URL → connection refused
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        broken_primary = LiteLlm(
            model="openai/gpt-4o",
            api_base="http://127.0.0.1:9999",
            api_key="sk-fake-key-not-used",
        )
        groq_fallback = LiteLlm(model="groq/llama-3.3-70b-versatile")

    llm = FallbackLlm(primary=broken_primary, fallback=groq_fallback)

    print(f"  Primary : {broken_primary.model} @ http://127.0.0.1:9999 (connection refused)")
    print(f"  Fallback: {groq_fallback.model}")

    request = LlmRequest(
        model=llm.model,
        contents=[Content(role="user", parts=[Part(text="Reply with exactly the word: PONG")])],
    )

    chunks = []
    async for response in llm.generate_content_async(request, stream=False):
        if response.content and response.content.parts:
            for part in response.content.parts:
                if part.text:
                    chunks.append(part.text)

    response_text = "".join(chunks).strip()
    print(f"  Response: {response_text[:80]!r}")

    # Check 1: response is non-empty
    assert response_text, "FallbackLlm returned empty response"

    # Check 2: structured log line was emitted
    triggered = capture.triggered_messages()
    if not triggered:
        raise AssertionError(
            "LLM_FALLBACK_TRIGGERED log line was NOT emitted — "
            "fallback fired silently or did not fire at all"
        )

    print(f"\n  Structured fallback log line:")
    for msg in triggered:
        print(f"  >>> {msg}")

    # Verify log line contains the expected fields
    log_line = triggered[0]
    assert "primary=" in log_line, f"Missing primary= in log: {log_line}"
    assert "error=" in log_line, f"Missing error= in log: {log_line}"
    assert "fallback=" in log_line, f"Missing fallback= in log: {log_line}"
    assert "groq" in log_line.lower(), f"Fallback provider not in log: {log_line}"

    factory_logger.removeHandler(capture)
    return response_text


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    logging.basicConfig(
        level=logging.WARNING,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )

    print(f"\n{'─' * 60}")
    print("  LLM FALLBACK INTEGRATION TEST")
    print(f"{'─' * 60}\n")

    failures = []

    for attempt in range(3):
        try:
            response = await test_fallback_fires_on_connection_refused()
            break
        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "503" in exc_str:
                wait = 60 * (attempt + 1)
                print(f"  [attempt {attempt + 1}/3 hit rate limit — waiting {wait}s]")
                await asyncio.sleep(wait)
                if attempt == 2:
                    failures.append(f"Rate limited after 3 attempts: {exc_str[:200]}")
            else:
                failures.append(str(exc))
                break

    print(f"\n{'═' * 60}")
    if failures:
        print("  LLM FALLBACK TEST — FAIL")
        for f in failures:
            print(f"  ✗ {f}")
    else:
        print("  LLM FALLBACK TEST — PASS")
    print(f"{'═' * 60}\n")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
