"""
tests/test_llm_factory.py — LLM factory unit test

Verifies:
  1. build_llm() returns a plain string for Gemini (no fallback configured)
  2. build_llm() returns a LiteLlm instance for Groq
  3. build_llm_with_fallback() returns a FallbackLlm when fallback is configured
  4. Each provider can actually complete a one-shot "say hi" — real API call

Budget: 1 Gemini call + 1 Groq call.

Usage:
    python tests/test_llm_factory.py
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

from core.llm_factory import FallbackLlm, build_llm, build_llm_with_fallback
from google.adk.models.base_llm import BaseLlm
from google.adk.models.lite_llm import LiteLlm

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(levelname)s %(name)s: %(message)s",
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

async def one_shot(llm: BaseLlm | str, prompt: str) -> str:
    """Run a single completion against any BaseLlm or plain-string model name."""
    import warnings
    from google.adk.models.llm_request import LlmRequest
    from google.genai.types import Content, Part

    if isinstance(llm, str):
        # build_llm() returns a plain string for native Gemini. For the live
        # call we need a BaseLlm instance — use LiteLlm with gemini/ prefix.
        # The structural test already verified the string return; this is just
        # checking API reachability.
        qualified = llm if "/" in llm else f"gemini/{llm}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            llm = LiteLlm(model=qualified)

    request = LlmRequest(
        model=llm.model,
        contents=[Content(role="user", parts=[Part(text=prompt)])],
    )
    chunks = []
    async for response in llm.generate_content_async(request, stream=False):
        if response.content and response.content.parts:
            for part in response.content.parts:
                if part.text:
                    chunks.append(part.text)
    return "".join(chunks)


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_gemini_returns_string_without_fallback():
    """build_llm() should return a plain string when LLM_PROVIDER=gemini and no fallback."""
    orig_fallback = os.environ.pop("LLM_FALLBACK_PROVIDER", None)
    try:
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["LLM_MODEL"] = "gemini-2.5-flash"
        result = build_llm()
        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert result == "gemini-2.5-flash"
        print("  [PASS] build_llm(gemini, no fallback) → plain string")
    finally:
        if orig_fallback is not None:
            os.environ["LLM_FALLBACK_PROVIDER"] = orig_fallback
        else:
            os.environ.pop("LLM_FALLBACK_PROVIDER", None)


def test_groq_returns_litellm():
    """build_llm() should return a LiteLlm instance for groq provider."""
    orig_provider = os.environ.get("LLM_PROVIDER")
    orig_model = os.environ.get("LLM_MODEL")
    orig_fallback = os.environ.pop("LLM_FALLBACK_PROVIDER", None)
    try:
        os.environ["LLM_PROVIDER"] = "groq"
        os.environ["LLM_MODEL"] = "groq/llama-3.3-70b-versatile"
        result = build_llm()
        assert isinstance(result, LiteLlm), f"Expected LiteLlm, got {type(result)}"
        assert result.model == "groq/llama-3.3-70b-versatile"
        print("  [PASS] build_llm(groq) → LiteLlm instance")
    finally:
        if orig_provider is not None:
            os.environ["LLM_PROVIDER"] = orig_provider
        if orig_model is not None:
            os.environ["LLM_MODEL"] = orig_model
        if orig_fallback is not None:
            os.environ["LLM_FALLBACK_PROVIDER"] = orig_fallback


def test_fallback_wrapper_constructed():
    """build_llm_with_fallback() should return a FallbackLlm when fallback is configured."""
    orig = {k: os.environ.get(k) for k in ["LLM_PROVIDER", "LLM_MODEL", "LLM_FALLBACK_PROVIDER", "LLM_FALLBACK_MODEL"]}
    try:
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["LLM_MODEL"] = "gemini-2.5-flash"
        os.environ["LLM_FALLBACK_PROVIDER"] = "groq"
        os.environ["LLM_FALLBACK_MODEL"] = "groq/llama-3.3-70b-versatile"
        result = build_llm_with_fallback()
        assert isinstance(result, FallbackLlm), f"Expected FallbackLlm, got {type(result)}"
        assert isinstance(result.primary, LiteLlm)
        assert isinstance(result.fallback, LiteLlm)
        assert "gemini" in result.primary.model
        assert "groq" in result.fallback.model
        print(f"  [PASS] build_llm_with_fallback() → FallbackLlm(primary={result.primary.model}, fallback={result.fallback.model})")
    finally:
        for k, v in orig.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)


async def test_gemini_live():
    """One real Gemini API call — verify non-empty response."""
    os.environ["LLM_PROVIDER"] = "gemini"
    os.environ["LLM_MODEL"] = "gemini-2.5-flash"
    os.environ.pop("LLM_FALLBACK_PROVIDER", None)

    llm = build_llm()
    print(f"  Calling Gemini ({llm!r})...")
    response = await one_shot(llm, "Reply with exactly the word: PONG")
    print(f"  Response: {response.strip()[:80]!r}")
    assert response.strip(), "Gemini returned empty response"
    assert len(response.strip()) > 0
    print("  [PASS] Gemini live call → non-empty")


async def test_groq_live():
    """One real Groq API call — verify non-empty response."""
    orig_provider = os.environ.get("LLM_PROVIDER")
    orig_model = os.environ.get("LLM_MODEL")
    orig_fallback = os.environ.pop("LLM_FALLBACK_PROVIDER", None)
    try:
        os.environ["LLM_PROVIDER"] = "groq"
        os.environ["LLM_MODEL"] = "groq/llama-3.3-70b-versatile"
        llm = build_llm()
        print(f"  Calling Groq ({llm.model})...")
        response = await one_shot(llm, "Reply with exactly the word: PONG")
        print(f"  Response: {response.strip()[:80]!r}")
        assert response.strip(), "Groq returned empty response"
        print("  [PASS] Groq live call → non-empty")
    finally:
        if orig_provider is not None:
            os.environ["LLM_PROVIDER"] = orig_provider
        if orig_model is not None:
            os.environ["LLM_MODEL"] = orig_model
        if orig_fallback is not None:
            os.environ["LLM_FALLBACK_PROVIDER"] = orig_fallback


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main():
    print(f"\n{'─' * 60}")
    print("  LLM FACTORY UNIT TEST")
    print(f"{'─' * 60}")

    failures = []

    # Structural tests — no API calls
    print("\n  [Structural — no API calls]")
    try:
        test_gemini_returns_string_without_fallback()
    except Exception as e:
        failures.append(f"test_gemini_returns_string: {e}")
        print(f"  [FAIL] {e}")

    try:
        test_groq_returns_litellm()
    except Exception as e:
        failures.append(f"test_groq_returns_litellm: {e}")
        print(f"  [FAIL] {e}")

    try:
        test_fallback_wrapper_constructed()
    except Exception as e:
        failures.append(f"test_fallback_wrapper_constructed: {e}")
        print(f"  [FAIL] {e}")

    # Live tests — real API calls
    print("\n  [Live — API calls]")
    try:
        await test_gemini_live()
    except Exception as e:
        failures.append(f"test_gemini_live: {e}")
        print(f"  [FAIL] {e}")

    try:
        await test_groq_live()
    except Exception as e:
        failures.append(f"test_groq_live: {e}")
        print(f"  [FAIL] {e}")

    print(f"\n{'═' * 60}")
    if failures:
        print(f"  LLM FACTORY TEST — {len(failures)} FAILURE(S)")
        for f in failures:
            print(f"  ✗ {f}")
        print(f"{'═' * 60}\n")
        sys.exit(1)
    else:
        print("  LLM FACTORY TEST — ALL PASS")
        print(f"{'═' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
