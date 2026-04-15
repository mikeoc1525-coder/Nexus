"""
core/llm_factory.py — Nexus LLM provider factory

Constructs the correct LLM instance for ADK agents based on .env config.
All agents call build_llm() or build_llm_with_fallback() — never os.getenv directly.

Provider routing:
  - LLM_PROVIDER=gemini  → plain string "gemini-2.5-flash" (native ADK path)
                           unless fallback is configured, in which case both
                           providers go through LiteLLM for a uniform BaseLlm
                           interface (see Decision 1 in feat/llm-provider-abstraction plan)
  - LLM_PROVIDER=groq    → LiteLlm(model="groq/llama-3.3-70b-versatile")
  - LLM_PROVIDER=anthropic → LiteLlm(model="anthropic/claude-sonnet-4-6")
  - LLM_PROVIDER=openai  → LiteLlm(model="openai/gpt-4o")

Fallback:
  - Set LLM_FALLBACK_PROVIDER + LLM_FALLBACK_MODEL + the provider's API key
  - FallbackLlm wraps primary + fallback as BaseLlm instances
  - Fallback triggers ONLY on transient provider-side failures:
      RateLimitError, ServiceUnavailableError, APIConnectionError,
      google.api_core ResourceExhausted / ServiceUnavailable
  - Hard errors (AuthenticationError, BadRequestError, NotFoundError) are
    raised immediately — fallback would hide real bugs

Backward compatibility:
  - Existing .env with only GOOGLE_API_KEY + LLM_MODEL=gemini-2.5-flash works
    with zero changes — build_llm() returns the same string it always did
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import AsyncGenerator, Union

from google.adk.models.base_llm import BaseLlm
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from typing_extensions import override

logger = logging.getLogger(__name__)

# ─── Error classification ─────────────────────────────────────────────────────

# LiteLLM exception types — imported lazily to avoid hard dependency when
# litellm is not installed (e.g., if someone runs without extensions).
def _get_litellm_transient_errors() -> tuple[type, ...]:
    try:
        import litellm
        return (
            litellm.RateLimitError,
            litellm.ServiceUnavailableError,
            litellm.APIConnectionError,
            # LiteLLM wraps connection-refused and DNS failures as
            # InternalServerError (not APIConnectionError) — tested empirically.
            # 500s from LLM providers are typically transient overload, not bugs.
            litellm.InternalServerError,
        )
    except ImportError:
        return ()


def _get_google_transient_errors() -> tuple[type, ...]:
    try:
        from google.api_core import exceptions as gexc
        return (
            gexc.ResourceExhausted,
            gexc.ServiceUnavailable,
        )
    except ImportError:
        return ()


# ─── Provider → LiteLLM model string mapping ─────────────────────────────────

# Maps LLM_PROVIDER env values to LiteLLM model string prefixes.
# When LLM_MODEL is already a fully-qualified LiteLLM string (e.g. "groq/llama-3.3-70b-versatile"),
# it is used as-is. When it is a bare model name (e.g. "gemini-2.5-flash"), the
# provider prefix is prepended.
_PROVIDER_PREFIX: dict[str, str] = {
    "gemini": "gemini",
    "groq": "groq",
    "anthropic": "anthropic",
    "openai": "openai",
    "ollama": "ollama",
    "mistral": "mistral",
    "deepseek": "deepseek",
}


def _litellm_model_string(provider: str, model: str) -> str:
    """Return a fully-qualified LiteLLM model string.

    If model already contains a '/' it is used as-is (already qualified).
    Otherwise prepend the provider prefix: e.g. "gemini-2.5-flash" → "gemini/gemini-2.5-flash".
    """
    if "/" in model:
        return model
    prefix = _PROVIDER_PREFIX.get(provider.lower(), provider.lower())
    return f"{prefix}/{model}"


# ─── FallbackLlm ─────────────────────────────────────────────────────────────

class FallbackLlm(BaseLlm):
    """A BaseLlm that transparently falls back to a secondary provider.

    On each generate_content_async() call:
      1. Try primary.
      2. If a transient provider-side error occurs, emit a structured WARNING
         log and retry the same request against fallback.
      3. Hard errors (auth, bad request, not found) propagate immediately.

    The structured log line is intentionally loud — it should be visible in
    the terminal during smoke tests so fallback is observable, not silent.
    Future sessions can pipe this into AgentDecision.challenge_reasoning.
    """

    primary: BaseLlm
    fallback: BaseLlm

    # Override model to be a computed descriptor — required by BaseLlm but
    # not meaningful for FallbackLlm (it's a wrapper, not a single model).
    model: str = "fallback-wrapper"

    @override
    async def generate_content_async(
        self,
        llm_request: LlmRequest,
        stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        transient = _get_litellm_transient_errors() + _get_google_transient_errors()

        # LiteLlm reads effective_model = llm_request.model or self.model.
        # FallbackLlm.model is "fallback-wrapper" (a placeholder), so we must
        # replace the request's model field with each sub-LLM's own model
        # before delegating, or LiteLlm will try to route "fallback-wrapper".
        primary_request = llm_request.model_copy(update={"model": self.primary.model})

        try:
            async for chunk in self.primary.generate_content_async(primary_request, stream):
                yield chunk
            return
        except Exception as exc:
            if transient and isinstance(exc, transient):
                # Structured fallback log — visible in terminal, parseable later
                logger.warning(
                    "LLM_FALLBACK_TRIGGERED primary=%s error=%s(%s) fallback=%s",
                    self.primary.model,
                    type(exc).__name__,
                    str(exc)[:120],
                    self.fallback.model,
                )
            else:
                raise  # Hard error — surface immediately

        # Fallback path
        fallback_request = llm_request.model_copy(update={"model": self.fallback.model})
        async for chunk in self.fallback.generate_content_async(fallback_request, stream):
            yield chunk

    @classmethod
    @override
    def supported_models(cls) -> list[str]:
        # FallbackLlm is constructed directly — never registered by model string
        return []


# ─── Public API ──────────────────────────────────────────────────────────────

def build_llm() -> Union[str, BaseLlm]:
    """Build the primary LLM instance from .env config.

    Returns a plain string for Gemini (native ADK path, no overhead) when no
    fallback is configured. Returns a LiteLlm instance for all other providers,
    and also for Gemini when LLM_FALLBACK_PROVIDER is set (uniform interface).

    Usage:
        agent = LlmAgent(model=build_llm(), ...)
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    fallback_provider = os.getenv("LLM_FALLBACK_PROVIDER", "").lower()

    # Native Gemini string path — only when no fallback is configured
    if provider == "gemini" and not fallback_provider:
        return model  # plain string, native ADK route

    # LiteLLM path for all other providers (or Gemini with fallback active)
    litellm_model = _litellm_model_string(provider, model)
    # Suppress ADK's "don't use Gemini via LiteLLM" warning — intentional here
    # because fallback requires a uniform BaseLlm interface (see plan Decision 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LiteLlm(model=litellm_model)


def build_llm_with_fallback() -> Union[str, BaseLlm]:
    """Build the primary LLM, wrapping with FallbackLlm if fallback is configured.

    When LLM_FALLBACK_PROVIDER is not set, behaves identically to build_llm().
    When configured, returns a FallbackLlm that wraps both providers.

    Usage:
        agent = LlmAgent(model=build_llm_with_fallback(), ...)
    """
    fallback_provider = os.getenv("LLM_FALLBACK_PROVIDER", "").lower()
    if not fallback_provider:
        return build_llm()

    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    fallback_model = os.getenv("LLM_FALLBACK_MODEL", "groq/llama-3.3-70b-versatile")

    primary_model_str = _litellm_model_string(provider, model)
    fallback_model_str = _litellm_model_string(fallback_provider, fallback_model)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        primary = LiteLlm(model=primary_model_str)
        fallback = LiteLlm(model=fallback_model_str)

    logger.info(
        "LLM_PROVIDER_CONFIG primary=%s fallback=%s",
        primary_model_str,
        fallback_model_str,
    )
    return FallbackLlm(primary=primary, fallback=fallback)
