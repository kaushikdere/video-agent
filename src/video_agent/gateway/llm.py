"""LiteLLM gateway — single egress for every model call via logical aliases."""
from __future__ import annotations

import time
from typing import Any

import structlog
from litellm import acompletion, completion_cost
from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    ServiceUnavailableError,
)
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from video_agent.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

# ─── Model alias → LiteLLM route ─────────────────────────────────────────────
# Prefer Gemini when available; fall back to OpenAI or Anthropic.
def _build_alias_map() -> dict[str, str]:
    if settings.gemini_api_key:
        return {
            "reasoning-high": "gemini/gemini-flash-latest",
            "reasoning-fast": "gemini/gemini-flash-lite-latest",
            "vision-default": "gemini/gemini-flash-latest",
            "embed-default": "gemini/text-embedding-004",
            "realtime-voice": "openai/gpt-4o-realtime-preview",
        }
    if settings.anthropic_api_key:
        return {
            "reasoning-high": "anthropic/claude-3-5-sonnet-20241022",
            "reasoning-fast": "anthropic/claude-3-haiku-20240307",
            "vision-default": "anthropic/claude-3-5-sonnet-20241022",
            "embed-default": "openai/text-embedding-3-small",
            "realtime-voice": "openai/gpt-4o-realtime-preview",
        }
    # Default to OpenAI
    return {
        "reasoning-high": "openai/gpt-4o",
        "reasoning-fast": "openai/gpt-4o-mini",
        "vision-default": "openai/gpt-4o",
        "embed-default": "openai/text-embedding-3-small",
        "realtime-voice": "openai/gpt-4o-realtime-preview",
    }


ALIAS_MAP: dict[str, str] = _build_alias_map()

_RETRYABLE = (RateLimitError, ServiceUnavailableError)


FALLBACK_ORDER: dict[str, list[str]] = {
    "reasoning-high": [
        "gemini/gemini-flash-latest",
        "anthropic/claude-3-5-sonnet-20241022",
        "openai/gpt-4o"
    ],
    "reasoning-fast": [
        "gemini/gemini-flash-lite-latest",
        "anthropic/claude-3-haiku-20240307",
        "openai/gpt-4o-mini"
    ],
    "vision-default": [
        "gemini/gemini-flash-latest",
        "anthropic/claude-3-5-sonnet-20241022",
        "openai/gpt-4o"
    ],
    "embed-default": [
        "gemini/text-embedding-004",
        "openai/text-embedding-3-small"
    ],
    "realtime-voice": [
        "openai/gpt-4o-realtime-preview"
    ]
}


async def llm_call(
    alias: str,
    messages: list[dict[str, Any]],
    *,
    response_format: dict[str, Any] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    trace_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Async LLM call via LiteLLM.
    Supports provider failover/fallback within the alias group.

    Returns:
        {"content": str, "model": str, "cost_usd": float, "tokens": int}

    Raises:
        RuntimeError if all fallback models fail.
    """
    models_to_try = FALLBACK_ORDER.get(alias, [alias])
    errors = []

    for model in models_to_try:
        # Check key availability for the provider
        api_key = None
        if model.startswith("gemini/"):
            if not settings.gemini_api_key:
                continue
            api_key = settings.gemini_api_key
        elif model.startswith("anthropic/"):
            if not settings.anthropic_api_key:
                continue
            api_key = settings.anthropic_api_key
        elif model.startswith("openai/"):
            if not settings.openai_api_key:
                continue
            api_key = settings.openai_api_key

        log = logger.bind(alias=alias, model=model, trace_id=trace_id)
        log.info("llm_call_attempting")

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_key": api_key,
        }
        if response_format:
            kwargs["response_format"] = response_format

        if trace_id:
            kwargs.setdefault("metadata", {})
            kwargs["metadata"]["trace_id"] = trace_id
            if metadata:
                kwargs["metadata"].update(metadata)

        start = time.perf_counter()
        response: Any = None
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(_RETRYABLE),
                stop=stop_after_attempt(3),
                wait=wait_exponential_jitter(initial=1, max=10),
                reraise=True,
            ):
                with attempt:
                    response = await acompletion(**kwargs)
            
            # Succeeded! Process response
            latency = time.perf_counter() - start
            cost = 0.0
            try:
                cost = completion_cost(completion_response=response)
            except Exception:
                pass

            content = getattr(response.choices[0].message, "content", "") if response and hasattr(response, "choices") else ""
            content = content or ""
            usage = getattr(response, "usage", None) if response else None
            tokens = getattr(usage, "total_tokens", 0) if usage else 0
            model_name = getattr(response, "model", alias) if response else alias

            log.info("llm_ok", latency=round(latency, 2), tokens=tokens, cost=round(cost, 6))

            return {
                "content": content,
                "model": model_name,
                "cost_usd": cost,
                "tokens": tokens,
            }

        except AuthenticationError as exc:
            log.error("llm_auth_error", error=str(exc))
            errors.append(f"{model}: Auth failed ({exc})")
        except BadRequestError as exc:
            log.error("llm_bad_request", error=str(exc))
            errors.append(f"{model}: Bad request ({exc})")
        except Exception as exc:
            log.warning("llm_model_failed_trying_next_fallback", error=str(exc))
            errors.append(f"{model}: {type(exc).__name__} ({exc})")

    # If we get here, all attempts failed
    raise RuntimeError(f"All fallback models failed for alias '{alias}': {'; '.join(errors)}")

