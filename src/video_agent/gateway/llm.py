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

    Returns:
        {"content": str, "model": str, "cost_usd": float, "tokens": int}

    Raises:
        RuntimeError on non-retryable failure.
    """
    model = ALIAS_MAP.get(alias, alias)
    log = logger.bind(alias=alias, model=model, trace_id=trace_id)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format
    # Select API key based on which provider the resolved model uses
    if model.startswith("gemini/") and settings.gemini_api_key:
        kwargs["api_key"] = settings.gemini_api_key
    elif model.startswith("anthropic/") and settings.anthropic_api_key:
        kwargs["api_key"] = settings.anthropic_api_key
    elif settings.openai_api_key:
        kwargs["api_key"] = settings.openai_api_key

    # Add Langfuse metadata via litellm
    if trace_id:
        kwargs.setdefault("metadata", {})
        kwargs["metadata"]["trace_id"] = trace_id
        if metadata:
            kwargs["metadata"].update(metadata)

    start = time.perf_counter()
    try:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(_RETRYABLE),
            stop=stop_after_attempt(3),
            wait=wait_exponential_jitter(initial=1, max=30),
            reraise=True,
        ):
            with attempt:
                response = await acompletion(**kwargs)
    except AuthenticationError as exc:
        log.error("llm_auth_error", error=str(exc))
        raise RuntimeError(f"LLM auth failed for alias '{alias}'") from exc
    except BadRequestError as exc:
        log.error("llm_bad_request", error=str(exc))
        raise RuntimeError(f"LLM bad request for alias '{alias}': {exc}") from exc
    except Exception as exc:
        log.error("llm_error", error=str(exc))
        raise

    latency = time.perf_counter() - start
    cost = 0.0
    try:
        cost = completion_cost(completion_response=response)
    except Exception:
        pass

    content = response.choices[0].message.content or ""
    tokens = response.usage.total_tokens if response.usage else 0

    log.info("llm_ok", latency=round(latency, 2), tokens=tokens, cost=round(cost, 6))

    return {
        "content": content,
        "model": response.model,
        "cost_usd": cost,
        "tokens": tokens,
    }
