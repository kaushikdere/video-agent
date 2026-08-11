"""Higgsfield MCP video provider adapter."""
from __future__ import annotations

import hashlib
import time
import uuid

import httpx
import structlog
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from video_agent.config import get_settings
from video_agent.providers.base import (
    AbstractVideoProvider,
    GenerationRequest,
    GenerationResult,
)

logger = structlog.get_logger(__name__)
settings = get_settings()

# Cost estimate: $0.10 per 10s clip (placeholder — adjust to actual pricing)
_COST_PER_CLIP_USD = 0.10


class HiggsFieldProvider(AbstractVideoProvider):
    """
    Higgsfield MCP adapter.

    Prompt construction follows: bible + beat action + camera move.
    Frame chaining: previous_frame_url is passed as image conditioning.
    Idempotency key on every work-creating POST.
    """

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.higgsfield_base_url,
            headers={
                "Authorization": f"Bearer {settings.higgsfield_api_key}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

    @property
    def name(self) -> str:
        return "higgsfield"

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        log = logger.bind(
            provider=self.name,
            job_id=request.job_id,
            shot=request.shot_index,
        )

        # Idempotency key — deterministic per job+shot
        idempotency_key = hashlib.sha256(
            f"{request.job_id}:shot:{request.shot_index}".encode()
        ).hexdigest()[:32]

        payload: dict = {
            "prompt": request.prompt,
            "duration": request.duration_seconds,
            "idempotency_key": idempotency_key,
        }
        if request.seed:
            payload["seed"] = request.seed
        if request.previous_frame_url:
            payload["conditioning_image_url"] = request.previous_frame_url

        log.info("higgsfield_generate_start", prompt_len=len(request.prompt))
        t0 = time.perf_counter()

        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(httpx.HTTPStatusError),
                stop=stop_after_attempt(3),
                wait=wait_exponential_jitter(initial=2, max=30),
                reraise=True,
            ):
                with attempt:
                    resp = await self._client.post("/generations", json=payload)
                    if resp.status_code in (429, 503):
                        resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            log.error("higgsfield_http_error", status=exc.response.status_code)
            raise RuntimeError(f"Higgsfield HTTP {exc.response.status_code}") from exc

        latency = time.perf_counter() - t0
        data = resp.json()

        return GenerationResult(
            clip_url=data["clip_url"],
            final_frame_url=data.get("final_frame_url", ""),
            thumbnail_url=data.get("thumbnail_url", ""),
            seed=data.get("seed", 0),
            model=data.get("model", "higgsfield-v1"),
            cost_usd=_COST_PER_CLIP_USD,
            latency_seconds=latency,
            raw_response=data,
        )

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self._client.aclose()
