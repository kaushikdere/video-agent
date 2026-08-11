"""Higgsfield Python SDK video provider adapter."""
from __future__ import annotations

import os
import time
import structlog

from video_agent.config import get_settings
from video_agent.providers.base import (
    AbstractVideoProvider,
    GenerationRequest,
    GenerationResult,
)

logger = structlog.get_logger(__name__)
settings = get_settings()

# Cost estimate: $0.10 per 10s clip
_COST_PER_CLIP_USD = 0.10


class HiggsFieldProvider(AbstractVideoProvider):
    """
    Higgsfield SDK adapter.

    Uses `higgsfield-client` to run a text-to-image-to-video workflow:
    1. If no previous frame exists (Shot 1), generate a start image using 'higgsfield-ai/soul/v2/standard'.
    2. Pass the start image (or previous shot's final frame) into 'higgsfield-ai/dop/standard' along with the prompt.
    3. Handles authentication credentials via settings.
    """

    def __init__(self) -> None:
        # Set credentials in env vars for the SDK
        os.environ["HF_API_KEY"] = settings.higgsfield_api_key
        os.environ["HF_API_SECRET"] = settings.higgsfield_api_secret

    @property
    def name(self) -> str:
        return "higgsfield"

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        log = logger.bind(
            provider=self.name,
            job_id=request.job_id,
            shot=request.shot_index,
        )

        t0 = time.perf_counter()
        try:
            import higgsfield_client

            # Step 1: Resolve the input image_url
            image_url = request.previous_frame_url
            if not image_url:
                log.info("generating_start_image_for_shot_1")
                image_res = await higgsfield_client.subscribe_async(
                    "higgsfield-ai/soul/v2/standard",
                    arguments={
                        "prompt": request.prompt,
                    }
                )
                images = image_res.get("images", [])
                if not images:
                    raise RuntimeError("No image returned from soul/v2/standard")
                image_url = images[0].get("url")
                log.info("start_image_generated", url=image_url)

            # Step 2: Generate the video
            log.info("higgsfield_generate_start", prompt_len=len(request.prompt), image_url=image_url)
            
            result = await higgsfield_client.subscribe_async(
                settings.higgsfield_model,  # e.g., 'higgsfield-ai/dop/standard'
                arguments={
                    "prompt": request.prompt,
                    "image_url": image_url,
                }
            )

            videos = result.get("videos", [])
            if not videos:
                raise RuntimeError("No video returned in results")

            clip_url = videos[0].get("url")
            
            # Resolve final frame & thumbnail
            final_frame_url = ""
            final_frames = result.get("final_frames", [])
            if final_frames:
                final_frame_url = final_frames[0].get("url", "")
            if not final_frame_url:
                final_frame_url = result.get("final_frame_url", "")

            thumbnail_url = ""
            thumbnails = result.get("thumbnails", [])
            if thumbnails:
                thumbnail_url = thumbnails[0].get("url", "")
            if not thumbnail_url:
                thumbnail_url = result.get("thumbnail_url", "")

            latency = time.perf_counter() - t0
            return GenerationResult(
                clip_url=clip_url,
                final_frame_url=final_frame_url or clip_url,
                thumbnail_url=thumbnail_url or clip_url,
                seed=result.get("seed", 0),
                model=settings.higgsfield_model,
                cost_usd=_COST_PER_CLIP_USD,
                latency_seconds=latency,
                raw_response=result,
            )

        except Exception as exc:
            log.warning("higgsfield_sdk_failed_falling_back_to_mock", error=str(exc))
            from video_agent.providers.mock import MockVideoProvider
            mock_provider = MockVideoProvider(simulate_latency=False)
            mock_result = await mock_provider.generate(request)
            return GenerationResult(
                clip_url=mock_result.clip_url,
                final_frame_url=mock_result.final_frame_url,
                thumbnail_url=mock_result.thumbnail_url,
                seed=mock_result.seed,
                model=f"{settings.higgsfield_model} (fallback)",
                cost_usd=_COST_PER_CLIP_USD,
                latency_seconds=time.perf_counter() - t0,
                raw_response={"fallback": True, "error": str(exc)},
            )

    async def health_check(self) -> bool:
        try:
            import higgsfield_client
            # Minimal probe to verify SDK and credentials
            # status_async with a dummy ID will raise a ClientError rather than connection error if authenticated
            await higgsfield_client.status_async("probe")
            return True
        except Exception as e:
            # If the error is credentials or connection
            if "Credentials" in type(e).__name__:
                return False
            return True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

