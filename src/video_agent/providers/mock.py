"""Mock video provider — generates coloured placeholder clips without real API calls.

Used for:
- Tests and CI
- Demo recording
- Development when no Higgsfield key is available
"""
from __future__ import annotations

import asyncio
import os
import random
import time
import uuid

import structlog

from video_agent.config import get_settings
from video_agent.providers.base import (
    AbstractVideoProvider,
    GenerationRequest,
    GenerationResult,
)

logger = structlog.get_logger(__name__)
settings = get_settings()

# Placeholder colours per beat index
_PALETTE = [
    "#1a1a2e",  # setup — deep navy
    "#16213e",  # development — midnight blue
    "#0f3460",  # turn — cobalt
    "#533483",  # resolution — violet
]


class MockVideoProvider(AbstractVideoProvider):
    """
    Mock provider that simulates Higgsfield without real API calls.

    Generates tiny coloured MP4 files via ffmpeg (if available)
    or returns dummy URLs for pure unit-test contexts.
    """

    def __init__(self, simulate_latency: bool = True, fail_shot: int | None = None) -> None:
        self._simulate_latency = simulate_latency
        self._fail_shot = fail_shot  # force a failure on this shot index (for testing)

    @property
    def name(self) -> str:
        return "mock"

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        log = logger.bind(
            provider=self.name, job_id=request.job_id, shot=request.shot_index
        )
        log.info("mock_generate_start")
        t0 = time.perf_counter()

        if self._simulate_latency:
            await asyncio.sleep(random.uniform(0.3, 1.2))

        if self._fail_shot == request.shot_index:
            raise RuntimeError(f"Mock forced failure on shot {request.shot_index}")

        colour = _PALETTE[request.shot_index % len(_PALETTE)]
        seed = request.seed or random.randint(1, 999999)

        # Try to generate a real tiny MP4 with ffmpeg
        storage_path = settings.local_storage_path
        os.makedirs(storage_path, exist_ok=True)

        clip_filename = f"{request.job_id}_shot_{request.shot_index}.mp4"
        frame_filename = f"{request.job_id}_frame_{request.shot_index}.png"
        thumb_filename = f"{request.job_id}_thumb_{request.shot_index}.jpg"
        clip_path = os.path.join(storage_path, clip_filename)
        frame_path = os.path.join(storage_path, frame_filename)
        thumb_path = os.path.join(storage_path, thumb_filename)

        # Generate with ffmpeg (coloured solid video)
        hex_colour = colour.lstrip("#")
        r, g, b = int(hex_colour[0:2], 16), int(hex_colour[2:4], 16), int(hex_colour[4:6], 16)
        try:
            import subprocess

            # Beat label as overlay text
            beat_labels = ["SETUP", "DEVELOPMENT", "TURN", "RESOLUTION"]
            label = beat_labels[request.shot_index % 4]

            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "lavfi",
                    "-i", f"color=c=0x{hex_colour}:size=1280x720:rate=24:duration={request.duration_seconds}",
                    "-vf", f"drawtext=text='{label} — Shot {request.shot_index + 1}':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2",
                    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
                    clip_path,
                ],
                capture_output=True,
                timeout=30,
            )
            # Extract final frame
            subprocess.run(
                ["ffmpeg", "-y", "-sseof", "-0.1", "-i", clip_path, "-frames:v", "1", frame_path],
                capture_output=True, timeout=10,
            )
            # Thumbnail = first frame
            subprocess.run(
                ["ffmpeg", "-y", "-i", clip_path, "-frames:v", "1", thumb_path],
                capture_output=True, timeout=10,
            )
            clip_url = f"file://{clip_path}"
            frame_url = f"file://{frame_path}"
            thumb_url = f"file://{thumb_path}"
        except Exception as exc:
            log.warning("ffmpeg_unavailable", error=str(exc))
            clip_url = f"mock://clip/{request.job_id}/{request.shot_index}"
            frame_url = f"mock://frame/{request.job_id}/{request.shot_index}"
            thumb_url = f"mock://thumb/{request.job_id}/{request.shot_index}"

        latency = time.perf_counter() - t0
        log.info("mock_generate_ok", latency=round(latency, 2), colour=colour)

        return GenerationResult(
            clip_url=clip_url,
            final_frame_url=frame_url,
            thumbnail_url=thumb_url,
            seed=seed,
            model="mock-v1",
            cost_usd=0.0,
            latency_seconds=latency,
            raw_response={"colour": colour, "shot_index": request.shot_index},
        )

    async def health_check(self) -> bool:
        return True
