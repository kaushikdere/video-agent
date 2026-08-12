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

        # Try to generate a real tiny MP4 with ffmpeg / OpenCV
        # Always use absolute paths — cv2.VideoWriter fails silently with relative paths on macOS
        storage_path = os.path.abspath(settings.local_storage_path)
        os.makedirs(storage_path, exist_ok=True)

        clip_filename = f"{request.job_id}_shot_{request.shot_index}.mp4"
        frame_filename = f"{request.job_id}_frame_{request.shot_index}.png"
        thumb_filename = f"{request.job_id}_thumb_{request.shot_index}.jpg"
        clip_path = os.path.join(storage_path, clip_filename)
        frame_path = os.path.join(storage_path, frame_filename)
        thumb_path = os.path.join(storage_path, thumb_filename)

        # Generate with ffmpeg or OpenCV fallback
        hex_colour = colour.lstrip("#")
        r, g, b = int(hex_colour[0:2], 16), int(hex_colour[2:4], 16), int(hex_colour[4:6], 16)
        beat_labels = ["SETUP", "DEVELOPMENT", "TURN", "RESOLUTION"]
        label = beat_labels[request.shot_index % 4]

        generated = False
        try:
            import subprocess
            res = subprocess.run(
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
            if res.returncode == 0:
                subprocess.run(
                    ["ffmpeg", "-y", "-sseof", "-0.1", "-i", clip_path, "-frames:v", "1", frame_path],
                    capture_output=True, timeout=10,
                )
                subprocess.run(
                    ["ffmpeg", "-y", "-i", clip_path, "-frames:v", "1", thumb_path],
                    capture_output=True, timeout=10,
                )
                generated = True
        except Exception as ffmpeg_exc:
            log.warning("ffmpeg_unavailable", error=str(ffmpeg_exc))

        if not generated:
            try:
                import cv2
                import numpy as np

                width, height, fps = 1280, 720, 24
                total_frames = int(fps * request.duration_seconds)
                fourcc = getattr(cv2, "VideoWriter_fourcc", lambda *a: 0)(*"mp4v")  # type: ignore[misc]
                # Use absolute path — cv2.VideoWriter silently fails with relative paths on macOS
                out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

                if not out.isOpened():
                    raise RuntimeError(f"cv2.VideoWriter could not open: {clip_path}")

                last_frame: "np.ndarray | None" = None
                first_frame: "np.ndarray | None" = None

                for f_idx in range(total_frames):
                    # Animated gradient — colours shift from dark to vivid
                    shift = int((f_idx / total_frames) * 60)
                    bg = np.zeros((height, width, 3), dtype=np.uint8)
                    bg[:, :, 0] = max(0, min(255, b + shift))        # B channel
                    bg[:, :, 1] = max(0, min(255, g + (shift // 2))) # G channel
                    bg[:, :, 2] = max(0, min(255, r + (shift // 3))) # R channel

                    title = f"Shot {request.shot_index + 1}: {label}"
                    cv2.putText(bg, title, (200, 320), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3, cv2.LINE_AA)
                    cv2.putText(bg, f"Model: {settings.higgsfield_model}", (280, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA)
                    cv2.putText(bg, f"Duration: {request.duration_seconds}s | Frame: {f_idx+1}/{total_frames}", (260, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160), 1, cv2.LINE_AA)

                    if first_frame is None:
                        first_frame = bg.copy()
                    last_frame = bg.copy()
                    out.write(bg)

                out.release()

                # Verify the file was actually written
                if not os.path.exists(clip_path) or os.path.getsize(clip_path) < 1000:
                    raise RuntimeError(f"cv2 wrote empty/missing file: {clip_path}")

                if last_frame is not None:
                    cv2.imwrite(frame_path, last_frame)
                if first_frame is not None:
                    cv2.imwrite(thumb_path, first_frame)
                log.info("opencv_clip_generated", path=clip_path, size=os.path.getsize(clip_path))
                generated = True
            except Exception as cv_exc:
                log.warning("opencv_fallback_failed", error=str(cv_exc))

        if generated:
            clip_url = f"file://{clip_path}"
            frame_url = f"file://{frame_path}"
            thumb_url = f"file://{thumb_path}"
        else:
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
