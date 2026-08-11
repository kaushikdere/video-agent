"""Node 5: assemble — ffmpeg stitch, normalise, presigned URLs, deliver artifacts."""
from __future__ import annotations

import json
import os
import time

import structlog

from video_agent.agent.state import AgentState, DeliveryArtifacts, JobStatus
from video_agent.config import get_settings
from video_agent.utils.ffmpeg import stitch_clips, add_music_bed
from video_agent.utils.storage import upload_file, presigned_url

logger = structlog.get_logger(__name__)
settings = get_settings()


async def assemble_node(state: AgentState) -> dict:
    """
    LangGraph node: stitch clips → deliver artifacts.

    Never returns nothing:
    - If all shots ok → SUCCESS
    - If some shots ok → PARTIAL (best-so-far, flagged)
    - If no shots ok → FAILED
    """
    log = logger.bind(job_id=state["job_id"], node="assemble")
    log.info("node_start")
    t0 = time.perf_counter()

    budget = dict(state["budget"])
    budget["iterations"] += 1
    budget["elapsed_seconds"] = time.time() - budget["started_at"]

    shots = state["shots"]
    ok_shots = [s for s in shots if s["status"] in ("ok", "repaired")]

    # Determine final status
    if len(ok_shots) == 0:
        log.error("no_ok_shots")
        return {
            "status": JobStatus.FAILED,
            "error_message": "No shots passed QC",
            "budget": budget,
        }

    is_partial = len(ok_shots) < settings.shots_per_story
    log.info("assembling", ok_shots=len(ok_shots), partial=is_partial)

    # Collect clip paths/URLs in order
    ordered_clips = sorted(ok_shots, key=lambda s: s["shot_index"])
    clip_urls = [s["clip_url"] for s in ordered_clips]
    frame_urls = [s["final_frame_url"] for s in ordered_clips]
    individual_clip_uploads: list[str] = []
    stitched_url = ""
    thumbnail_url = ""

    storage_path = settings.local_storage_path
    os.makedirs(storage_path, exist_ok=True)

    # Upload individual clips
    for shot in ordered_clips:
        if shot["clip_url"].startswith("file://") or shot["clip_url"].startswith("mock://"):
            individual_clip_uploads.append(shot["clip_url"])
        else:
            uploaded = await upload_file(shot["clip_url"], f"{state['job_id']}/shot_{shot['shot_index']}.mp4")
            individual_clip_uploads.append(presigned_url(uploaded))

    # Stitch with ffmpeg
    stitched_path = os.path.join(storage_path, f"{state['job_id']}_stitched.mp4")
    try:
        local_clip_paths = [
            url.replace("file://", "") for url in clip_urls if url.startswith("file://")
        ]
        if local_clip_paths and len(local_clip_paths) == len(clip_urls):
            stitch_clips(local_clip_paths, stitched_path)
            stitched_url = f"file://{stitched_path}"
        else:
            stitched_url = clip_urls[0] if clip_urls else "mock://stitched"
    except Exception as exc:
        log.warning("stitch_failed", error=str(exc))
        stitched_url = clip_urls[0] if clip_urls else "mock://stitched"

    # Optional music bed
    if settings.app_env != "production":
        pass  # skip music in dev for speed

    # Save StoryPlan + ContinuityBible as JSON
    plan_path = os.path.join(storage_path, f"{state['job_id']}_story_plan.json")
    bible_path = os.path.join(storage_path, f"{state['job_id']}_continuity_bible.json")

    with open(plan_path, "w") as f:
        json.dump(state["story_plan"], f, indent=2)
    with open(bible_path, "w") as f:
        json.dump(state["continuity_bible"], f, indent=2)

    thumbnail_url = ordered_clips[0]["thumbnail_url"] if ordered_clips else ""

    artifacts = DeliveryArtifacts(
        stitched_mp4_url=stitched_url,
        individual_clips=individual_clip_uploads,
        thumbnail_url=thumbnail_url,
        continuity_frames=frame_urls,
        story_plan_url=f"file://{plan_path}",
        continuity_bible_url=f"file://{bible_path}",
    )

    final_status = JobStatus.PARTIAL if is_partial else JobStatus.SUCCESS
    latency = time.perf_counter() - t0

    log.info(
        "assemble_ok",
        status=final_status,
        shots_delivered=len(ok_shots),
        latency=round(latency, 2),
    )

    return {
        "artifacts": artifacts,
        "status": final_status,
        "budget": budget,
    }
