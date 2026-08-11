"""Node 3: generate_shot — build prompt from bible + beat, call provider sequentially."""
from __future__ import annotations

import time
from typing import Any

import structlog

from video_agent.agent.state import AgentState, BudgetState, JobStatus, ShotResult
from video_agent.config import get_settings
from video_agent.providers.base import GenerationRequest
from video_agent.providers.higgsfield import HiggsFieldProvider
from video_agent.providers.mock import MockVideoProvider

logger = structlog.get_logger(__name__)
settings = get_settings()


def _get_provider() -> Any:
    if settings.video_provider == "higgsfield":
        return HiggsFieldProvider()
    return MockVideoProvider()


def _build_shot_prompt(state: AgentState, shot_index: int) -> str:
    """
    Construct the generation prompt:
      bible + beat action + camera move
    """
    bible = state["continuity_bible"]
    story_plan = state["story_plan"]

    if bible is None or story_plan is None:
        raise ValueError("Missing continuity_bible or story_plan in AgentState")

    beat = story_plan["beats"][shot_index]

    return (
        f"Cinematic short film clip, {beat['duration_seconds']} seconds. "
        f"Protagonist: {bible['protagonist']}. "
        f"Wearing: {bible['wardrobe']}. "
        f"Location: {bible['location']}. "
        f"Lighting: {bible['lighting']}. "
        f"Palette: {bible['colour_palette']}. "
        f"Lens: {bible['lens_language']}. "
        f"Style: {', '.join(bible['style_tags'])}. "
        f"Scene: {beat['action']}. "
        f"Camera: {beat['camera_move']}. "
        f"Mood: {beat['mood']}."
    )


async def generate_shot_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: generate the current shot.

    Deliberate trade-off: shots run sequentially, not in parallel.
    Parallel is ~4× faster but breaks frame chaining.
    Frame chaining is what makes the product work.
    """
    shot_index = state["current_shot_index"]
    log = logger.bind(job_id=state["job_id"], node="generate_shot", shot=shot_index)
    log.info("node_start")

    budget: BudgetState = state["budget"].copy()
    budget["iterations"] += 1
    budget["elapsed_seconds"] = time.time() - budget["started_at"]

    prompt = _build_shot_prompt(state, shot_index)

    # Frame chaining — condition on final frame of previous shot
    previous_frame_url: str | None = None
    shots = state["shots"]
    # Find last successful shot before this one
    for shot in reversed(shots):
        if shot["shot_index"] < shot_index and shot["status"] != "failed":
            previous_frame_url = shot["final_frame_url"]
            break

    request = GenerationRequest(
        prompt=prompt,
        duration_seconds=settings.shot_duration_seconds,
        previous_frame_url=previous_frame_url,
        seed=None,
        job_id=state["job_id"],
        shot_index=shot_index,
    )

    t0 = time.perf_counter()
    provider = _get_provider()

    try:
        result = await provider.generate(request)
    except Exception as exc:
        log.error("generation_failed", error=str(exc))
        failure_sig = f"shot:{shot_index}:generation_error"
        failure_signatures = list(state["failure_signatures"])
        failure_signatures.append(failure_sig)

        # Detect no-progress: same failure signature twice
        if failure_signatures.count(failure_sig) >= 2:
            return {
                "status": JobStatus.FAILED_NO_PROGRESS,
                "error_message": f"Same failure twice on shot {shot_index}: {exc}",
                "failure_signatures": failure_signatures,
                "budget": budget,
            }

        # Record failed shot for partial delivery
        failed_shot = ShotResult(
            shot_index=shot_index,
            status="failed",
            clip_url="",
            final_frame_url="",
            thumbnail_url="",
            qc_score=0.0,
            qc_attempts=0,
            prompt_used=prompt,
            model="none",
            seed=None,
            cost_usd=0.0,
            latency_seconds=time.perf_counter() - t0,
            failure_signature=failure_sig,
        )
        existing = [s for s in shots if s["shot_index"] != shot_index]
        return {
            "shots": existing + [failed_shot],
            "failure_signatures": failure_signatures,
            "budget": budget,
        }

    budget["cost_usd"] += result.cost_usd
    latency = time.perf_counter() - t0

    shot_result = ShotResult(
        shot_index=shot_index,
        status="pending_qc",
        clip_url=result.clip_url,
        final_frame_url=result.final_frame_url,
        thumbnail_url=result.thumbnail_url,
        qc_score=0.0,
        qc_attempts=0,
        prompt_used=prompt,
        model=result.model,
        seed=result.seed,
        cost_usd=result.cost_usd,
        latency_seconds=latency,
        failure_signature=None,
    )

    existing = [s for s in shots if s["shot_index"] != shot_index]
    log.info("node_ok", latency=round(latency, 2), url=result.clip_url[:60])

    return {
        "shots": existing + [shot_result],
        "status": JobStatus.QC,
        "budget": budget,
    }

