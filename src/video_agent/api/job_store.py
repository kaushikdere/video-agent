"""In-memory job store — used when running without PostgreSQL."""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog

from video_agent.agent.state import AgentState, JobStatus, initial_state
from video_agent.agent.graph import get_graph
from video_agent.api.schemas import (
    ArtifactInfo,
    BudgetInfo,
    CreateJobResponse,
    JobResponse,
    ShotInfo,
    StoryPlanInfo,
)
from video_agent.config import get_settings
from video_agent.observability.langfuse_client import create_trace, score_job

logger = structlog.get_logger(__name__)
settings = get_settings()

# Simple in-memory store (keyed by job_id)
_jobs: dict[str, dict] = {}


def _make_record(job_id: str, prompt: str, provider: str) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "job_id": job_id,
        "prompt": prompt,
        "provider": provider,
        "status": JobStatus.PLANNING,
        "state": None,
        "error_message": None,
        "created_at": now,
        "updated_at": now,
    }


async def create_job(prompt: str, provider: str | None = None) -> CreateJobResponse:
    job_id = str(uuid.uuid4())
    p = provider or settings.video_provider
    trace_id = create_trace(job_id, prompt, {"provider": p})

    record = _make_record(job_id, prompt, p)
    _jobs[job_id] = record

    # Run agent in background
    asyncio.create_task(_run_agent(job_id, prompt, trace_id, p))

    logger.info("job_created", job_id=job_id, provider=p)
    return CreateJobResponse(
        job_id=job_id,
        status=JobStatus.PLANNING,
        message=f"Job {job_id} created. Poll GET /jobs/{job_id} for status.",
    )


async def _run_agent(job_id: str, prompt: str, trace_id: str, provider: str) -> None:
    log = logger.bind(job_id=job_id)
    record = _jobs[job_id]

    # Temporarily override provider setting for this run
    import video_agent.agent.nodes.generator as gen_mod
    original_provider = settings.video_provider
    if provider == "mock":
        from video_agent.providers.mock import MockVideoProvider
        gen_mod._get_provider = lambda: MockVideoProvider()
    elif provider == "higgsfield":
        from video_agent.providers.higgsfield import HiggsFieldProvider
        gen_mod._get_provider = lambda: HiggsFieldProvider()

    try:
        state = initial_state(job_id=job_id, user_prompt=prompt, trace_id=trace_id)
        graph = get_graph()

        config = {"configurable": {"thread_id": job_id}}
        async for event in graph.astream(state, config=config):
            # Update in-memory record with latest state
            for node_name, node_state in event.items():
                if isinstance(node_state, dict):
                    current = _jobs[job_id].get("state") or {}
                    current.update(node_state)
                    _jobs[job_id]["state"] = current
                    _jobs[job_id]["updated_at"] = datetime.now(timezone.utc)
                    if "status" in node_state:
                        _jobs[job_id]["status"] = node_state["status"]
                    log.info("node_complete", node=node_name, status=_jobs[job_id]["status"])

        # Post continuity score to Langfuse
        final_state = _jobs[job_id].get("state", {})
        shots = final_state.get("shots", [])
        if shots:
            avg_score = sum(s.get("qc_score", 0) for s in shots) / len(shots)
            score_job(trace_id, "continuity_score", avg_score)

    except Exception as exc:
        log.error("agent_error", error=str(exc))
        _jobs[job_id]["status"] = JobStatus.FAILED
        _jobs[job_id]["error_message"] = str(exc)
        _jobs[job_id]["updated_at"] = datetime.now(timezone.utc)
    finally:
        # Restore provider
        gen_mod._get_provider = lambda: (
            __import__("video_agent.providers.higgsfield", fromlist=["HiggsFieldProvider"]).HiggsFieldProvider()
            if original_provider == "higgsfield"
            else __import__("video_agent.providers.mock", fromlist=["MockVideoProvider"]).MockVideoProvider()
        )


def get_job(job_id: str) -> JobResponse | None:
    record = _jobs.get(job_id)
    if not record:
        return None

    state: dict = record.get("state") or {}
    shots_raw = state.get("shots", [])
    shots = [
        ShotInfo(
            shot_index=s["shot_index"],
            status=s["status"],
            clip_url=s["clip_url"],
            thumbnail_url=s.get("thumbnail_url", ""),
            qc_score=s.get("qc_score", 0.0),
            qc_attempts=s.get("qc_attempts", 0),
            model=s.get("model", ""),
            cost_usd=s.get("cost_usd", 0.0),
            latency_seconds=s.get("latency_seconds", 0.0),
        )
        for s in shots_raw
    ]

    artifacts = None
    if state.get("artifacts"):
        a = state["artifacts"]
        artifacts = ArtifactInfo(**a)

    story_plan = None
    if state.get("story_plan"):
        sp = state["story_plan"]
        story_plan = StoryPlanInfo(**sp)

    budget = None
    if state.get("budget"):
        b = state["budget"]
        budget = BudgetInfo(
            tokens_used=b.get("tokens_used", 0),
            cost_usd=b.get("cost_usd", 0.0),
            iterations=b.get("iterations", 0),
            elapsed_seconds=b.get("elapsed_seconds", 0.0),
        )

    return JobResponse(
        job_id=job_id,
        status=record["status"],
        prompt=record["prompt"],
        story_plan=story_plan,
        shots=shots,
        artifacts=artifacts,
        budget=budget,
        error_message=record.get("error_message") or state.get("error_message"),
        created_at=record["created_at"],
        updated_at=record["updated_at"],
        provider=record["provider"],
    )


def list_jobs() -> list[dict]:
    return [
        {
            "job_id": r["job_id"],
            "status": r["status"],
            "prompt": r["prompt"][:80] + "..." if len(r["prompt"]) > 80 else r["prompt"],
            "provider": r["provider"],
            "created_at": r["created_at"].isoformat(),
        }
        for r in reversed(list(_jobs.values()))
    ]
