"""API routes — POST /jobs, GET /jobs/{id}, GET /jobs, GET /health."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from video_agent import __version__
from video_agent.api.schemas import (
    CreateJobRequest,
    CreateJobResponse,
    HealthResponse,
    JobResponse,
)
from video_agent.api.job_store import create_job, get_job, list_jobs
from video_agent.config import get_settings
from video_agent.observability.langfuse_client import get_langfuse

router = APIRouter()
settings = get_settings()


@router.post(
    "/jobs",
    response_model=CreateJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create a new Video Agent job",
    description=(
        "Submit a text prompt. The agent will plan a 4-beat story, lock a ContinuityBible, "
        "generate 4×10-second shots sequentially (frame chaining), run QC, stitch, and deliver "
        "a 40-second MP4. Poll GET /jobs/{job_id} for status."
    ),
)
async def create_job_endpoint(body: CreateJobRequest) -> CreateJobResponse:
    return await create_job(prompt=body.prompt, provider=body.provider)


@router.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    summary="Poll job status and results",
)
async def get_job_endpoint(job_id: str) -> JobResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@router.get(
    "/jobs",
    summary="List recent jobs",
)
async def list_jobs_endpoint() -> list[dict]:
    return list_jobs()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
async def health_endpoint() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        provider=settings.video_provider,
        langfuse=get_langfuse() is not None,
    )
