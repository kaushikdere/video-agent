"""API request / response schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from video_agent.agent.state import JobStatus


# ─── Request schemas ──────────────────────────────────────────────────────────

class CreateJobRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Your story idea. One sentence to a paragraph.",
        examples=["A lone astronaut discovers a hidden garden on the surface of Mars at sunset."],
    )
    provider: str | None = Field(
        default=None,
        description="Override video provider: 'higgsfield' | 'mock'. Auto-selected if omitted.",
    )


# ─── Response schemas ─────────────────────────────────────────────────────────

class BudgetInfo(BaseModel):
    tokens_used: int
    cost_usd: float
    iterations: int
    elapsed_seconds: float


class ShotInfo(BaseModel):
    shot_index: int
    status: str
    clip_url: str
    thumbnail_url: str
    qc_score: float
    qc_attempts: int
    model: str
    cost_usd: float
    latency_seconds: float


class ArtifactInfo(BaseModel):
    stitched_mp4_url: str
    individual_clips: list[str]
    thumbnail_url: str
    continuity_frames: list[str]
    story_plan_url: str
    continuity_bible_url: str


class StoryPlanInfo(BaseModel):
    title: str
    genre: str
    beats: list[dict[str, Any]]
    total_duration_seconds: int


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    prompt: str
    story_plan: StoryPlanInfo | None = None
    shots: list[ShotInfo] = []
    artifacts: ArtifactInfo | None = None
    budget: BudgetInfo | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    provider: str


class CreateJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    provider: str
    langfuse: bool
