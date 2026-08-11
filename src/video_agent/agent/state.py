"""Agent state — the single source of truth flowing through the LangGraph."""
from __future__ import annotations

import time
from enum import Enum
from typing import Any
from typing_extensions import TypedDict


class JobStatus(str, Enum):
    PLANNING = "planning"
    GENERATING = "generating"
    QC = "qc"
    ASSEMBLING = "assembling"
    SUCCESS = "success"
    PARTIAL = "partial"       # best-so-far, budget exhausted
    FAILED = "failed"
    FAILED_NO_PROGRESS = "failed_no_progress"
    ESCALATED = "escalated"


class Beat(TypedDict):
    """One 10-second narrative beat."""
    index: int                    # 0-3
    label: str                    # setup | development | turn | resolution
    duration_seconds: int         # always 10
    action: str                   # what happens on screen
    camera_move: str              # e.g. "slow push-in", "tracking shot"
    mood: str


class StoryPlan(TypedDict):
    """Four-beat narrative arc produced by the planner node."""
    title: str
    genre: str
    beats: list[Beat]
    total_duration_seconds: int   # must == 40


class ContinuityBible(TypedDict):
    """Immutable visual contract for the life of the job."""
    protagonist: str              # physical description
    wardrobe: str
    location: str
    lighting: str
    colour_palette: str           # hex codes or descriptive
    lens_language: str            # focal length, depth-of-field
    style_tags: list[str]


class ShotResult(TypedDict):
    """Result of generating (and QC-ing) a single shot."""
    shot_index: int
    status: str                   # "ok" | "failed" | "repaired"
    clip_url: str
    final_frame_url: str          # used to condition next shot
    thumbnail_url: str
    qc_score: float               # 0.0-1.0 continuity score
    qc_attempts: int
    prompt_used: str
    model: str
    seed: int | None
    cost_usd: float
    latency_seconds: float
    failure_signature: str | None


class BudgetState(TypedDict):
    tokens_used: int
    cost_usd: float
    iterations: int
    started_at: float             # epoch seconds
    elapsed_seconds: float


class DeliveryArtifacts(TypedDict):
    """URLs delivered to the caller on job completion."""
    stitched_mp4_url: str
    individual_clips: list[str]
    thumbnail_url: str
    continuity_frames: list[str]
    story_plan_url: str
    continuity_bible_url: str


class AgentState(TypedDict):
    """Full mutable state that flows between LangGraph nodes."""
    # Identity
    job_id: str
    trace_id: str

    # Input
    user_prompt: str

    # Outputs from planning nodes
    story_plan: StoryPlan | None
    continuity_bible: ContinuityBible | None

    # Shot results (populated sequentially)
    shots: list[ShotResult]

    # Budget
    budget: BudgetState

    # Status
    status: JobStatus
    error_message: str | None
    failure_signatures: list[str]   # detect no-progress loops

    # Delivery
    artifacts: DeliveryArtifacts | None

    # Internal routing
    current_shot_index: int
    repair_count: int               # repair attempts for current shot


def initial_state(job_id: str, user_prompt: str, trace_id: str) -> AgentState:
    """Return a fresh AgentState for a new job."""
    return AgentState(
        job_id=job_id,
        trace_id=trace_id,
        user_prompt=user_prompt,
        story_plan=None,
        continuity_bible=None,
        shots=[],
        budget=BudgetState(
            tokens_used=0,
            cost_usd=0.0,
            iterations=0,
            started_at=time.time(),
            elapsed_seconds=0.0,
        ),
        status=JobStatus.PLANNING,
        error_message=None,
        failure_signatures=[],
        artifacts=None,
        current_shot_index=0,
        repair_count=0,
    )
