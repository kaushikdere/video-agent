"""SQLAlchemy models for Job and Shot persistence."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Boolean,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship

from video_agent.agent.state import JobStatus


class Base(DeclarativeBase):
    pass


class Job(Base):
    """Persistent job record — system of record for a video generation job."""

    __tablename__ = "jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trace_id = Column(String(64), index=True)
    prompt = Column(Text, nullable=False)
    provider = Column(String(32), nullable=False, default="mock")
    status = Column(
        Enum(JobStatus, name="job_status_enum"),
        nullable=False,
        default=JobStatus.PLANNING,
        index=True,
    )
    story_plan = Column(JSONB, nullable=True)
    continuity_bible = Column(JSONB, nullable=True)
    artifacts = Column(JSONB, nullable=True)
    error_message = Column(Text, nullable=True)

    # Budget
    cost_usd = Column(Float, default=0.0)
    tokens_used = Column(Integer, default=0)
    iterations = Column(Integer, default=0)
    elapsed_seconds = Column(Float, default=0.0)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    shots = relationship("Shot", back_populates="job", order_by="Shot.shot_index")


class Shot(Base):
    """Individual shot result within a job."""

    __tablename__ = "shots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    shot_index = Column(Integer, nullable=False)
    status = Column(String(32), nullable=False, default="pending_qc")
    clip_url = Column(Text, default="")
    final_frame_url = Column(Text, default="")
    thumbnail_url = Column(Text, default="")
    qc_score = Column(Float, default=0.0)
    qc_attempts = Column(Integer, default=0)
    prompt_used = Column(Text, default="")
    model = Column(String(64), default="")
    seed = Column(Integer, nullable=True)
    cost_usd = Column(Float, default=0.0)
    latency_seconds = Column(Float, default=0.0)
    failure_signature = Column(String(256), nullable=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    job = relationship("Job", back_populates="shots")
