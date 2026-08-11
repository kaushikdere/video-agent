"""Async database repository — all DB operations in one place."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from video_agent.config import get_settings
from video_agent.db.models import Base, Job, Shot
from video_agent.agent.state import JobStatus

logger = structlog.get_logger(__name__)
settings = get_settings()

# Engine and session factory (created lazily)
_engine = None
_session_factory = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            settings.database_url,
            echo=False,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
    return _engine


def get_session_factory() -> async_sessionmaker:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            get_engine(), class_=AsyncSession, expire_on_commit=False
        )
    return _session_factory


async def create_tables() -> None:
    """Create all tables (expand-only migrations in prod via Alembic)."""
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


class JobRepository:
    """All database access for the Job aggregate."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, job_id: str, prompt: str, provider: str, trace_id: str) -> Job:
        job = Job(
            id=uuid.UUID(job_id) if len(job_id) == 36 else uuid.uuid4(),
            trace_id=trace_id,
            prompt=prompt,
            provider=provider,
            status=JobStatus.PLANNING,
        )
        self.session.add(job)
        await self.session.commit()
        await self.session.refresh(job)
        return job

    async def get(self, job_id: str) -> Job | None:
        result = await self.session.execute(
            select(Job).where(Job.id == uuid.UUID(job_id))
        )
        return result.scalar_one_or_none()

    async def update_status(self, job_id: str, status: JobStatus, **kwargs: Any) -> None:
        values = {"status": status, "updated_at": datetime.now(timezone.utc), **kwargs}
        await self.session.execute(
            update(Job).where(Job.id == uuid.UUID(job_id)).values(**values)
        )
        await self.session.commit()

    async def upsert_shot(self, job_id: str, shot_data: dict) -> None:
        result = await self.session.execute(
            select(Shot).where(
                Shot.job_id == uuid.UUID(job_id),
                Shot.shot_index == shot_data["shot_index"],
            )
        )
        shot = result.scalar_one_or_none()
        if shot:
            for k, v in shot_data.items():
                if hasattr(shot, k):
                    setattr(shot, k, v)
        else:
            shot = Shot(job_id=uuid.UUID(job_id), **shot_data)
            self.session.add(shot)
        await self.session.commit()

    async def list_recent(self, limit: int = 20) -> list[Job]:
        result = await self.session.execute(
            select(Job).order_by(Job.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())
