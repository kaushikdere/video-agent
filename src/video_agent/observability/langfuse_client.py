"""Langfuse observability client — traces, generations, scores."""
from __future__ import annotations

from typing import Any

import structlog

from video_agent.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()

_langfuse = None


def get_langfuse():
    global _langfuse
    if _langfuse is None and settings.langfuse_public_key:
        try:
            from langfuse import Langfuse  # type: ignore

            _langfuse = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        except ImportError:
            logger.warning("langfuse_not_installed")
        except Exception as exc:
            logger.warning("langfuse_init_failed", error=str(exc))
    return _langfuse


def create_trace(job_id: str, user_prompt: str, metadata: dict[str, Any] | None = None) -> str:
    """Create a Langfuse trace and return trace_id."""
    lf = get_langfuse()
    if lf is None:
        return job_id  # fallback trace_id

    try:
        trace = lf.trace(
            id=job_id,
            name="video-agent-job",
            input={"prompt": user_prompt},
            metadata=metadata or {},
            tags=["video-agent", settings.app_env],
        )
        return trace.id
    except Exception as exc:
        logger.warning("langfuse_trace_failed", error=str(exc))
        return job_id


def score_job(trace_id: str, score_name: str, value: float, comment: str = "") -> None:
    """Post a score to a Langfuse trace."""
    lf = get_langfuse()
    if lf is None:
        return
    try:
        lf.score(trace_id=trace_id, name=score_name, value=value, comment=comment)
    except Exception as exc:
        logger.warning("langfuse_score_failed", error=str(exc))


def flush() -> None:
    """Flush pending Langfuse events (call before shutdown)."""
    lf = get_langfuse()
    if lf:
        try:
            lf.flush()
        except Exception:
            pass
