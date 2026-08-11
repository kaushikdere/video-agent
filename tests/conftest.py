"""Shared pytest fixtures."""
from __future__ import annotations

import pytest
import os

# Set test environment variables before any imports
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("STORAGE_BACKEND", "local")
os.environ.setdefault("LOCAL_STORAGE_PATH", "/tmp/video-agent-test-artifacts")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "")


@pytest.fixture(autouse=True)
def reset_settings_cache():
    """Reset the lru_cache on settings between tests."""
    from video_agent.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def clear_job_store():
    """Clear in-memory job store between tests."""
    try:
        from video_agent.api.job_store import _jobs
        _jobs.clear()
    except ImportError:
        pass
    yield
    try:
        from video_agent.api.job_store import _jobs
        _jobs.clear()
    except ImportError:
        pass
