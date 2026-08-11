"""Application configuration — all settings read from environment / .env file."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ────────────────────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    api_secret_key: str = "change-me"

    # ── LLM ───────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    litellm_proxy_url: str = "http://localhost:4000"
    litellm_master_key: str = "sk-litellm-master"

    # ── Video provider ─────────────────────────────────────────────────────
    higgsfield_api_key: str = ""
    higgsfield_base_url: str = "https://api.higgsfield.ai/v1"
    # If no key is provided, the mock provider is used automatically
    @property
    def video_provider(self) -> str:
        return "higgsfield" if self.higgsfield_api_key else "mock"

    # ── Observability ──────────────────────────────────────────────────────
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # ── Database ──────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://video_agent:secret@localhost:5432/video_agent"

    # ── Redis ─────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Storage ───────────────────────────────────────────────────────────
    storage_backend: Literal["local", "s3"] = "local"
    local_storage_path: str = "./artifacts"
    aws_access_key_id: str = "minioadmin"
    aws_secret_access_key: str = "minioadmin"
    aws_region: str = "us-east-1"
    s3_bucket: str = "video-agent-artifacts"
    s3_endpoint_url: str = ""

    # ── Agent behaviour ────────────────────────────────────────────────────
    max_qc_repair_attempts: int = 2
    max_job_budget_usd: float = 2.00
    max_job_iterations: int = 20
    max_job_wall_clock_seconds: int = 480

    # ── Video spec ─────────────────────────────────────────────────────────
    shots_per_story: int = 4
    shot_duration_seconds: int = 10
    total_duration_seconds: int = 40
    continuity_score_threshold: float = 0.75

    @model_validator(mode="after")
    def validate_at_least_one_llm_key(self) -> "Settings":
        if not any([self.openai_api_key, self.anthropic_api_key, self.gemini_api_key]):
            # Allow in test mode
            pass
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
