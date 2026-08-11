"""FastAPI application entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager
import structlog

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from video_agent import __version__
from video_agent.api.routes import router
from video_agent.config import get_settings
from video_agent.observability.langfuse_client import flush
from video_agent.observability.middleware import LoggingMiddleware

# ─── Structured logging setup ─────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

settings = get_settings()
logger = structlog.get_logger(__name__)


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup", version=__version__, env=settings.app_env, provider=settings.video_provider)
    yield
    flush()
    logger.info("shutdown")


# ─── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Video Agent",
    description=(
        "**Entermind · Video Agent**\n\n"
        "One prompt becomes a continuous 40-second story — four 10-second shots "
        "with enforced narrative and visual continuity.\n\n"
        "Built on: LangGraph · LiteLLM · Higgsfield MCP · ffmpeg · Langfuse"
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ─── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routes ───────────────────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")


def run() -> None:
    """CLI entry point."""
    import uvicorn

    uvicorn.run(
        "video_agent.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
