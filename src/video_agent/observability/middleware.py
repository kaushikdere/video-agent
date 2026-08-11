"""Request logging middleware — JSON structured logs with trace_id."""
from __future__ import annotations

import time
import uuid

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Per-request structured logging.
    Attaches a request_id and logs method, path, status, latency.
    Never logs: credentials, raw PII, full media payloads.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        log = logger.bind(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        log.info("request_start")

        try:
            response = await call_next(request)
        except Exception as exc:
            log.error("request_error", error=str(exc))
            raise

        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        log.info(
            "request_end",
            status=response.status_code,
            latency_ms=latency_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response
