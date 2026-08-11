"""Budget tracker — tokens, USD, iterations, wall-clock enforcement."""
from __future__ import annotations

import time

from video_agent.agent.state import BudgetState
from video_agent.config import get_settings

settings = get_settings()


def is_budget_exhausted(budget: BudgetState) -> tuple[bool, str]:
    """
    Returns (exhausted: bool, reason: str).
    Hard budget caps: cost, iterations, wall-clock.
    """
    elapsed = time.time() - budget["started_at"]

    if budget["cost_usd"] >= settings.max_job_budget_usd:
        return True, f"cost_usd={budget['cost_usd']:.4f} >= {settings.max_job_budget_usd}"

    if budget["iterations"] >= settings.max_job_iterations:
        return True, f"iterations={budget['iterations']} >= {settings.max_job_iterations}"

    if elapsed >= settings.max_job_wall_clock_seconds:
        return True, f"elapsed={elapsed:.0f}s >= {settings.max_job_wall_clock_seconds}s"

    return False, ""


def update_elapsed(budget: BudgetState) -> BudgetState:
    """Return a copy of budget with elapsed_seconds updated."""
    updated = dict(budget)
    updated["elapsed_seconds"] = time.time() - budget["started_at"]
    return updated  # type: ignore[return-value]
