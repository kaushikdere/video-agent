"""Node 1: plan_story — LLM produces a 4-beat StoryPlan."""
from __future__ import annotations

import json
import time
from typing import Any, cast

import structlog

from video_agent.agent.state import AgentState, BudgetState, JobStatus, StoryPlan
from video_agent.config import get_settings
from video_agent.gateway.llm import llm_call

logger = structlog.get_logger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """You are a cinematic story architect for a video AI platform.

Given a user prompt, you produce a 4-beat narrative arc for a 40-second short film.
Each beat is exactly 10 seconds long. The four beats are:
  0. setup        — establish world and protagonist
  1. development  — introduce tension or movement
  2. turn         — a moment of change or revelation
  3. resolution   — closure or open ending

Rules:
- Total duration MUST equal 40 seconds (4 × 10s).
- Each beat must have a vivid, specific camera move.
- Keep 'action' under 40 words per beat — it becomes a video prompt.
- Output ONLY valid JSON matching the schema exactly. No markdown fences.

Schema:
{
  "title": "string",
  "genre": "string",
  "total_duration_seconds": 40,
  "beats": [
    {
      "index": 0,
      "label": "setup",
      "duration_seconds": 10,
      "action": "string (≤40 words)",
      "camera_move": "string",
      "mood": "string"
    }
  ]
}"""


def _extract_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start : end + 1]
    return json.loads(raw)  # type: ignore[no-any-return]


async def plan_story_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: produce a StoryPlan from the user prompt."""
    log = logger.bind(job_id=state["job_id"], node="plan_story")
    log.info("node_start")
    t0 = time.perf_counter()

    budget: BudgetState = state["budget"].copy()
    budget["iterations"] += 1
    budget["elapsed_seconds"] = time.time() - budget["started_at"]

    try:
        result = await llm_call(
            alias="reasoning-high",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": state["user_prompt"]},
            ],
            temperature=0.7,
            max_tokens=2048,
            response_format={"type": "json_object"},
            trace_id=state["trace_id"],
            metadata={"job_id": state["job_id"], "node": "plan_story"},
        )
    except Exception as exc:
        log.error("plan_story_failed", error=str(exc))
        return {
            "status": JobStatus.FAILED,
            "error_message": f"Planner LLM error: {exc}",
            "budget": budget,
        }

    # Parse JSON
    try:
        plan_data = _extract_json(result["content"])
        _validate_plan(plan_data)
        story_plan: StoryPlan = cast(StoryPlan, plan_data)
    except Exception as exc:
        log.error("plan_parse_failed", error=str(exc), raw=result["content"][:200])
        return {
            "status": JobStatus.FAILED,
            "error_message": f"Planner JSON parse error: {exc}",
            "budget": budget,
        }

    budget["cost_usd"] += result["cost_usd"]
    budget["tokens_used"] += result["tokens"]

    log.info(
        "node_ok",
        title=story_plan.get("title", ""),
        latency=round(time.perf_counter() - t0, 2),
        cost=result["cost_usd"],
    )

    return {
        "story_plan": story_plan,
        "status": JobStatus.GENERATING,
        "budget": budget,
    }


def _validate_plan(data: dict[str, Any]) -> None:
    beats = data.get("beats", [])
    if len(beats) != 4:
        raise ValueError(f"Expected 4 beats, got {len(beats)}")
    total = sum(b.get("duration_seconds", 0) for b in beats)
    if total != 40:
        raise ValueError(f"Total duration must be 40s, got {total}s")
    labels = {"setup", "development", "turn", "resolution"}
    got_labels = {b.get("label") for b in beats}
    if not labels <= got_labels:
        raise ValueError(f"Missing beat labels: {labels - got_labels}")

