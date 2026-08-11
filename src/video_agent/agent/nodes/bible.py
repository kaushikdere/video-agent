"""Node 2: lock_bible — LLM produces the immutable ContinuityBible."""
from __future__ import annotations

import json
import time
from typing import Any, cast

import structlog

from video_agent.agent.state import AgentState, BudgetState, ContinuityBible, JobStatus
from video_agent.config import get_settings
from video_agent.gateway.llm import llm_call

logger = structlog.get_logger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """You are a film continuity supervisor.

Given a user prompt and a 4-beat story plan, produce a ContinuityBible —
the immutable visual contract that will be embedded in EVERY shot prompt to
enforce consistency across all four clips.

Rules:
- Be very specific about physical appearance (height, hair, clothing colours, accessories).
- Colour palette: list 4–6 hex codes OR named film references.
- Lens language: focal length (e.g. 35mm), depth-of-field, aspect ratio.
- Output ONLY valid JSON. No markdown fences.

Schema:
{
  "protagonist": "string (detailed physical description)",
  "wardrobe": "string (clothing, shoes, accessories — exact colours)",
  "location": "string (precise setting — architecture, textures, scale)",
  "lighting": "string (time of day, quality, direction, shadows)",
  "colour_palette": "string (hex codes or film reference)",
  "lens_language": "string (focal length, DoF, aspect ratio)",
  "style_tags": ["string"]
}"""


def _extract_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start : end + 1]
    return json.loads(raw)  # type: ignore[no-any-return]


async def lock_bible_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: produce and lock the ContinuityBible."""
    log = logger.bind(job_id=state["job_id"], node="lock_bible")
    log.info("node_start")
    t0 = time.perf_counter()

    budget: BudgetState = state["budget"].copy()
    budget["iterations"] += 1
    budget["elapsed_seconds"] = time.time() - budget["started_at"]

    story_plan = state["story_plan"]
    if story_plan is None:
        log.error("missing_story_plan")
        return {
            "status": JobStatus.FAILED,
            "error_message": "Story plan is missing for bible node.",
            "budget": budget,
        }

    beats_summary = "\n".join(
        f"Beat {b['index']} ({b['label']}): {b['action']}" for b in story_plan["beats"]
    )

    user_content = (
        f"User prompt: {state['user_prompt']}\n\n"
        f"Story: {story_plan['title']} ({story_plan['genre']})\n\n"
        f"Beats:\n{beats_summary}"
    )

    try:
        result = await llm_call(
            alias="reasoning-high",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.5,
            max_tokens=2048,
            response_format={"type": "json_object"},
            trace_id=state["trace_id"],
            metadata={"job_id": state["job_id"], "node": "lock_bible"},
        )
    except Exception as exc:
        log.error("lock_bible_failed", error=str(exc))
        return {
            "status": JobStatus.FAILED,
            "error_message": f"Bible LLM error: {exc}",
            "budget": budget,
        }

    try:
        bible_data = _extract_json(result["content"])
        bible: ContinuityBible = cast(ContinuityBible, bible_data)
    except Exception as exc:
        log.error("bible_parse_failed", error=str(exc))
        return {
            "status": JobStatus.FAILED,
            "error_message": f"Bible JSON parse error: {exc}",
            "budget": budget,
        }

    budget["cost_usd"] += result["cost_usd"]
    budget["tokens_used"] += result["tokens"]

    log.info(
        "node_ok",
        location=bible.get("location", ""),
        latency=round(time.perf_counter() - t0, 2),
    )

    return {
        "continuity_bible": bible,
        "budget": budget,
    }

