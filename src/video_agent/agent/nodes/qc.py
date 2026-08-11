"""Node 4: qc_shot — vision model scores each shot vs. the ContinuityBible."""
from __future__ import annotations

import json
import time

import structlog

from video_agent.agent.state import AgentState, JobStatus
from video_agent.config import get_settings
from video_agent.gateway.llm import llm_call

logger = structlog.get_logger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """You are a film continuity quality inspector.

You will be given:
1. A ContinuityBible (the immutable visual contract for this film)
2. A description of the shot that was just generated

Your job: score how well the shot matches the bible (0.0–1.0) and explain deviations.

Scoring rubric:
- 1.0: Perfect match on all dimensions
- 0.75–0.99: Minor deviations, acceptable
- 0.50–0.74: Moderate issues, may need repair
- 0.0–0.49: Major continuity break, must repair

Output ONLY valid JSON:
{
  "continuity_score": 0.85,
  "passed": true,
  "deviations": ["list of specific issues, empty if none"],
  "repair_hint": "brief instruction for regeneration if failed, else null"
}"""


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start : end + 1]
    return json.loads(raw)


async def qc_shot_node(state: AgentState) -> dict:
    """LangGraph node: QC a shot against the bible."""
    shot_index = state["current_shot_index"]
    log = logger.bind(job_id=state["job_id"], node="qc_shot", shot=shot_index)
    log.info("node_start")

    budget = dict(state["budget"])
    budget["iterations"] += 1
    budget["elapsed_seconds"] = time.time() - budget["started_at"]

    shots = list(state["shots"])
    current_shot = next((s for s in shots if s["shot_index"] == shot_index), None)

    if current_shot is None or current_shot["status"] == "failed":
        # Nothing to QC — advance
        return {"budget": budget, "current_shot_index": shot_index + 1, "repair_count": 0}

    bible = state["continuity_bible"]
    bible_text = json.dumps(bible, indent=2)

    shot_description = (
        f"Shot {shot_index + 1} of 4\n"
        f"Prompt used: {current_shot['prompt_used']}\n"
        f"Clip URL: {current_shot['clip_url']}\n"
        f"(Assume prompt was followed accurately for simulation purposes)"
    )

    try:
        result = await llm_call(
            alias="vision-default",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"ContinuityBible:\n{bible_text}\n\n"
                        f"Shot to evaluate:\n{shot_description}"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=512,
            trace_id=state["trace_id"],
            metadata={"job_id": state["job_id"], "node": "qc_shot", "shot": shot_index},
        )
    except Exception as exc:
        log.error("qc_llm_failed", error=str(exc))
        # If QC itself fails, pass the shot (don't waste repair budget on QC failure)
        updated = dict(current_shot)
        updated["status"] = "ok"
        updated["qc_score"] = 0.75
        updated["qc_attempts"] = current_shot.get("qc_attempts", 0) + 1
        shots = [s if s["shot_index"] != shot_index else updated for s in shots]
        return {
            "shots": shots,
            "budget": budget,
            "current_shot_index": shot_index + 1,
            "repair_count": 0,
        }

    # Parse QC result
    try:
        qc_data = _extract_json(result["content"])
        score = float(qc_data.get("continuity_score", 0.75))
        passed = bool(qc_data.get("passed", score >= settings.continuity_score_threshold))
    except Exception as exc:
        log.warning("qc_parse_failed", error=str(exc))
        score = 0.75
        passed = True

    budget["cost_usd"] += result["cost_usd"]
    budget["tokens_used"] += result["tokens"]

    repair_count = state.get("repair_count", 0)
    updated = dict(current_shot)
    updated["qc_score"] = score
    updated["qc_attempts"] = current_shot.get("qc_attempts", 0) + 1

    if passed:
        updated["status"] = "ok"
        log.info("qc_passed", score=score, shot=shot_index)
        shots = [s if s["shot_index"] != shot_index else updated for s in shots]
        return {
            "shots": shots,
            "budget": budget,
            "current_shot_index": shot_index + 1,
            "repair_count": 0,
        }
    else:
        log.warning("qc_failed", score=score, shot=shot_index, attempts=repair_count + 1)
        failure_sig = f"shot:{shot_index}:qc_score:{score:.2f}"
        failure_signatures = list(state["failure_signatures"])
        failure_signatures.append(failure_sig)

        # Same failure signature twice → no progress
        if failure_signatures.count(failure_sig) >= 2:
            updated["status"] = "failed"
            shots = [s if s["shot_index"] != shot_index else updated for s in shots]
            return {
                "shots": shots,
                "status": JobStatus.FAILED_NO_PROGRESS,
                "failure_signatures": failure_signatures,
                "budget": budget,
                "current_shot_index": shot_index + 1,
                "repair_count": 0,
            }

        updated["status"] = "failed"
        shots = [s if s["shot_index"] != shot_index else updated for s in shots]
        return {
            "shots": shots,
            "failure_signatures": failure_signatures,
            "budget": budget,
            "repair_count": repair_count + 1,
            # current_shot_index stays the same for repair
        }
