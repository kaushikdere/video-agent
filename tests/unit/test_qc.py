"""Unit tests for the QC node."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from video_agent.agent.nodes.qc import qc_shot_node
from video_agent.agent.state import JobStatus, ShotResult, initial_state, StoryPlan, ContinuityBible

BIBLE: ContinuityBible = {
    "protagonist": "Dr. Yuki Tanaka",
    "wardrobe": "White NASA jumpsuit",
    "location": "Mars surface",
    "lighting": "Golden hour",
    "colour_palette": "#C1440E #FFD700",
    "lens_language": "35mm anamorphic",
    "style_tags": ["cinematic"],
}

STORY_PLAN: StoryPlan = {
    "title": "Test",
    "genre": "sci-fi",
    "total_duration_seconds": 40,
    "beats": [
        {"index": 0, "label": "setup", "duration_seconds": 10,
         "action": "Astronaut stands alone", "camera_move": "push-in", "mood": "lonely"},
        {"index": 1, "label": "development", "duration_seconds": 10,
         "action": "She hears a sound", "camera_move": "pan", "mood": "alert"},
        {"index": 2, "label": "turn", "duration_seconds": 10,
         "action": "Light appears", "camera_move": "zoom", "mood": "wonder"},
        {"index": 3, "label": "resolution", "duration_seconds": 10,
         "action": "She walks toward it", "camera_move": "track", "mood": "hopeful"},
    ],
}


def _state_with_shot(shot_index: int, status: str = "pending_qc") -> dict:
    state = initial_state("job-q01", "Test", "trace-q01")
    state["story_plan"] = STORY_PLAN
    state["continuity_bible"] = BIBLE
    state["current_shot_index"] = shot_index
    shot: ShotResult = {
        "shot_index": shot_index,
        "status": status,
        "clip_url": f"file:///tmp/shot_{shot_index}.mp4",
        "final_frame_url": f"file:///tmp/frame_{shot_index}.png",
        "thumbnail_url": f"file:///tmp/thumb_{shot_index}.jpg",
        "qc_score": 0.0,
        "qc_attempts": 0,
        "prompt_used": "test prompt",
        "model": "mock-v1",
        "seed": 42,
        "cost_usd": 0.0,
        "latency_seconds": 0.5,
        "failure_signature": None,
    }
    state["shots"] = [shot]
    return state


@pytest.mark.asyncio
async def test_qc_passed():
    state = _state_with_shot(0)
    qc_response = {"continuity_score": 0.92, "passed": True, "deviations": [], "repair_hint": None}

    with patch("video_agent.agent.nodes.qc.llm_call", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "content": json.dumps(qc_response),
            "model": "gpt-4o",
            "cost_usd": 0.001,
            "tokens": 100,
        }
        result = await qc_shot_node(state)

    assert result["shots"][0]["status"] == "ok"
    assert result["shots"][0]["qc_score"] == pytest.approx(0.92)
    assert result["current_shot_index"] == 1
    assert result["repair_count"] == 0


@pytest.mark.asyncio
async def test_qc_failed_triggers_repair():
    state = _state_with_shot(1)
    state["repair_count"] = 0
    qc_response = {"continuity_score": 0.40, "passed": False,
                   "deviations": ["Wrong wardrobe colour"], "repair_hint": "Use white jumpsuit"}

    with patch("video_agent.agent.nodes.qc.llm_call", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "content": json.dumps(qc_response),
            "model": "gpt-4o",
            "cost_usd": 0.001,
            "tokens": 100,
        }
        result = await qc_shot_node(state)

    assert result["shots"][0]["status"] == "failed"
    assert result["repair_count"] == 1
    # current_shot_index stays the same so router sends back to generator
    assert result.get("current_shot_index") is None or result.get("current_shot_index") == 1


@pytest.mark.asyncio
async def test_qc_no_progress_detection():
    state = _state_with_shot(2)
    state["repair_count"] = 1
    # Same failure signature twice
    state["failure_signatures"] = ["shot:2:qc_score:0.40"]

    qc_response = {"continuity_score": 0.40, "passed": False,
                   "deviations": ["Same error"], "repair_hint": "try again"}

    with patch("video_agent.agent.nodes.qc.llm_call", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "content": json.dumps(qc_response),
            "model": "gpt-4o",
            "cost_usd": 0.001,
            "tokens": 100,
        }
        result = await qc_shot_node(state)

    assert result["status"] == JobStatus.FAILED_NO_PROGRESS
