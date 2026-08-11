"""Unit tests for the planner node."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from video_agent.agent.nodes.planner import plan_story_node, _validate_plan
from video_agent.agent.state import JobStatus, initial_state


VALID_PLAN = {
    "title": "The Last Signal",
    "genre": "sci-fi drama",
    "total_duration_seconds": 40,
    "beats": [
        {"index": 0, "label": "setup", "duration_seconds": 10,
         "action": "Astronaut floats in empty control room", "camera_move": "slow push-in", "mood": "lonely"},
        {"index": 1, "label": "development", "duration_seconds": 10,
         "action": "Console flickers with alien symbols", "camera_move": "rack focus", "mood": "tense"},
        {"index": 2, "label": "turn", "duration_seconds": 10,
         "action": "Astronaut touches screen — hologram blooms", "camera_move": "360 orbit", "mood": "wonder"},
        {"index": 3, "label": "resolution", "duration_seconds": 10,
         "action": "She smiles at stars through porthole", "camera_move": "slow zoom out", "mood": "hopeful"},
    ],
}


@pytest.mark.asyncio
async def test_plan_story_node_success():
    state = initial_state("job-001", "A lone astronaut finds an alien signal", "trace-001")

    with patch("video_agent.agent.nodes.planner.llm_call", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "content": json.dumps(VALID_PLAN),
            "model": "gpt-4o",
            "cost_usd": 0.002,
            "tokens": 350,
        }
        result = await plan_story_node(state)

    assert result["story_plan"]["title"] == "The Last Signal"
    assert len(result["story_plan"]["beats"]) == 4
    assert result["story_plan"]["total_duration_seconds"] == 40
    assert result["status"] == JobStatus.GENERATING
    assert result["budget"]["cost_usd"] == pytest.approx(0.002)


@pytest.mark.asyncio
async def test_plan_story_node_strips_markdown_fences():
    state = initial_state("job-002", "Space drama", "trace-002")
    wrapped = f"```json\n{json.dumps(VALID_PLAN)}\n```"

    with patch("video_agent.agent.nodes.planner.llm_call", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {"content": wrapped, "model": "gpt-4o", "cost_usd": 0.001, "tokens": 200}
        result = await plan_story_node(state)

    assert result["story_plan"]["title"] == "The Last Signal"


@pytest.mark.asyncio
async def test_plan_story_node_llm_failure():
    state = initial_state("job-003", "Test prompt", "trace-003")

    with patch("video_agent.agent.nodes.planner.llm_call", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = RuntimeError("API timeout")
        result = await plan_story_node(state)

    assert result["status"] == JobStatus.FAILED
    assert "API timeout" in result["error_message"]


def test_validate_plan_wrong_beat_count():
    bad_plan = dict(VALID_PLAN, beats=VALID_PLAN["beats"][:3])
    with pytest.raises(ValueError, match="4 beats"):
        _validate_plan(bad_plan)


def test_validate_plan_wrong_duration():
    beats = [dict(b, duration_seconds=15) for b in VALID_PLAN["beats"]]
    bad_plan = dict(VALID_PLAN, beats=beats)
    with pytest.raises(ValueError, match="40s"):
        _validate_plan(bad_plan)
