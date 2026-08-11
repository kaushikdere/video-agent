"""Unit tests for the bible node."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from video_agent.agent.nodes.bible import lock_bible_node
from video_agent.agent.state import JobStatus, initial_state, StoryPlan

VALID_BIBLE = {
    "protagonist": "Dr. Yuki Tanaka, 30s, Japanese-American, short black hair, determined eyes",
    "wardrobe": "White NASA jumpsuit with orange trim, silver boots, wedding ring",
    "location": "Mars surface at sunset — red rocky terrain, dust haze on horizon",
    "lighting": "Golden hour, warm amber sun low on horizon, long purple shadows",
    "colour_palette": "#C1440E #8B6F47 #2C1810 #FFD700 #4A0E0E",
    "lens_language": "35mm, shallow DoF, 2.39:1 anamorphic, slight lens flare",
    "style_tags": ["cinematic", "sci-fi", "intimate", "golden-hour"],
}

STORY_PLAN = StoryPlan(
    title="The Last Signal",
    genre="sci-fi drama",
    total_duration_seconds=40,
    beats=[
        {"index": 0, "label": "setup", "duration_seconds": 10,
         "action": "Astronaut stands alone", "camera_move": "push-in", "mood": "lonely"},
        {"index": 1, "label": "development", "duration_seconds": 10,
         "action": "She hears a sound", "camera_move": "pan left", "mood": "alert"},
        {"index": 2, "label": "turn", "duration_seconds": 10,
         "action": "Light appears on horizon", "camera_move": "zoom", "mood": "wonder"},
        {"index": 3, "label": "resolution", "duration_seconds": 10,
         "action": "She walks toward it", "camera_move": "tracking shot", "mood": "hopeful"},
    ],
)


@pytest.mark.asyncio
async def test_lock_bible_success():
    state = initial_state("job-b01", "Mars drama", "trace-b01")
    state["story_plan"] = STORY_PLAN

    with patch("video_agent.agent.nodes.bible.llm_call", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "content": json.dumps(VALID_BIBLE),
            "model": "gpt-4o",
            "cost_usd": 0.003,
            "tokens": 400,
        }
        result = await lock_bible_node(state)

    assert result["continuity_bible"]["location"].startswith("Mars")
    assert len(result["continuity_bible"]["style_tags"]) > 0
    assert result["budget"]["cost_usd"] == pytest.approx(0.003)


@pytest.mark.asyncio
async def test_lock_bible_parse_failure():
    state = initial_state("job-b02", "Mars drama", "trace-b02")
    state["story_plan"] = STORY_PLAN

    with patch("video_agent.agent.nodes.bible.llm_call", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = {
            "content": "not json at all !!",
            "model": "gpt-4o",
            "cost_usd": 0.001,
            "tokens": 50,
        }
        result = await lock_bible_node(state)

    assert result["status"] == JobStatus.FAILED
    assert "JSON parse error" in result["error_message"]
