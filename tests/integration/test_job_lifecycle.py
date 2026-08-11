"""Integration test — full job lifecycle with mock provider."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient, ASGITransport

from video_agent.agent.state import JobStatus
from video_agent.main import app


@pytest.mark.asyncio
async def test_full_job_lifecycle():
    """
    End-to-end test: POST /jobs → poll GET /jobs/{id} until terminal status.
    Uses mock provider + mocked LLM calls.
    """
    valid_plan = {
        "title": "Integration Test Story",
        "genre": "drama",
        "total_duration_seconds": 40,
        "beats": [
            {"index": 0, "label": "setup", "duration_seconds": 10,
             "action": "Hero enters empty room", "camera_move": "push-in", "mood": "calm"},
            {"index": 1, "label": "development", "duration_seconds": 10,
             "action": "Phone rings loudly", "camera_move": "pan", "mood": "tense"},
            {"index": 2, "label": "turn", "duration_seconds": 10,
             "action": "He picks it up — silence", "camera_move": "close-up", "mood": "dread"},
            {"index": 3, "label": "resolution", "duration_seconds": 10,
             "action": "He walks to window", "camera_move": "slow zoom out", "mood": "quiet"},
        ],
    }
    valid_bible = {
        "protagonist": "James, 40s, weathered face",
        "wardrobe": "Grey wool coat, black trousers",
        "location": "1970s London flat, dark wallpaper",
        "lighting": "Overcast afternoon, cool blue light",
        "colour_palette": "#2B2D42 #8D99AE #EDF2F4 #EF233C",
        "lens_language": "50mm, deep DoF, 1.85:1",
        "style_tags": ["moody", "British", "intimate"],
    }
    valid_qc = {
        "continuity_score": 0.88,
        "passed": True,
        "deviations": [],
        "repair_hint": None,
    }

    llm_responses = [
        {"content": json.dumps(valid_plan), "model": "gpt-4o", "cost_usd": 0.002, "tokens": 300},
        {"content": json.dumps(valid_bible), "model": "gpt-4o", "cost_usd": 0.002, "tokens": 300},
        # QC for each shot
        {"content": json.dumps(valid_qc), "model": "gpt-4o", "cost_usd": 0.001, "tokens": 100},
        {"content": json.dumps(valid_qc), "model": "gpt-4o", "cost_usd": 0.001, "tokens": 100},
        {"content": json.dumps(valid_qc), "model": "gpt-4o", "cost_usd": 0.001, "tokens": 100},
        {"content": json.dumps(valid_qc), "model": "gpt-4o", "cost_usd": 0.001, "tokens": 100},
    ]

    call_count = 0

    async def mock_llm(**kwargs):
        nonlocal call_count
        resp = llm_responses[min(call_count, len(llm_responses) - 1)]
        call_count += 1
        return resp

    with patch("video_agent.gateway.llm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        # Patch at the node level instead
        with patch("video_agent.agent.nodes.planner.llm_call", new_callable=AsyncMock) as p_llm, \
             patch("video_agent.agent.nodes.bible.llm_call", new_callable=AsyncMock) as b_llm, \
             patch("video_agent.agent.nodes.qc.llm_call", new_callable=AsyncMock) as q_llm:

            p_llm.return_value = llm_responses[0]
            b_llm.return_value = llm_responses[1]
            q_llm.return_value = llm_responses[2]

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                # Create job
                resp = await client.post(
                    "/api/v1/jobs",
                    json={"prompt": "A detective uncovers a century-old conspiracy in London", "provider": "mock"},
                )
                assert resp.status_code == 202
                data = resp.json()
                job_id = data["job_id"]
                assert job_id

                # Wait for completion (max 30 seconds)
                terminal_statuses = {
                    JobStatus.SUCCESS, JobStatus.PARTIAL, JobStatus.FAILED,
                    JobStatus.FAILED_NO_PROGRESS, JobStatus.ESCALATED
                }
                for _ in range(30):
                    await asyncio.sleep(1)
                    poll = await client.get(f"/api/v1/jobs/{job_id}")
                    assert poll.status_code == 200
                    job_data = poll.json()
                    if job_data["status"] in [s.value for s in terminal_statuses]:
                        break

                # Verify terminal state
                assert job_data["status"] in [s.value for s in terminal_statuses]
                assert job_data["story_plan"] is not None or job_data["status"] == JobStatus.FAILED.value


@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_create_job_validation():
    """Short prompts should be rejected."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/v1/jobs", json={"prompt": "hi"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_get_nonexistent_job():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/jobs/nonexistent-job-id")
    assert resp.status_code == 404
