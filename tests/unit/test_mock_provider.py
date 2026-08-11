"""Mock provider unit tests."""
from __future__ import annotations

import pytest

from video_agent.providers.mock import MockVideoProvider
from video_agent.providers.base import GenerationRequest


@pytest.mark.asyncio
async def test_mock_provider_generates_result():
    provider = MockVideoProvider(simulate_latency=False)
    request = GenerationRequest(
        prompt="A lone astronaut on Mars",
        duration_seconds=10,
        previous_frame_url=None,
        seed=42,
        job_id="test-job",
        shot_index=0,
    )
    result = await provider.generate(request)

    assert result.clip_url != ""
    assert result.seed == 42
    assert result.model == "mock-v1"
    assert result.cost_usd == 0.0
    assert result.latency_seconds > 0


@pytest.mark.asyncio
async def test_mock_provider_forced_failure():
    provider = MockVideoProvider(simulate_latency=False, fail_shot=1)

    ok_request = GenerationRequest(
        prompt="First shot", duration_seconds=10,
        previous_frame_url=None, seed=None, job_id="test-job", shot_index=0,
    )
    fail_request = GenerationRequest(
        prompt="Second shot", duration_seconds=10,
        previous_frame_url=None, seed=None, job_id="test-job", shot_index=1,
    )

    ok_result = await provider.generate(ok_request)
    assert ok_result.clip_url != ""

    with pytest.raises(RuntimeError, match="Mock forced failure"):
        await provider.generate(fail_request)


@pytest.mark.asyncio
async def test_mock_provider_health_check():
    provider = MockVideoProvider()
    assert await provider.health_check() is True


def test_mock_provider_name():
    assert MockVideoProvider().name == "mock"
