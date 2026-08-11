"""Provider abstraction — AbstractVideoProvider protocol."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationRequest:
    """All inputs for a single shot generation."""
    prompt: str
    duration_seconds: int
    previous_frame_url: str | None   # frame chaining input
    seed: int | None
    job_id: str
    shot_index: int


@dataclass
class GenerationResult:
    """Output from a single shot generation."""
    clip_url: str
    final_frame_url: str
    thumbnail_url: str
    seed: int
    model: str
    cost_usd: float
    latency_seconds: float
    raw_response: dict


class AbstractVideoProvider(ABC):
    """
    Provider abstraction — capability negotiation + failover.
    Swapping providers is a config change with zero code diff.
    """

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate a single video clip from a prompt."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier for logging."""
        ...
