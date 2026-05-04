from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.schemas.research_engine import ResearchProviderError, ResearchSource, SearchProviderResult


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ProviderBudget:
    max_searches: int
    max_sources_per_search: int
    max_total_sources: int
    timeout_seconds: float


class WebResearchProvider:
    async def search(self, query: str, limit: int = 10) -> SearchProviderResult:
        raise NotImplementedError


class AnalysisProvider:
    async def analyze_json(self, *, prompt: str, system: str | None = None, max_tokens: int | None = None) -> dict[str, Any]:
        raise NotImplementedError


def truncate_sources(sources: list[ResearchSource], limit: int) -> list[ResearchSource]:
    if limit <= 0:
        return []
    return sources[:limit]

