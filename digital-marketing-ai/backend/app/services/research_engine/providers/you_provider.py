from __future__ import annotations

import logging
from typing import Any

import httpx
from datetime import datetime, timezone

from app.core import config
from app.schemas.research_engine import ResearchProviderError, ResearchSource, SearchProviderResult
from app.services.research_engine.source_quality import extract_domain, score_source

logger = logging.getLogger(__name__)


class YouWebResearchProvider:
    """
    Minimal You.com provider.

    If YOU_API_KEY is missing, callers must treat this as a hard failure (no mock fallback).
    """

    def __init__(self, api_key: str | None = None, timeout_seconds: float | None = None) -> None:
        self._api_key = (api_key or config.YOU_API_KEY or "").strip()
        self._timeout = timeout_seconds if timeout_seconds is not None else config.RESEARCH_TIMEOUT_SECONDS

    def _ensure_key(self) -> None:
        if not self._api_key:
            raise ResearchProviderError("YOU_API_KEY is not configured")

    async def search(self, query: str, limit: int = 10) -> SearchProviderResult:
        self._ensure_key()
        q = (query or "").strip()
        if not q:
            raise ResearchProviderError("empty query")

        # NOTE: You.com APIs vary by plan/version. This implementation is intentionally defensive:
        # - It captures raw payload for debugging.
        # - It tries multiple likely shapes for web results.
        url = "https://api.you.com/api/search"
        headers = {"X-API-Key": self._api_key}
        params = {"query": q, "num_web_results": max(1, min(int(limit), 20))}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as http:
                resp = await http.get(url, headers=headers, params=params)
                resp.raise_for_status()
                raw: dict[str, Any] = resp.json()
        except httpx.HTTPStatusError as exc:
            raise ResearchProviderError(f"You.com returned {exc.response.status_code}: {exc.response.text[:200]}") from exc
        except httpx.RequestError as exc:
            raise ResearchProviderError(f"You.com request failed: {exc}") from exc
        except Exception as exc:
            raise ResearchProviderError(f"You.com response parse failed: {exc}") from exc

        sources: list[ResearchSource] = []

        # Common shapes we try:
        # - raw["web_results"] = [{"title","url","snippet"}]
        # - raw["results"]["web"] = [...]
        candidates: list[dict[str, Any]] = []
        if isinstance(raw.get("web_results"), list):
            candidates = [c for c in raw["web_results"] if isinstance(c, dict)]
        elif isinstance(raw.get("results"), dict) and isinstance(raw["results"].get("web"), list):
            candidates = [c for c in raw["results"]["web"] if isinstance(c, dict)]

        idx = 1
        for item in candidates:
            if idx > limit:
                break
            url_value = str(item.get("url") or item.get("link") or "").strip()
            if not url_value:
                continue
            title = str(item.get("title") or "").strip() or None
            snippet = str(item.get("snippet") or item.get("description") or "").strip() or None
            domain = extract_domain(url_value)
            credibility = score_source(url_value, source_type="web")

            sources.append(
                ResearchSource(
                    source_index=idx,
                    title=title,
                    url=url_value,
                    domain=domain,
                    snippet=snippet,
                    source_type="web",
                    fetched_at=datetime.now(timezone.utc),
                    credibility_score=credibility,
                    raw=item if isinstance(item, dict) else None,
                )
            )
            idx += 1

        return SearchProviderResult(query=q, sources=sources, raw_payload=raw)
