from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_TAVILY_BASE_URL = "https://api.tavily.com"
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class TavilyClientError(RuntimeError):
    """Raised when a Tavily request fails or cannot complete."""


class TavilyResearchClient:
    """Minimal async Tavily client using the official REST API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = _TAVILY_BASE_URL,
        timeout: float = 30.0,       # ✅ 90.0 → 30.0 per request
        max_retries: int = 3,        # ✅ retry count
        max_concurrent_requests: int = 3,  # ✅ Rate limit: max 3 concurrent requests
    ) -> None:
        self._api_key = (api_key or os.getenv("TAVILY_API_KEY") or os.getenv("api_key") or "").strip()
        self._project_id = (os.getenv("TAVILY_PROJECT") or "").strip() or None
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)  # ✅ Rate limiter

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    async def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        payload = {"query": query, **kwargs}
        return await self._post("/search", payload)

    async def extract(self, urls: str | list[str], **kwargs: Any) -> dict[str, Any]:
        payload = {"urls": urls, **kwargs}
        return await self._post("/extract", payload)

    async def map(self, url: str, **kwargs: Any) -> dict[str, Any]:
        payload = {"url": url, **kwargs}
        return await self._post("/map", payload)

    async def crawl(self, url: str, **kwargs: Any) -> dict[str, Any]:
        payload = {"url": url, **kwargs}
        return await self._post("/crawl", payload)

    async def research(self, input_text: str, **kwargs: Any) -> dict[str, Any]:
        payload = {"input": input_text, **kwargs}
        return await self._post("/research", payload)

    async def get_research(self, request_id: str) -> dict[str, Any]:
        return await self._get(f"/research/{request_id}")

    async def research_and_wait(
        self,
        input_text: str,
        *,
        timeout_seconds: float = 180.0,  # ✅ 120 → 180
        poll_interval: float = 5.0,      # ✅ 3 → 5
        **kwargs: Any,
    ) -> dict[str, Any]:
        created = await self.research(input_text, **kwargs)
        request_id = created.get("request_id")
        if not request_id:
            raise TavilyClientError("Tavily research did not return a request_id.")

        deadline = time.monotonic() + timeout_seconds
        latest = created

        while time.monotonic() < deadline:
            await asyncio.sleep(poll_interval)
            latest = await self.get_research(request_id)
            status = (latest.get("status") or "").lower()
            if status in {"completed", "failed", "error", "cancelled"}:
                return latest

        raise TavilyClientError("Timed out while waiting for Tavily research task to finish.")

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    def _make_timeout(self) -> httpx.Timeout:
        """✅ Granular timeout — connect aur read alag alag"""
        return httpx.Timeout(
            connect=10.0,        # connection banane ka time
            read=self._timeout,  # response padhne ka time
            write=10.0,          # request likhne ka time
            pool=5.0,            # pool se connection lene ka time
        )

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.configured:
            raise TavilyClientError("Tavily API key is not configured.")

        async with self._semaphore:  # ✅ Rate limit: max concurrent requests
            for attempt in range(1, self._max_retries + 1):
                async with httpx.AsyncClient(timeout=self._make_timeout()) as client:
                    try:
                        response = await client.post(
                            f"{self._base_url}{path}",
                            json=payload,
                            headers=self._headers(),
                        )
                        response.raise_for_status()
                        return response.json()

                    except httpx.HTTPStatusError as exc:
                        body = exc.response.text[:500]
                        logger.error(
                            "Tavily API error %s on %s (attempt %d/%d): %s",
                            exc.response.status_code, path, attempt, self._max_retries, body
                        )
                        # ✅ 429 aur 5xx par retry karo
                        if attempt < self._max_retries and exc.response.status_code in _RETRYABLE_STATUS_CODES:
                            wait = 2 ** attempt  # 2s, 4s, 8s
                            logger.warning("Retrying %s in %.1fs...", path, wait)
                            await asyncio.sleep(wait)
                            continue
                        raise TavilyClientError(
                            f"Tavily API returned {exc.response.status_code} for {path}: {body[:200]}"
                        ) from exc

                    except httpx.RequestError as exc:
                        # ✅ Fix: type naam bhi log karo — empty message problem solve
                        error_detail = f"{type(exc).__name__}: {exc}"
                        logger.error(
                            "Tavily request failed on %s (attempt %d/%d): %s",
                            path, attempt, self._max_retries, error_detail
                        )
                        # ✅ Network error par bhi retry karo
                        if attempt < self._max_retries:
                            wait = 2 ** attempt  # 2s, 4s, 8s
                            logger.warning("Retrying %s in %.1fs...", path, wait)
                            await asyncio.sleep(wait)
                            continue
                        raise TavilyClientError(
                            f"Tavily request failed for {path}: {error_detail}"
                        ) from exc

    async def _get(self, path: str) -> dict[str, Any]:
        if not self.configured:
            raise TavilyClientError("Tavily API key is not configured.")

        async with self._semaphore:  # ✅ Rate limit: max concurrent requests
            for attempt in range(1, self._max_retries + 1):
                async with httpx.AsyncClient(timeout=self._make_timeout()) as client:
                    try:
                        response = await client.get(
                            f"{self._base_url}{path}",
                            headers=self._headers(),
                        )
                        response.raise_for_status()
                        return response.json()

                    except httpx.HTTPStatusError as exc:
                        body = exc.response.text[:500]
                        logger.error(
                            "Tavily API error %s on %s (attempt %d/%d): %s",
                            exc.response.status_code, path, attempt, self._max_retries, body
                        )
                        if attempt < self._max_retries and exc.response.status_code in _RETRYABLE_STATUS_CODES:
                            wait = 2 ** attempt
                            logger.warning("Retrying %s in %.1fs...", path, wait)
                            await asyncio.sleep(wait)
                            continue
                        raise TavilyClientError(
                            f"Tavily API returned {exc.response.status_code} for {path}: {body[:200]}"
                        ) from exc

                    except httpx.RequestError as exc:
                        error_detail = f"{type(exc).__name__}: {exc}"
                        logger.error(
                            "Tavily request failed on %s (attempt %d/%d): %s",
                            path, attempt, self._max_retries, error_detail
                        )
                        if attempt < self._max_retries:
                            wait = 2 ** attempt
                            logger.warning("Retrying %s in %.1fs...", path, wait)
                            await asyncio.sleep(wait)
                            continue
                        raise TavilyClientError(
                            f"Tavily request failed for {path}: {error_detail}"
                        ) from exc

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._project_id:
            headers["X-Project-ID"] = self._project_id
        return headers