"""
LLM Client — single wrapper around the configured language model provider.

Reads configuration from environment variables.  Supports Cerebras
(OpenAI-compatible endpoint) by default; falls back gracefully to the
OpenAI SDK when CEREBRAS_API_KEY is absent.

All profile-engine services call this module exclusively — no direct
HTTP/API calls in service files.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

_CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
_OPENAI_BASE_URL = "https://api.openai.com/v1"
_DEFAULT_CONTEXT_LIMIT = 32000
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def _active_provider() -> tuple[str, str, str]:
    """Return (base_url, api_key, model) for the configured provider."""
    cerebras_key = os.getenv("CEREBRAS_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    cerebras_model = os.getenv("CEREBRAS_MODEL", "llama3.1-8b")

    if cerebras_key:
        return _CEREBRAS_BASE_URL, cerebras_key, cerebras_model
    if openai_key:
        return _OPENAI_BASE_URL, openai_key, os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Demo/dev fallback:
    # return a provider tuple even when no key is present so downstream services
    # can catch LLMClientError and use their heuristic fallbacks instead of
    # crashing the entire request with a 500.
    logger.warning("No LLM provider configured; running in fallback mode.")
    return _CEREBRAS_BASE_URL, "", cerebras_model


# ---------------------------------------------------------------------------
# Core client
# ---------------------------------------------------------------------------


class LLMClient:
    """
    Lightweight async wrapper around any OpenAI-compatible chat completion API.

    Usage::

        client = LLMClient()
        text = await client.complete("Classify this business: ...")
        data = await client.complete_json("Extract JSON from: ...", schema_hint="...")
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 60.0,
    ) -> None:
        detected_url, detected_key, detected_model = _active_provider()
        self._base_url = (base_url or detected_url).rstrip("/")
        self._api_key = api_key or detected_key
        self._model = model or detected_model
        env_temp = float(os.getenv("CEREBRAS_TEMPERATURE", "0.3"))
        env_max = int(os.getenv("CEREBRAS_MAX_TOKENS", "4096"))
        self._temperature = temperature if temperature is not None else env_temp
        self._max_tokens = max_tokens or env_max
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Send a chat completion request and return the assistant message text.

        Args:
            prompt: User-turn content.
            system: Optional system message prepended to the conversation.
            temperature: Override instance temperature for this call.
            max_tokens: Override max tokens for this call.

        Returns:
            The raw assistant reply as a string.
        """
        payload_messages = self._fit_messages_to_context(
            messages=[{"role": "user", "content": prompt}],
            system=system,
            max_tokens=max_tokens or self._max_tokens,
        )

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": payload_messages,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens or self._max_tokens,
        }

        response = await self._post("/chat/completions", payload)
        return response["choices"][0]["message"]["content"].strip()

    async def complete_messages(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Send a multi-turn chat completion request and return assistant text.
        """
        payload_messages = self._fit_messages_to_context(
            messages=messages,
            system=system,
            max_tokens=max_tokens or self._max_tokens,
        )

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": payload_messages,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens or self._max_tokens,
        }

        response = await self._post("/chat/completions", payload)
        return response["choices"][0]["message"]["content"].strip()

    async def complete_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Send a prompt that expects a JSON object in the response.
        Strips markdown code fences and parses the result.

        Returns:
            Parsed JSON as a dict.  Raises ValueError on parse failure.
        """
        raw = await self.complete(
            prompt=prompt,
            system=system or "You are a precise JSON extractor. Respond ONLY with valid JSON. No markdown, no explanation.",
            temperature=temperature if temperature is not None else 0.1,
            max_tokens=max_tokens or self._max_tokens,
        )
        return _parse_json_response(raw)

    # ------------------------------------------------------------------
    # Internal HTTP helpers
    # ------------------------------------------------------------------

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        max_attempts = max(1, int(os.getenv("LLM_MAX_RETRIES", "5")))
        backoff_seconds = max(0.25, float(os.getenv("LLM_RETRY_BACKOFF_SECONDS", "2.0")))

        async with httpx.AsyncClient(timeout=self._timeout) as http:
            for attempt in range(1, max_attempts + 1):
                try:
                    response = await http.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    return response.json()
                except httpx.HTTPStatusError as exc:
                    status_code = exc.response.status_code
                    response_text = exc.response.text[:500]
                    logger.error("LLM API error %s: %s", status_code, response_text)

                    if attempt < max_attempts and status_code in _RETRYABLE_STATUS_CODES:
                        await asyncio.sleep(backoff_seconds ** attempt)
                        continue

                    error_cls = LLMRateLimitError if status_code == 429 else LLMClientError
                    raise error_cls(
                        message=f"LLM provider returned {status_code}: {exc.response.text[:200]}",
                        status_code=status_code,
                        response_text=exc.response.text[:500],
                    ) from exc
                except httpx.RequestError as exc:
                    logger.error("LLM request failed: %s", exc)
                    if attempt < max_attempts:
                        await asyncio.sleep(backoff_seconds ** attempt)
                        continue
                    raise LLMClientError(f"LLM request failed: {exc}") from exc

    def _fit_messages_to_context(
        self,
        messages: list[dict[str, str]],
        system: str | None,
        max_tokens: int,
    ) -> list[dict[str, str]]:
        """
        Keep the newest messages and trim oversized content so the request stays
        below the provider context limit.
        """
        context_limit = int(os.getenv("LLM_CONTEXT_LIMIT", str(_DEFAULT_CONTEXT_LIMIT)))
        input_budget = max(1200, context_limit - max_tokens - 512)

        fitted: list[dict[str, str]] = []
        used = 0

        if system:
            system_budget = max(600, int(input_budget * 0.55))
            system_content = self._truncate_text(system, system_budget)
            fitted.append({"role": "system", "content": system_content})
            used += self._estimate_tokens(system_content)

        kept_reversed: list[dict[str, str]] = []
        for message in reversed(messages):
            role = message.get("role", "user")
            content = message.get("content", "")
            if not content:
                continue

            remaining = input_budget - used
            if remaining <= 120:
                break

            truncated_content = self._truncate_text(content, remaining)
            message_tokens = self._estimate_tokens(truncated_content)
            if message_tokens > remaining:
                continue

            kept_reversed.append({"role": role, "content": truncated_content})
            used += message_tokens

        fitted.extend(reversed(kept_reversed))
        return fitted

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def _truncate_text(self, text: str, max_tokens: int) -> str:
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        marker = "\n[Middle content trimmed for context length.]\n"
        available = max(0, max_chars - len(marker))
        head_chars = int(available * 0.6)
        tail_chars = max(0, available - head_chars)
        head = text[:head_chars].rstrip()
        tail = text[-tail_chars:].lstrip() if tail_chars else ""
        return f"{head}{marker}{tail}"


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_default_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    return LLMClient()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LLMClientError(RuntimeError):
    """Raised when the LLM provider returns an error or times out."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_text: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


class LLMRateLimitError(LLMClientError):
    """Raised when the LLM provider is temporarily rate limited."""


# ---------------------------------------------------------------------------
# JSON parse helper
# ---------------------------------------------------------------------------


def _parse_json_response(raw: str) -> dict[str, Any]:
    """Strip markdown fences and parse JSON from an LLM response."""
    text = raw.strip()
    # Remove ```json ... ``` or ``` ... ``` wrappers
    if text.startswith("```"):
        lines = text.splitlines()
        inner = [
            l for l in lines
            if not l.strip().startswith("```")
        ]
        text = "\n".join(inner).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse LLM JSON response: %s | raw=%s", exc, raw[:300])
        raise ValueError(f"LLM did not return valid JSON: {exc}") from exc
