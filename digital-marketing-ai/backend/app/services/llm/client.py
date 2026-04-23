"""
LLM Client — single wrapper around the configured language model provider.

Reads configuration from environment variables.  Supports Cerebras
(OpenAI-compatible endpoint) by default; falls back gracefully to the
OpenAI SDK when CEREBRAS_API_KEY is absent.

All profile-engine services call this module exclusively — no direct
HTTP/API calls in service files.
"""

from __future__ import annotations

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
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
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
        async with httpx.AsyncClient(timeout=self._timeout) as http:
            try:
                response = await http.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "LLM API error %s: %s",
                    exc.response.status_code,
                    exc.response.text[:500],
                )
                raise LLMClientError(
                    f"LLM provider returned {exc.response.status_code}: {exc.response.text[:200]}"
                ) from exc
            except httpx.RequestError as exc:
                logger.error("LLM request failed: %s", exc)
                raise LLMClientError(f"LLM request failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_default_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Return a module-level singleton LLMClient (lazy-initialised)."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LLMClientError(RuntimeError):
    """Raised when the LLM provider returns an error or times out."""


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
