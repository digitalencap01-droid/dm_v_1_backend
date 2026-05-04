from __future__ import annotations

from typing import Any

from app.core import config
from app.services.llm.client import LLMClient, LLMClientError, get_llm_client
from app.schemas.research_engine import ResearchProviderError


class CerebrasAnalysisProvider:
    """
    Analysis provider that uses the existing LLMClient (OpenAI-compatible),
    configured for Cerebras via env (CEREBRAS_API_KEY, CEREBRAS_MODEL).

    Rule: never use this provider for live facts; only analyze provided sources.
    """

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or get_llm_client()

    async def analyze_json(
        self,
        *,
        prompt: str,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        try:
            return await self._llm.complete_json(
                prompt=prompt,
                system=system,
                max_tokens=max_tokens or config.MAX_ANALYSIS_TOKENS,
            )
        except (LLMClientError, ValueError) as exc:
            raise ResearchProviderError(str(exc)) from exc

