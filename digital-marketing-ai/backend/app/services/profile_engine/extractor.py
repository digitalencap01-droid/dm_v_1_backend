"""
Extractor — extracts structured business details from raw free-form input.

Responsibility: Call LLM with the extraction prompt, parse the JSON response,
and return a validated ExtractedData instance.  No classification or scoring.
"""

from __future__ import annotations

import logging

from app.schemas.profile_engine import ExtractedData
from app.services.llm.client import LLMClient, LLMClientError
from app.services.llm.prompts import SYSTEM_EXTRACTION, build_extraction_prompt

logger = logging.getLogger(__name__)


async def extract(raw_input: str, llm: LLMClient) -> ExtractedData:
    """
    Extract structured business information from raw free-form text.

    Args:
        raw_input: Raw business description provided by the user.
        llm: LLM client instance.

    Returns:
        ExtractedData populated from the LLM response.

    Raises:
        ExtractionError: When the LLM returns unusable output.
    """
    prompt = build_extraction_prompt(raw_input)
    try:
        data = await llm.complete_json(prompt=prompt, system=SYSTEM_EXTRACTION)
    except (LLMClientError, ValueError) as exc:
        logger.warning("Extraction LLM call failed: %s — using minimal fallback", exc)
        return _minimal_fallback(raw_input)

    return _parse_llm_response(data, raw_input)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_llm_response(data: dict, raw_input: str) -> ExtractedData:
    """Safely build ExtractedData from the LLM JSON payload."""
    try:
        return ExtractedData(
            business_name=_safe_str(data.get("business_name")),
            description=_safe_str(data.get("description")) or _truncate(raw_input, 200),
            target_audience=_safe_str(data.get("target_audience")),
            product_or_service=_safe_str(data.get("product_or_service")),
            revenue_model=_safe_str(data.get("revenue_model")),
            current_challenges=_safe_list(data.get("current_challenges")),
            mentioned_goals=_safe_list(data.get("mentioned_goals")),
            mentioned_channels=_safe_list(data.get("mentioned_channels")),
            raw_persona_hint=_safe_str(data.get("raw_persona_hint")),
            raw_industry_hint=_safe_str(data.get("raw_industry_hint")),
            raw_readiness_hint=_safe_str(data.get("raw_readiness_hint")),
            confidence_hints=_safe_confidence(data.get("confidence_hints")),
        )
    except Exception as exc:
        logger.warning("Failed to parse extraction response: %s", exc)
        return _minimal_fallback(raw_input)


def _minimal_fallback(raw_input: str) -> ExtractedData:
    """Return a bare-minimum ExtractedData when LLM fails."""
    return ExtractedData(
        description=_truncate(raw_input, 200),
        confidence_hints={"persona": 0.1, "industry": 0.1, "readiness": 0.1},
    )


def _safe_str(value: object) -> str | None:
    if value is None or value == "":
        return None
    return str(value).strip() or None


def _safe_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if item]


def _safe_confidence(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for k, v in value.items():
        try:
            result[str(k)] = max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            pass
    return result


def _truncate(text: str, max_len: int) -> str:
    return text[:max_len].strip() if text else ""


class ExtractionError(RuntimeError):
    """Raised when extraction cannot produce usable output."""
