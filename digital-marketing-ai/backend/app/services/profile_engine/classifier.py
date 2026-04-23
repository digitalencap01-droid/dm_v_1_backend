"""
Classifier — classifies persona type and industry from extracted data.

Responsibility: Call LLM with the classification prompt, parse response,
normalise enum values, and return ClassifiedProfile.
"""

from __future__ import annotations

import logging

from app.schemas.profile_engine import ClassifiedProfile, ExtractedData, IndustryType, PersonaType
from app.services.llm.client import LLMClient, LLMClientError
from app.services.llm.prompts import SYSTEM_CLASSIFICATION, build_classification_prompt
from app.services.profile_engine.normalizer import normalize_industry, normalize_persona

logger = logging.getLogger(__name__)


async def classify(extracted: ExtractedData, llm: LLMClient) -> ClassifiedProfile:
    """
    Classify persona and industry from extracted business data.

    Args:
        extracted: Output from the extractor service.
        llm: LLM client instance.

    Returns:
        ClassifiedProfile with typed enum values and confidence scores.
    """
    prompt = build_classification_prompt(
        description=extracted.description or "",
        target_audience=extracted.target_audience,
        product_or_service=extracted.product_or_service,
        raw_persona_hint=extracted.raw_persona_hint,
        raw_industry_hint=extracted.raw_industry_hint,
    )

    try:
        data = await llm.complete_json(prompt=prompt, system=SYSTEM_CLASSIFICATION)
    except (LLMClientError, ValueError) as exc:
        logger.warning("Classification LLM call failed: %s — using fallback", exc)
        return _fallback_from_hints(extracted)

    return _parse_llm_response(data, extracted)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_llm_response(data: dict, extracted: ExtractedData) -> ClassifiedProfile:
    """Safely build ClassifiedProfile from LLM output."""
    try:
        raw_persona = data.get("persona", "")
        raw_industry = data.get("industry", "")

        persona = normalize_persona(str(raw_persona))
        industry = normalize_industry(str(raw_industry))

        persona_confidence = _clamp(data.get("persona_confidence", 0.5))
        industry_confidence = _clamp(data.get("industry_confidence", 0.5))

        # If normalisation fell back to UNKNOWN, reduce confidence
        if persona == PersonaType.UNKNOWN:
            persona_confidence = min(persona_confidence, 0.3)
        if industry == IndustryType.OTHER:
            industry_confidence = min(industry_confidence, 0.4)

        return ClassifiedProfile(
            persona=persona,
            industry=industry,
            persona_confidence=persona_confidence,
            industry_confidence=industry_confidence,
            raw_persona=str(raw_persona) if raw_persona else None,
            raw_industry=str(raw_industry) if raw_industry else None,
        )
    except Exception as exc:
        logger.warning("Failed to parse classification response: %s", exc)
        return _fallback_from_hints(extracted)


def _fallback_from_hints(extracted: ExtractedData) -> ClassifiedProfile:
    """Derive a best-effort classification from raw hint strings."""
    persona = normalize_persona(extracted.raw_persona_hint)
    industry = normalize_industry(extracted.raw_industry_hint)
    hints = extracted.confidence_hints
    return ClassifiedProfile(
        persona=persona,
        industry=industry,
        persona_confidence=hints.get("persona", 0.2),
        industry_confidence=hints.get("industry", 0.2),
        raw_persona=extracted.raw_persona_hint,
        raw_industry=extracted.raw_industry_hint,
    )


def _clamp(value: object, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.5
