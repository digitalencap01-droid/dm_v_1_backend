"""
Readiness — assesses the business maturity/readiness level.

Responsibility: Combine extracted data with any follow-up answers,
call LLM, normalise, and return ReadinessAssessment.
"""

from __future__ import annotations

import logging

from app.schemas.profile_engine import (
    ClassifiedProfile,
    ExtractedData,
    ReadinessAssessment,
    ReadinessLevel,
)
from app.services.llm.client import LLMClient, LLMClientError
from app.services.llm.prompts import SYSTEM_READINESS, build_readiness_prompt
from app.services.profile_engine.normalizer import normalize_readiness

logger = logging.getLogger(__name__)


async def assess_readiness(
    extracted: ExtractedData,
    classified: ClassifiedProfile,
    answers: dict[str, str],
    llm: LLMClient,
) -> ReadinessAssessment:
    """
    Determine the business readiness / maturity stage.

    Args:
        extracted: Output from the extractor service.
        classified: Output from the classifier service.
        answers: Dict of follow-up question answers keyed by question key.
        llm: LLM client instance.

    Returns:
        ReadinessAssessment with typed enum and confidence score.
    """
    prompt = build_readiness_prompt(
        description=extracted.description or "",
        challenges=extracted.current_challenges,
        goals=extracted.mentioned_goals,
        raw_readiness_hint=extracted.raw_readiness_hint,
        answers=answers,
    )

    try:
        data = await llm.complete_json(prompt=prompt, system=SYSTEM_READINESS)
    except (LLMClientError, ValueError) as exc:
        logger.warning("Readiness LLM call failed: %s — using fallback", exc)
        return _fallback_from_hints(extracted)

    return _parse_llm_response(data, extracted)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_llm_response(data: dict, extracted: ExtractedData) -> ReadinessAssessment:
    try:
        raw_level = data.get("readiness_level", "")
        readiness_level = normalize_readiness(str(raw_level))

        readiness_confidence = _clamp(data.get("readiness_confidence", 0.4))
        if readiness_level == ReadinessLevel.UNKNOWN:
            readiness_confidence = min(readiness_confidence, 0.3)

        return ReadinessAssessment(
            readiness_level=readiness_level,
            readiness_confidence=readiness_confidence,
            reasoning=_safe_str(data.get("reasoning")),
            raw_readiness=_safe_str(data.get("raw_readiness")),
        )
    except Exception as exc:
        logger.warning("Failed to parse readiness response: %s", exc)
        return _fallback_from_hints(extracted)


def _fallback_from_hints(extracted: ExtractedData) -> ReadinessAssessment:
    level = normalize_readiness(extracted.raw_readiness_hint)
    confidence = extracted.confidence_hints.get("readiness", 0.2)
    return ReadinessAssessment(
        readiness_level=level,
        readiness_confidence=confidence,
        reasoning="Derived from text hints due to LLM unavailability.",
        raw_readiness=extracted.raw_readiness_hint,
    )


def _clamp(value: object, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.4


def _safe_str(value: object) -> str | None:
    if value is None or value == "":
        return None
    return str(value).strip() or None
