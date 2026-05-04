"""
Need Routing — maps the business profile to a primary need-state.

Responsibility: Combine classified profile, readiness, and answers,
call LLM, normalise, and return NeedRouting with primary and secondary needs.
"""

from __future__ import annotations

import logging
import json

from app.schemas.profile_engine import (
    ClassifiedProfile,
    ExtractedData,
    NeedRouting,
    NeedState,
    ReadinessAssessment,
)
from app.services.llm.client import LLMClient, LLMClientError
from app.services.llm.prompts import SYSTEM_NEED_ROUTING, build_need_routing_prompt
from app.services.profile_engine.normalizer import normalize_need_list, normalize_need_state

logger = logging.getLogger(__name__)


async def route_need(
    extracted: ExtractedData,
    classified: ClassifiedProfile,
    readiness: ReadinessAssessment,
    answers: dict[str, str],
    llm: LLMClient,
) -> NeedRouting:
    """
    Determine the primary and secondary business needs.

    Args:
        extracted: Output from the extractor service.
        classified: Output from the classifier service.
        readiness: Output from the readiness service.
        answers: Dict of follow-up question answers.
        llm: LLM client instance.

    Returns:
        NeedRouting with primary need, secondary needs, and confidence.
    """
    prompt = build_need_routing_prompt(
        description=extracted.description or "",
        persona=classified.persona.value,
        industry=classified.industry.value,
        readiness_level=readiness.readiness_level.value,
        challenges=extracted.current_challenges,
        goals=extracted.mentioned_goals,
        answers=answers,
    )

    try:
        data = await llm.complete_json(prompt=prompt, system=SYSTEM_NEED_ROUTING)
    except (LLMClientError, ValueError) as exc:
        logger.warning("Need-routing LLM call failed: %s — using heuristic fallback", exc)
        return _heuristic_fallback(extracted, readiness, answers)

    return _parse_llm_response(data, extracted, readiness, answers)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_llm_response(
    data: dict,
    extracted: ExtractedData,
    readiness: ReadinessAssessment,
    answers: dict[str, str],
) -> NeedRouting:
    try:
        raw_primary = data.get("primary_need", "")
        raw_secondary = data.get("secondary_needs", [])
        if not isinstance(raw_secondary, list):
            raw_secondary = []

        primary_need = normalize_need_state(str(raw_primary))
        secondary_needs = normalize_need_list([str(n) for n in raw_secondary])
        # Ensure primary is not duplicated in secondary
        secondary_needs = [n for n in secondary_needs if n != primary_need]

        need_confidence = _clamp(data.get("need_confidence", 0.5))
        if primary_need == NeedState.UNKNOWN:
            need_confidence = min(need_confidence, 0.3)

        if primary_need == NeedState.UNKNOWN:
            fallback = _goal_to_primary_need_fallback(answers)
            if fallback is not None:
                primary_need = fallback
                need_confidence = max(need_confidence, 0.5)

        return NeedRouting(
            primary_need=primary_need,
            secondary_needs=secondary_needs,
            need_confidence=need_confidence,
            reasoning=_safe_str(data.get("reasoning"))
            or ("Deterministic fallback from declared goals." if primary_need != NeedState.UNKNOWN else None),
        )
    except Exception as exc:
        logger.warning("Failed to parse need-routing response: %s", exc)
        return _heuristic_fallback(extracted, readiness, answers)


def _heuristic_fallback(
    extracted: ExtractedData,
    readiness: ReadinessAssessment,
    answers: dict[str, str],
) -> NeedRouting:
    """
    Simple rule-based fallback when LLM is unavailable.
    Maps readiness level to a sensible default need state.
    """
    from app.schemas.profile_engine import ReadinessLevel

    level_to_need: dict[ReadinessLevel, NeedState] = {
        ReadinessLevel.IDEA_STAGE: NeedState.BRAND_AWARENESS,
        ReadinessLevel.MVP: NeedState.LEAD_GENERATION,
        ReadinessLevel.EARLY_TRACTION: NeedState.LEAD_GENERATION,
        ReadinessLevel.SCALING: NeedState.REVENUE_GROWTH,
        ReadinessLevel.MATURE: NeedState.CUSTOMER_RETENTION,
        ReadinessLevel.UNKNOWN: NeedState.UNKNOWN,
    }
    primary = level_to_need.get(readiness.readiness_level, NeedState.UNKNOWN)
    if primary == NeedState.UNKNOWN:
        fallback = _goal_to_primary_need_fallback(answers)
        if fallback is not None:
            primary = fallback
    return NeedRouting(
        primary_need=primary,
        secondary_needs=[],
        need_confidence=0.5 if primary != NeedState.UNKNOWN else 0.25,
        reasoning=(
            "Deterministic fallback from declared goals."
            if primary != NeedState.UNKNOWN
            else "Heuristic fallback based on readiness level."
        ),
    )


def _goal_to_primary_need_fallback(answers: dict[str, str]) -> NeedState | None:
    raw = answers.get("declared_goals")
    if not isinstance(raw, str) or not raw.strip():
        return None

    parsed: list[str] = []
    text = raw.strip()
    if text.startswith("["):
        try:
            value = json.loads(text)
            if isinstance(value, list):
                parsed = [str(x) for x in value]
        except Exception:
            parsed = []
    else:
        parsed = [p.strip() for p in text.split(",") if p.strip()]

    mapping: dict[str, NeedState] = {
        "brand_awareness": NeedState.BRAND_AWARENESS,
        "lead_generation": NeedState.LEAD_GENERATION,
        "customer_retention": NeedState.CUSTOMER_RETENTION,
        "revenue_growth": NeedState.REVENUE_GROWTH,
        "product_launch": NeedState.PRODUCT_LAUNCH,
        "market_expansion": NeedState.MARKET_EXPANSION,
    }
    for item in parsed:
        key = str(item).strip().lower()
        if key in mapping:
            return mapping[key]

    return None


def _clamp(value: object, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(value)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.5


def _safe_str(value: object) -> str | None:
    if value is None or value == "":
        return None
    return str(value).strip() or None
