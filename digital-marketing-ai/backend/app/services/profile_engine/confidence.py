"""
Confidence — calculates the overall profile confidence score.

Responsibility: Aggregate per-dimension confidence scores into a single
0.0–1.0 float that drives question selection and profile readiness gating.

Pure function — no LLM calls, no I/O.
"""

from __future__ import annotations

import logging

from app.schemas.profile_engine import (
    ClassifiedProfile,
    NeedRouting,
    NeedState,
    PersonaType,
    ReadinessAssessment,
    ReadinessLevel,
    SessionState,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weight configuration
# ---------------------------------------------------------------------------

# Weights must sum to 1.0
_WEIGHTS: dict[str, float] = {
    "persona": 0.20,
    "industry": 0.20,
    "readiness": 0.20,
    "need": 0.25,
    "answers_bonus": 0.15,
}

# Bonus per answered question (up to the answers_bonus weight ceiling)
_ANSWER_BONUS_PER_QUESTION = 0.05


def calculate_confidence(state: SessionState) -> float:
    """
    Compute a composite confidence score [0.0, 1.0] for the current session.

    Factors:
    - Individual dimension confidences from service outputs.
    - Whether key fields are still UNKNOWN/OTHER.
    - Number of follow-up answers already provided.

    Args:
        state: Current SessionState with partial or full service outputs.

    Returns:
        Normalised confidence float.
    """
    score = 0.0

    if state.classified:
        score += _WEIGHTS["persona"] * _persona_score(state.classified)
        score += _WEIGHTS["industry"] * _industry_score(state.classified)
    else:
        # Before classification, use hints from extraction
        if state.extracted:
            hints = state.extracted.confidence_hints
            score += _WEIGHTS["persona"] * hints.get("persona", 0.0)
            score += _WEIGHTS["industry"] * hints.get("industry", 0.0)

    if state.readiness:
        score += _WEIGHTS["readiness"] * _readiness_score(state.readiness)
    elif state.extracted:
        hints = state.extracted.confidence_hints
        score += _WEIGHTS["readiness"] * hints.get("readiness", 0.0)

    if state.need_routing:
        score += _WEIGHTS["need"] * _need_score(state.need_routing)

    # Answer bonus
    num_answers = len(state.answers)
    raw_bonus = min(num_answers * _ANSWER_BONUS_PER_QUESTION, _WEIGHTS["answers_bonus"])
    score += raw_bonus

    result = round(min(max(score, 0.0), 1.0), 4)
    logger.debug("Confidence for session %s: %.4f", state.session_id, result)
    return result


# ---------------------------------------------------------------------------
# Dimension scorers
# ---------------------------------------------------------------------------


def _persona_score(classified: ClassifiedProfile) -> float:
    base = classified.persona_confidence
    if classified.persona == PersonaType.UNKNOWN:
        return base * 0.3
    return base


def _industry_score(classified: ClassifiedProfile) -> float:
    base = classified.industry_confidence
    return base


def _readiness_score(readiness: ReadinessAssessment) -> float:
    base = readiness.readiness_confidence
    if readiness.readiness_level == ReadinessLevel.UNKNOWN:
        return base * 0.3
    return base


def _need_score(need_routing: NeedRouting) -> float:
    base = need_routing.need_confidence
    if need_routing.primary_need == NeedState.UNKNOWN:
        return base * 0.2
    return base
