"""
Question Selector — chooses the next question to ask the user.

Tiered selection:
1) Baseline required (same for everyone): stage, team size, goals (multi-select)
2) Optional follow-ups only when the user explicitly opts in (allow_optional)

Pure function — no LLM calls, no I/O.
"""

from __future__ import annotations

import logging

from app.schemas.profile_engine import Question, QuestionType, SessionState
from app.services.llm.prompts import QUESTION_BANK

logger = logging.getLogger(__name__)

# Maximum total questions to ask per session (prevents infinite loops)
_MAX_QUESTIONS_PER_SESSION = 4

_BASELINE_KEYS: list[str] = ["declared_stage", "team_size", "declared_goals"]


def select_question(state: SessionState) -> Question | None:
    """
    Select the next question to ask, or None if the session should proceed.

    Selection priority:
    1. Baseline required questions not yet answered.
    2. Optional questions only if allow_optional=True.
    3. None — enough information collected.

    Args:
        state: Current orchestrator session state.

    Returns:
        The next Question to present, or None if collection is complete.
    """
    if len(state.asked_questions) >= _MAX_QUESTIONS_PER_SESSION:
        logger.debug("Max questions reached for session %s", state.session_id)
        return None

    answered = set(state.asked_questions) | set(state.answers.keys())

    # --- Tier 1: Baseline required (fixed order) ---
    for key in _BASELINE_KEYS:
        if key not in answered:
            q_def = _lookup_def(key)
            if q_def:
                logger.debug("Selecting baseline question: %s", key)
                return _to_question(q_def)
            return Question(
                key=key,
                text=key,
                question_type=QuestionType.REQUIRED,
                context="Baseline required question.",
            )

    # --- Tier 2: Optional only if user opts in ---
    if state.allow_optional:
        for q_def in QUESTION_BANK:
            if q_def["type"] == QuestionType.OPTIONAL and q_def["key"] not in answered:
                logger.debug("Selecting optional question: %s", q_def["key"])
                return _to_question(q_def)

    logger.debug(
        "No more questions needed for session %s (confidence=%.2f)",
        state.session_id,
        state.confidence_score,
    )
    return None


def has_more_questions(state: SessionState) -> bool:
    """Convenience predicate used by the orchestrator."""
    return select_question(state) is not None


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _to_question(q_def: dict) -> Question:
    return Question(
        key=q_def["key"],
        text=q_def["text"],
        question_type=QuestionType(q_def.get("type", QuestionType.OPTIONAL)),
        context=q_def.get("context"),
    )


def _lookup_def(key: str) -> dict | None:
    for q_def in QUESTION_BANK:
        if q_def.get("key") == key:
            return q_def
    return None
