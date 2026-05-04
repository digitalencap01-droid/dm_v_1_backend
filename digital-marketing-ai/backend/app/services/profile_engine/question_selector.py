"""
Question Selector — chooses the next question to ask the user.

Tiered selection:
1) Baseline required (same for everyone): stage, team size, goals (multi-select)
2) Optional follow-ups only when the user explicitly opts in (allow_optional)

Pure function — no LLM calls, no I/O.
"""

from __future__ import annotations

import logging

from app.schemas.profile_engine import Question, QuestionOption, QuestionType, SessionState, ReadinessLevel
from app.services.llm.prompts import QUESTION_BANK

logger = logging.getLogger(__name__)

# Maximum total questions to ask per session (prevents infinite loops)
# Keep this high enough to allow baseline + dynamic required + optional follow-ups.
_MAX_QUESTIONS_PER_SESSION = 10

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
    answered = set(state.asked_questions) | set(state.answers.keys())

    if len(answered) >= _MAX_QUESTIONS_PER_SESSION:
        logger.debug("Max questions reached for session %s", state.session_id)
        return None

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
            q_type = q_def.get("type", QuestionType.OPTIONAL)
            try:
                q_type_enum = q_type if isinstance(q_type, QuestionType) else QuestionType(str(q_type))
            except ValueError:
                q_type_enum = QuestionType.OPTIONAL

            # Hard UX rule: never ask website for idea-stage users.
            if state.declared_stage == ReadinessLevel.IDEA_STAGE and q_def.get("key") == "website_url":
                continue

            if q_type_enum == QuestionType.OPTIONAL and q_def.get("key") not in answered:
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
    """Convert question definition (from QUESTION_BANK) to Question schema.
    
    Handles new optional fields with safe fallbacks for backward compatibility.
    """
    # Parse options from question definition
    options = []
    if "options" in q_def:
        for opt in q_def["options"]:
            if not isinstance(opt, dict):
                continue
            options.append(
                QuestionOption(
                    value=str(opt.get("value") or ""),
                    label=str(opt.get("label") or ""),
                    requires_text=bool(opt.get("requires_text") or False),
                    text_placeholder=(
                        str(opt.get("text_placeholder")).strip()
                        if opt.get("text_placeholder") is not None
                        else None
                    ),
                )
            )
    
    return Question(
        key=q_def["key"],
        text=q_def["text"],
        question_type=QuestionType(q_def.get("type", QuestionType.OPTIONAL)),
        context=q_def.get("context"),
        options=options,
        input_type=q_def.get("input_type", "text"),
        allow_multiple=q_def.get("allow_multiple", False),
        allow_custom=q_def.get("allow_custom", True),
    )


def _lookup_def(key: str) -> dict | None:
    for q_def in QUESTION_BANK:
        if q_def.get("key") == key:
            return q_def
    return None
