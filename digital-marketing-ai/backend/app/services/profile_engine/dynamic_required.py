"""
Dynamic required questions (slot-based).

Purpose:
- Keep required questioning minimal and business-specific.
- Represent \"bare minimum\" as required slots: OFFER + ICP.
- Generate the next best required question via LLM (with safe fallbacks).
- Extract a clean slot value from the user's answer and mark it filled.
"""

from __future__ import annotations

import logging

from app.schemas.profile_engine import (
    ResearchSlot,
    SessionState,
    Question,
    QuestionOption,
    QuestionType,
)
from app.services.llm.client import LLMClient, LLMClientError
from app.services.llm.prompts import (
    SYSTEM_REQUIRED_QUESTION_GENERATOR,
    SYSTEM_SLOT_FILLER,
    build_required_question_prompt_with_options,
    build_slot_fill_prompt,
)

logger = logging.getLogger(__name__)


def missing_required_slots(state: SessionState) -> list[ResearchSlot]:
    """
    Determine which required slots are still missing.

    This is deterministic.  We treat extracted signals as already filling slots.
    """
    filled = dict(state.required_slots_filled or {})

    # Seed from extraction if present
    if state.extracted:
        if state.extracted.product_or_service and ResearchSlot.OFFER not in filled:
            filled[ResearchSlot.OFFER] = state.extracted.product_or_service
        if state.extracted.target_audience and ResearchSlot.ICP not in filled:
            filled[ResearchSlot.ICP] = state.extracted.target_audience

    required = [ResearchSlot.OFFER, ResearchSlot.ICP]
    missing = [slot for slot in required if slot not in filled or not filled[slot].strip()]
    state.required_slots_filled = filled
    state.required_slots_missing = missing
    return missing


async def generate_required_question(state: SessionState, llm: LLMClient) -> Question:
    """
    Generate a dynamic required question for the next missing slot.

    Increments state.required_questions_asked.
    """
    missing = missing_required_slots(state)
    slot = missing[0] if missing else ResearchSlot.OFFER
    n = (state.required_questions_asked or 0) + 1

    baseline = {
        "declared_stage": state.declared_stage.value,
        "declared_goals": [g.value for g in state.declared_goals],
        "team_size": state.answers.get("team_size"),
    }
    extracted = state.extracted.model_dump() if state.extracted else {}

    try:
        payload = await llm.complete_json(
            prompt=build_required_question_prompt_with_options(
                slot=slot.value,
                raw_input=state.raw_input or "",
                extracted=extracted,
                baseline=baseline,
            ),
            system=SYSTEM_REQUIRED_QUESTION_GENERATOR,
        )

        question_text = str(payload.get("question") or "").strip()
        question_key = str(payload.get("question_key") or f"req_{slot.value}_{n}").strip()
        reason = str(payload.get("reason") or "Required research-minimum slot.").strip()
        out_slot = str(payload.get("slot") or slot.value).strip()
        slot_enum = ResearchSlot(out_slot) if out_slot in (ResearchSlot.OFFER.value, ResearchSlot.ICP.value) else slot

        input_type = str(payload.get("input_type") or "text").strip()
        if input_type not in ("text", "single_select", "multi_select"):
            input_type = "text"

        allow_multiple = payload.get("allow_multiple")
        if not isinstance(allow_multiple, bool):
            allow_multiple = (input_type == "multi_select")

        allow_custom = payload.get("allow_custom")
        if not isinstance(allow_custom, bool):
            allow_custom = True

        options: list[QuestionOption] = []
        raw_options = payload.get("options") or []
        if isinstance(raw_options, list) and input_type in ("single_select", "multi_select"):
            for opt in raw_options[:12]:
                if not isinstance(opt, dict):
                    continue
                value = str(opt.get("value") or "").strip()
                label = str(opt.get("label") or "").strip()
                if value and label:
                    requires_text = opt.get("requires_text")
                    if not isinstance(requires_text, bool):
                        requires_text = False
                    text_placeholder = opt.get("text_placeholder")
                    if text_placeholder is not None:
                        text_placeholder = str(text_placeholder).strip() or None

                    options.append(
                        QuestionOption(
                            value=value,
                            label=label,
                            requires_text=requires_text,
                            text_placeholder=text_placeholder,
                        )
                    )

        if not question_text:
            raise ValueError("empty question")

        # Force stable keys so we can infer slot from the key during answer submission.
        if not question_key.startswith("req_"):
            question_key = f"req_{slot_enum.value}_{n}"

        state.required_questions_asked = n
        return Question(
            key=question_key,
            text=question_text,
            question_type=QuestionType.REQUIRED,
            context=reason,
            slot=slot_enum,
            options=options,
            input_type=input_type,
            allow_multiple=allow_multiple,
            allow_custom=allow_custom,
        )
    except (LLMClientError, ValueError, Exception) as exc:
        logger.warning("Dynamic required question failed (%s); using fallback.", exc)
        state.required_questions_asked = n
        return _fallback_question(slot, n)


async def fill_slot_from_answer(
    state: SessionState,
    slot: ResearchSlot,
    question_text: str,
    answer_text: str,
    llm: LLMClient,
) -> None:
    """
    Extract a concise slot value and store it into state.required_slots_filled.
    """
    try:
        payload = await llm.complete_json(
            prompt=build_slot_fill_prompt(slot=slot.value, question=question_text, answer=answer_text),
            system=SYSTEM_SLOT_FILLER,
        )
        value = str(payload.get("value") or "").strip()
        if not value:
            raise ValueError("empty slot value")
        state.required_slots_filled[slot] = value
    except (LLMClientError, ValueError, Exception) as exc:
        logger.warning("Slot fill failed (%s); using raw answer as value.", exc)
        state.required_slots_filled[slot] = (answer_text or "").strip()

    missing_required_slots(state)


def _fallback_question(slot: ResearchSlot, n: int) -> Question:
    if slot == ResearchSlot.OFFER:
        return Question(
            key=f"req_offer_{n}",
            text="In one sentence, what product or service do you sell?",
            question_type=QuestionType.REQUIRED,
            context="Required to understand the offer for research.",
            slot=ResearchSlot.OFFER,
        )
    return Question(
        key=f"req_icp_{n}",
        text="Who is your ideal customer? (role, industry, company size)",
        question_type=QuestionType.REQUIRED,
        context="Required to understand who you sell to for research.",
        slot=ResearchSlot.ICP,
    )

