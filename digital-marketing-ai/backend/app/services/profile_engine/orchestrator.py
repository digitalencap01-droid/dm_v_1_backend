"""
Orchestrator — main controller for the Profile Intelligence Engine pipeline.

Responsibility: Own the session state machine and call all sub-services in
the correct order.  Boundaries:
- Input enters via public methods.
- Each step calls exactly one service and updates SessionState.
- The orchestrator decides what happens next (next question vs. build profile).
- Error handling at this layer converts internal exceptions into structured results.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
import json
import re

from app.schemas.profile_engine import (
    FinalProfile,
    GoalType,
    Question,
    ReadinessLevel,
    ExtractedData,
    ResearchSlot,
    SessionState,
    SessionStatus,
)
from app.services.llm.client import LLMClient, get_llm_client
from app.services.profile_engine import (
    classifier,
    confidence,
    dynamic_required,
    extractor,
    need_routing,
    profile_builder,
    question_selector,
    readiness,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Answer normalization (deterministic preprocessing)
# ---------------------------------------------------------------------------


_NULLISH_TOKENS = {
    "",
    "na",
    "n/a",
    "none",
    "no",
    "nope",
    "nil",
    "not available",
    "not sure",
    "unknown",
    "dont know",
    "don't know",
    "dont have",
    "don't have",
    "no website",
    "no site",
}


def _is_nullish_text(text: str | None) -> bool:
    if not text:
        return True
    normalized = str(text).strip().lower()
    return normalized in _NULLISH_TOKENS


def _parse_monthly_revenue_to_int(value: str | None) -> int | None:
    """
    Best-effort deterministic numeric parsing.

    Examples:
    - "0", "no", "none" -> 0
    - "1000", "1,000", "₹1000", "1000rs" -> 1000
    - "10k", "10 K" -> 10000
    - "1 lakh", "1 lac" -> 100000
    - "2 crore" -> 20000000
    """
    if value is None:
        return None
    raw = str(value).strip().lower()
    if _is_nullish_text(raw):
        return 0

    raw = raw.replace(",", " ")
    raw = raw.replace("₹", " ")
    raw = raw.replace("$", " ")
    raw = raw.replace("inr", " ")
    raw = raw.replace("rs.", " ").replace("rs", " ")
    raw = raw.replace("usd", " ").replace("dollars", " ").replace("dollar", " ")
    raw = re.sub(r"\s+", " ", raw).strip()

    m = re.search(r"(\d+(?:\.\d+)?)\s*(k|lakh|lac|crore|cr)?\b", raw)
    if not m:
        return None

    try:
        number = float(m.group(1))
    except ValueError:
        return None

    unit = (m.group(2) or "").strip()
    multiplier = 1
    if unit == "k":
        multiplier = 1_000
    elif unit in ("lakh", "lac"):
        multiplier = 100_000
    elif unit in ("crore", "cr"):
        multiplier = 10_000_000

    out = int(round(number * multiplier))
    return max(out, 0)


# ---------------------------------------------------------------------------
# Result container returned from every orchestrator call
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorResult:
    """
    Outcome of a single orchestrator step.

    Exactly one of `next_question` or `profile` will be set when the
    operation succeeds.  Both may be None if an error occurred.
    """

    session_id: uuid.UUID
    status: SessionStatus
    next_question: Question | None = None
    profile: FinalProfile | None = None
    confidence_score: float = 0.0
    error: str | None = None
    message: str | None = None


# ---------------------------------------------------------------------------
# Main orchestrator class
# ---------------------------------------------------------------------------


class ProfileEngineOrchestrator:
    """
    Stateless controller — all session state is passed in/out explicitly.

    Call :meth:`process_input` after the user submits their initial description.
    Call :meth:`process_answer` for each follow-up answer.
    """

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or get_llm_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_initial_pipeline(self, state: SessionState) -> SessionState:
        """
        Run extract → classify → readiness → need routing → confidence.

        Intended for scripts and tests that hold a mutable ``SessionState``
        and need the same stages as :meth:`process_input` without discarding
        the populated state object.
        """
        return await self._run_pipeline(state)

    async def process_input(
        self,
        session_id: uuid.UUID,
        raw_input: str,
        existing_answers: dict[str, str] | None = None,
    ) -> OrchestratorResult:
        """
        Run the full initial pipeline for a new business description.

        Steps:
        1. Extract → 2. Classify → 3. Readiness → 4. NeedRouting
        5. Confidence → 6. QuestionSelector → [7. ProfileBuilder if ready]

        Args:
            session_id: The active session UUID.
            raw_input: Raw business description text.
            existing_answers: Any previously stored answers to reuse.

        Returns:
            OrchestratorResult with next question or completed profile.
        """
        state = SessionState(
            session_id=session_id,
            status=SessionStatus.COLLECTING_INPUT,
            raw_input=raw_input,
            answers=existing_answers or {},
        )

        try:
            self._normalize_answers_in_place(state)
            self._sync_baseline_fields(state)
            self._hydrate_extracted_from_baseline(state)

            # Baseline-first: do not call AI until baseline is complete.
            if not state.baseline_complete:
                return self._decide_next_step(state)

            state = await self._run_pipeline(state)
            await self._maybe_ask_dynamic_required(state)
        except Exception as exc:
            logger.exception("Pipeline error for session %s: %s", session_id, exc)
            return OrchestratorResult(
                session_id=session_id,
                status=SessionStatus.FAILED,
                error=str(exc),
                message="An error occurred during profile processing.",
            )

        return self._decide_next_step(state)

    async def process_answer(
        self,
        state: SessionState,
        question_key: str,
        answer: str,
    ) -> OrchestratorResult:
        """
        Process a follow-up answer and continue the pipeline.

        The orchestrator re-runs readiness + need_routing with the new answer
        to update its assessments before deciding the next step.

        Args:
            state: Current full session state (loaded from repository).
            question_key: The key of the question that was answered.
            answer: The user's answer text.

        Returns:
            OrchestratorResult with next question or completed profile.
        """
        # Record the answer
        state.answers[question_key] = answer
        if question_key not in state.asked_questions:
            state.asked_questions.append(question_key)
        state.pending_question = None

        try:
            self._normalize_answers_in_place(state, changed_key=question_key)
            self._sync_baseline_fields(state)
            self._hydrate_extracted_from_baseline(state)
            self._hydrate_extracted_from_required_slots(state)

            # If baseline is not complete yet, keep collecting baseline.
            if not state.baseline_complete:
                state.confidence_score = confidence.calculate_confidence(state)
                return self._decide_next_step(state)

            # If baseline just completed and pipeline not yet run, run it once.
            if state.extracted is None:
                state = await self._run_pipeline(state)
                await self._maybe_ask_dynamic_required(state)
                return self._decide_next_step(state)

            # If this was a dynamic required slot question, fill the slot.
            slot = self._slot_from_question_key(question_key)
            if slot is not None:
                slot_answer_text = answer
                if isinstance(slot_answer_text, str) and slot_answer_text.strip().startswith("["):
                    try:
                        parsed = json.loads(slot_answer_text)
                        if isinstance(parsed, list):
                            cleaned = [str(x).strip() for x in parsed if str(x).strip()]
                            if cleaned:
                                slot_answer_text = ", ".join(cleaned)
                    except Exception:
                        pass
                await dynamic_required.fill_slot_from_answer(
                    state=state,
                    slot=slot,
                    question_text=question_key,
                    answer_text=slot_answer_text,
                    llm=self._llm,
                )
                self._hydrate_extracted_from_required_slots(state)
                # If we materially improved extracted context, re-run classification once.
                if state.extracted and state.classified and (
                    state.classified.persona.value == "unknown"
                    or state.classified.industry.value == "other"
                ):
                    state.classified = await classifier.classify(
                        extracted=state.extracted,
                        llm=self._llm,
                    )
                self._apply_deterministic_persona_fallback(state)

            # Re-run readiness and need_routing with new information
            if state.extracted:
                self._apply_deterministic_persona_fallback(state)
                state.readiness = await readiness.assess_readiness(
                    extracted=state.extracted,
                    classified=state.classified,  # type: ignore[arg-type]
                    answers=state.answers,
                    llm=self._llm,
                )
                if state.classified:
                    state.need_routing = await need_routing.route_need(
                        extracted=state.extracted,
                        classified=state.classified,
                        readiness=state.readiness,
                        answers=state.answers,
                        llm=self._llm,
                    )

            state.confidence_score = confidence.calculate_confidence(state)
            await self._maybe_ask_dynamic_required(state)
        except Exception as exc:
            logger.exception(
                "Error processing answer for session %s: %s", state.session_id, exc
            )
            return OrchestratorResult(
                session_id=state.session_id,
                status=SessionStatus.FAILED,
                error=str(exc),
                message="Failed to process your answer.",
            )

        return self._decide_next_step(state)

    async def _maybe_ask_dynamic_required(self, state: SessionState) -> None:
        """
        If required research slots are missing and we are under the cap,
        generate one dynamic required question and set it as pending_question.
        """
        missing = dynamic_required.missing_required_slots(state)
        if not missing:
            return
        if (state.required_questions_asked or 0) >= (state.max_required_questions or 0):
            return
        # Generate exactly one next required question.
        state.pending_question = await dynamic_required.generate_required_question(state, self._llm)

    # ------------------------------------------------------------------
    # Baseline helpers
    # ------------------------------------------------------------------

    def _ensure_extracted_metadata(self, state: SessionState) -> ExtractedData:
        if state.extracted is None:
            state.extracted = ExtractedData()
        if state.extracted.metadata is None:
            state.extracted.metadata = {}
        return state.extracted

    def _record_raw_answer(self, state: SessionState, key: str, raw_value: str) -> None:
        extracted = self._ensure_extracted_metadata(state)
        answers_raw = extracted.metadata.setdefault("answers_raw", {})
        if isinstance(answers_raw, dict) and key not in answers_raw:
            answers_raw[key] = raw_value

    def _set_normalized_fact(
        self,
        state: SessionState,
        key: str,
        value: object,
        source: str,
        confidence_value: float,
    ) -> None:
        extracted = self._ensure_extracted_metadata(state)
        normalized = extracted.metadata.setdefault("normalized_facts", {})
        if not isinstance(normalized, dict):
            extracted.metadata["normalized_facts"] = {}
            normalized = extracted.metadata["normalized_facts"]
        normalized[key] = {
            "value": value,
            "source": source,
            "confidence": float(confidence_value),
        }

    def _normalize_answers_in_place(self, state: SessionState, changed_key: str | None = None) -> None:
        """
        Deterministically normalise raw answers into structured semantics.

        - Keep original raw answer text for auditability (state.extracted.metadata["answers_raw"]).
        - Do not change the overall answers payload shape (dict[str, str]).
        """
        answers = state.answers or {}

        # Record raw answer once, before any mutation.
        if changed_key and changed_key in answers and isinstance(answers[changed_key], str):
            self._record_raw_answer(state, changed_key, answers[changed_key])

        # website_url: treat "no/none/na" as no website (falsy url) and derive has_website
        website_raw = answers.get("website_url")
        if isinstance(website_raw, str):
            website = website_raw.strip()
            if _is_nullish_text(website):
                answers["website_url"] = ""
                answers["has_website"] = "false"
                self._set_normalized_fact(
                    state=state,
                    key="has_website",
                    value=False,
                    source="normalized_user_answer",
                    confidence_value=0.9,
                )
            elif website:
                answers["has_website"] = "true"
                self._set_normalized_fact(
                    state=state,
                    key="has_website",
                    value=True,
                    source="normalized_user_answer",
                    confidence_value=0.9,
                )

        # monthly_revenue: numeric parsing + has_revenue derived boolean
        revenue_raw = answers.get("monthly_revenue")
        if isinstance(revenue_raw, str):
            parsed = _parse_monthly_revenue_to_int(revenue_raw)
            if parsed is not None:
                answers["monthly_revenue_normalized"] = str(parsed)
                answers["has_revenue"] = "true" if parsed > 0 else "false"
                self._set_normalized_fact(
                    state=state,
                    key="monthly_revenue",
                    value=parsed,
                    source="normalized_user_answer",
                    confidence_value=0.8,
                )
                self._set_normalized_fact(
                    state=state,
                    key="has_revenue",
                    value=(parsed > 0),
                    source="derived_from_monthly_revenue",
                    confidence_value=0.8,
                )

        # req_icp: normalize "not sure/unknown/na" as an explicit unknown signal
        icp_key_candidates = [k for k in answers.keys() if str(k).lower().startswith("req_icp")]
        for key in icp_key_candidates:
            raw = answers.get(key)
            if isinstance(raw, str) and _is_nullish_text(raw):
                answers[key] = ""
                answers["icp_unknown"] = "true"
                self._set_normalized_fact(
                    state=state,
                    key="icp_unknown",
                    value=True,
                    source="normalized_user_answer",
                    confidence_value=0.9,
                )

        state.answers = answers

    def _hydrate_extracted_from_required_slots(self, state: SessionState) -> None:
        """
        Hydrate extracted fields from required slot values with overwrite policy.

        Policy:
        - Fill null/empty extracted fields first.
        - Only overwrite when the new signal is stronger than existing signal.
        - Persist provenance to state.extracted.metadata for debugging.
        """
        if not state.extracted:
            return

        extracted = state.extracted
        slots = state.required_slots_filled or {}
        provenance = extracted.metadata.setdefault("field_provenance", {})
        if not isinstance(provenance, dict):
            extracted.metadata["field_provenance"] = {}
            provenance = extracted.metadata["field_provenance"]

        def set_field(field: str, value: str, source: str, confidence_value: float) -> None:
            if not value.strip():
                return
            current = getattr(extracted, field, None)
            current_text = (str(current).strip() if isinstance(current, str) else "")
            current_prov = provenance.get(field) if isinstance(provenance, dict) else None
            current_conf = 0.0
            if isinstance(current_prov, dict):
                try:
                    current_conf = float(current_prov.get("confidence") or 0.0)
                except (TypeError, ValueError):
                    current_conf = 0.0

            should_fill = not current_text
            should_overwrite = bool(current_text) and (confidence_value > current_conf)

            if should_fill or should_overwrite:
                setattr(extracted, field, value.strip())
                provenance[field] = {
                    "source": source,
                    "confidence": float(confidence_value),
                }

        offer = slots.get(ResearchSlot.OFFER)
        if isinstance(offer, str):
            set_field(
                field="product_or_service",
                value=offer,
                source="required_slot_offer",
                confidence_value=0.85,
            )

        icp = slots.get(ResearchSlot.ICP)
        if isinstance(icp, str) and state.answers.get("icp_unknown") != "true":
            set_field(
                field="target_audience",
                value=icp,
                source="required_slot_icp",
                confidence_value=0.85,
            )

    def _hydrate_extracted_from_baseline(self, state: SessionState) -> None:
        """
        Copy baseline signals into extracted fields where safe and helpful.

        This ensures "basic inference" fields (goals/description) are present
        even if extraction had low signal.
        """
        extracted = self._ensure_extracted_metadata(state)

        # Goals: mirror declared_goals into extracted.mentioned_goals if empty.
        if not extracted.mentioned_goals and state.declared_goals:
            extracted.mentioned_goals = [g.value for g in state.declared_goals]
            extracted.metadata.setdefault("field_provenance", {})
            prov = extracted.metadata.get("field_provenance")
            if isinstance(prov, dict):
                prov["mentioned_goals"] = {
                    "source": "baseline_declared_goals",
                    "confidence": 0.95,
                }

        # Description: build a minimal deterministic description when missing.
        if not extracted.description:
            offer = ""
            if state.required_slots_filled and ResearchSlot.OFFER in state.required_slots_filled:
                offer = (state.required_slots_filled.get(ResearchSlot.OFFER) or "").strip()
            if not offer:
                offer = (state.answers.get("product_or_service") or "").strip()

            icp = ""
            if state.required_slots_filled and ResearchSlot.ICP in state.required_slots_filled:
                icp = (state.required_slots_filled.get(ResearchSlot.ICP) or "").strip()
            if not icp and state.answers.get("icp_unknown") != "true":
                icp = (state.answers.get("target_audience") or "").strip()

            geo = (state.answers.get("target_market_geo") or "").strip()
            stage = state.declared_stage.value if state.declared_stage else ""

            parts: list[str] = []
            if stage:
                parts.append(stage.replace("_", " "))
            if geo:
                parts.append(geo)
            if offer:
                parts.append(f"{offer}")
            if icp:
                parts.append(f"targeting {icp}")

            if parts:
                extracted.description = " ".join(parts).strip().rstrip(".") + "."
                extracted.metadata.setdefault("field_provenance", {})
                prov = extracted.metadata.get("field_provenance")
                if isinstance(prov, dict):
                    prov["description"] = {
                        "source": "deterministic_baseline_summary",
                        "confidence": 0.7,
                    }

    def _apply_deterministic_persona_fallback(self, state: SessionState) -> None:
        """
        If classification is weak, pick a conservative default persona based on baseline signals.
        """
        if state.classified is None:
            return
        if state.classified.persona.value != "unknown":
            return

        team_size = (state.answers.get("team_size") or "").strip().lower()
        stage = state.declared_stage
        if stage in (ReadinessLevel.IDEA_STAGE, ReadinessLevel.MVP, ReadinessLevel.UNKNOWN) and team_size not in (
            "",
            "prefer_not_say",
        ):
            from app.schemas.profile_engine import PersonaType

            state.classified.persona = PersonaType.FOUNDER
            state.classified.persona_confidence = max(state.classified.persona_confidence, 0.5)
            if not state.classified.raw_persona:
                state.classified.raw_persona = "deterministic_baseline_fallback"

            if state.extracted and isinstance(state.extracted.metadata, dict):
                state.extracted.metadata.setdefault("field_provenance", {})
                prov = state.extracted.metadata.get("field_provenance")
                if isinstance(prov, dict):
                    prov["persona"] = {
                        "source": "deterministic_baseline_fallback",
                        "confidence": float(state.classified.persona_confidence),
                    }

    def _sync_baseline_fields(self, state: SessionState) -> None:
        """
        Derive baseline flags from answers.

        This lets the API treat baseline questions as normal /answers submissions,
        while still keeping typed fields on SessionState.
        """
        answers = state.answers or {}

        # Baseline completion
        baseline_keys = {"declared_stage", "team_size", "declared_goals"}
        state.baseline_complete = baseline_keys.issubset(set(answers.keys()))

        # Stage
        raw_stage = answers.get("declared_stage")
        if isinstance(raw_stage, str):
            normalized = raw_stage.strip().lower()
            if normalized == "not_sure":
                state.declared_stage = ReadinessLevel.UNKNOWN
            else:
                try:
                    state.declared_stage = ReadinessLevel(normalized)
                except ValueError:
                    state.declared_stage = ReadinessLevel.UNKNOWN

        # Goals (may be stored as JSON list string or comma-separated)
        raw_goals = answers.get("declared_goals")
        parsed: list[str] = []
        if isinstance(raw_goals, str) and raw_goals.strip():
            text = raw_goals.strip()
            if text.startswith("["):
                try:
                    value = json.loads(text)
                    if isinstance(value, list):
                        parsed = [str(x) for x in value]
                except Exception:
                    parsed = []
            else:
                parsed = [p.strip() for p in text.split(",") if p.strip()]

        goals: list[GoalType] = []
        for item in parsed:
            try:
                goals.append(GoalType(item))
            except ValueError:
                continue
        state.declared_goals = goals

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    async def _run_pipeline(self, state: SessionState) -> SessionState:
        """Run all pipeline stages in order, updating state after each."""

        # Stage 1: Extract
        logger.info("Stage[extract] session=%s", state.session_id)
        state.extracted = await extractor.extract(
            raw_input=state.raw_input or "",
            llm=self._llm,
        )

        # Stage 2: Classify
        logger.info("Stage[classify] session=%s", state.session_id)
        state.classified = await classifier.classify(
            extracted=state.extracted,
            llm=self._llm,
        )

        # Stage 3: Readiness
        logger.info("Stage[readiness] session=%s", state.session_id)
        state.readiness = await readiness.assess_readiness(
            extracted=state.extracted,
            classified=state.classified,
            answers=state.answers,
            llm=self._llm,
        )

        # Stage 4: Need routing
        logger.info("Stage[need_routing] session=%s", state.session_id)
        state.need_routing = await need_routing.route_need(
            extracted=state.extracted,
            classified=state.classified,
            readiness=state.readiness,
            answers=state.answers,
            llm=self._llm,
        )

        # Stage 5: Confidence
        state.confidence_score = confidence.calculate_confidence(state)
        logger.info(
            "Stage[confidence] session=%s score=%.3f",
            state.session_id,
            state.confidence_score,
        )

        return state

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _decide_next_step(self, state: SessionState) -> OrchestratorResult:
        """
        Choose between asking another question or building the final profile.
        """
        # Dynamic required question takes priority once baseline is complete.
        if state.baseline_complete and state.pending_question is not None:
            next_q = state.pending_question
        else:
            next_q = question_selector.select_question(state)

        if next_q is not None:
            state.status = SessionStatus.ASKING_QUESTIONS
            state.pending_question = next_q
            return OrchestratorResult(
                session_id=state.session_id,
                status=SessionStatus.ASKING_QUESTIONS,
                next_question=next_q,
                confidence_score=state.confidence_score,
                message="Additional information needed.",
            )

        # No more questions — ready to start research (finalize)
        return OrchestratorResult(
            session_id=state.session_id,
            status=SessionStatus.BUILDING_PROFILE,
            confidence_score=state.confidence_score,
            message="Sufficient information collected. Building profile.",
        )

    @staticmethod
    def _slot_from_question_key(question_key: str) -> ResearchSlot | None:
        key = (question_key or "").strip().lower()
        if key.startswith("req_offer_") or key.startswith("req_offer"):
            return ResearchSlot.OFFER
        if key.startswith("req_icp_") or key.startswith("req_icp"):
            return ResearchSlot.ICP
        return None

    async def build_final_profile(self, state: SessionState) -> OrchestratorResult:
        """
        Build and return the final profile for a session that is ready.

        Args:
            state: Complete session state.

        Returns:
            OrchestratorResult with populated :attr:`profile`.
        """
        try:
            logger.info("Stage[profile_builder] session=%s", state.session_id)
            state.status = SessionStatus.BUILDING_PROFILE

            # Ensure we don't build a profile from stale/under-hydrated state.
            self._normalize_answers_in_place(state)
            self._sync_baseline_fields(state)
            self._hydrate_extracted_from_baseline(state)
            self._hydrate_extracted_from_required_slots(state)

            if state.extracted and state.classified and (
                state.classified.persona.value == "unknown"
                or state.classified.industry.value == "other"
            ):
                state.classified = await classifier.classify(extracted=state.extracted, llm=self._llm)

            self._apply_deterministic_persona_fallback(state)

            if state.extracted and state.classified and state.readiness and (
                state.need_routing is None or state.need_routing.primary_need.value == "unknown"
            ):
                state.need_routing = await need_routing.route_need(
                    extracted=state.extracted,
                    classified=state.classified,
                    readiness=state.readiness,
                    answers=state.answers,
                    llm=self._llm,
                )

            final = await profile_builder.build_profile(state=state, llm=self._llm)
            state.status = SessionStatus.COMPLETE
            return OrchestratorResult(
                session_id=state.session_id,
                status=SessionStatus.COMPLETE,
                profile=final,
                confidence_score=state.confidence_score,
                message="Profile successfully generated.",
            )
        except Exception as exc:
            logger.exception(
                "Profile build error for session %s: %s", state.session_id, exc
            )
            return OrchestratorResult(
                session_id=state.session_id,
                status=SessionStatus.FAILED,
                error=str(exc),
                message="Profile generation failed.",
            )

    async def enable_optional_questions(self, state: SessionState) -> OrchestratorResult:
        """
        Handle 'answer more' requests to enable optional follow-up questions.
        
        This allows users to provide additional information to enhance results.
        
        Args:
            state: Current session state.
        
        Returns:
            OrchestratorResult with the next optional question or profile.
        """
        try:
            logger.info("Enabling optional questions for session %s", state.session_id)
            
            # Enable optional questions
            state.allow_optional = True
            
            # Select the next optional question
            next_q = question_selector.select_question(state)
            
            if next_q is not None:
                state.status = SessionStatus.ASKING_QUESTIONS
                state.pending_question = next_q
                logger.info("Selected optional question %s for session %s", next_q.key, state.session_id)
                return OrchestratorResult(
                    session_id=state.session_id,
                    status=SessionStatus.ASKING_QUESTIONS,
                    next_question=next_q,
                    confidence_score=state.confidence_score,
                    message="Great! Here's an optional question to enhance your profile.",
                )
            else:
                # No optional questions available, proceed to profile
                logger.info("No optional questions available for session %s, building profile", state.session_id)
                return await self.build_final_profile(state)
        except Exception as exc:
            logger.exception(
                "Error enabling optional questions for session %s: %s", state.session_id, exc
            )
            return OrchestratorResult(
                session_id=state.session_id,
                status=SessionStatus.FAILED,
                error=str(exc),
                message="Failed to enable optional questions.",
            )
