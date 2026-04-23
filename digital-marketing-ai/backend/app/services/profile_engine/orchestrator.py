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

from app.schemas.profile_engine import (
    FinalProfile,
    GoalType,
    Question,
    ReadinessLevel,
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
            self._sync_baseline_fields(state)

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
            self._sync_baseline_fields(state)

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
                await dynamic_required.fill_slot_from_answer(
                    state=state,
                    slot=slot,
                    question_text=question_key,
                    answer_text=answer,
                    llm=self._llm,
                )

            # Re-run readiness and need_routing with new information
            if state.extracted:
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
