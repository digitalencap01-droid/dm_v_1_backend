"""
Profile Engine Repository — all persistence operations for the engine.

Responsibility: Save, load, and update ProfileEngineSession,
ProfileEngineAnswer, and ProfileEngineProfile records.

Uses SQLAlchemy AsyncSession.  All methods are async and raise
RepositoryError on unexpected DB failures.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.profile_engine import (
    ClassifiedProfile,
    ExtractedData,
    FinalProfile,
    GoalType,
    NeedRouting,
    ReadinessAssessment,
    ReadinessLevel,
    ResearchSlot,
    SessionState,
    SessionStatus,
)
from app.services.profile_engine.model import (
    ProfileEngineAnswer,
    ProfileEngineProfile,
    ProfileEngineSession,
)

logger = logging.getLogger(__name__)


class ProfileEngineRepository:
    """
    Data-access layer for the Profile Intelligence Engine.

    All methods accept an AsyncSession injected by the caller (FastAPI deps).
    """

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Session operations
    # ------------------------------------------------------------------

    async def create_session(
        self,
        session_id: uuid.UUID | None = None,
        user_id: str | None = None,
    ) -> ProfileEngineSession:
        """Create and persist a new profile engine session."""
        record = ProfileEngineSession(
            id=session_id or uuid.uuid4(),
            user_id=user_id,
            status=SessionStatus.PENDING.value,
        )
        self._db.add(record)
        await self._db.flush()
        logger.info("Created ProfileEngineSession %s", record.id)
        return record

    async def get_session(self, session_id: uuid.UUID) -> ProfileEngineSession | None:
        """Fetch a session by ID. Returns None if not found."""
        result = await self._db.execute(
            select(ProfileEngineSession).where(ProfileEngineSession.id == session_id)
        )
        return result.scalar_one_or_none()

    async def update_session_from_state(self, state: SessionState) -> None:
        """
        Persist all intermediate service outputs from a SessionState back
        to the database record.
        """
        record = await self._get_session_or_raise(state.session_id)

        record.status = state.status.value
        record.raw_input = state.raw_input
        record.confidence_score = state.confidence_score
        record.asked_questions = state.asked_questions

        # Persist extracted + baseline metadata in the same JSON blob to avoid
        # adding new DB columns during the ideation phase.
        extracted_payload: dict | None = None
        if state.extracted:
            extracted_payload = state.extracted.model_dump()
        else:
            extracted_payload = record.extracted_data or {"metadata": {}}

        if "metadata" not in extracted_payload or not isinstance(extracted_payload["metadata"], dict):
            extracted_payload["metadata"] = {}

        extracted_payload["metadata"].update(
            {
                "baseline_complete": state.baseline_complete,
                "allow_optional": state.allow_optional,
                "declared_stage": state.declared_stage.value
                if isinstance(state.declared_stage, ReadinessLevel)
                else str(state.declared_stage),
                "declared_goals": [g.value for g in state.declared_goals],
                "required_slots_filled": {
                    (k.value if isinstance(k, ResearchSlot) else str(k)): v
                    for k, v in (state.required_slots_filled or {}).items()
                },
                "required_questions_asked": int(state.required_questions_asked or 0),
                "max_required_questions": int(state.max_required_questions or 0),
            }
        )

        record.extracted_data = extracted_payload
        if state.classified:
            record.classified_data = state.classified.model_dump()
        if state.readiness:
            record.readiness_data = state.readiness.model_dump()
        if state.need_routing:
            record.need_routing_data = state.need_routing.model_dump()

        await self._db.flush()

    async def update_session_status(
        self, session_id: uuid.UUID, status: SessionStatus
    ) -> None:
        """Update only the status field of a session."""
        record = await self._get_session_or_raise(session_id)
        record.status = status.value
        await self._db.flush()

    # ------------------------------------------------------------------
    # Answer operations
    # ------------------------------------------------------------------

    async def save_answer(
        self,
        session_id: uuid.UUID,
        question_key: str,
        question_text: str,
        answer_text: str,
    ) -> ProfileEngineAnswer:
        """
        Upsert a question/answer pair.  If the question was already answered,
        update the existing row.
        """
        result = await self._db.execute(
            select(ProfileEngineAnswer).where(
                ProfileEngineAnswer.session_id == session_id,
                ProfileEngineAnswer.question_key == question_key,
            )
        )
        existing = result.scalar_one_or_none()

        if existing:
            existing.answer_text = answer_text
            existing.question_text = question_text
            await self._db.flush()
            return existing

        answer_record = ProfileEngineAnswer(
            session_id=session_id,
            question_key=question_key,
            question_text=question_text,
            answer_text=answer_text,
        )
        self._db.add(answer_record)
        await self._db.flush()
        return answer_record

    async def get_answers(self, session_id: uuid.UUID) -> dict[str, str]:
        """Return all answers for a session as {question_key: answer_text}."""
        result = await self._db.execute(
            select(ProfileEngineAnswer).where(
                ProfileEngineAnswer.session_id == session_id
            )
        )
        rows = result.scalars().all()
        return {row.question_key: row.answer_text for row in rows}

    async def list_answer_rows(self, session_id: uuid.UUID) -> list[ProfileEngineAnswer]:
        """Return answer rows ordered by creation time (oldest first)."""
        result = await self._db.execute(
            select(ProfileEngineAnswer)
            .where(ProfileEngineAnswer.session_id == session_id)
            .order_by(ProfileEngineAnswer.created_at.asc())
        )
        return list(result.scalars().all())

    # ------------------------------------------------------------------
    # Profile operations
    # ------------------------------------------------------------------

    async def save_profile(self, profile: FinalProfile) -> ProfileEngineProfile:
        """Persist the final profile (create or overwrite)."""
        result = await self._db.execute(
            select(ProfileEngineProfile).where(
                ProfileEngineProfile.session_id == profile.session_id
            )
        )
        existing = result.scalar_one_or_none()

        data = _profile_to_record_dict(profile)

        if existing:
            for key, value in data.items():
                setattr(existing, key, value)
            await self._db.flush()
            return existing

        record = ProfileEngineProfile(session_id=profile.session_id, **data)
        self._db.add(record)
        await self._db.flush()
        logger.info("Saved ProfileEngineProfile for session %s", profile.session_id)
        return record

    async def get_profile(
        self, session_id: uuid.UUID
    ) -> ProfileEngineProfile | None:
        """Fetch the final profile for a session."""
        result = await self._db.execute(
            select(ProfileEngineProfile).where(
                ProfileEngineProfile.session_id == session_id
            )
        )
        return result.scalar_one_or_none()

    # ------------------------------------------------------------------
    # State reconstruction
    # ------------------------------------------------------------------

    async def load_session_state(self, session_id: uuid.UUID) -> SessionState | None:
        """
        Reconstruct a SessionState from persisted data.
        Returns None if the session does not exist.
        """
        record = await self.get_session(session_id)
        if record is None:
            return None

        answers = await self.get_answers(session_id)

        state = SessionState(
            session_id=session_id,
            status=SessionStatus(record.status),
            raw_input=record.raw_input,
            extracted=ExtractedData(**record.extracted_data) if record.extracted_data else None,
            classified=ClassifiedProfile(**record.classified_data) if record.classified_data else None,
            readiness=ReadinessAssessment(**record.readiness_data) if record.readiness_data else None,
            need_routing=NeedRouting(**record.need_routing_data) if record.need_routing_data else None,
            answers=answers,
            asked_questions=record.asked_questions or [],
            confidence_score=record.confidence_score or 0.0,
        )

        # Restore baseline flags from extracted.metadata if present.
        meta: dict = (state.extracted.metadata if state.extracted else {}) or {}
        state.baseline_complete = bool(meta.get("baseline_complete", False))
        state.allow_optional = bool(meta.get("allow_optional", False))

        declared_stage = meta.get("declared_stage")
        if isinstance(declared_stage, str):
            try:
                state.declared_stage = ReadinessLevel(declared_stage)
            except ValueError:
                state.declared_stage = ReadinessLevel.UNKNOWN

        declared_goals = meta.get("declared_goals", [])
        if isinstance(declared_goals, list):
            parsed: list[GoalType] = []
            for item in declared_goals:
                try:
                    parsed.append(GoalType(str(item)))
                except ValueError:
                    pass
            state.declared_goals = parsed

        # Restore required slot state
        state.required_questions_asked = int(meta.get("required_questions_asked", 0) or 0)
        state.max_required_questions = int(meta.get("max_required_questions", state.max_required_questions) or 0)
        filled_meta = meta.get("required_slots_filled", {}) or {}
        if isinstance(filled_meta, dict):
            restored: dict[ResearchSlot, str] = {}
            for k, v in filled_meta.items():
                try:
                    restored[ResearchSlot(str(k))] = str(v)
                except ValueError:
                    continue
            state.required_slots_filled = restored

        return state

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _get_session_or_raise(self, session_id: uuid.UUID) -> ProfileEngineSession:
        record = await self.get_session(session_id)
        if record is None:
            raise RepositoryError(f"Session {session_id} not found.")
        return record


def _profile_to_record_dict(profile: FinalProfile) -> dict[str, Any]:
    return {
        "persona": profile.persona.value,
        "industry": profile.industry.value,
        "readiness_level": profile.readiness_level.value,
        "primary_need": profile.primary_need.value,
        "secondary_needs": [n.value for n in profile.secondary_needs],
        "confidence_score": profile.confidence_score,
        "business_name": profile.business_name,
        "target_audience": profile.target_audience,
        "product_or_service": profile.product_or_service,
        "revenue_model": profile.revenue_model,
        "current_challenges": profile.current_challenges,
        "goals": profile.goals,
        "recommended_channels": profile.recommended_channels,
        "summary": profile.summary,
        "raw_data": profile.raw_data,
    }


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class RepositoryError(RuntimeError):
    """Raised on unexpected repository failures."""
