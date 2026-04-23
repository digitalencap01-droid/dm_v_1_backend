"""
Profile Intelligence Engine — FastAPI route handlers.

HTTP surface only.  No business logic here.
All decisions are delegated to ProfileEngineOrchestrator and
ProfileEngineRepository.

Endpoints:
  POST   /profile-engine/sessions                         - start session
  POST   /profile-engine/sessions/{session_id}/input      - submit business input
  POST   /profile-engine/sessions/{session_id}/answers    - answer a follow-up question
  GET    /profile-engine/sessions/{session_id}/profile    - get final profile
  GET    /profile-engine/sessions/{session_id}            - get session status
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.repositories.profile_engine_repository import (
    ProfileEngineRepository,
    RepositoryError,
)
from app.schemas.profile_engine import (
    AnswerQuestionRequest,
    EngineMessage,
    EngineStepResponse,
    ErrorResponse,
    FinalProfile,
    MessageRole,
    MessagesResponse,
    ProfileResponse,
    SessionResponse,
    SessionStatus,
    StartSessionRequest,
    SubmitInputRequest,
)
from app.services.profile_engine.orchestrator import (
    OrchestratorResult,
    ProfileEngineOrchestrator,
)
from app.services.profile_engine import question_selector
from app.services.profile_engine import dynamic_required
from app.services.llm.client import get_llm_client, LLMClient
from app.services.llm.prompts import QUESTION_BANK

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/profile-engine",
    tags=["Profile Intelligence Engine"],
)


# ---------------------------------------------------------------------------
# Dependency providers
# ---------------------------------------------------------------------------


def get_repository(db: AsyncSession = Depends(get_db_session)) -> ProfileEngineRepository:
    return ProfileEngineRepository(db)


def get_orchestrator(
    llm: LLMClient = Depends(get_llm_client),
) -> ProfileEngineOrchestrator:
    return ProfileEngineOrchestrator(llm=llm)


DbDep = Annotated[AsyncSession, Depends(get_db_session)]
RepoDep = Annotated[ProfileEngineRepository, Depends(get_repository)]
OrchestratorDep = Annotated[ProfileEngineOrchestrator, Depends(get_orchestrator)]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start a new profile engine session",
)
async def start_session(
    body: StartSessionRequest,
    repo: RepoDep,
    db: DbDep,
) -> SessionResponse:
    """
    Create a new profile engine session and return the session ID.
    The caller uses the returned session_id for all subsequent requests.
    """
    try:
        record = await repo.create_session(user_id=body.user_id)
        await db.commit()
        return SessionResponse(
            session_id=record.id,
            status=SessionStatus(record.status),
            created_at=record.created_at,
            user_id=record.user_id,
        )
    except Exception as exc:
        await db.rollback()
        logger.exception("Failed to create session: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create session.",
        ) from exc


@router.post(
    "/sessions/{session_id}/input",
    response_model=EngineStepResponse,
    summary="Submit initial business description",
)
async def submit_input(
    session_id: uuid.UUID,
    body: SubmitInputRequest,
    repo: RepoDep,
    orchestrator: OrchestratorDep,
    db: DbDep,
) -> EngineStepResponse:
    """
    Submit the initial free-form business description.

    The engine will:
    1. Extract and classify information from the text.
    2. Assess readiness and need-state.
    3. Return either the first follow-up question or signal profile readiness.
    """
    session_record = await repo.get_session(session_id)
    if session_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found.",
        )

    if session_record.status in (SessionStatus.COMPLETE.value, SessionStatus.FAILED.value):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Session is already in terminal state: {session_record.status}.",
        )

    try:
        from app.schemas.profile_engine import SessionState

        # Baseline-first: persist raw input, then start baseline questions.
        state = await repo.load_session_state(session_id) or SessionState(session_id=session_id)
        state.raw_input = body.raw_input
        state.status = SessionStatus.ASKING_QUESTIONS

        next_q = question_selector.select_question(state)
        state.pending_question = next_q

        await repo.update_session_from_state(state)

        await db.commit()
        return EngineStepResponse(
            session_id=session_id,
            status=state.status,
            next_question=next_q,
            profile_ready=False,
            message="Additional information needed.",
            confidence_score=state.confidence_score,
            recommended_action=None,
            baseline_remaining=_baseline_remaining(state),
        )

    except HTTPException:
        raise
    except Exception as exc:
        await db.rollback()
        logger.exception("Error processing input for session %s: %s", session_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error during input processing.",
        ) from exc


@router.post(
    "/sessions/{session_id}/answers",
    response_model=EngineStepResponse,
    summary="Answer a follow-up question",
)
async def answer_question(
    session_id: uuid.UUID,
    body: AnswerQuestionRequest,
    repo: RepoDep,
    orchestrator: OrchestratorDep,
    db: DbDep,
) -> EngineStepResponse:
    """
    Submit an answer to the current follow-up question.

    Returns either the next question or signals that the profile is ready.
    """
    state = await repo.load_session_state(session_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found.",
        )

    if state.status == SessionStatus.COMPLETE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Session is already complete. Fetch the profile instead.",
        )

    if state.status == SessionStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Session has failed. Start a new session.",
        )

    # Look up the question text for the given key
    question_text = _get_question_text(body.question_key)

    try:
        answer_text = _answer_to_text(body.answer)

        await repo.save_answer(
            session_id=session_id,
            question_key=body.question_key,
            question_text=question_text,
            answer_text=answer_text,
        )

        result = await orchestrator.process_answer(
            state=state,
            question_key=body.question_key,
            answer=answer_text,
        )

        if result.status == SessionStatus.FAILED:
            await db.rollback()
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=result.message or "Processing answer failed.",
            )

        state.status = result.status
        state.confidence_score = result.confidence_score
        state.pending_question = result.next_question
        await repo.update_session_from_state(state)

        await db.commit()
        return EngineStepResponse(
            session_id=session_id,
            status=result.status,
            next_question=result.next_question,
            profile_ready=(result.status == SessionStatus.BUILDING_PROFILE),
            message=result.message,
            confidence_score=result.confidence_score,
            recommended_action=_recommended_action(result.confidence_score),
            baseline_remaining=_baseline_remaining(state),
        )

    except HTTPException:
        raise
    except Exception as exc:
        await db.rollback()
        logger.exception("Error answering question for session %s: %s", session_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error processing answer.",
        ) from exc


@router.get(
    "/sessions/{session_id}/profile",
    response_model=ProfileResponse,
    summary="Get the completed profile for a session",
)
async def get_profile(
    session_id: uuid.UUID,
    repo: RepoDep,
) -> ProfileResponse:
    """
    Retrieve the final completed profile for a session.

    Returns 404 if the session does not exist and 425 (Too Early) if the
    profile is not yet ready.
    """
    session_record = await repo.get_session(session_id)
    if session_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found.",
        )

    if session_record.status != SessionStatus.COMPLETE.value:
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail=(
                f"Profile is not yet ready. Session status: {session_record.status}. "
                "Continue answering questions or wait for processing to finish."
            ),
        )

    profile_record = await repo.get_profile(session_id)
    final_profile: FinalProfile | None = None

    if profile_record:
        from app.schemas.profile_engine import (
            IndustryType, NeedState, PersonaType, ReadinessLevel
        )
        final_profile = FinalProfile(
            session_id=session_id,
            persona=PersonaType(profile_record.persona),
            industry=IndustryType(profile_record.industry),
            readiness_level=ReadinessLevel(profile_record.readiness_level),
            primary_need=NeedState(profile_record.primary_need),
            secondary_needs=[NeedState(n) for n in (profile_record.secondary_needs or [])],
            confidence_score=profile_record.confidence_score,
            business_name=profile_record.business_name,
            target_audience=profile_record.target_audience,
            product_or_service=profile_record.product_or_service,
            revenue_model=profile_record.revenue_model,
            current_challenges=profile_record.current_challenges or [],
            goals=profile_record.goals or [],
            recommended_channels=profile_record.recommended_channels or [],
            summary=profile_record.summary,
            raw_data=profile_record.raw_data or {},
        )

    return ProfileResponse(
        session_id=session_id,
        status=SessionStatus(session_record.status),
        profile=final_profile,
        confidence_score=session_record.confidence_score,
        message="Profile retrieved successfully." if final_profile else "Profile record missing.",
    )


@router.get(
    "/sessions/{session_id}/messages",
    response_model=MessagesResponse,
    summary="Get session message history (Q/A thread)",
)
async def get_messages(
    session_id: uuid.UUID,
    repo: RepoDep,
) -> MessagesResponse:
    state = await repo.load_session_state(session_id)
    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found.",
        )

    msgs: list[EngineMessage] = []

    if state.raw_input:
        msgs.append(
            EngineMessage(
                role=MessageRole.USER,
                content=state.raw_input,
                key="raw_input",
                created_at=None,
            )
        )

    answer_rows = await repo.list_answer_rows(session_id)
    for row in answer_rows:
        msgs.append(
            EngineMessage(
                role=MessageRole.ASSISTANT,
                content=row.question_text,
                key=row.question_key,
                created_at=row.created_at,
            )
        )
        msgs.append(
            EngineMessage(
                role=MessageRole.USER,
                content=row.answer_text,
                key=row.question_key,
                created_at=row.created_at,
            )
        )

    if state.pending_question and state.pending_question.key not in (state.answers or {}):
        msgs.append(
            EngineMessage(
                role=MessageRole.ASSISTANT,
                content=state.pending_question.text,
                key=state.pending_question.key,
                created_at=None,
            )
        )

    return MessagesResponse(session_id=session_id, messages=msgs)


@router.post(
    "/sessions/{session_id}/answer-more",
    response_model=EngineStepResponse,
    summary="Opt in to optional questions",
)
async def answer_more(
    session_id: uuid.UUID,
    repo: RepoDep,
    db: DbDep,
) -> EngineStepResponse:
    """
    Enables optional questions for the session and returns the next optional
    question if available.
    """
    state = await repo.load_session_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    state.allow_optional = True
    next_q = question_selector.select_question(state)
    state.pending_question = next_q
    if next_q is not None:
        state.status = SessionStatus.ASKING_QUESTIONS

    await repo.update_session_from_state(state)
    await db.commit()
    return EngineStepResponse(
        session_id=session_id,
        status=state.status,
        next_question=next_q,
        profile_ready=(next_q is None and state.status == SessionStatus.BUILDING_PROFILE),
        message="Optional questions enabled.",
        confidence_score=state.confidence_score,
        recommended_action=_recommended_action(state.confidence_score),
        baseline_remaining=_baseline_remaining(state),
    )


@router.post(
    "/sessions/{session_id}/start-research",
    response_model=ProfileResponse,
    summary="Finalize and generate the profile",
)
async def start_research(
    session_id: uuid.UUID,
    repo: RepoDep,
    orchestrator: OrchestratorDep,
    db: DbDep,
) -> ProfileResponse:
    """
    Generates the final profile (research output) for the session and persists it.
    """
    state = await repo.load_session_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    if not state.baseline_complete:
        raise HTTPException(status_code=409, detail="Baseline questions are not complete.")

    # Ensure the AI pipeline has run at least once after baseline.
    if state.extracted is None:
        state = await orchestrator.run_initial_pipeline(state)
        await repo.update_session_from_state(state)

    # Do not allow finalize while required slots are missing (unless cap reached).
    missing = dynamic_required.missing_required_slots(state)
    if missing and (state.required_questions_asked or 0) < (state.max_required_questions or 0):
        raise HTTPException(status_code=409, detail="Required information still missing.")

    build_result = await orchestrator.build_final_profile(state)
    if not build_result.profile:
        await db.rollback()
        raise HTTPException(status_code=422, detail=build_result.message or "Profile build failed.")

    await repo.save_profile(build_result.profile)
    await repo.update_session_status(session_id, SessionStatus.COMPLETE)
    await db.commit()

    return ProfileResponse(
        session_id=session_id,
        status=SessionStatus.COMPLETE,
        profile=build_result.profile,
        confidence_score=build_result.profile.confidence_score,
        message=build_result.message,
    )

@router.get(
    "/sessions/{session_id}",
    response_model=SessionResponse,
    summary="Get session status",
)
async def get_session_status(
    session_id: uuid.UUID,
    repo: RepoDep,
) -> SessionResponse:
    """Retrieve the current status of a profile engine session."""
    record = await repo.get_session(session_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found.",
        )
    return SessionResponse(
        session_id=record.id,
        status=SessionStatus(record.status),
        created_at=record.created_at,
        user_id=record.user_id,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_question_text(question_key: str) -> str:
    """Look up the canonical question text for a given key."""
    for q in QUESTION_BANK:
        if q["key"] == question_key:
            return q["text"]
    return question_key  # Fallback: use key as text for custom questions


def _answer_to_text(answer: str | list[str]) -> str:
    if isinstance(answer, list):
        return json.dumps(answer)
    return answer


def _baseline_remaining(state) -> int:
    keys = ["declared_stage", "team_size", "declared_goals"]
    present = set(state.answers.keys()) if state.answers else set()
    return len([k for k in keys if k not in present])


def _recommended_action(confidence_score: float) -> str:
    return "start_research" if confidence_score >= 0.70 else "answer_more"
