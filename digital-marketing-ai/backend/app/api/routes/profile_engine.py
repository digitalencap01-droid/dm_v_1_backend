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
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
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
from app.services.llm.client import get_llm_client, LLMClient, LLMClientError, LLMRateLimitError
from app.services.llm.prompts import (
    QUESTION_BANK,
    SYSTEM_BIZMENTOR_RESEARCH_REPORT,
    build_bizmentor_research_report_prompt,
    load_bizmentor_demo_prompt_compact,
)
from app.services.research.biz_research import build_biz_research_bundle
from app.services.research.tavily_client import TavilyClientError, TavilyResearchClient
from app.services.pdf_generator import generate_bizmentor_pdf

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/profile-engine",
    tags=["Profile Intelligence Engine"],
)


class DemoChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str = Field(..., min_length=1)


class DemoChatRequest(BaseModel):
    system: str | None = None
    messages: list[DemoChatMessage] = Field(default_factory=list)
    max_tokens: int = Field(default=1500, ge=1, le=8000)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)


class DemoChatResponse(BaseModel):
    reply: str


class DemoResearchRequest(BaseModel):
    stage_type: str = Field(..., alias="stageType", min_length=1)
    report_mode: str = Field(..., alias="reportMode", min_length=1)
    business_idea: str = Field(..., alias="businessIdea", min_length=10)
    target_audience: str | None = Field(default=None, alias="targetAudience")
    stage: str | None = None
    team_size: str | None = Field(default=None, alias="teamSize")
    goals: list[str] = Field(default_factory=list)
    budget_revenue: str | None = Field(default=None, alias="budgetRevenue")
    location: str | None = None
    context: str | None = None
    conversation_history: list[DemoChatMessage] = Field(default_factory=list, alias="conversationHistory")

    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }


class DemoResearchResponse(BaseModel):
    pdf_path: str
    report_text: str = Field(description="Text version of the report for reference")
    report: str = Field(description="Alias for report_text for frontend compatibility")
    research_bundle: dict[str, Any] = Field(default_factory=dict)
    sources: list[dict[str, str]] = Field(default_factory=list)


class DemoIdeaValidationRequest(BaseModel):
    business_idea: str = Field(..., alias="businessIdea", min_length=3)
    stage_type: str | None = Field(default=None, alias="stageType")
    target_audience: str | None = Field(default=None, alias="targetAudience")
    context: str | None = None

    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }


class DemoIdeaValidationResponse(BaseModel):
    is_valid: bool = Field(alias="isValid")
    reason: str
    recommendation: str | None = None

    model_config = {
        "populate_by_name": True,
    }


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
    "/demo/validate-idea",
    response_model=DemoIdeaValidationResponse,
    summary="Validate whether the submitted business idea is actionable",
)
async def validate_demo_idea(
    body: DemoIdeaValidationRequest,
    llm: LLMClient = Depends(get_llm_client),
) -> DemoIdeaValidationResponse:
    validation = await _validate_demo_business_idea(
        llm=llm,
        business_idea=body.business_idea,
        stage_type=body.stage_type,
        target_audience=body.target_audience,
        context=body.context,
    )
    return DemoIdeaValidationResponse(**validation)


@router.post(
    "/demo/chat",
    response_model=DemoChatResponse,
    summary="Demo chat proxy for the static frontend",
)
async def demo_chat(
    body: DemoChatRequest,
    llm: LLMClient = Depends(get_llm_client),
) -> DemoChatResponse:
    """
    Proxy demo chat requests through the backend so the browser does not call
    the upstream LLM provider directly.
    """
    if not body.messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one message is required.",
        )

    try:
        reply = await llm.complete_messages(
            messages=[m.model_dump() for m in body.messages],
            system=load_bizmentor_demo_prompt_compact(),
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )
        return DemoChatResponse(reply=reply)
    except LLMRateLimitError as exc:
        logger.warning("Demo chat hit LLM rate limit: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="AI service is busy right now. Please try again in a few seconds.",
        ) from exc
    except LLMClientError as exc:
        logger.exception("Demo chat LLM request failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI service request failed. Please try again shortly.",
        ) from exc
    except Exception as exc:
        logger.exception("Demo chat request failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider request failed. Check API key, provider availability, or network.",
        ) from exc


@router.post(
    "/demo/research-report",
    response_model=DemoResearchResponse,
    summary="Generate a Tavily-backed research report for the demo frontend",
)
async def demo_research_report(
    body: DemoResearchRequest,
    llm: LLMClient = Depends(get_llm_client),
) -> DemoResearchResponse:
    """
    Generate a full researched report from the saved intake form and chat
    context. Tavily performs the evidence gathering first; only then is the
    final report written by the LLM.
    """
    validation = await _validate_demo_business_idea(
        llm=llm,
        business_idea=body.business_idea,
        stage_type=body.stage_type,
        target_audience=body.target_audience,
        context=body.context,
    )
    if not validation["isValid"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=validation["reason"],
        )

    intake = body.model_dump(by_alias=False, exclude={"conversation_history"})
    conversation_history = [message.model_dump() for message in body.conversation_history]

    tavily = TavilyResearchClient()
    try:
        # Step 1: Research bundle banana
        research_bundle = await build_biz_research_bundle(
            intake=intake,
            conversation_history=conversation_history,
            client=tavily,
        )

        # Fix 3: Warnings log karo
        if research_bundle.get("warnings"):
            for warning in research_bundle["warnings"]:
                logger.warning("Research bundle warning: %s", warning)

        # Step 2: Report prompt banana
        prompt = build_bizmentor_research_report_prompt(
            intake=intake,
            conversation_history=conversation_history,
            research_bundle=research_bundle,
        )

        # Fix 1: max_tokens 3200 → 8192
        report_text = await llm.complete(
            prompt=prompt,
            system=SYSTEM_BIZMENTOR_RESEARCH_REPORT,
            temperature=0.2,
            max_tokens=4096,
        )

        # Fix 2: Saare sources combine karo — research_task + search_runs
        research_sources = (
            research_bundle
            .get("research_task", {})
            .get("data", {})
            .get("sources", [])
        )
        search_sources = [
            {"title": r.get("title", ""), "url": r.get("url", "")}
            for run in research_bundle.get("search_runs", [])
            for r in run.get("results", [])
            if r.get("url")
        ]
        # Deduplicate by URL
        seen_urls: set[str] = set()
        sources: list[dict[str, str]] = []
        for s in research_sources + search_sources:
            url = s.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append(s)
        sources = sources[:20]

        # Step 3: PDF banana
        pdf_path = generate_bizmentor_pdf(
            intake=intake,
            report_content=report_text,
            sources=sources,
            output_filename=None,
        )
        pdf_filename = Path(pdf_path).name

        return DemoResearchResponse(
            pdf_path=pdf_filename,
            report_text=report_text,
            report=report_text,
            research_bundle=research_bundle,
            sources=sources,
        )

    except TavilyClientError as exc:
        logger.exception("Tavily research failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Tavily research failed: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Research report generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate researched report.",
        ) from exc


@router.get(
    "/demo/download-report/{filename}",
    summary="Download a generated PDF report",
)
async def download_report(filename: str):
    """Download a generated PDF report file"""
    reports_dir = Path(__file__).resolve().parents[4] / "reports"
    safe_filename = Path(filename).name
    file_path = reports_dir / safe_filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Report file not found")

    return FileResponse(
        path=file_path,
        media_type='application/pdf',
        filename=safe_filename
    )


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

    if state.extracted is None:
        state = await orchestrator.run_initial_pipeline(state)
        await repo.update_session_from_state(state)

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
    return question_key


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


def _basic_idea_quality_checks(business_idea: str) -> str | None:
    text = business_idea.strip()
    if len(text) < 12:
        return "Business idea is too short. Please describe what you want to sell or build in one clear sentence."

    words = re.findall(r"[A-Za-z]+", text)
    if len(words) < 3:
        return "Business idea is too unclear. Please write it as a real business idea, not just a few letters or fragments."

    unique_words = {word.lower() for word in words}
    if len(unique_words) <= 2 and len(words) >= 4:
        return "Business idea looks repetitive or unclear. Please describe the actual product, service, or business model."

    alpha_chars = re.findall(r"[A-Za-z]", text)
    if alpha_chars:
        vowel_count = len(re.findall(r"[aeiouAEIOU]", "".join(alpha_chars)))
        vowel_ratio = vowel_count / max(1, len(alpha_chars))
        if len(alpha_chars) >= 10 and vowel_ratio < 0.15:
            return "Business idea looks like random text instead of a real business concept. Please enter a meaningful idea."

    if re.fullmatch(r"[\W\d_]+", text):
        return "Business idea cannot contain only symbols or numbers. Please describe the business in words."

    return None


async def _validate_demo_business_idea(
    llm: LLMClient,
    business_idea: str,
    stage_type: str | None = None,
    target_audience: str | None = None,
    context: str | None = None,
) -> dict[str, Any]:
    heuristic_failure = _basic_idea_quality_checks(business_idea)
    if heuristic_failure:
        return {
            "isValid": False,
            "reason": heuristic_failure,
            "recommendation": "Example: 'I want to start a D2C earbuds brand for college students in India.'",
        }

    prompt = f"""
Validate whether this is a meaningful business idea that should proceed to AI business analysis.

Business idea: {business_idea}
Stage type: {stage_type or "not provided"}
Target audience: {target_audience or "not provided"}
Extra context: {context or "not provided"}

Return JSON only:
{{
  "is_valid": true,
  "reason": "<short reason>",
  "recommendation": "<short rewrite suggestion if invalid, else optional improvement tip>"
}}

Rules:
- Mark invalid if the text is nonsense, random characters, only filler, or not an actual business idea.
- Mark valid if it clearly describes a product, service, store, agency, app, brand, manufacturing idea, or business concept.
- Be strict but practical.
- Keep the reason under 25 words.
"""
    try:
        result = await llm.complete_json(prompt=prompt, temperature=0.0, max_tokens=180)
        is_valid = bool(result.get("is_valid"))
        reason = str(result.get("reason") or "").strip() or (
            "Idea looks valid for further analysis."
            if is_valid else
            "Idea is too unclear for analysis."
        )
        recommendation = result.get("recommendation")
        return {
            "isValid": is_valid,
            "reason": reason,
            "recommendation": str(recommendation).strip() if recommendation else None,
        }
    except Exception as exc:
        logger.warning("Idea validation fallback triggered: %s", exc)
        return {
            "isValid": True,
            "reason": "Idea passed basic validation.",
            "recommendation": None,
        }