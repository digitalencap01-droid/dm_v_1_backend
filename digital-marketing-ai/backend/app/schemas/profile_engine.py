"""
Profile Intelligence Engine — shared Pydantic schemas, enums, and DTO contracts.

All request/response DTOs, internal structured LLM output schemas, and domain
enums live here.  No business logic — pure data contracts only.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PersonaType(str, Enum):
    FOUNDER = "founder"
    MARKETER = "marketer"
    SALES_LEAD = "sales_lead"
    AGENCY = "agency"
    ENTERPRISE = "enterprise"
    SMB = "smb"
    UNKNOWN = "unknown"


class IndustryType(str, Enum):
    SAAS = "saas"
    ECOMMERCE = "ecommerce"
    PROFESSIONAL_SERVICES = "professional_services"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    REAL_ESTATE = "real_estate"
    FINANCE = "finance"
    OTHER = "other"


class ReadinessLevel(str, Enum):
    IDEA_STAGE = "idea_stage"
    MVP = "mvp"
    EARLY_TRACTION = "early_traction"
    SCALING = "scaling"
    MATURE = "mature"
    UNKNOWN = "unknown"


class NeedState(str, Enum):
    BRAND_AWARENESS = "brand_awareness"
    LEAD_GENERATION = "lead_generation"
    CUSTOMER_RETENTION = "customer_retention"
    REVENUE_GROWTH = "revenue_growth"
    PRODUCT_LAUNCH = "product_launch"
    MARKET_EXPANSION = "market_expansion"
    UNKNOWN = "unknown"


class GoalType(str, Enum):
    BRAND_AWARENESS = "brand_awareness"
    LEAD_GENERATION = "lead_generation"
    CUSTOMER_RETENTION = "customer_retention"
    REVENUE_GROWTH = "revenue_growth"
    PRODUCT_LAUNCH = "product_launch"
    MARKET_EXPANSION = "market_expansion"
    OTHER = "other"


class ResearchSlot(str, Enum):
    OFFER = "offer"
    ICP = "icp"


class SessionStatus(str, Enum):
    PENDING = "pending"
    COLLECTING_INPUT = "collecting_input"
    ASKING_QUESTIONS = "asking_questions"
    BUILDING_PROFILE = "building_profile"
    COMPLETE = "complete"
    FAILED = "failed"


class QuestionType(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


# ---------------------------------------------------------------------------
# Internal structured LLM output schemas
# (used between service layers — not exposed to the HTTP surface)
# ---------------------------------------------------------------------------


class ExtractedData(BaseModel):
    """Normalised extraction of raw business input text."""

    business_name: str | None = None
    description: str | None = None
    target_audience: str | None = None
    product_or_service: str | None = None
    revenue_model: str | None = None
    current_challenges: list[str] = Field(default_factory=list)
    mentioned_goals: list[str] = Field(default_factory=list)
    mentioned_channels: list[str] = Field(default_factory=list)
    raw_persona_hint: str | None = None
    raw_industry_hint: str | None = None
    raw_readiness_hint: str | None = None
    confidence_hints: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClassifiedProfile(BaseModel):
    """Classification output from the classifier service."""

    persona: PersonaType = PersonaType.UNKNOWN
    industry: IndustryType = IndustryType.OTHER
    persona_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    industry_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    raw_persona: str | None = None
    raw_industry: str | None = None


class ReadinessAssessment(BaseModel):
    """Output from the readiness service."""

    readiness_level: ReadinessLevel = ReadinessLevel.UNKNOWN
    readiness_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    reasoning: str | None = None
    raw_readiness: str | None = None


class NeedRouting(BaseModel):
    """Output from the need-routing service."""

    primary_need: NeedState = NeedState.UNKNOWN
    secondary_needs: list[NeedState] = Field(default_factory=list)
    need_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    reasoning: str | None = None


class Question(BaseModel):
    """A single follow-up question for the user."""

    key: str
    text: str
    question_type: QuestionType = QuestionType.OPTIONAL
    context: str | None = None
    slot: ResearchSlot | None = None


class SessionState(BaseModel):
    """
    Transient in-memory state passed between orchestrator steps.
    Not persisted directly — the repository persists the individual fields.
    """

    session_id: uuid.UUID
    status: SessionStatus = SessionStatus.PENDING
    raw_input: str | None = None
    extracted: ExtractedData | None = None
    classified: ClassifiedProfile | None = None
    readiness: ReadinessAssessment | None = None
    need_routing: NeedRouting | None = None
    answers: dict[str, str] = Field(default_factory=dict)
    asked_questions: list[str] = Field(default_factory=list)
    pending_question: Question | None = None
    confidence_score: float = 0.0

    # Baseline + optional gating
    baseline_complete: bool = False
    allow_optional: bool = False
    declared_stage: ReadinessLevel = ReadinessLevel.UNKNOWN
    declared_goals: list[GoalType] = Field(default_factory=list)

    # Dynamic required slot tracking (research-minimum)
    required_slots_missing: list[ResearchSlot] = Field(default_factory=list)
    required_slots_filled: dict[ResearchSlot, str] = Field(default_factory=dict)
    required_questions_asked: int = 0
    max_required_questions: int = 3


class FinalProfile(BaseModel):
    """Fully assembled profile — output of the profile_builder service."""

    session_id: uuid.UUID
    persona: PersonaType
    industry: IndustryType
    readiness_level: ReadinessLevel
    primary_need: NeedState
    secondary_needs: list[NeedState] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    business_name: str | None = None
    target_audience: str | None = None
    product_or_service: str | None = None
    revenue_model: str | None = None
    current_challenges: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    recommended_channels: list[str] = Field(default_factory=list)
    summary: str | None = None
    raw_data: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# HTTP Request DTOs
# ---------------------------------------------------------------------------


class StartSessionRequest(BaseModel):
    """Payload for starting a new profile engine session."""

    user_id: str | None = Field(default=None, description="Optional caller user ID")

    @field_validator("user_id")
    @classmethod
    def strip_user_id(cls, v: str | None) -> str | None:
        return v.strip() if v else None


class SubmitInputRequest(BaseModel):
    """Payload for submitting the initial business description."""

    raw_input: str = Field(
        min_length=10,
        max_length=5000,
        description="Free-form business description",
    )

    @field_validator("raw_input")
    @classmethod
    def strip_input(cls, v: str) -> str:
        return v.strip()


class AnswerQuestionRequest(BaseModel):
    """Payload for answering a follow-up question."""

    question_key: str = Field(description="Key of the question being answered")
    answer: str | list[str] = Field(
        description="User's answer text (string) or multi-select values (list of strings).",
    )

    @field_validator("answer")
    @classmethod
    def strip_answer(cls, v: str | list[str]) -> str | list[str]:
        if isinstance(v, str):
            vv = v.strip()
            if not vv:
                raise ValueError("answer must not be empty")
            if len(vv) > 2000:
                raise ValueError("answer is too long")
            return vv
        if isinstance(v, list):
            cleaned = [str(x).strip() for x in v if str(x).strip()]
            if not cleaned:
                raise ValueError("answer must not be empty")
            return cleaned
        raise ValueError("invalid answer type")


# ---------------------------------------------------------------------------
# HTTP Response DTOs
# ---------------------------------------------------------------------------


class SessionResponse(BaseModel):
    """Response for session creation or status queries."""

    session_id: uuid.UUID
    status: SessionStatus
    created_at: datetime
    user_id: str | None = None


class EngineStepResponse(BaseModel):
    """
    Generic response after submitting input or answering a question.
    Either contains a follow-up question OR signals profile completion.
    """

    session_id: uuid.UUID
    status: SessionStatus
    next_question: Question | None = None
    profile_ready: bool = False
    message: str | None = None

    confidence_score: float = 0.0
    recommended_action: str | None = None
    baseline_remaining: int | None = None


class ProfileResponse(BaseModel):
    """Response containing the completed profile."""

    session_id: uuid.UUID
    status: SessionStatus
    profile: FinalProfile | None = None
    confidence_score: float
    message: str | None = None


class EngineMessage(BaseModel):
    role: MessageRole
    content: str
    key: str | None = None
    created_at: datetime | None = None


class MessagesResponse(BaseModel):
    session_id: uuid.UUID
    messages: list[EngineMessage] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    error: str
    detail: str | None = None
    session_id: uuid.UUID | None = None
