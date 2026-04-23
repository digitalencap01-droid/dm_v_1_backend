"""
SQLAlchemy models for the Profile Intelligence Engine.

Three tables:
- profile_engine_sessions  : one row per engine session
- profile_engine_answers   : one row per follow-up answer in a session
- profile_engine_profiles  : one row per completed final profile

Uses SQLAlchemy 2.0 declarative style with mapped_column().
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Declarative base — isolated from any other base the project may define
# so this module stays self-contained during the ideation phase.
# ---------------------------------------------------------------------------


class ProfileEngineBase(DeclarativeBase):
    """Base class for all profile-engine ORM models."""
    pass


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class ProfileEngineSession(ProfileEngineBase):
    """
    Tracks an individual profile engine session from start to completion.
    """

    __tablename__ = "profile_engine_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="pending", index=True
    )
    raw_input: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Serialised intermediate state (JSON blob — used to reconstruct SessionState)
    extracted_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    classified_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    readiness_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    need_routing_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    confidence_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    asked_questions: Mapped[list | None] = mapped_column(JSON, nullable=True, default=list)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    # Relationships
    answers: Mapped[list["ProfileEngineAnswer"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    profile: Mapped["ProfileEngineProfile | None"] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        uselist=False,
        lazy="selectin",
    )

    def __repr__(self) -> str:
        return f"<ProfileEngineSession id={self.id} status={self.status}>"


class ProfileEngineAnswer(ProfileEngineBase):
    """
    Stores each follow-up question and answer pair for a session.
    """

    __tablename__ = "profile_engine_answers"
    __table_args__ = (
        UniqueConstraint("session_id", "question_key", name="uq_session_question"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("profile_engine_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question_key: Mapped[str] = mapped_column(String(100), nullable=False)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    session: Mapped["ProfileEngineSession"] = relationship(back_populates="answers")

    def __repr__(self) -> str:
        return f"<ProfileEngineAnswer session={self.session_id} key={self.question_key}>"


class ProfileEngineProfile(ProfileEngineBase):
    """
    The fully assembled final profile for a completed session.
    One-to-one with ProfileEngineSession.
    """

    __tablename__ = "profile_engine_profiles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("profile_engine_sessions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # Enum values stored as strings
    persona: Mapped[str] = mapped_column(String(50), nullable=False)
    industry: Mapped[str] = mapped_column(String(50), nullable=False)
    readiness_level: Mapped[str] = mapped_column(String(50), nullable=False)
    primary_need: Mapped[str] = mapped_column(String(50), nullable=False)
    secondary_needs: Mapped[list | None] = mapped_column(JSON, nullable=True)

    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)

    business_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    target_audience: Mapped[str | None] = mapped_column(Text, nullable=True)
    product_or_service: Mapped[str | None] = mapped_column(Text, nullable=True)
    revenue_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    current_challenges: Mapped[list | None] = mapped_column(JSON, nullable=True)
    goals: Mapped[list | None] = mapped_column(JSON, nullable=True)
    recommended_channels: Mapped[list | None] = mapped_column(JSON, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Full raw snapshot for audit / reprocessing
    raw_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    session: Mapped["ProfileEngineSession"] = relationship(back_populates="profile")

    def __repr__(self) -> str:
        return f"<ProfileEngineProfile session={self.session_id} persona={self.persona}>"
