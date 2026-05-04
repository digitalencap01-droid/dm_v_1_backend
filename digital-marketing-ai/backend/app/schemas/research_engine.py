"""
Research Engine schemas (Phase 3).

Strict rules:
- No mock data.
- No fabricated insights.
- If sources are missing/weak, return `insufficient_data`.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


ResearchRunStatus = Literal["pending", "running", "completed", "failed", "partial"]
ResearchCategoryStatus = Literal[
    "completed",
    "insufficient_data",
    "failed",
    "not_applicable",
]


class ResearchRunCreate(BaseModel):
    session_id: uuid.UUID
    project_id: uuid.UUID | None = None


class ResearchSource(BaseModel):
    source_index: int = Field(ge=1, description="1-based index used for citations.")
    title: str | None = None
    url: str
    domain: str | None = None
    snippet: str | None = None
    source_type: str = "web"
    fetched_at: datetime
    credibility_score: float = Field(ge=0.0, le=1.0, default=0.0)
    raw: dict[str, Any] | None = None


class SearchProviderResult(BaseModel):
    query: str
    sources: list[ResearchSource] = Field(default_factory=list)
    raw_payload: dict[str, Any] | None = None


class ResearchCategoryResult(BaseModel):
    category_key: str
    category_name: str
    status: ResearchCategoryStatus
    score: int | None = Field(default=None, ge=0, le=100)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    summary: str | None = None
    findings: dict[str, Any] = Field(default_factory=dict)
    sources: list[ResearchSource] = Field(default_factory=list)
    raw_provider_response: dict[str, Any] | None = None
    error_message: str | None = None


class ResearchRunResponse(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    project_id: uuid.UUID | None = None
    status: ResearchRunStatus
    final_decision: str | None = None
    final_score: float | None = None
    confidence_score: float | None = None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    error_message: str | None = None


class ResearchStatusResponse(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    status: ResearchRunStatus
    completed_categories: int = 0
    total_categories: int = 0
    current_category: str | None = None


class ResearchReportResponse(BaseModel):
    research_run: ResearchRunResponse
    categories: list[ResearchCategoryResult] = Field(default_factory=list)
    strategic_recommendation: dict[str, Any] = Field(default_factory=dict)


class ResearchProviderError(RuntimeError):
    pass


class CitationViolationError(ValueError):
    pass


def enforce_citations(findings: dict[str, Any], max_source_index: int) -> None:
    """
    Enforce that any list of findings includes `source_indexes` where applicable.

    This is intentionally lightweight for MVP: it checks common patterns used by the analysis layer.
    """
    def _check_obj(obj: Any) -> None:
        if isinstance(obj, dict):
            if "finding" in obj:
                indexes = obj.get("source_indexes")
                if not isinstance(indexes, list) or not indexes:
                    raise CitationViolationError("finding is missing non-empty source_indexes")
                for idx in indexes:
                    if not isinstance(idx, int) or idx < 1 or idx > max_source_index:
                        raise CitationViolationError(f"invalid source index: {idx}")
            for v in obj.values():
                _check_obj(v)
        elif isinstance(obj, list):
            for item in obj:
                _check_obj(item)

    _check_obj(findings)

