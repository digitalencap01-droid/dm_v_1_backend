"""
Profile Builder — assembles the final structured marketing profile.

Responsibility: Call LLM to synthesise all collected signal into a
comprehensive FinalProfile, filling in recommended channels and summary.
"""

from __future__ import annotations

import logging
import uuid

from app.schemas.profile_engine import (
    ClassifiedProfile,
    ExtractedData,
    FinalProfile,
    NeedRouting,
    NeedState,
    ReadinessAssessment,
    ReadinessLevel,
    SessionState,
)
from app.services.llm.client import LLMClient, LLMClientError
from app.services.llm.prompts import SYSTEM_PROFILE_BUILDER, build_profile_prompt

logger = logging.getLogger(__name__)


async def build_profile(state: SessionState, llm: LLMClient) -> FinalProfile:
    """
    Build the final profile by synthesising all service outputs.

    Args:
        state: Complete (or near-complete) SessionState.
        llm: LLM client instance.

    Returns:
        FinalProfile ready for persistence and API response.
    """
    classified = state.classified or ClassifiedProfile()
    readiness = state.readiness or ReadinessAssessment()
    need_routing = state.need_routing or NeedRouting()
    extracted = state.extracted or ExtractedData()

    prompt = build_profile_prompt(
        session_id=str(state.session_id),
        persona=classified.persona.value,
        industry=classified.industry.value,
        readiness_level=readiness.readiness_level.value,
        primary_need=need_routing.primary_need.value,
        secondary_needs=[n.value for n in need_routing.secondary_needs],
        extracted=extracted.model_dump(),
        answers=state.answers,
        confidence_score=state.confidence_score,
    )

    try:
        data = await llm.complete_json(prompt=prompt, system=SYSTEM_PROFILE_BUILDER)
        enriched = _parse_llm_response(data)
    except (LLMClientError, ValueError) as exc:
        logger.warning("Profile builder LLM call failed: %s — using raw extracted data", exc)
        enriched = {}

    # Stage-field eligibility guardrails (idea-stage should not imply operational maturity).
    if readiness.readiness_level == ReadinessLevel.IDEA_STAGE:
        if not extracted.revenue_model and not state.answers.get("revenue_model"):
            enriched["revenue_model"] = None

    return FinalProfile(
        session_id=state.session_id,
        persona=classified.persona,
        industry=classified.industry,
        readiness_level=readiness.readiness_level,
        primary_need=need_routing.primary_need,
        secondary_needs=need_routing.secondary_needs,
        confidence_score=state.confidence_score,
        business_name=enriched.get("business_name") or extracted.business_name,
        target_audience=enriched.get("target_audience") or extracted.target_audience,
        product_or_service=enriched.get("product_or_service") or extracted.product_or_service,
        revenue_model=enriched.get("revenue_model") or extracted.revenue_model,
        current_challenges=enriched.get("current_challenges") or extracted.current_challenges,
        goals=enriched.get("goals") or extracted.mentioned_goals,
        recommended_channels=enriched.get("recommended_channels", []),
        summary=enriched.get("summary"),
        raw_data={
            "extracted": extracted.model_dump(),
            "classified": classified.model_dump(),
            "readiness": readiness.model_dump(),
            "need_routing": need_routing.model_dump(),
            "answers": state.answers,
        },
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_llm_response(data: dict) -> dict:
    """Extract safe values from the LLM profile payload."""
    return {
        "business_name": _safe_str(data.get("business_name")),
        "target_audience": _safe_str(data.get("target_audience")),
        "product_or_service": _safe_str(data.get("product_or_service")),
        "revenue_model": _safe_str(data.get("revenue_model")),
        "current_challenges": _safe_list(data.get("current_challenges")),
        "goals": _safe_list(data.get("goals")),
        "recommended_channels": _safe_list(data.get("recommended_channels")),
        "summary": _safe_str(data.get("summary")),
    }


def _safe_str(v: object) -> str | None:
    if v is None:
        return None
    text = str(v).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"null", "none", "na", "n/a", "undefined"}:
        return None
    return text


def _safe_list(v: object) -> list[str]:
    if not isinstance(v, list):
        return []
    return [str(item).strip() for item in v if item]
