"""
Required-fields policy for the Profile Intelligence Engine.

This module is intentionally deterministic (no LLM calls).
Given the current SessionState (including pre-categorization outputs),
it returns a prioritized list of question keys that are REQUIRED to reach
\"research-ready\" minimum context.
"""

from __future__ import annotations

from app.schemas.profile_engine import NeedState, ReadinessLevel, SessionState


def required_question_keys(state: SessionState) -> list[str]:
    """
    Return required question keys in priority order.

    Notes:
    - Only called after baseline is complete and the AI pre-categorization
      pipeline has run (so state.extracted/classified/readiness/need_routing
      may be present).
    - Keep the list short; the selector may cap how many are asked.
    """
    answers = state.answers or {}
    extracted = state.extracted
    readiness = state.readiness
    need = state.need_routing

    required: list[str] = []

    # 1) Research identifier: website_url OR business_name
    if not answers.get("website_url"):
        if not (extracted and extracted.business_name):
            required.append("website_url")

    # 2) Offer clarity (only if extraction couldn't infer it)
    if not answers.get("product_or_service"):
        if not (extracted and extracted.product_or_service):
            required.append("product_or_service")

    # 3) Readiness calibration
    if not answers.get("monthly_revenue"):
        if readiness and readiness.readiness_level in (ReadinessLevel.UNKNOWN, ReadinessLevel.MVP):
            required.append("monthly_revenue")

    # 4) Channel/need-specific
    if need and need.primary_need == NeedState.LEAD_GENERATION:
        if not answers.get("primary_channel"):
            required.append("primary_channel")

    if need and need.primary_need == NeedState.BRAND_AWARENESS:
        if not answers.get("target_market_geo"):
            required.append("target_market_geo")

    # 5) Bottleneck is generally useful if need is unknown
    if not answers.get("biggest_bottleneck") and (not need or need.primary_need == NeedState.UNKNOWN):
        required.append("biggest_bottleneck")

    return required

