from __future__ import annotations

from typing import Any

from app.core import config
from app.schemas.research_engine import ResearchCategoryResult, ResearchProviderError, enforce_citations
from app.services.research_engine.providers.cerebras_provider import CerebrasAnalysisProvider


SYSTEM_SYNTHESIS = """
You synthesize category results into a strategic recommendation.
Rules:
- Use ONLY the provided category outputs.
- Do NOT fabricate facts.
- If upstream categories are insufficient/failed, decision must be insufficient_data.
- Return ONLY valid JSON.
""".strip()


async def run(
    *,
    demand: ResearchCategoryResult,
    competitors: ResearchCategoryResult,
    pain_points: ResearchCategoryResult,
    llm: CerebrasAnalysisProvider,
) -> dict[str, Any]:
    upstream = [demand, competitors, pain_points]
    if all(r.status != "completed" for r in upstream):
        return {
            "decision": "insufficient_data",
            "overall_score": 0,
            "confidence": 0.2,
            "best_niche": None,
            "best_first_product": None,
            "best_growth_channel": None,
            "key_risks": [],
            "next_30_days": [],
            "reasoning": "All live-data categories returned insufficient data or failed.",
        }

    payload = {
        "demand_trend": demand.model_dump(),
        "competitor_discovery": competitors.model_dump(),
        "audience_pain_points": pain_points.model_dump(),
    }

    schema = {
        "decision": "proceed|pivot|avoid|insufficient_data",
        "overall_score": 0,
        "confidence": 0.0,
        "best_niche": "string|null",
        "best_first_product": "string|null",
        "best_growth_channel": "string|null",
        "key_risks": ["string"],
        "next_30_days": ["string"],
        "reasoning": "string",
    }

    prompt = f"""
Upstream category outputs:
{payload}

Return strict JSON with this schema:
{schema}
""".strip()

    try:
        return await llm.analyze_json(prompt=prompt, system=SYSTEM_SYNTHESIS, max_tokens=config.MAX_ANALYSIS_TOKENS)
    except ResearchProviderError:
        return {
            "decision": "insufficient_data",
            "overall_score": 0,
            "confidence": 0.2,
            "best_niche": None,
            "best_first_product": None,
            "best_growth_channel": None,
            "key_risks": [],
            "next_30_days": [],
            "reasoning": "Synthesis model unavailable.",
        }

