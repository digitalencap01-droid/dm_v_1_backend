from __future__ import annotations

from typing import Any

from app.core import config
from app.schemas.research_engine import ResearchCategoryResult, ResearchProviderError, enforce_citations
from app.schemas.profile_engine import FinalProfile, SessionState
from app.services.research_engine.providers.cerebras_provider import CerebrasAnalysisProvider
from app.services.research_engine.providers.you_provider import YouWebResearchProvider


SYSTEM_DEMAND_TREND = """
You analyze live web research sources for market demand & trend.
Rules:
- Use ONLY the provided sources.
- Do NOT use outside knowledge.
- Do NOT fabricate metrics.
- Every finding object MUST include source_indexes (list of ints).
- If sources are insufficient/weak, set status="insufficient_data".
Return ONLY valid JSON.
""".strip()


async def run(
    *,
    state: SessionState,
    profile: FinalProfile | None,
    web: YouWebResearchProvider,
    llm: CerebrasAnalysisProvider,
) -> ResearchCategoryResult:
    category_key = "demand_trend"
    category_name = "Demand & Trend Research"

    idea = (state.extracted.description if state.extracted else None) or (state.raw_input or "")
    location = (state.answers.get("target_market_geo") if state.answers else None) or ""
    industry = (profile.industry.value if profile else "") if profile else ""

    if not idea.strip():
        return ResearchCategoryResult(
            category_key=category_key,
            category_name=category_name,
            status="insufficient_data",
            confidence=0.0,
            summary="No idea/description available to research.",
        )

    queries = [
        f"{idea} demand {location}".strip(),
        f"{industry} market trend {location}".strip(),
        f"{idea} trends 2025 2026".strip(),
    ]
    queries = [q for q in queries if q and q != "market trend" and q != "demand"]
    queries = queries[: config.MAX_SEARCHES_PER_CATEGORY]

    all_sources = []
    raw_payloads: list[dict[str, Any]] = []

    try:
        for q in queries:
            result = await web.search(q, limit=config.MAX_SOURCES_PER_CATEGORY)
            raw_payloads.append(result.raw_payload or {})
            all_sources.extend(result.sources)
            if len(all_sources) >= config.MAX_TOTAL_SOURCES_PER_RUN:
                break
    except ResearchProviderError as exc:
        return ResearchCategoryResult(
            category_key=category_key,
            category_name=category_name,
            status="failed",
            confidence=0.0,
            error_message=str(exc),
        )

    if not all_sources:
        return ResearchCategoryResult(
            category_key=category_key,
            category_name=category_name,
            status="insufficient_data",
            confidence=0.2,
            summary="No reliable live sources found for this query set.",
            raw_provider_response={"queries": queries, "raw": raw_payloads},
        )

    # Reindex sources globally so citations are unambiguous across multiple searches.
    all_sources = [s.model_copy(update={"source_index": i}) for i, s in enumerate(all_sources, start=1)]

    sources_block = [
        {
            "source_index": s.source_index,
            "title": s.title,
            "url": s.url,
            "domain": s.domain,
            "snippet": s.snippet,
        }
        for s in all_sources[: config.MAX_SOURCES_PER_CATEGORY]
    ]

    schema = {
        "status": "completed|insufficient_data",
        "score": 0,
        "confidence": 0.0,
        "summary": "string or null",
        "findings": {
            "demand_signals": [{"finding": "string", "source_indexes": [1], "confidence": 0.0}],
            "trend_direction": "rising|stable|declining|unknown",
            "geo_relevance": "low|medium|high|unknown",
            "risks": [{"finding": "string", "source_indexes": [1], "confidence": 0.0}],
        },
    }

    prompt = f"""
Business idea / context:
{idea}

Location: {location or "unknown"}
Industry: {industry or "unknown"}

Live sources (use ONLY these):
{sources_block}

Return strict JSON with this schema:
{schema}
""".strip()

    try:
        analysis = await llm.analyze_json(prompt=prompt, system=SYSTEM_DEMAND_TREND)
    except ResearchProviderError as exc:
        return ResearchCategoryResult(
            category_key=category_key,
            category_name=category_name,
            status="failed",
            confidence=0.0,
            error_message=str(exc),
            sources=all_sources[: config.MAX_SOURCES_PER_CATEGORY],
        )

    status = str(analysis.get("status") or "completed").strip()
    if status not in ("completed", "insufficient_data"):
        status = "completed"

    findings = analysis.get("findings") if isinstance(analysis.get("findings"), dict) else {}
    try:
        enforce_citations(findings, max_source_index=max(s.source_index for s in all_sources))
    except Exception as exc:
        return ResearchCategoryResult(
            category_key=category_key,
            category_name=category_name,
            status="failed",
            confidence=0.0,
            error_message=f"citation_enforcement_failed: {exc}",
            sources=all_sources[: config.MAX_SOURCES_PER_CATEGORY],
            raw_provider_response={"analysis": analysis},
        )

    return ResearchCategoryResult(
        category_key=category_key,
        category_name=category_name,
        status=status,  # type: ignore[arg-type]
        score=int(analysis.get("score") or 0) if status == "completed" else None,
        confidence=float(analysis.get("confidence") or 0.0),
        summary=(str(analysis.get("summary")).strip() if analysis.get("summary") else None),
        findings=findings,
        sources=all_sources[: config.MAX_SOURCES_PER_CATEGORY],
        raw_provider_response={"queries": queries, "raw": raw_payloads},
    )
