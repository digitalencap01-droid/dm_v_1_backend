from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core import config
from app.repositories.profile_engine_repository import ProfileEngineRepository
from app.repositories.research_engine_repository import ResearchEngineRepository
from app.schemas.profile_engine import FinalProfile, SessionState
from app.schemas.research_engine import (
    ResearchCategoryResult,
    ResearchReportResponse,
    ResearchRunCreate,
    ResearchRunResponse,
    ResearchStatusResponse,
    ResearchSource,
)
from app.services.research_engine.categories import (
    run_audience_pain_points,
    run_competitor_discovery,
    run_demand_trend,
)
from app.services.research_engine.categories.strategic_synthesis import run as run_synthesis
from app.services.research_engine.providers.cerebras_provider import CerebrasAnalysisProvider
from app.services.research_engine.providers.you_provider import YouWebResearchProvider


MVP_CATEGORIES = [
    ("demand_trend", "Demand & Trend Research"),
    ("competitor_discovery", "Competitor Discovery"),
    ("audience_pain_points", "Audience & Pain-Point Research"),
]


class ResearchEngineOrchestrator:
    def __init__(
        self,
        *,
        profile_db: AsyncSession,
        research_db: AsyncSession,
    ) -> None:
        self._profile_repo = ProfileEngineRepository(profile_db)
        self._research_repo = ResearchEngineRepository(research_db)
        self._web = YouWebResearchProvider()
        self._llm = CerebrasAnalysisProvider()

    async def start_and_run(self, body: ResearchRunCreate) -> uuid.UUID:
        state = await self._profile_repo.load_session_state(body.session_id)
        if state is None:
            raise ValueError(f"Session {body.session_id} not found.")

        profile_record = await self._profile_repo.get_profile(body.session_id)
        profile: FinalProfile | None = None
        if profile_record:
            from app.schemas.profile_engine import IndustryType, NeedState, PersonaType, ReadinessLevel

            profile = FinalProfile(
                session_id=profile_record.session_id,
                persona=PersonaType(profile_record.persona),
                industry=IndustryType(profile_record.industry),
                readiness_level=ReadinessLevel(profile_record.readiness_level),
                primary_need=NeedState(profile_record.primary_need),
                secondary_needs=[NeedState(x) for x in (profile_record.secondary_needs or [])],
                confidence_score=float(profile_record.confidence_score or 0.0),
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

        input_snapshot = {
            "session_id": str(body.session_id),
            "project_id": str(body.project_id) if body.project_id else None,
            "state": state.model_dump(),
            "profile": profile.model_dump() if profile else None,
        }

        run = await self._research_repo.create_run(
            session_id=body.session_id,
            project_id=body.project_id,
            input_snapshot=input_snapshot,
        )

        # Hard requirement for MVP live research.
        if config.RESEARCH_MODE != "live" or not config.YOU_API_KEY:
            await self._research_repo.set_run_status(
                run.id,
                status="failed",
                error_message="Live research provider unavailable (YOU_API_KEY missing or RESEARCH_MODE not live).",
                completed=True,
            )
            return run.id

        # Run categories sequentially.
        results: dict[str, ResearchCategoryResult] = {}

        demand = await run_demand_trend(state=state, profile=profile, web=self._web, llm=self._llm)
        await self._persist_category(run.id, demand)
        results[demand.category_key] = demand

        competitors = await run_competitor_discovery(state=state, profile=profile, web=self._web, llm=self._llm)
        await self._persist_category(run.id, competitors)
        results[competitors.category_key] = competitors

        pain = await run_audience_pain_points(state=state, profile=profile, web=self._web, llm=self._llm)
        await self._persist_category(run.id, pain)
        results[pain.category_key] = pain

        synthesis = await run_synthesis(demand=demand, competitors=competitors, pain_points=pain, llm=self._llm)

        # MVP final status: completed if at least one completed category, else partial/failed.
        any_completed = any(r.status == "completed" for r in results.values())
        status = "completed" if any_completed else "partial"
        if not any_completed and any(r.status == "failed" for r in results.values()):
            status = "failed"

        final_score = float(synthesis.get("overall_score") or 0.0)
        final_decision = str(synthesis.get("decision") or "insufficient_data")
        confidence_score = float(synthesis.get("confidence") or 0.0)

        await self._research_repo.set_run_status(
            run.id,
            status=status,
            final_score=final_score,
            final_decision=final_decision,
            confidence_score=confidence_score,
            completed=True,
        )

        # Persist synthesis as a category-like result for report retrieval.
        await self._research_repo.upsert_category_result(
            run_id=run.id,
            category_key="strategic_synthesis",
            category_name="Strategic Recommendation Synthesis",
            status="completed" if final_decision != "insufficient_data" else "insufficient_data",
            score=None,
            confidence=confidence_score,
            summary=str(synthesis.get("reasoning") or "") or None,
            findings=synthesis,
            raw_provider_response=None,
            error_message=None,
        )

        return run.id

    async def _persist_category(self, run_id: uuid.UUID, result: ResearchCategoryResult) -> None:
        await self._research_repo.upsert_category_result(
            run_id=run_id,
            category_key=result.category_key,
            category_name=result.category_name,
            status=result.status,
            score=result.score,
            confidence=result.confidence,
            summary=result.summary,
            findings=result.findings,
            raw_provider_response=result.raw_provider_response,
            error_message=result.error_message,
        )
        await self._research_repo.replace_sources(
            run_id=run_id,
            category_key=result.category_key,
            sources=[s.model_dump() for s in (result.sources or [])],
        )

    async def get_status(self, run_id: uuid.UUID) -> ResearchStatusResponse:
        run = await self._research_repo.get_run(run_id)
        if run is None:
            raise ValueError("Research run not found.")
        categories = await self._research_repo.list_category_results(run_id)
        completed = len([c for c in categories if c.status in ("completed", "insufficient_data", "failed", "not_applicable")])
        return ResearchStatusResponse(
            id=run.id,
            session_id=run.session_id,
            status=run.status,  # type: ignore[arg-type]
            completed_categories=completed,
            total_categories=len(MVP_CATEGORIES) + 1,  # + synthesis
            current_category=None,
        )

    async def get_report(self, run_id: uuid.UUID) -> ResearchReportResponse:
        run = await self._research_repo.get_run(run_id)
        if run is None:
            raise ValueError("Research run not found.")

        categories_rows = await self._research_repo.list_category_results(run_id)
        sources_rows = await self._research_repo.list_sources(run_id)

        # Map sources to category
        sources_by_category: dict[str, list[dict[str, Any]]] = {}
        for s in sources_rows:
            sources_by_category.setdefault(s.category_key, []).append(
                {
                    "source_index": s.source_index,
                    "title": s.title,
                    "url": s.url,
                    "domain": s.domain,
                    "snippet": s.snippet,
                    "source_type": s.source_type,
                    "fetched_at": s.fetched_at,
                    "credibility_score": float(s.credibility_score or 0.0),
                    "raw": s.raw,
                }
            )

        categories: list[ResearchCategoryResult] = []
        synthesis_payload: dict[str, Any] = {}

        for row in categories_rows:
            cat_sources = sources_by_category.get(row.category_key, [])
            item = ResearchCategoryResult(
                category_key=row.category_key,
                category_name=row.category_name,
                status=row.status,  # type: ignore[arg-type]
                score=row.score,
                confidence=float(row.confidence or 0.0),
                summary=row.summary,
                findings=row.findings or {},
                sources=[ResearchSource(**s) for s in cat_sources],
                raw_provider_response=row.raw_provider_response,
                error_message=row.error_message,
            )
            categories.append(item)
            if row.category_key == "strategic_synthesis":
                synthesis_payload = row.findings or {}

        run_resp = ResearchRunResponse(
            id=run.id,
            session_id=run.session_id,
            project_id=run.project_id,
            status=run.status,  # type: ignore[arg-type]
            final_decision=run.final_decision,
            final_score=run.final_score,
            confidence_score=run.confidence_score,
            created_at=run.created_at,
            updated_at=run.updated_at,
            completed_at=run.completed_at,
            error_message=run.error_message,
        )

        return ResearchReportResponse(
            research_run=run_resp,
            categories=categories,
            strategic_recommendation=synthesis_payload,
        )
