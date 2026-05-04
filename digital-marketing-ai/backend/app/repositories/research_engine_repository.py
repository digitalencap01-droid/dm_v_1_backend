from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.research_engine import ResearchCategoryResult, ResearchRun, ResearchSource


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def make_json_safe(value: Any) -> Any:
    if isinstance(value, uuid.UUID):
        return str(value)

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [make_json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [make_json_safe(v) for v in value]

    if hasattr(value, "model_dump"):
        return make_json_safe(value.model_dump())

    if hasattr(value, "dict"):
        return make_json_safe(value.dict())

    return value


class ResearchEngineRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create_run(
        self,
        *,
        session_id: uuid.UUID,
        project_id: uuid.UUID | None,
        input_snapshot: dict[str, Any] | None,
    ) -> ResearchRun:
        run = ResearchRun(
            session_id=str(session_id),
            project_id=str(project_id) if project_id else None,
            status="running",
            input_snapshot=make_json_safe(input_snapshot or {}),
        )
        self._db.add(run)
        await self._db.flush()
        return run

    async def set_run_status(
        self,
        run_id: uuid.UUID,
        *,
        status: str,
        error_message: str | None = None,
        final_score: float | None = None,
        final_decision: str | None = None,
        confidence_score: float | None = None,
        completed: bool = False,
    ) -> None:
        run = await self.get_run(run_id)
        if run is None:
            return

        run.status = status
        run.error_message = error_message
        run.final_score = final_score
        run.final_decision = final_decision
        run.confidence_score = confidence_score

        if completed:
            run.completed_at = _utcnow()

        await self._db.flush()

    async def get_run(self, run_id: uuid.UUID | str) -> ResearchRun | None:
        result = await self._db.execute(
            select(ResearchRun).where(ResearchRun.id == str(run_id))
        )
        return result.scalar_one_or_none()

    async def list_category_results(self, run_id: uuid.UUID | str) -> list[ResearchCategoryResult]:
        result = await self._db.execute(
            select(ResearchCategoryResult).where(
                ResearchCategoryResult.research_run_id == str(run_id)
            )
        )
        return list(result.scalars().all())

    async def list_sources(
        self,
        run_id: uuid.UUID | str,
        *,
        category_key: str | None = None,
    ) -> list[ResearchSource]:
        stmt = select(ResearchSource).where(ResearchSource.research_run_id == str(run_id))

        if category_key:
            stmt = stmt.where(ResearchSource.category_key == category_key)

        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    async def upsert_category_result(
        self,
        *,
        run_id: uuid.UUID | str,
        category_key: str,
        category_name: str,
        status: str,
        score: int | None,
        confidence: float | None,
        summary: str | None,
        findings: dict | None,
        raw_provider_response: dict | None,
        error_message: str | None,
    ) -> ResearchCategoryResult:
        result = await self._db.execute(
            select(ResearchCategoryResult).where(
                ResearchCategoryResult.research_run_id == str(run_id),
                ResearchCategoryResult.category_key == category_key,
            )
        )

        existing = result.scalar_one_or_none()

        safe_findings = make_json_safe(findings or {})
        safe_raw_provider_response = make_json_safe(raw_provider_response or {})

        if existing:
            existing.category_name = category_name
            existing.status = status
            existing.score = score
            existing.confidence = confidence
            existing.summary = summary
            existing.findings = safe_findings
            existing.raw_provider_response = safe_raw_provider_response
            existing.error_message = error_message
            await self._db.flush()
            return existing

        row = ResearchCategoryResult(
            research_run_id=str(run_id),
            category_key=category_key,
            category_name=category_name,
            status=status,
            score=score,
            confidence=confidence,
            summary=summary,
            findings=safe_findings,
            raw_provider_response=safe_raw_provider_response,
            error_message=error_message,
        )

        self._db.add(row)
        await self._db.flush()
        return row

    async def replace_sources(
        self,
        *,
        run_id: uuid.UUID | str,
        category_key: str,
        sources: list[dict[str, Any]],
    ) -> None:
        existing = await self.list_sources(run_id, category_key=category_key)

        for row in existing:
            await self._db.delete(row)

        for item in sources:
            safe_item = make_json_safe(item)

            self._db.add(
                ResearchSource(
                    research_run_id=str(run_id),
                    category_key=category_key,
                    source_index=int(safe_item.get("source_index") or 0) or 0,
                    title=safe_item.get("title"),
                    url=str(safe_item.get("url") or ""),
                    domain=safe_item.get("domain"),
                    snippet=safe_item.get("snippet"),
                    source_type=str(safe_item.get("source_type") or "web"),
                    credibility_score=float(safe_item.get("credibility_score") or 0.0),
                    raw=make_json_safe(safe_item.get("raw") or {}),
                )
            )

        await self._db.flush()