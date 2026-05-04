from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.db.research_session import get_research_db_session
from app.schemas.research_engine import ResearchReportResponse, ResearchRunCreate, ResearchStatusResponse
from app.services.research_engine.orchestrator import ResearchEngineOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["Research Engine"])


def get_profile_db(db: AsyncSession = Depends(get_db_session)) -> AsyncSession:
    return db


def get_research_db(db: AsyncSession = Depends(get_research_db_session)) -> AsyncSession:
    return db


@router.post(
    "/start",
    status_code=status.HTTP_201_CREATED,
    response_model=dict,
    summary="Start a research run (MVP: runs immediately)",
)
async def start_research(
    body: ResearchRunCreate,
    profile_db: AsyncSession = Depends(get_profile_db),
    research_db: AsyncSession = Depends(get_research_db),
) -> dict:
    try:
        orchestrator = ResearchEngineOrchestrator(profile_db=profile_db, research_db=research_db)
        run_id = await orchestrator.start_and_run(body)
        await research_db.commit()
        return {"research_run_id": run_id, "status": "running"}
    except ValueError as exc:
        await research_db.rollback()
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        await research_db.rollback()
        logger.exception("Failed to start research: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to start research.") from exc


@router.get(
    "/{research_run_id}",
    response_model=ResearchStatusResponse,
    summary="Get research run status/progress",
)
async def get_research_status(
    research_run_id: uuid.UUID,
    profile_db: AsyncSession = Depends(get_profile_db),
    research_db: AsyncSession = Depends(get_research_db),
) -> ResearchStatusResponse:
    try:
        orchestrator = ResearchEngineOrchestrator(profile_db=profile_db, research_db=research_db)
        return await orchestrator.get_status(research_run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/{research_run_id}/report",
    response_model=ResearchReportResponse,
    summary="Get full research report",
)
async def get_research_report(
    research_run_id: uuid.UUID,
    profile_db: AsyncSession = Depends(get_profile_db),
    research_db: AsyncSession = Depends(get_research_db),
) -> ResearchReportResponse:
    try:
        orchestrator = ResearchEngineOrchestrator(profile_db=profile_db, research_db=research_db)
        return await orchestrator.get_report(research_run_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/{research_run_id}/rerun",
    response_model=dict,
    summary="Re-run an existing research run (MVP: creates a new run)",
)
async def rerun_research(
    research_run_id: uuid.UUID,
    profile_db: AsyncSession = Depends(get_profile_db),
    research_db: AsyncSession = Depends(get_research_db),
) -> dict:
    # MVP behavior: treat rerun as start with same session_id.
    from app.repositories.research_engine_repository import ResearchEngineRepository

    repo = ResearchEngineRepository(research_db)
    run = await repo.get_run(research_run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Research run not found.")

    try:
        orchestrator = ResearchEngineOrchestrator(profile_db=profile_db, research_db=research_db)
        new_id = await orchestrator.start_and_run(
            ResearchRunCreate(session_id=run.session_id, project_id=run.project_id)
        )
        await research_db.commit()
        return {"research_run_id": new_id, "status": "running"}
    except Exception as exc:
        await research_db.rollback()
        logger.exception("Failed to rerun research: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to rerun research.") from exc

