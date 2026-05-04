from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

_BACKEND_DIR = Path(__file__).resolve().parents[1]

# Load backend/.env early so other modules that read env at import-time
# (like the LLM client) can see those values.
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(dotenv_path=_BACKEND_DIR / ".env", override=True)
except Exception:
    # python-dotenv is optional; env can be provided by the shell instead.
    pass

from app.api.routes import profile_engine
from app.db.session import engine
from app.services.profile_engine.model import ProfileEngineBase


def _cors_origins() -> list[str]:
    raw = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://localhost:4173,"
        "http://127.0.0.1:3000,http://127.0.0.1:5173,http://127.0.0.1:4173",
    )
    return [o.strip() for o in raw.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Create tables for the profile engine (demo/dev convenience).
    async with engine.begin() as conn:
        await conn.run_sync(ProfileEngineBase.metadata.create_all)
    yield


app = FastAPI(title="Digital Marketing AI Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(profile_engine.router, prefix="/api/v1")

# Avoid noisy browser console 404s.
@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)

# Optional same-origin demo page (served at /demo/)
app.mount(
    "/demo",
    StaticFiles(directory=str(_BACKEND_DIR / "app" / "static"), html=True),
    name="demo",
)
