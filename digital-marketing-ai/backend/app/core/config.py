from __future__ import annotations

import os

from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_DIR / ".env")

def env(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


# ---------------------------------------------------------------------------
# Profile engine DB (existing)
# ---------------------------------------------------------------------------

DATABASE_URL = env("DATABASE_URL", "sqlite+aiosqlite:///./local_profile_engine.db")


# ---------------------------------------------------------------------------
# Research engine (Phase 3)
# ---------------------------------------------------------------------------

RESEARCH_DATABASE_URL = env(
    "RESEARCH_DATABASE_URL",
    "sqlite+aiosqlite:///./local_research_engine.db",
)

RESEARCH_MODE = env("RESEARCH_MODE", "live")  # live|disabled
WEB_RESEARCH_PROVIDER = env("WEB_RESEARCH_PROVIDER", "you")  # you
LLM_PROVIDER = env("LLM_PROVIDER", "cerebras")  # cerebras|openai (via LLMClient)

YOU_API_KEY = env("YOU_API_KEY")

MAX_SEARCHES_PER_CATEGORY = int(env("MAX_SEARCHES_PER_CATEGORY", "3") or 3)
MAX_SOURCES_PER_CATEGORY = int(env("MAX_SOURCES_PER_CATEGORY", "8") or 8)
MAX_TOTAL_SOURCES_PER_RUN = int(env("MAX_TOTAL_SOURCES_PER_RUN", "30") or 30)
RESEARCH_TIMEOUT_SECONDS = float(env("RESEARCH_TIMEOUT_SECONDS", "30") or 30)
MAX_ANALYSIS_TOKENS = int(env("MAX_ANALYSIS_TOKENS", "3000") or 3000)

