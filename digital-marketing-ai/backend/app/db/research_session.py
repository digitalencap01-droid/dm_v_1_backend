from __future__ import annotations

import os
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def _research_database_url() -> str:
    """
    Return async SQLAlchemy database URL for the research engine.

    Defaults to a local SQLite file.
    """
    return os.getenv(
        "RESEARCH_DATABASE_URL",
        "sqlite+aiosqlite:///./local_research_engine.db",
    )


research_engine: AsyncEngine = create_async_engine(
    _research_database_url(),
    echo=False,
    future=True,
)

ResearchAsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=research_engine,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_research_db_session() -> AsyncIterator[AsyncSession]:
    async with ResearchAsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise

