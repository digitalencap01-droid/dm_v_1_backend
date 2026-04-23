from __future__ import annotations

import os
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def _database_url() -> str:
    """
    Return the async SQLAlchemy database URL.

    Defaults to a local SQLite file for demo/dev.
    """
    return os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./local_profile_engine.db")


engine: AsyncEngine = create_async_engine(
    _database_url(),
    echo=False,
    future=True,
)

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """
    FastAPI dependency that yields an AsyncSession.

    Route handlers own commit/rollback; this dependency guarantees rollback
    on unexpected exceptions and always closes the session.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
