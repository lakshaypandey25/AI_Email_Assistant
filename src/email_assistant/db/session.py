"""Async engine / session factory with SQLite hardening (WAL, foreign keys, busy timeout)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import event
from sqlalchemy.engine.url import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from email_assistant.db import models  # noqa: F401  (register tables on Base.metadata)
from email_assistant.db.base import Base


def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()


class Database:
    """Owns the async engine and hands out transactional sessions."""

    def __init__(self, url: str, *, echo: bool = False) -> None:
        kwargs: dict[str, Any] = {}
        parsed = make_url(url)
        is_sqlite = parsed.get_backend_name() == "sqlite"
        if is_sqlite:
            if parsed.database in (None, "", ":memory:"):
                kwargs.update(poolclass=StaticPool, connect_args={"check_same_thread": False})
            else:
                Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
        self.engine: AsyncEngine = create_async_engine(url, echo=echo, **kwargs)
        if is_sqlite:
            event.listen(self.engine.sync_engine, "connect", _sqlite_pragmas)
        self._maker = async_sessionmaker(self.engine, expire_on_commit=False)

    async def create_all(self) -> None:
        """Create tables if they do not exist (use Alembic for real migrations in production)."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Transactional scope: commit on success, rollback on error."""
        async with self._maker() as session:
            try:
                yield session
                await session.commit()
            except BaseException:
                await session.rollback()
                raise

    async def dispose(self) -> None:
        await self.engine.dispose()
