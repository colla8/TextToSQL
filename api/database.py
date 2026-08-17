# -*- coding: utf-8 -*-
"""
Lightweight persistence for the API itself (not the business data in
Postgres). Every incoming question is written here first as "pending",
then updated to "done"/"error" once the pipeline finishes.

Why this exists:
  - Multiple Telegram users can message at the same time. Without any
    server-side record of "who has a question in flight right now", two
    fast messages from the same user (or a client retry) would both kick
    off a full retrieval+LLM+SQL pipeline concurrently, and it's easy to
    lose track of which reply belongs to which request.
  - This gives every request a durable row and a status, so /question can
    reject a second request while one is still pending for that user
    (see api/main.py), and /history/{telegram_id} can show past Q&A even
    after a restart.

SQLite (via aiosqlite) is used because it needs zero extra infrastructure
to deploy - it's a single file under data/. WAL mode is enabled so reads
and writes don't block each other under concurrent access. If usage grows
enough that SQLite becomes a bottleneck, APP_DB_URL can be pointed at
Postgres instead (SQLAlchemy's async API doesn't change).
"""

import datetime as dt
from pathlib import Path
from typing import AsyncIterator

from sqlalchemy import String, Text, BigInteger, DateTime, Integer, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import NullPool

from app.core.config import setting

# Make sure the sqlite file's parent directory exists before the engine
# tries to open it (SQLite will not create missing directories itself).
if setting.APP_DB_URL.startswith("sqlite"):
    _sqlite_path = setting.APP_DB_URL.split("///", 1)[-1]
    Path(_sqlite_path).parent.mkdir(parents=True, exist_ok=True)

STATUS_PENDING = "pending"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_NOT_FOUND = "not_found"

_is_sqlite = setting.APP_DB_URL.startswith("sqlite")

_engine = create_async_engine(
    setting.APP_DB_URL,
    echo=False,
    poolclass=NullPool if _is_sqlite else None,
)
_SessionLocal = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    telegram_id: Mapped[int] = mapped_column(BigInteger, index=True, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), default=STATUS_PENDING, nullable=False
    )
    sql: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    completed_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


async def init_db() -> None:
    if _is_sqlite:
        async with _engine.begin() as conn:
            await conn.exec_driver_sql("PRAGMA journal_mode=WAL")
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def reset_stale_pending() -> int:
    """Any row still 'pending' when the server starts up belongs to a
    request that was cut off mid-flight by a crash/restart/shutdown --
    the process that would have marked it done/error no longer exists.
    Left as-is, has_pending_question() would block that user's future
    questions forever (see api/main.py). Called once from lifespan()
    right after init_db(), before any request can come in.
    """
    async with _SessionLocal() as session:
        result = await session.execute(
            select(Question).where(Question.status == STATUS_PENDING)
        )
        rows = list(result.scalars().all())
        for row in rows:
            row.status = STATUS_ERROR
            row.error = "درخواست به دلیل راه‌اندازی مجدد سرویس، ناتمام ماند."
            row.completed_at = dt.datetime.now(dt.timezone.utc)
        if rows:
            await session.commit()
        return len(rows)


async def dispose_db() -> None:
    await _engine.dispose()


async def get_session() -> AsyncIterator[AsyncSession]:
    async with _SessionLocal() as session:
        yield session


async def has_pending_question(session: AsyncSession, telegram_id: int) -> bool:
    result = await session.execute(
        select(Question.id).where(
            Question.telegram_id == telegram_id,
            Question.status == STATUS_PENDING,
        )
    )
    return result.first() is not None


async def create_question(
    session: AsyncSession, telegram_id: int, question: str
) -> Question:
    row = Question(telegram_id=telegram_id, question=question, status=STATUS_PENDING)
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row


async def mark_done(session: AsyncSession, row_id: int, sql: str, answer: str) -> None:
    row = await session.get(Question, row_id)
    if row is None:
        return
    row.status = STATUS_DONE
    row.sql = sql
    row.answer = answer
    row.completed_at = dt.datetime.now(dt.timezone.utc)
    await session.commit()


async def mark_not_found(session: AsyncSession, row_id: int, reason: str) -> None:
    row = await session.get(Question, row_id)
    if row is None:
        return
    row.status = STATUS_NOT_FOUND
    row.answer = reason
    row.completed_at = dt.datetime.now(dt.timezone.utc)
    await session.commit()


async def mark_error(
    session: AsyncSession, row_id: int, error: str, sql: str | None = None
) -> None:
    row = await session.get(Question, row_id)
    if row is None:
        return
    row.status = STATUS_ERROR
    row.error = error
    row.sql = sql
    row.completed_at = dt.datetime.now(dt.timezone.utc)
    await session.commit()


async def get_history(
    session: AsyncSession, telegram_id: int, limit: int = 20
) -> list[Question]:
    result = await session.execute(
        select(Question)
        .where(Question.telegram_id == telegram_id)
        .order_by(Question.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())
