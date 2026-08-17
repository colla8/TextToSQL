# -*- coding: utf-8 -*-
"""
FastAPI entrypoint for the Text-to-SQL service.

Run locally:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Run in production (see README.md / Dockerfile for the full setup):
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
"""

import asyncio
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import setting
from app.db import executor as db_executor
from app.pipeline import TextToSQLPipeline

from api import database
from api.deps import configure_logging, get_pipeline, set_pipeline, logger
from api.schemas import (
    ErrorResponse,
    HealthResponse,
    HistoryItem,
    HistoryResponse,
    QuestionRequest,
    QuestionResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(setting.LOG_LEVEL)
    logger.info("در حال راه‌اندازی سرویس Text-to-SQL...")

    await database.init_db()
    logger.info("پایگاه‌داده وضعیت درخواست‌ها آماده شد.")

    reset_count = await database.reset_stale_pending()
    if reset_count:
        logger.info(
            "%d درخواست ناتمام از اجرای قبلی پاک‌سازی شد (pending -> error).",
            reset_count,
        )

    # Loads the embedding model + FAISS index exactly once for the whole
    # process lifetime - this is the expensive part, so it must not run
    # per-request.
    pipeline = await asyncio.to_thread(TextToSQLPipeline)
    set_pipeline(pipeline)
    logger.info("پایپ‌لاین Text-to-SQL آماده شد.")

    yield

    logger.info("در حال خاموش کردن سرویس...")
    db_executor.close_pool()
    await database.dispose_db()


app = FastAPI(
    title=setting.PROJECT_NAME,
    description="سرویس پرسش‌وپاسخ زبان طبیعی روی پایگاه‌داده فروشگاه (Text-to-SQL)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=setting.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("خطای پیش‌بینی‌نشده در پردازش درخواست %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(detail="خطای داخلی سرور رخ داد.").model_dump(),
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{setting.OLLAMA_BASE_URL}/api/tags")
            ollama_ok = resp.status_code == 200
    except httpx.HTTPError:
        ollama_ok = False

    db_ok = True
    try:
        async for session in database.get_session():
            await session.execute(select(database.Question.id).limit(1))
            break
    except Exception:
        db_ok = False

    overall = "ok" if (ollama_ok and db_ok) else "degraded"
    return HealthResponse(status=overall, ollama=ollama_ok, database=db_ok)


@app.post(
    "/question",
    response_model=QuestionResponse,
    responses={
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["qa"],
)
async def question(
    request: QuestionRequest,
    session: AsyncSession = Depends(database.get_session),
):
    if len(request.question) > setting.MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"سؤال طولانی‌تر از {setting.MAX_QUESTION_LENGTH} کاراکتر است.",
        )

    # Guard against the same user firing a second request while their first
    # one is still being processed (double taps, client retries, etc). This
    # is what keeps concurrent users from stepping on each other / piling up
    # duplicate work against the same pipeline.
    if await database.has_pending_question(session, request.telegram_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="سؤال قبلی شما هنوز در حال پردازش است. لطفاً منتظر بمانید.",
        )

    row = await database.create_question(session, request.telegram_id, request.question)
    pipeline = get_pipeline()

    try:
        # pipeline.ask() is fully synchronous (network calls to Ollama,
        # Postgres, FAISS) - running it directly here would block the
        # entire event loop and every other user's request. to_thread hands
        # it to a worker thread instead. No wait_for/timeout here on
        # purpose: on this hardware a single question can legitimately
        # take several minutes, and cutting it off would both waste the
        # work already done and (before the reset_stale_pending fix)
        # leave the row stuck as "pending". If the pipeline finishes,
        # this always returns -- it's not going to hang forever on a
        # network call, since OllamaClient itself still has its own
        # (currently 300s) socket-level timeout as a last-resort guard.
        result = await asyncio.to_thread(pipeline.ask, request.question)
    except Exception as exc:
        logger.exception(
            "خطا در اجرای پایپ‌لاین برای telegram_id=%s", request.telegram_id
        )
        await database.mark_error(session, row.id, str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="خطا در پردازش سؤال.",
        )

    if result.get("not_found"):
        await database.mark_not_found(session, row.id, result["reason"])
        return QuestionResponse(answer=result["reason"], sql=None)

    if result.get("error"):
        await database.mark_error(
            session, row.id, result["error"], sql=result.get("sql")
        )
        return QuestionResponse(answer=result["error"], sql=result.get("sql"))

    await database.mark_done(
        session, row.id, sql=result.get("sql", ""), answer=result["answer"]
    )
    return QuestionResponse(answer=result["answer"], sql=result.get("sql"))


@app.get("/history/{telegram_id}", response_model=HistoryResponse, tags=["qa"])
async def history(
    telegram_id: int,
    limit: int = 20,
    session: AsyncSession = Depends(database.get_session),
):
    rows = await database.get_history(session, telegram_id, limit=limit)
    return HistoryResponse(
        telegram_id=telegram_id,
        items=[HistoryItem.model_validate(r) for r in rows],
    )
