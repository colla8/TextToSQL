# -*- coding: utf-8 -*-
"""
Tests of the FastAPI service (api/main.py) through FastAPI's TestClient,
against a throw-away SQLite request-log (the module-level engine in
api/database.py is swapped for a temp-file one, so the project's real
data/state/app_state.db is never touched). The pipeline itself is faked --
its own behavior is covered by tests/test_pipeline.py.

Run with: python -m unittest tests.test_api -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from api import database
from api.main import app


class FakePipeline:
    def __init__(self, result=None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.questions: list[str] = []

    def ask(self, question: str):
        self.questions.append(question)
        if self.error:
            raise self.error
        return self.result


class TestAPI(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        engine = create_async_engine(
            f"sqlite+aiosqlite:///{Path(tmp.name) / 'app_state.db'}",
            poolclass=NullPool,
        )
        session_local = async_sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )
        for target, value in (
            ("_engine", engine),
            ("_SessionLocal", session_local),
        ):
            patcher = mock.patch.object(database, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_app(self, pipeline: FakePipeline):
        """Starts the app (lifespan included) with `pipeline` injected in
        place of the real TextToSQLPipeline."""
        patcher = mock.patch("api.main.TextToSQLPipeline", return_value=pipeline)
        patcher.start()
        self.addCleanup(patcher.stop)
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client

    def test_question_success_is_logged_and_in_history(self):
        pipeline = FakePipeline({"sql": "SELECT 1", "answer": "پاسخ"})
        client = self.run_app(pipeline)

        resp = client.post("/question", json={"telegram_id": 7, "question": "سلام؟"})

        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json(), {"answer": "پاسخ", "sql": "SELECT 1"})
        self.assertEqual(pipeline.questions, ["سلام؟"])

        hist = client.get("/history/7").json()
        self.assertEqual(len(hist["items"]), 1)
        self.assertEqual(hist["items"][0]["status"], "done")

    def test_not_found_and_pipeline_error_results(self):
        client = self.run_app(FakePipeline({"not_found": True, "reason": "داده نیست"}))
        resp = client.post("/question", json={"telegram_id": 1, "question": "q"})
        self.assertEqual(resp.json(), {"answer": "داده نیست", "sql": None})
        self.assertEqual(client.get("/history/1").json()["items"][0]["status"], "not_found")

    def test_pipeline_exception_returns_500_and_does_not_block_user(self):
        pipeline = FakePipeline(error=RuntimeError("boom"))
        client = self.run_app(pipeline)

        resp = client.post("/question", json={"telegram_id": 2, "question": "q"})
        self.assertEqual(resp.status_code, 500)
        self.assertEqual(client.get("/history/2").json()["items"][0]["status"], "error")

        # row was marked error (not left pending) -> the user can ask again
        pipeline.error = None
        pipeline.result = {"sql": "SELECT 1", "answer": "ok"}
        again = client.post("/question", json={"telegram_id": 2, "question": "q"})
        self.assertEqual(again.status_code, 200)

    def test_too_long_question_is_rejected(self):
        client = self.run_app(FakePipeline({"sql": "x", "answer": "y"}))
        resp = client.post("/question", json={"telegram_id": 3, "question": "ا" * 5000})
        self.assertIn(resp.status_code, (413, 422))

    def test_second_request_while_pending_gets_409(self):
        import asyncio

        client = self.run_app(FakePipeline({"sql": "x", "answer": "y"}))

        async def seed_pending():
            async with database._SessionLocal() as s:
                await database.create_question(s, 9, "در حال پردازش")

        asyncio.run(seed_pending())
        resp = client.post("/question", json={"telegram_id": 9, "question": "دوباره"})
        self.assertEqual(resp.status_code, 409)

    def test_stale_pending_rows_are_reset_on_startup(self):
        import asyncio

        async def prepare():
            await database.init_db()
            async with database._SessionLocal() as s:
                await database.create_question(s, 11, "قدیمی")

        asyncio.run(prepare())
        client = self.run_app(FakePipeline({"sql": "x", "answer": "y"}))
        # startup ran reset_stale_pending(): user 11 is no longer blocked
        resp = client.post("/question", json={"telegram_id": 11, "question": "جدید"})
        self.assertEqual(resp.status_code, 200)

    def test_health_reports_degraded_without_ollama(self):
        client = self.run_app(FakePipeline())
        with mock.patch("api.main.setting.OLLAMA_BASE_URL", "http://127.0.0.1:9"):
            body = client.get("/health").json()
        self.assertEqual(body["status"], "degraded")
        self.assertFalse(body["ollama"])
        self.assertTrue(body["database"])


if __name__ == "__main__":
    unittest.main()
