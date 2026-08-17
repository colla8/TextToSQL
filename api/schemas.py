# -*- coding: utf-8 -*-
import datetime as dt

from pydantic import BaseModel, Field, field_validator


class QuestionRequest(BaseModel):
    telegram_id: int
    question: str = Field(..., min_length=1)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("سؤال نمی‌تواند خالی باشد.")
        return v


class QuestionResponse(BaseModel):
    answer: str
    sql: str | None = None


class HistoryItem(BaseModel):
    id: int
    question: str
    status: str
    answer: str | None
    created_at: dt.datetime

    model_config = {"from_attributes": True}


class HistoryResponse(BaseModel):
    telegram_id: int
    items: list[HistoryItem]


class HealthResponse(BaseModel):
    status: str
    ollama: bool
    database: bool


class ErrorResponse(BaseModel):
    detail: str
