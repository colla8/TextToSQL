# -*- coding: utf-8 -*-
"""
app/llm/ollama_client.py's generate() returns only resp.json()["response"]
and discards Ollama's own timing breakdown. Rather than patch app/ (this
project's own README states app/ stays behavior-unchanged except two named
bugfixes), this subclass duplicates the request just enough to also keep the
full JSON, then delegates string extraction to the parent's return contract.

Ollama's /api/generate (stream=False) response includes, in nanoseconds:
  total_duration       -- everything, including queueing behind other requests
  load_duration         -- time to load the model into memory (0 if already
                            resident thanks to keep_alive)
  prompt_eval_count      -- tokens in the prompt
  prompt_eval_duration   -- prefill time
  eval_count              -- tokens generated
  eval_duration           -- decode time

tokens/sec for generation = eval_count / (eval_duration / 1e9). That number,
tracked across your benchmark question set, tells you whether "it's slow"
means "the model is thinking a long time" (eval_count is huge -- your
think=True budget of 800 tokens is being used up) vs "the box is slow at
each token" (eval_count is normal but ns/token is high, i.e. CPU offload) vs
"model reload every call" (load_duration > 0 every time -- keep_alive isn't
holding, or repeated OllamaClient() instantiation with different params is
counted as a different model config).
"""

from __future__ import annotations

import requests

from app.core.config import setting
from app.llm.ollama_client import OllamaClient


class InstrumentedOllamaClient(OllamaClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_timing: dict | None = None
        # SQLGenerator can call .generate() more than once (repair loop);
        # last_timing alone would silently lose the first call. Keep all of
        # them, in order, so a stage that wraps multiple calls can slice
        # this list instead of only seeing the final one.
        self.call_history: list[dict] = []

    def generate(
        self, prompt: str, system: str | None = None, max_tokens: int | None = None
    ) -> str:
        # Same per-call override contract as app/llm/ollama_client.py's
        # generate() -- AnswerGenerator.generate() calls this with an
        # explicit max_tokens sized to the row count, so this subclass has
        # to accept it too or the answer_generation stage raises TypeError.
        num_predict = self.max_tokens if max_tokens is None else max_tokens

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": setting.OLLAMA_KEEP_ALIVE,
            "options": {
                "temperature": self.temperature,
                "num_predict": num_predict,
                "num_ctx": setting.NUM_CTX,
            },
        }
        if setting.NUM_GPU >= 0:
            payload["options"]["num_gpu"] = setting.NUM_GPU
        if system:
            payload["system"] = system
        if self.think is not None:
            payload["think"] = self.think

        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout if self.timeout else None,
        )
        if not resp.ok:
            raise requests.exceptions.HTTPError(
                f"{resp.status_code} از Ollama: {resp.text[:2000]}", response=resp
            )

        body = resp.json()
        self.last_timing = {
            "model": self.model,
            "think": self.think,
            "total_duration_ns": body.get("total_duration"),
            "load_duration_ns": body.get("load_duration"),
            "prompt_eval_count": body.get("prompt_eval_count"),
            "prompt_eval_duration_ns": body.get("prompt_eval_duration"),
            "eval_count": body.get("eval_count"),
            "eval_duration_ns": body.get("eval_duration"),
        }
        d = self.last_timing
        if d["eval_count"] and d["eval_duration_ns"]:
            d["tokens_per_sec"] = d["eval_count"] / (d["eval_duration_ns"] / 1e9)
        else:
            d["tokens_per_sec"] = None
        self.call_history.append(d)

        return body["response"]
