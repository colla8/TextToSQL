# -*- coding: utf-8 -*-
"""
Minimal client for a local Ollama server (setting.OLLAMA_BASE_URL),
using the /api/generate endpoint with stream=False for simplicity.
"""

import requests

from app.core.config import setting


class OllamaClient:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        think: bool | None = None,
    ):
        self.model = model or setting.OLLAMA_MODEL
        self.base_url = (base_url or setting.OLLAMA_BASE_URL).rstrip("/")
        self.temperature = setting.TEMPERATURE if temperature is None else temperature
        self.max_tokens = max_tokens or setting.MAX_TOKENS
        self.timeout = timeout or setting.OLLAMA_TIMEOUT_SECONDS
        self.think = think

    def generate(
        self, prompt: str, system: str | None = None, max_tokens: int | None = None
    ) -> str:
        # max_tokens here overrides self.max_tokens for this one call only
        # (e.g. AnswerGenerator sizing num_predict to how many rows it's
        # summarizing) -- it never mutates the client's own default.
        num_predict = self.max_tokens if max_tokens is None else max_tokens

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            # keep_alive: keeps qwen3 resident in memory between requests
            # so it isn't reloaded (real seconds lost) after every idle gap.
            "keep_alive": setting.OLLAMA_KEEP_ALIVE,
            "options": {
                "temperature": self.temperature,
                "num_predict": num_predict,
                # num_ctx: default Ollama context reserves far more
                # KV-cache VRAM than this prompt needs -- shrinking it
                # frees VRAM for more model layers on the GPU.
                "num_ctx": setting.NUM_CTX,
            },
        }

        # num_gpu: only sent if explicitly configured (see config.py) --
        # forcing this too high crashes Ollama with a CUDA OOM instead of
        # just falling back to CPU, so -1 (the default) leaves it to
        # Ollama's own placement instead of guessing.
        if setting.NUM_GPU >= 0:
            payload["options"]["num_gpu"] = setting.NUM_GPU

        if system:
            payload["system"] = system

        if self.think is not None:
            payload["think"] = self.think

        # timeout=None -> wait indefinitely. On this hardware a single
        # generation can legitimately take several minutes; cutting the
        # socket off early just turns slow-but-working into a hard error.
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout if self.timeout else None,
        )
        if not resp.ok:
            # requests' default HTTPError message doesn't include the
            # response body, which for Ollama is where the actual reason
            # lives (e.g. "CUDA error: out of memory" when num_gpu forces
            # more layers onto the GPU than fit). Surface it so failures
            # are diagnosable from the FastAPI log alone.
            raise requests.exceptions.HTTPError(
                f"{resp.status_code} از Ollama: {resp.text[:2000]}",
                response=resp,
            )

        return resp.json()["response"]
