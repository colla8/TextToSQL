# -*- coding: utf-8 -*-
"""
`Stage` is a context manager: wraps one pipeline step, records wall-clock
time, and slices the ResourceSampler's samples down to that exact window.
Optionally attach Ollama's own per-call timing (see
instrumented_ollama_client.py) so an LLM stage's wall time is broken into
load / prompt-eval (prefill) / eval (decode) without needing external
sampling to infer that split.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from benchmarks.resource_monitor import ResourceSampler, summarize_window


@dataclass
class StageRecord:
    name: str
    wall_s: float
    resources: dict
    ollama_calls: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


class Stage:
    def __init__(
        self,
        name: str,
        sampler: ResourceSampler,
        sink: list[StageRecord],
        meta: dict | None = None,
    ):
        self.name = name
        self.sampler = sampler
        self.sink = sink
        self.meta = meta or {}
        self.ollama_calls: list[dict] = []
        self._t0 = None

    def record_ollama_call(self, timing: dict):
        """Call site (instrumented_ollama_client) pushes each raw Ollama
        timing dict here so multi-call stages (e.g. SQL generation with a
        repair attempt) keep all calls, not just the last."""
        self.ollama_calls.append(timing)

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        t1 = time.time()
        window = self.sampler.window(self._t0, t1)
        self.sink.append(
            StageRecord(
                name=self.name,
                wall_s=t1 - self._t0,
                resources=summarize_window(window),
                ollama_calls=self.ollama_calls,
                meta=self.meta,
            )
        )
        return False  # never swallow exceptions
