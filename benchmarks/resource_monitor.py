# -*- coding: utf-8 -*-
"""
Background resource sampler.

Why this exists: the pipeline's actual cost (LLM inference, embedding,
Postgres) does NOT happen inside the Python process that calls
pipeline.ask(). It happens in `ollama` (a separate server process, possibly
partly on GPU) and `postgres`. Sampling psutil.Process() on the caller only
tells you the caller is idle while it waits on a socket -- useless.

This sampler finds the ollama / postgres processes by name and polls their
CPU% and RSS on a fixed interval from a background thread, plus GPU
utilization and VRAM via `nvidia-smi` (no extra dependency -- shells out,
same as `ollama ps` already does informally). Samples are timestamped so a
Stage (see stage_timer.py) can slice out exactly the window it covers.

Caveats, said plainly:
- psutil's cpu_percent() is an *average over the interval since the last
  call* for that process, not instantaneous. Interval is set by how often
  the background thread loops (default 0.1s) -- fine-grained enough for
  LLM calls that take seconds, not for sub-100ms stages like sqlglot
  validation (those will show 0.0 CPU because no sample lands inside the
  window; that's expected, not a bug, since a wall-clock timer already
  covers stages that fast).
- nvidia-smi is polled via subprocess each tick. That has non-trivial
  overhead (tens of ms) -- don't drop the interval below ~0.1s or the
  sampler thread itself becomes a confound.
- If nvidia-smi isn't on PATH or there's no GPU, GPU sampling silently
  no-ops (fields come back as None) rather than crashing the benchmark.
  Check the report for `gpu_available: False` before concluding "GPU
  isn't the bottleneck" -- it might just not have been measured.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field

import psutil


def _gpu_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def _sample_gpu() -> dict | None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        # first GPU only -- extend to parse all lines if you have >1 GPU
        util, mem_used, mem_total = out.stdout.strip().splitlines()[0].split(",")
        return {
            "util_pct": float(util.strip()),
            "mem_used_mb": float(mem_used.strip()),
            "mem_total_mb": float(mem_total.strip()),
        }
    except Exception:
        return None


@dataclass
class Sample:
    t: float
    self_cpu_pct: float
    self_rss_mb: float
    proc_cpu_pct: dict[str, float] = field(default_factory=dict)
    proc_rss_mb: dict[str, float] = field(default_factory=dict)
    gpu: dict | None = None


class ResourceSampler:
    def __init__(
        self,
        interval: float = 0.1,
        watch_process_names=("ollama", "llama-server", "postgres"),
    ):
        # "ollama" (ollama serve) is just the thin API wrapper -- on a snap
        # install it's typically near-idle (0% CPU) while a separate child
        # process, "llama-server" (the actual llama.cpp inference backend
        # ollama spawns per loaded model), does 100% of the CPU/GPU work.
        # Matching "ollama" alone silently records zero for the stage that
        # matters most. Verify with `ps aux | grep -E "ollama|llama-server"`
        # on your machine and adjust this tuple if your build's process
        # name differs (e.g. "ollama_llama_server" on some builds).
        self.interval = interval
        self.watch_process_names = watch_process_names
        self.samples: list[Sample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._self_proc = psutil.Process()
        self.gpu_available = _gpu_available()
        self.permission_warnings: set[str] = set()

    def _find_named_processes(self) -> dict[str, psutil.Process]:
        found: dict[str, psutil.Process] = {}
        for p in psutil.process_iter(["pid", "name"]):
            name = (p.info.get("name") or "").lower()
            for watch in self.watch_process_names:
                if watch in name and watch not in found:
                    try:
                        proc = psutil.Process(p.info["pid"])
                        # snap-installed ollama runs its worker as root;
                        # if the benchmark runs as a normal user, prime
                        # access here so a permission problem surfaces
                        # once at startup (self.permission_warnings) instead
                        # of silently zeroing every sample for the rest of
                        # the run.
                        proc.cpu_percent()
                        proc.memory_info()
                        found[watch] = proc
                    except psutil.NoSuchProcess:
                        pass
                    except (psutil.AccessDenied, PermissionError):
                        self.permission_warnings.add(watch)
        return found

    def _loop(self):
        named = self._find_named_processes()
        # prime cpu_percent() -- first call always returns 0.0 / garbage
        self._self_proc.cpu_percent()
        for p in named.values():
            try:
                p.cpu_percent()
            except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
                pass

        while not self._stop.is_set():
            time.sleep(self.interval)
            t = time.time()

            try:
                self_cpu = self._self_proc.cpu_percent()
                self_rss = self._self_proc.memory_info().rss / 1e6
            except psutil.NoSuchProcess:
                self_cpu, self_rss = 0.0, 0.0

            proc_cpu, proc_rss = {}, {}
            for name in list(named.keys()):
                p = named[name]
                try:
                    proc_cpu[name] = p.cpu_percent()
                    proc_rss[name] = p.memory_info().rss / 1e6
                except psutil.NoSuchProcess:
                    del named[name]
                except (psutil.AccessDenied, PermissionError):
                    self.permission_warnings.add(name)
                    del named[name]
            # a watched process that wasn't running yet at start() may have
            # started since (e.g. ollama lazily spawning a runner) -- re-scan
            # occasionally so we don't miss it for the rest of the run
            if len(named) < len(self.watch_process_names):
                named.update(self._find_named_processes())

            gpu = _sample_gpu() if self.gpu_available else None

            self.samples.append(
                Sample(
                    t=t,
                    self_cpu_pct=self_cpu,
                    self_rss_mb=self_rss,
                    proc_cpu_pct=proc_cpu,
                    proc_rss_mb=proc_rss,
                    gpu=gpu,
                )
            )

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        # let it take a couple of samples before anything starts timing
        time.sleep(self.interval * 2)
        if self.permission_warnings:
            names = ", ".join(sorted(self.permission_warnings))
            print(
                f"[!] دسترسی به پردازش(های) «{names}» نداری (احتمالاً root روی snap) -- "
                f"CPU/RAM آن‌ها ثبت نمی‌شود. با sudo اجرا کن یا کاربر ollama/root را برای "
                f"این پردازش‌ها استفاده کن اگر این داده لازم است."
            )

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def window(self, t0: float, t1: float) -> list[Sample]:
        return [s for s in self.samples if t0 <= s.t <= t1]


def summarize_window(samples: list[Sample]) -> dict:
    """Reduces a list of Sample into peak/avg stats. Empty window (stage
    faster than the sampling interval) returns None fields, not zeros --
    zero would falsely claim "measured, and it was idle"."""
    if not samples:
        return {
            "n_samples": 0,
            "self_cpu_avg": None,
            "self_cpu_max": None,
            "self_rss_max_mb": None,
            "proc_cpu_avg": {},
            "proc_cpu_max": {},
            "proc_rss_max_mb": {},
            "gpu_util_avg": None,
            "gpu_util_max": None,
            "gpu_mem_max_mb": None,
        }

    proc_names = set()
    for s in samples:
        proc_names |= set(s.proc_cpu_pct.keys())

    proc_cpu_avg = {
        n: _avg([s.proc_cpu_pct.get(n, 0.0) for s in samples]) for n in proc_names
    }
    proc_cpu_max = {
        n: max(s.proc_cpu_pct.get(n, 0.0) for s in samples) for n in proc_names
    }
    proc_rss_max = {
        n: max(s.proc_rss_mb.get(n, 0.0) for s in samples) for n in proc_names
    }

    gpu_samples = [s.gpu for s in samples if s.gpu is not None]
    gpu_util_avg = _avg([g["util_pct"] for g in gpu_samples]) if gpu_samples else None
    gpu_util_max = max((g["util_pct"] for g in gpu_samples), default=None)
    gpu_mem_max = max((g["mem_used_mb"] for g in gpu_samples), default=None)

    return {
        "n_samples": len(samples),
        "self_cpu_avg": _avg([s.self_cpu_pct for s in samples]),
        "self_cpu_max": max(s.self_cpu_pct for s in samples),
        "self_rss_max_mb": max(s.self_rss_mb for s in samples),
        "proc_cpu_avg": proc_cpu_avg,
        "proc_cpu_max": proc_cpu_max,
        "proc_rss_max_mb": proc_rss_max,
        "gpu_util_avg": gpu_util_avg,
        "gpu_util_max": gpu_util_max,
        "gpu_mem_max_mb": gpu_mem_max,
    }


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0
