# -*- coding: utf-8 -*-
"""
Usage (from the Text-to-SQL/ project root, same venv as the API):

    python -m benchmarks.run_benchmark --repeats 5

Requires the same live dependencies api/main.py needs: Postgres reachable
(app/core/config.py PG_* settings) and `ollama serve` running with the
configured model already pulled. This does NOT start those for you.

Why repeats > 1 matters: a single run tells you what happened once. LLM
decode length depends on the question (think=True means the chain-of-
thought itself varies in length), and wall-clock is noisy under CPU-heavy
inference. `report.py` needs several samples per question to give you a
median and a p95 that mean anything -- one pass is an anecdote.

Cold start is NOT specially handled here: model load only happens once
regardless (OLLAMA_KEEP_ALIVE keeps qwen3 resident across every question
in the run), so at most the very first Ollama call across the whole
benchmark pays load_duration_ns > 0. Look at that field directly in the
report instead of trying to force repeated cold starts -- deliberately
reloading the model between every question would benchmark something you
don't actually do in production.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from app.cache.query_cache import QueryCache
from benchmarks.instrumented_pipeline import InstrumentedPipeline
from benchmarks.questions import QUESTIONS
from benchmarks.resource_monitor import ResourceSampler


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark the Text-to-SQL pipeline stage by stage."
    )
    ap.add_argument(
        "--repeats", type=int, default=5, help="Repetitions per question (default: 5)."
    )
    ap.add_argument(
        "--sample-interval",
        type=float,
        default=0.1,
        help="Resource sampling interval, seconds.",
    )
    ap.add_argument("--out", type=str, default=None, help="Output JSONL path.")
    ap.add_argument(
        "--only-category",
        type=str,
        default=None,
        help="Only run questions from this category (see questions.py).",
    )
    ap.add_argument(
        "--with-cache",
        action="store_true",
        help=(
            "Enable the SQL/retrieval/result caches for this run. Changes "
            "what's being measured: repeat_idx > 0 (and any repeated "
            "question text within a repeat) will mostly hit cache, so "
            "per-stage medians stop reflecting raw pipeline cost. Use this "
            "to measure cache-hit speedup specifically, not for the normal "
            "stage-by-stage benchmark."
        ),
    )
    args = ap.parse_args()

    questions = QUESTIONS
    if args.only_category:
        questions = [q for q in questions if q["category"] == args.only_category]
        if not questions:
            print(
                f"شناسه دسته‌ی '{args.only_category}' هیچ سوالی نداشت.", file=sys.stderr
            )
            sys.exit(1)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = (
        Path(args.out)
        if args.out
        else out_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    )

    print("[*] بارگذاری pipeline (embedding model + FAISS index)...")
    sampler = ResourceSampler(interval=args.sample_interval)
    sampler.start()
    if not sampler.gpu_available:
        print("[!] nvidia-smi پیدا نشد -- داده‌های GPU در این اجرا ثبت نمی‌شوند.")

    if args.with_cache:
        print("[!] --with-cache enabled: repeats will mostly measure cache hits.")
    pipeline = InstrumentedPipeline(sampler, cache=QueryCache(enabled=args.with_cache))

    total = len(questions) * args.repeats
    done = 0
    t_run_start = time.time()

    with open(out_path, "w", encoding="utf-8") as f:
        for repeat_idx in range(args.repeats):
            for q in questions:
                done += 1
                print(
                    f"[{done}/{total}] repeat={repeat_idx} id={q['id']} ({q['category']})",
                    end=" ... ",
                    flush=True,
                )
                t0 = time.time()
                row = {
                    "question_id": q["id"],
                    "category": q["category"],
                    "expect": q["expect"],
                    "repeat_idx": repeat_idx,
                    "question_text": q["text"],
                }
                try:
                    result, stage_records = pipeline.ask(q["text"])
                    row["outcome"] = (
                        "not_found"
                        if result.get("not_found")
                        else "error" if result.get("error") else "ok"
                    )
                    row["error_detail"] = result.get("error") or result.get("reason")
                    row["cache"] = result.get("cache")
                    row["stages"] = [asdict(sr) for sr in stage_records]
                    row["total_wall_s"] = time.time() - t0
                    print(f"{row['outcome']} ({row['total_wall_s']:.1f}s)")
                except Exception as e:
                    row["outcome"] = "exception"
                    row["error_detail"] = f"{type(e).__name__}: {e}"
                    row["traceback"] = traceback.format_exc()
                    row["stages"] = []
                    row["total_wall_s"] = time.time() - t0
                    print(f"EXCEPTION: {e}")

                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                f.flush()

    sampler.stop()
    print(f"[*] تمام شد در {time.time() - t_run_start:.0f} ثانیه. نتایج: {out_path}")
    print(f"[*] برای گزارش: python -m benchmarks.report {out_path}")


if __name__ == "__main__":
    main()
