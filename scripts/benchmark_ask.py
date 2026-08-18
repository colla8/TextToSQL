#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone CLI for the benchmark, kept for offline testing when the FastAPI
server/bot isn't running. Builds its own pipeline (so it needs Postgres +
Ollama reachable, same as scripts/ask.py) and calls the exact same
`benchmark.live.run_benchmark()` used by the live API path in api/main.py --
no duplicated logic between the two.

For measuring against the REAL, already-running bot/API (recommended --
reflects actual warm state, real connection pool, real Ollama keep_alive),
send `"benchmark": true` on a normal POST /question request instead; see
benchmark/README.md.

Usage:
    python scripts/benchmark_ask.py "چند تا سفارش امروز ثبت شده؟"
    python scripts/benchmark_ask.py                 # prompts interactively
    python scripts/benchmark_ask.py --runs 5 "..."
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline import TextToSQLPipeline
from benchmark.live import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "question", nargs="*", help="Question text (or omit to be prompted)."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of real (measured) runs. Default: 3.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warm-up runs (excluded). Default: 1.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.05,
        help="Resource sampler interval, seconds.",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to dump raw results as JSON.",
    )
    args = parser.parse_args()

    question = " ".join(args.question).strip() or input("سؤال: ").strip()
    if not question:
        print("No question given.", file=sys.stderr)
        sys.exit(1)

    print("Loading pipeline (embedding model + FAISS index)...", file=sys.stderr)
    t0 = time.perf_counter()
    pipeline = TextToSQLPipeline()
    init_time_s = time.perf_counter() - t0
    print(f"  loaded in {init_time_s:.3f}s", file=sys.stderr)

    _, report_text, saved_path = run_benchmark(
        pipeline,
        question,
        runs=args.runs,
        warmup=args.warmup,
        interval=args.interval,
        init_time_s=init_time_s,
        log=lambda msg: print(f"  {msg}", file=sys.stderr),
        save_json=True,
        json_path=Path(args.json) if args.json else None,
    )

    print()
    print(report_text)
    if saved_path:
        print(f"\nRaw results saved to: {saved_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
