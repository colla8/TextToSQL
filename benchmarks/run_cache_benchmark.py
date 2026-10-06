#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    python -m benchmarks.run_cache_benchmark

Purpose-built harness for measuring the SQL/retrieval/result caches
(app/cache/query_cache.py) in isolation from general pipeline performance.

benchmarks/run_benchmark.py's --repeats loop is NOT a clean cache A/B: its
whole point is averaging out inference noise across several *uncached*
runs of the same question (see its own docstring), so it defaults caching
off, and even with --with-cache passed, its median/p95 mixes one cold pass
in with several warm ones -- exactly the confusion this tool exists to
avoid.

For every question in benchmarks/cache_questions.py, this runs, in order,
against a cache that starts empty for that question:

  1. cold                -- the canonical question text. Every stage
                             should run: sql_cache=miss,
                             retrieval_cache=miss, result_cache=miss.
  2. warm_repeat          -- the exact same text again. Should be
                             sql_cache=hit (retrieval + generation +
                             validation + execution all skipped).
  3. warm_paraphrase      -- one pass per configured paraphrase: surface
                             variations only (whitespace, punctuation,
                             Arabic/Persian letterforms and digits -- see
                             app/cache/keys.py's normalize_question()).
                             Should ALSO be sql_cache=hit, which is the
                             real thing being tested here: that
                             normalization -- not just byte-identical
                             text -- is what drives cache hits.
  4. warm_retrieval_only  -- canonical text again, but with only this
                             question's *sql* cache entry deleted first
                             (retrieval + result entries from the cold
                             pass are left in place). app/pipeline.py only
                             ever consults the retrieval cache on a
                             sql-cache miss, so passes 1-3 never exercise
                             it in isolation -- a full sql_cache hit
                             shortcuts straight past the retrieval check.
                             Expected: sql_cache=miss,
                             retrieval_cache=hit. The SQL-generation LLM
                             call still runs (only embed+FAISS is
                             skipped), so this isolates the retrieval
                             cache's own contribution, not the sql cache's.
  5. warm_result_only     -- canonical text again, with both this
                             question's sql *and* retrieval cache entries
                             deleted (result entry from the cold pass is
                             left in place). Forces full retrieval +
                             SQL-generation, then checks whether the
                             (presumably identical) generated SQL still
                             hits the result cache by its own hash.
                             Expected: sql_cache=miss,
                             retrieval_cache=miss, result_cache=hit. If
                             SQL generation isn't deterministic this can
                             legitimately miss even when nothing is
                             broken -- see the [!] note it prints.

Writes one JSONL row per pass, then prints a cold-vs-warm summary (median
total wall time per pass type, with the speedup vs. cold) and flags any
pass whose cache_info didn't match what that pass type expects -- for
passes 1-3 that's a normalization bug worth looking at in
app/cache/keys.py; for passes 4-5 it can also mean SQL generation isn't
reproducing byte-identical SQL for the same question, which is worth
knowing but isn't a cache bug by itself.

Each row's "stages" field is the same full per-stage record shape (wall_s,
resources, ollama_calls, meta) that run_benchmark.py writes, keyed by
"pass_type" here instead of "repeat_idx" -- benchmarks/aggregate.py's
grouping functions don't care which of those two fields is present (they
group by question_id/stage and read repeat_idx with a default), so
report_cache_benchmark.py calls straight into aggregate.py for LLM
performance and resource usage instead of re-deriving them, and only
keeps its own cold-vs-warm comparison logic, which aggregate.py has no
equivalent for.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from app.cache.keys import question_cache_key
from app.cache.query_cache import NS_RETRIEVAL, NS_SQL, QueryCache
from benchmarks.cache_questions import CACHE_QUESTIONS
from benchmarks.instrumented_pipeline import InstrumentedPipeline
from benchmarks.resource_monitor import ResourceSampler

# Cache dimensions app/pipeline.py reports in result["cache"], in the order
# they're consulted per request.
CACHE_DIMENSIONS = ["sql_cache", "retrieval_cache", "result_cache"]


def _invalidate_sql_only(cache: QueryCache, question: str) -> None:
    """Deletes just this question's sql-cache entry, leaving its retrieval
    and result entries (from an earlier cold pass) in place -- forces the
    pipeline to fall through to the retrieval-cache check instead of
    short-circuiting on a sql_cache hit."""
    cache.store.delete(NS_SQL, question_cache_key(question))


def _invalidate_sql_and_retrieval(cache: QueryCache, question: str) -> None:
    """Deletes this question's sql-cache AND retrieval-cache entries,
    leaving only its result-cache entry in place -- forces the pipeline
    through embed+FAISS and SQL generation again, so a result_cache hit
    afterward reflects that layer alone, not a shortcut through the other
    two."""
    key = question_cache_key(question)
    cache.store.delete(NS_SQL, key)
    cache.store.delete(NS_RETRIEVAL, key)

# Isolated from the production cache (setting.CACHE_DB_PATH) and from
# run_benchmark.py's --with-cache runs on purpose -- this tool calls
# invalidate_all() liberally to guarantee cold starts, which would be
# destructive if pointed at the bot's real cache.
CACHE_DB_PATH = Path(__file__).resolve().parent / "results" / "cache_benchmark_state.db"


def run_pass(
    pipeline,
    question_text,
    pass_type,
    question_id,
    category,
    expectations: dict[str, str] | None = None,
):
    """expectations maps a subset of CACHE_DIMENSIONS to the value
    ('hit'/'miss') that pass type should produce -- e.g.
    {"sql_cache": "miss", "retrieval_cache": "hit"}. Dimensions not
    mentioned aren't asserted (some, like result_cache on a
    warm_retrieval_only pass, depend on SQL-generation determinism and
    aren't a cache bug either way -- see module docstring)."""
    import time

    expectations = expectations or {}
    t0 = time.time()
    try:
        result, stage_records = pipeline.ask(question_text)
    except Exception as e:
        return {
            "question_id": question_id,
            "category": category,
            "pass_type": pass_type,
            "question_text": question_text,
            "expectations": expectations,
            "outcome": "exception",
            "error_detail": str(e),
            "cache": {},
            "total_wall_s": time.time() - t0,
            "stages": [],
        }
    total_wall = time.time() - t0

    outcome = "not_found" if result.get("not_found") else "error" if result.get("error") else "ok"
    return {
        "question_id": question_id,
        "category": category,
        "pass_type": pass_type,
        "question_text": question_text,
        "expectations": expectations,
        "outcome": outcome,
        "error_detail": result.get("error") or result.get("reason"),
        "cache": result.get("cache", {}),
        "total_wall_s": total_wall,
        # Full per-stage records (wall_s + resources + ollama_calls + meta)
        # -- same shape run_benchmark.py writes (see aggregate.py's
        # StageRecord expectations) instead of just a stage->wall_s dict,
        # so report_cache_benchmark.py can reuse aggregate.py's LLM-
        # performance and resource-usage aggregation instead of throwing
        # that data away. Stages a cache hit skipped just aren't in this
        # list -- same "missing, not a fast zero" convention as before.
        "stages": [asdict(sr) for sr in stage_records],
    }


def _check_expectations(row: dict) -> list[str]:
    """Returns a list of human-readable mismatch descriptions, empty if
    every expectation on this row's pass_type was met."""
    problems = []
    for dim, expected in row.get("expectations", {}).items():
        actual = row["cache"].get(dim)
        if actual != expected:
            problems.append(f"{dim}: expected {expected}, got {actual}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default=None, help="Output JSONL path.")
    ap.add_argument("--sample-interval", type=float, default=0.1)
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = (
        Path(args.out)
        if args.out
        else out_dir / f"cache_run_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    )

    print("[*] در حال بارگذاری پایپ‌لاین...")
    sampler = ResourceSampler(interval=args.sample_interval)
    sampler.start()

    cache = QueryCache(db_path=CACHE_DB_PATH, enabled=True)
    cache.invalidate_all()
    pipeline = InstrumentedPipeline(sampler, cache=cache)

    rows: list[dict] = []
    mismatches: list[dict] = []

    for q in CACHE_QUESTIONS:
        print(f"\n[*] {q['id']} ({q['category']})")

        # isolate this question's cold pass from anything a previous
        # question left behind
        cache.invalidate_all()

        row = run_pass(pipeline, q["canonical"], "cold", q["id"], q["category"])
        rows.append(row)
        print(
            f"    cold                  : {row['total_wall_s']:6.2f}s  "
            f"sql={row['cache'].get('sql_cache')} "
            f"retrieval={row['cache'].get('retrieval_cache')} "
            f"result={row['cache'].get('result_cache')}"
        )

        row = run_pass(
            pipeline,
            q["canonical"],
            "warm_repeat",
            q["id"],
            q["category"],
            expectations={"sql_cache": "hit"},
        )
        rows.append(row)
        problems = _check_expectations(row)
        flag = "" if not problems else f"  [!] {'; '.join(problems)}"
        print(
            f"    warm_repeat           : {row['total_wall_s']:6.2f}s  "
            f"sql_cache={row['cache'].get('sql_cache')}{flag}"
        )
        if problems:
            mismatches.append(row)

        for i, para in enumerate(q.get("paraphrases", [])):
            row = run_pass(
                pipeline,
                para,
                "warm_paraphrase",
                q["id"],
                q["category"],
                expectations={"sql_cache": "hit"},
            )
            rows.append(row)
            problems = _check_expectations(row)
            flag = "" if not problems else "  [!] normalization miss"
            print(
                f"    warm_paraphrase[{i}]   : {row['total_wall_s']:6.2f}s  "
                f"sql_cache={row['cache'].get('sql_cache')}{flag}"
            )
            if problems:
                mismatches.append(row)

        # -- retrieval cache in isolation --------------------------------
        # only this question's sql entry is gone; retrieval + result
        # entries from the cold pass above are still there.
        _invalidate_sql_only(cache, q["canonical"])
        row = run_pass(
            pipeline,
            q["canonical"],
            "warm_retrieval_only",
            q["id"],
            q["category"],
            expectations={"sql_cache": "miss", "retrieval_cache": "hit"},
        )
        rows.append(row)
        problems = _check_expectations(row)
        flag = "" if not problems else f"  [!] {'; '.join(problems)}"
        print(
            f"    warm_retrieval_only   : {row['total_wall_s']:6.2f}s  "
            f"sql={row['cache'].get('sql_cache')} "
            f"retrieval={row['cache'].get('retrieval_cache')}{flag}"
        )
        if problems:
            mismatches.append(row)

        # -- result cache in isolation ------------------------------------
        # sql + retrieval entries both gone now; only the result entry
        # from the cold pass survives.
        _invalidate_sql_and_retrieval(cache, q["canonical"])
        row = run_pass(
            pipeline,
            q["canonical"],
            "warm_result_only",
            q["id"],
            q["category"],
            expectations={
                "sql_cache": "miss",
                "retrieval_cache": "miss",
                "result_cache": "hit",
            },
        )
        rows.append(row)
        problems = _check_expectations(row)
        # a result_cache miss here can legitimately mean SQL generation
        # produced non-identical (but still correct) SQL text, not a
        # cache bug -- flagged, but worded differently from the others.
        flag = "" if not problems else f"  [!] {'; '.join(problems)} (may be sql-gen nondeterminism)"
        print(
            f"    warm_result_only      : {row['total_wall_s']:6.2f}s  "
            f"sql={row['cache'].get('sql_cache')} "
            f"retrieval={row['cache'].get('retrieval_cache')} "
            f"result={row['cache'].get('result_cache')}{flag}"
        )
        if problems:
            mismatches.append(row)

    sampler.stop()

    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    print(f"\n[*] نتایج خام: {out_path}")
    _print_summary(rows)

    if mismatches:
        by_pass = {}
        for row in mismatches:
            by_pass.setdefault(row["pass_type"], 0)
            by_pass[row["pass_type"]] += 1
        detail = ", ".join(f"{k}={v}" for k, v in by_pass.items())
        print(
            f"\n[!] {len(mismatches)} مورد ({detail}) انتظارشون درست از آب درنیومد -- "
            "برای cold/warm_repeat/warm_paraphrase یعنی مشکل در "
            "app/cache/keys.py normalize_question() هست؛ برای "
            "warm_retrieval_only/warm_result_only ممکنه نشونه‌ی همون باشه یا "
            "غیرقطعی بودن خروجی SQL generation (توضیح در docstring بالای فایل)."
        )
        sys.exit(1)

    print("\n[*] همه‌ی hit های موردانتظار درست بودن.")


def _print_summary(rows: list[dict]) -> None:
    by_pass: dict[str, list[float]] = {}
    for row in rows:
        by_pass.setdefault(row["pass_type"], []).append(row["total_wall_s"])

    print("\n=== Cold vs Warm (کل زمان هر پاس) ===")
    cold_times = by_pass.get("cold")
    cold_median = statistics.median(cold_times) if cold_times else None

    for pass_type in [
        "cold",
        "warm_repeat",
        "warm_paraphrase",
        "warm_retrieval_only",
        "warm_result_only",
    ]:
        times = by_pass.get(pass_type)
        if not times:
            continue
        med = statistics.median(times)
        speedup = ""
        if cold_median and pass_type != "cold" and med > 0:
            speedup = f"  ({cold_median / med:.1f}x faster than cold)"
        print(
            f"{pass_type:20s} n={len(times):2d}  median={med:6.2f}s  "
            f"min={min(times):6.2f}s  max={max(times):6.2f}s{speedup}"
        )

    print("\n=== Cache Type Hit Rate (warm passes only) ===")
    for dim, pass_types in [
        ("sql_cache", ("warm_repeat", "warm_paraphrase")),
        ("retrieval_cache", ("warm_retrieval_only",)),
        ("result_cache", ("warm_result_only",)),
    ]:
        vals = [
            r["cache"].get(dim)
            for r in rows
            if r["pass_type"] in pass_types and r.get("cache", {}).get(dim) is not None
        ]
        if not vals:
            continue
        hits = sum(1 for v in vals if v == "hit")
        print(f"{dim:16s} {hits}/{len(vals)} hit ({hits / len(vals) * 100:.0f}%)")


if __name__ == "__main__":
    main()