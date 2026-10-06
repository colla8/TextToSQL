#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    python -m benchmarks.report_cache_benchmark benchmarks/results/cache_run_20260819_143658.jsonl

Reads the JSONL that benchmarks/run_cache_benchmark.py wrote and prints a
per-question cold-vs-warm report, plus writes <input>.report.txt,
<input>.summary.json, and (if matplotlib is available) <input>.png next
to it -- same convention as benchmarks/report.py.

Covers all three cache dimensions, not just the sql cache: alongside the
per-question cold/warm_repeat/warm_paraphrase table, this reads the
warm_retrieval_only and warm_result_only passes (see
run_cache_benchmark.py's docstring) to report retrieval-cache and
result-cache hit rates in isolation, plus an overall "Cache Type
Breakdown" table across every pass type -- the same hit/miss style as
benchmarks/report.py's "Cache Hit Rate" line, computed the same way
(counts include 'skipped' rows in the denominator, same as
benchmarks/aggregate.py's _cache_counts, so the two reports stay directly
comparable).

Row shape is now the same as run_benchmark.py's ("stages" is a full list
of per-stage records with resources/ollama_calls, not just a wall_s
number), keyed by "pass_type" instead of "repeat_idx" -- so this reuses
benchmarks/aggregate.py directly (load_rows, _cache_counts, and the LLM-
performance / resource-usage aggregation) instead of re-deriving the same
things by hand. Only the cold-vs-warm comparison table below is specific
to this tool, since aggregate.py has no notion of pass types.
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from benchmarks.aggregate import (
    _cache_counts,
    _stage_wall,
    llm_performance_rows,
    load_rows,
    resource_usage_rows,
)

STAGE_NAMES = ["retrieval", "sql_generation", "sql_execution", "answer_generation"]
STAGE_LABEL = {
    "retrieval": "Retrieval",
    "sql_generation": "SQL Gen",
    "sql_execution": "SQL Exec",
    "answer_generation": "Answer Gen",
}


def fmt_s(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}s"


def fmt_num(x, nd=1):
    return "—" if x is None else f"{x:.{nd}f}"


def fmt_mb(x, nd=0):
    return "—" if x is None else f"{x:.{nd}f} MB"


def fmt_pct(x, nd=0):
    return "—" if x is None else f"{x:.{nd}f}%"


def _hit_rate_str(counts: dict[str, int] | None) -> str:
    """'3/5 hit (60%)' -- same convention as benchmarks/report.py: the
    denominator includes 'skipped' rows on purpose, since a dimension that
    a given pass type never even consults (e.g. retrieval_cache on a
    warm_repeat pass, which short-circuits before reaching it) is exactly
    what a low hit rate there should communicate."""
    if not counts:
        return "—"
    total = sum(counts.values())
    hits = counts.get("hit", 0)
    return f"{hits}/{total} hit ({hits / total * 100:.0f}%)"


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render_row(cells):
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    lines = [render_row(headers), render_row(["-" * w for w in widths])]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def build_report(rows: list[dict]) -> dict:
    by_qid: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_qid[r["question_id"]].append(r)

    per_question = []
    for qid, qrows in by_qid.items():
        cold = next((r for r in qrows if r["pass_type"] == "cold"), None)
        warm_repeat = next((r for r in qrows if r["pass_type"] == "warm_repeat"), None)
        paraphrases = [r for r in qrows if r["pass_type"] == "warm_paraphrase"]
        retrieval_only = next(
            (r for r in qrows if r["pass_type"] == "warm_retrieval_only"), None
        )
        result_only = next(
            (r for r in qrows if r["pass_type"] == "warm_result_only"), None
        )

        cold_total = cold["total_wall_s"] if cold else None
        warm_repeat_total = warm_repeat["total_wall_s"] if warm_repeat else None
        para_totals = [r["total_wall_s"] for r in paraphrases]
        para_median = statistics.median(para_totals) if para_totals else None
        retrieval_only_total = retrieval_only["total_wall_s"] if retrieval_only else None
        result_only_total = result_only["total_wall_s"] if result_only else None

        def speedup(warm):
            if cold_total and warm and warm > 0:
                return cold_total / warm
            return None

        para_hits = sum(1 for r in paraphrases if r["cache"].get("sql_cache") == "hit")

        per_question.append(
            {
                "question_id": qid,
                "category": qrows[0].get("category"),
                "question_text": cold["question_text"] if cold else qrows[0]["question_text"],
                "cold_total_s": cold_total,
                "warm_repeat_total_s": warm_repeat_total,
                "warm_repeat_speedup": speedup(warm_repeat_total),
                "warm_repeat_hit": warm_repeat["cache"].get("sql_cache") == "hit" if warm_repeat else None,
                "para_median_s": para_median,
                "para_speedup": speedup(para_median),
                "para_hits": para_hits,
                "para_total": len(paraphrases),
                "retrieval_only_total_s": retrieval_only_total,
                "retrieval_only_speedup": speedup(retrieval_only_total),
                "retrieval_only_hit": (
                    retrieval_only["cache"].get("retrieval_cache") == "hit"
                    if retrieval_only
                    else None
                ),
                "result_only_total_s": result_only_total,
                "result_only_speedup": speedup(result_only_total),
                "result_only_hit": (
                    result_only["cache"].get("result_cache") == "hit"
                    if result_only
                    else None
                ),
                # per-stage wall time, pulled from the full stage records
                # via aggregate.py's own _stage_wall -- reused, not
                # re-derived, so this stays consistent with how
                # benchmarks/report.py reads the same field shape.
                "cold_stage_wall_s": {
                    stage: _stage_wall(cold, stage) for stage in STAGE_NAMES
                }
                if cold
                else {},
                "warm_stage_wall_s": {
                    stage: _stage_wall(warm_repeat, stage) for stage in STAGE_NAMES
                }
                if warm_repeat
                else {},
            }
        )

    all_cold = [q["cold_total_s"] for q in per_question if q["cold_total_s"] is not None]
    all_warm = [
        t
        for q in per_question
        for t in (
            [q["warm_repeat_total_s"]] if q["warm_repeat_total_s"] is not None else []
        )
        + ([q["para_median_s"]] if q["para_median_s"] is not None else [])
    ]
    total_para_hits = sum(q["para_hits"] for q in per_question)
    total_para = sum(q["para_total"] for q in per_question)
    total_repeat_hits = sum(1 for q in per_question if q["warm_repeat_hit"])
    total_repeat = sum(1 for q in per_question if q["warm_repeat_hit"] is not None)

    cache_counts = {
        key: _cache_counts(rows, key)
        for key in ("sql_cache", "retrieval_cache", "result_cache")
    }

    overall = {
        "n_questions": len(per_question),
        "cold_median_s": statistics.median(all_cold) if all_cold else None,
        "warm_median_s": statistics.median(all_warm) if all_warm else None,
        "overall_speedup": (
            statistics.median(all_cold) / statistics.median(all_warm)
            if all_cold and all_warm and statistics.median(all_warm) > 0
            else None
        ),
        "sql_cache_hit_rate": (
            (total_repeat_hits + total_para_hits) / (total_repeat + total_para)
            if (total_repeat + total_para)
            else None
        ),
        "cache_counts": cache_counts,
    }

    return {"overall": overall, "per_question": per_question}


def render_report(rows: list[dict]) -> tuple[str, dict]:
    data = build_report(rows)
    o = data["overall"]
    buf = io.StringIO()

    print("=== Cache Benchmark Summary ===", file=buf)
    print(f"Questions: {o['n_questions']}", file=buf)
    print(f"Cold median (total wall time): {fmt_s(o['cold_median_s'])}", file=buf)
    print(f"Warm median (total wall time): {fmt_s(o['warm_median_s'])}", file=buf)
    if o["overall_speedup"] is not None:
        print(f"Overall speedup: {o['overall_speedup']:.1f}x", file=buf)
    if o["sql_cache_hit_rate"] is not None:
        print(f"SQL cache hit rate (warm passes): {o['sql_cache_hit_rate'] * 100:.0f}%", file=buf)
    print(file=buf)

    cc = o["cache_counts"]
    if any(cc.values()):
        print("=== Cache Type Breakdown (all passes) ===", file=buf)
        print(
            "Denominator includes passes that never consult a given cache "
            "(e.g. retrieval_cache is 'skipped' whenever sql_cache hits) -- "
            "same convention as benchmarks/report.py.",
            file=buf,
        )
        headers = ["Cache", "Hit Rate"]
        trows = [
            ["SQL", _hit_rate_str(cc.get("sql_cache"))],
            ["Retrieval", _hit_rate_str(cc.get("retrieval_cache"))],
            ["Result", _hit_rate_str(cc.get("result_cache"))],
        ]
        print(render_table(headers, trows), file=buf)
        print(file=buf)

    print("=== Per Question -- Cold vs Warm ===", file=buf)
    headers = [
        "Question",
        "Category",
        "Cold",
        "Warm (repeat)",
        "Speedup",
        "Warm (paraphrase, median)",
        "Speedup",
        "Normalization",
    ]
    trows = []
    for q in data["per_question"]:
        trows.append(
            [
                q["question_id"],
                q["category"] or "—",
                fmt_s(q["cold_total_s"]),
                fmt_s(q["warm_repeat_total_s"]),
                f"{q['warm_repeat_speedup']:.1f}x" if q["warm_repeat_speedup"] else "—",
                fmt_s(q["para_median_s"]),
                f"{q['para_speedup']:.1f}x" if q["para_speedup"] else "—",
                f"{q['para_hits']}/{q['para_total']} hit",
            ]
        )
    print(render_table(headers, trows), file=buf)
    print(file=buf)

    print("Question text:", file=buf)
    for q in data["per_question"]:
        print(f"  {q['question_id']}: {q['question_text']}", file=buf)
    print(file=buf)

    has_isolation = any(
        q["retrieval_only_total_s"] is not None or q["result_only_total_s"] is not None
        for q in data["per_question"]
    )
    if has_isolation:
        print("=== Per Question -- Retrieval & Result Cache Isolation ===", file=buf)
        print(
            "warm_retrieval_only: this question's sql-cache entry deleted, "
            "retrieval kept. warm_result_only: sql + retrieval both deleted, "
            "result kept. See run_cache_benchmark.py's docstring.",
            file=buf,
        )
        headers = [
            "Question",
            "Warm (retrieval only)",
            "Speedup",
            "Retrieval hit",
            "Warm (result only)",
            "Speedup",
            "Result hit",
        ]
        trows = []
        for q in data["per_question"]:
            trows.append(
                [
                    q["question_id"],
                    fmt_s(q["retrieval_only_total_s"]),
                    f"{q['retrieval_only_speedup']:.1f}x" if q["retrieval_only_speedup"] else "—",
                    "—" if q["retrieval_only_hit"] is None else ("yes" if q["retrieval_only_hit"] else "no"),
                    fmt_s(q["result_only_total_s"]),
                    f"{q['result_only_speedup']:.1f}x" if q["result_only_speedup"] else "—",
                    "—" if q["result_only_hit"] is None else ("yes" if q["result_only_hit"] else "no"),
                ]
            )
        print(render_table(headers, trows), file=buf)
        print(file=buf)

    print("=== Stage Breakdown -- Cold vs Warm (repeat) ===", file=buf)
    print(
        "A '—' on the warm side means that stage was skipped entirely (a cache hit), not that it ran instantly.",
        file=buf,
    )
    headers = ["Question", "Stage", "Cold", "Warm (repeat)"]
    trows = []
    for q in data["per_question"]:
        for stage in STAGE_NAMES:
            cold_v = q["cold_stage_wall_s"].get(stage)
            warm_v = q["warm_stage_wall_s"].get(stage)
            if cold_v is None and warm_v is None:
                continue
            trows.append(
                [q["question_id"], STAGE_LABEL[stage], fmt_s(cold_v), fmt_s(warm_v)]
            )
    print(render_table(headers, trows), file=buf)
    print(file=buf)

    # LLM performance / resource usage -- reused straight from
    # aggregate.py (same functions benchmarks/report.py calls), computed
    # over the cold passes only: warm passes skip stages entirely on a
    # cache hit, so there's nothing to average there for a stage that
    # never ran, and mixing in the ones that did would understate the
    # real uncached cost.
    cold_rows = [r for r in rows if r["pass_type"] == "cold"]

    print("=== LLM Performance (cold pass) ===", file=buf)
    headers = [
        "Question",
        "Stage",
        "Model",
        "Prompt Tokens",
        "Output Tokens",
        "Tok/s",
        "Generation Time",
        "Load Time",
    ]
    trows = []
    for c in llm_performance_rows(cold_rows):
        trows.append(
            [
                c["question_id"],
                STAGE_LABEL.get(c["stage"], c["stage"]),
                c["model"] or "—",
                fmt_num(c["avg_prompt_tokens"], 0),
                fmt_num(c["avg_output_tokens"], 0),
                fmt_num(c["avg_tokens_per_sec"], 1),
                fmt_s(c["avg_generation_time_s"]),
                fmt_s(c["avg_load_time_s"]),
            ]
        )
    if trows:
        print(render_table(headers, trows), file=buf)
    else:
        print("(no LLM calls recorded)", file=buf)
    print(file=buf)

    print("=== Resource Usage (cold pass) ===", file=buf)
    headers = ["Question", "Stage", "CPU", "GPU %", "GPU Memory", "RAM"]
    trows = []
    for u in resource_usage_rows(cold_rows):
        if not u["measured"]:
            trows.append([u["question_id"], STAGE_LABEL.get(u["stage"], u["stage"]), "—", "—", "—", "—"])
            continue
        trows.append(
            [
                u["question_id"],
                STAGE_LABEL.get(u["stage"], u["stage"]),
                fmt_pct(u["cpu_pct"]),
                fmt_pct(u["gpu_pct"]),
                fmt_mb(u["gpu_mem_mb"]),
                fmt_mb(u["ram_mb"]),
            ]
        )
    print(render_table(headers, trows), file=buf)

    return buf.getvalue(), data


PASS_TYPE_LABEL = {
    "cold": "cold",
    "warm_repeat": "warm\n(repeat)",
    "warm_paraphrase": "warm\n(paraphrase)",
    "warm_retrieval_only": "warm\n(retrieval only)",
    "warm_result_only": "warm\n(result only)",
}
PASS_TYPE_ORDER = list(PASS_TYPE_LABEL)


def _maybe_chart(path: Path, rows: list[dict], skip: bool) -> None:
    """Bar chart of median total wall time per pass type, next to the
    input JSONL -- same convention (matplotlib, Agg backend, value labels
    on each bar, skipped silently if matplotlib isn't installed) as
    benchmarks/report.py's _maybe_chart."""
    if skip:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "[matplotlib نصب نیست -- نمودار رد شد. pip install matplotlib برای فعال‌سازی]"
        )
        return

    by_pass: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_pass[r["pass_type"]].append(r["total_wall_s"])

    pass_types = [p for p in PASS_TYPE_ORDER if p in by_pass]
    if not pass_types:
        return

    medians = [statistics.median(by_pass[p]) for p in pass_types]
    labels = [PASS_TYPE_LABEL[p] for p in pass_types]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, medians)
    ax.set_ylabel("median total wall time (s)")
    ax.set_title(f"Cache benchmark — median wall time per pass — {path.stem}")
    for i, v in enumerate(medians):
        ax.text(i, v, f"{v:.2f}s", ha="center", va="bottom")
    plt.xticks(rotation=0)
    plt.tight_layout()
    out_png = path.with_suffix(".png")
    fig.savefig(out_png, dpi=150)
    print(f"[نمودار ذخیره شد: {out_png}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=str)
    ap.add_argument("--no-chart", action="store_true")
    args = ap.parse_args()

    path = Path(args.path)
    rows = load_rows(path)
    if not rows:
        print("هیچ رکوردی در فایل نبود.", file=sys.stderr)
        sys.exit(1)

    text, data = render_report(rows)
    print(text)

    report_path = path.with_suffix(".report.txt")
    summary_path = path.with_suffix(".summary.json")
    report_path.write_text(text, encoding="utf-8")
    summary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    _maybe_chart(path, rows, skip=args.no_chart)
    print(f"\n[گزارش متنی ذخیره شد: {report_path}]")
    print(f"[خلاصه‌ی JSON ذخیره شد: {summary_path}]")


if __name__ == "__main__":
    main()