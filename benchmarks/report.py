# -*- coding: utf-8 -*-
"""
Usage:
    python -m benchmarks.report benchmarks/results/run_20260818_121500.jsonl

Reads the raw JSONL a benchmark run produced (unchanged -- see
run_benchmark.py / stage_timer.py / resource_monitor.py for how it's
measured) and prints a concise, human-readable report:

  === Benchmark Summary ===   overall totals + success rate
  === Per Question ===        one row per question_id, averaged across
                               every repeat_idx that question has
  === LLM Performance ===     one row per (question, LLM-calling stage),
                               averaged across repeats and any repair calls
  === Resource Usage ===      one row per (question, stage), averaged
                               across repeats

None of this changes or re-measures anything -- benchmarks/aggregate.py
just reshapes the rows already sitting in the JSONL file. The full raw
JSONL is left exactly where run_benchmark.py wrote it, for anyone who
wants to dig into a single call's timing or resource samples directly.

Besides the console tables, this also writes two files next to the input
JSONL:
  <run>.summary.json   -- the aggregated data as JSON (for other tooling)
  <run>.report.txt      -- the same tables as plain text

Also writes a stacked bar chart (median stage time per category) next to
the input file if matplotlib is available -- skipped silently otherwise.
"""

from __future__ import annotations

import argparse
import io
import statistics
import sys
from pathlib import Path

from benchmarks.aggregate import (
    STAGE_ORDER,
    build_report,
    load_rows,
)

STAGE_LABEL = {
    "retrieval": "Retrieval",
    "sql_generation": "SQL Gen",
    "sql_execution": "SQL Exec",
    "answer_generation": "Answer Gen",
}


def fmt_s(x, nd=2):
    """Seconds, formatted, '—' for missing."""
    if x is None:
        return "—"
    return f"{x:.{nd}f}s"


def fmt_pct(x, nd=0):
    if x is None:
        return "—"
    return f"{x:.{nd}f}%"


def fmt_num(x, nd=0):
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def fmt_mb(x, nd=0):
    if x is None:
        return "—"
    return f"{x:.{nd}f} MB"


def fmt_bool(x):
    if x is None:
        return "—"
    if x == "mixed":
        return "mixed"
    return "on" if x else "off"


def _hit_rate_str(counts: dict[str, int] | None) -> str:
    """'3/5 hit (60%)' style summary for one cache dimension's counts, or
    '—' if this run never recorded cache info (older JSONL / caching off)."""
    if not counts:
        return "—"
    total = sum(counts.values())
    hits = counts.get("hit", 0)
    return f"{hits}/{total} hit ({hits / total * 100:.0f}%)"


def _print_cache_line(cache_counts: dict, file) -> None:
    """One-liner in the overall summary -- omitted entirely if this run
    has no cache info at all (nothing to be misleadingly silent about)."""
    if not any(cache_counts.values()):
        return
    parts = [
        f"SQL: {_hit_rate_str(cache_counts.get('sql_cache'))}",
        f"Retrieval: {_hit_rate_str(cache_counts.get('retrieval_cache'))}",
        f"Result: {_hit_rate_str(cache_counts.get('result_cache'))}",
    ]
    print(f"Cache Hit Rate -- {' | '.join(parts)}", file=file)


def _retrieval_method_str(counts: dict[str, int] | None) -> str:
    """'4/5 lexical (80%), 1/5 embedding_fallback (20%)' style summary,
    or '—' if this run never recorded a retrieval method (older JSONL,
    or every question was a retrieval-cache hit)."""
    if not counts:
        return "—"
    total = sum(counts.values())
    parts = [
        f"{n}/{total} {method} ({n / total * 100:.0f}%)"
        for method, n in sorted(counts.items(), key=lambda kv: -kv[1])
    ]
    return ", ".join(parts)


def _print_retrieval_method_line(retrieval_method_counts: dict | None, file) -> None:
    """Overall-summary one-liner showing how often the fast lexical stage
    (see app/embeddings/retriever.py) answered vs. how often it had to
    fall back to embedding + FAISS -- omitted if this run has none of
    that info."""
    if not retrieval_method_counts or not any(retrieval_method_counts.values()):
        return
    print(
        f"Retrieval Method -- {_retrieval_method_str(retrieval_method_counts)}",
        file=file,
    )


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Plain markdown table, column widths padded for console readability."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render_row(cells):
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    lines = [render_row(headers), render_row(["-" * w for w in widths])]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def render_report(rows: list[dict]) -> tuple[str, dict]:
    """Builds the full text report (str) plus the raw aggregated data
    (dict, JSON-serializable) used to write benchmark_summary.json."""
    data = build_report(rows)
    o = data["overall"]
    buf = io.StringIO()

    print("=== Benchmark Summary ===", file=buf)
    print(f"Questions: {o['n_questions']}", file=buf)
    print(f"Runs: {o['n_runs']}", file=buf)
    print(f"Success Rate: {fmt_pct((o['success_rate'] or 0) * 100)}", file=buf)
    print(f"Average Total Latency: {fmt_s(o['avg_total_s'])}", file=buf)
    print(
        f"Average SQL Generation Latency: {fmt_s(o['avg_sql_generation_s'])}", file=buf
    )
    print(
        f"Average Answer Generation Latency: {fmt_s(o['avg_answer_generation_s'])}",
        file=buf,
    )
    print(f"Average SQL Execution Latency: {fmt_s(o['avg_sql_execution_s'])}", file=buf)
    print(f"Average Tokens/sec: {fmt_num(o['avg_tokens_per_sec'], 1)}", file=buf)
    print(
        f"Average / Max GPU Memory: {fmt_mb(o['avg_gpu_mem_mb'])} / {fmt_mb(o['max_gpu_mem_mb'])}",
        file=buf,
    )
    if o["avg_ram_mb"] is not None:
        print(
            f"Average / Max RAM: {fmt_mb(o['avg_ram_mb'])} / {fmt_mb(o['max_ram_mb'])}",
            file=buf,
        )
    _print_cache_line(o["cache_counts"], buf)
    _print_retrieval_method_line(o["retrieval_method_counts"], buf)
    print(file=buf)

    print("=== Per Question ===", file=buf)
    headers = [
        "Question",
        "Category",
        "Runs",
        "Avg Total",
        "Min",
        "Max",
        "Retrieval",
        "SQL Gen",
        "SQL Exec",
        "Answer Gen",
        "Outcome",
    ]
    trows = []
    for q in data["per_question"]:
        trows.append(
            [
                q["question_id"],
                q["category"] or "—",
                str(q["n_runs"]),
                fmt_s(q["avg_total_s"]),
                fmt_s(q["min_total_s"]),
                fmt_s(q["max_total_s"]),
                fmt_s(q["avg_retrieval_s"]),
                fmt_s(q["avg_sql_generation_s"]),
                fmt_s(q["avg_sql_execution_s"]),
                fmt_s(q["avg_answer_generation_s"]),
                q["outcome"],
            ]
        )
    print(render_table(headers, trows), file=buf)
    print(file=buf)

    if any(any(q["cache_counts"].values()) for q in data["per_question"]):
        print("=== Cache (per question) ===", file=buf)
        headers = ["Question", "SQL Cache", "Retrieval Cache", "Result Cache"]
        trows = []
        for q in data["per_question"]:
            cc = q["cache_counts"]
            trows.append(
                [
                    q["question_id"],
                    _hit_rate_str(cc.get("sql_cache")),
                    _hit_rate_str(cc.get("retrieval_cache")),
                    _hit_rate_str(cc.get("result_cache")),
                ]
            )
        print(render_table(headers, trows), file=buf)
        print(file=buf)

    if any(q["retrieval_method_counts"] for q in data["per_question"]):
        print("=== Retrieval Method (per question) ===", file=buf)
        headers = ["Question", "Method Split", "Avg Confidence"]
        trows = []
        for q in data["per_question"]:
            trows.append(
                [
                    q["question_id"],
                    _retrieval_method_str(q["retrieval_method_counts"]),
                    fmt_num(q["avg_retrieval_confidence"], 3),
                ]
            )
        print(render_table(headers, trows), file=buf)
        print(file=buf)

    # question_text lookup, printed once below the table instead of inside
    # it -- Persian question text next to 10+ latency columns makes the
    # table unreadable in a terminal.
    print("Question text:", file=buf)
    for q in data["per_question"]:
        print(f"  {q['question_id']}: {q['question_text']}", file=buf)
    print(file=buf)

    print("=== LLM Performance ===", file=buf)
    headers = [
        "Question",
        "Stage",
        "Model",
        "Thinking",
        "Prompt Tokens",
        "Output Tokens",
        "Tok/s",
        "Generation Time",
        "Load Time",
    ]
    trows = []
    for c in data["llm_performance"]:
        trows.append(
            [
                c["question_id"],
                STAGE_LABEL.get(c["stage"], c["stage"]),
                c["model"] or "—",
                fmt_bool(c["thinking"]),
                fmt_num(c["avg_prompt_tokens"]),
                fmt_num(c["avg_output_tokens"]),
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

    print("=== Resource Usage ===", file=buf)
    headers = ["Question", "Stage", "CPU", "Llama CPU", "GPU %", "GPU Memory", "RAM"]
    trows = []
    for u in data["resource_usage"]:
        if not u["measured"]:
            trows.append(
                [
                    u["question_id"],
                    STAGE_LABEL.get(u["stage"], u["stage"]),
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                ]
            )
            continue
        trows.append(
            [
                u["question_id"],
                STAGE_LABEL.get(u["stage"], u["stage"]),
                fmt_pct(u["cpu_pct"]),
                fmt_pct(u["llama_server_cpu_pct"]),
                fmt_pct(u["gpu_pct"]),
                fmt_mb(u["gpu_mem_mb"]),
                fmt_mb(u["ram_mb"]),
            ]
        )
    print(render_table(headers, trows), file=buf)

    return buf.getvalue(), data


def main():
    ap = argparse.ArgumentParser()
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
    import json

    summary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"[گزارش متنی ذخیره شد: {report_path}]")
    print(f"[خلاصه‌ی JSON ذخیره شد: {summary_path}]")

    _maybe_chart(path, rows, skip=args.no_chart)


def _maybe_chart(path: Path, rows: list[dict], skip: bool):
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

    from collections import defaultdict

    wall_by_stage_cat: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        for stage in r.get("stages", []):
            wall_by_stage_cat[stage["name"]][r["category"]].append(stage["wall_s"])

    stage_names = [s for s in STAGE_ORDER if s in wall_by_stage_cat] + [
        s for s in wall_by_stage_cat if s not in STAGE_ORDER
    ]
    if not stage_names:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    medians = [
        statistics.median([w for ws in wall_by_stage_cat[s].values() for w in ws])
        for s in stage_names
    ]
    ax.bar(stage_names, medians)
    ax.set_ylabel("median wall time (s)")
    ax.set_title(f"Per-stage median latency — {path.stem}")
    for i, v in enumerate(medians):
        ax.text(i, v, f"{v:.2f}s", ha="center", va="bottom")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out_png = path.with_suffix(".png")
    fig.savefig(out_png, dpi=150)
    print(f"[نمودار ذخیره شد: {out_png}]")


if __name__ == "__main__":
    main()
