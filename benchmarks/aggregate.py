# -*- coding: utf-8 -*-
"""
Aggregation layer for benchmark results.

This module does NOT change how data is measured (see stage_timer.py /
resource_monitor.py / instrumented_pipeline.py / run_benchmark.py). It only
reshapes the already-recorded raw JSONL rows into summaries that are easy to
read: per-question stats, an overall run summary, an LLM-call performance
table, and a resource-usage table. `report.py` renders these; other tools
(a notebook, a dashboard) can import this module directly and get plain
dicts/lists back instead of parsing printed tables.

Grouping rule used everywhere here: rows are grouped by `question_id`
across every `repeat_idx` that exists for it (0, 1, 2, ... however many
`--repeats` produced), and the LLM/resource tables are further grouped by
pipeline stage within that question. Nothing about question count or repeat
count is hardcoded -- both come from whatever is actually in the JSONL file.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any

STAGE_ORDER = ["retrieval", "sql_generation", "sql_execution", "answer_generation"]


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean(xs: list[float | None]) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def _min(xs: list[float | None]) -> float | None:
    xs = [x for x in xs if x is not None]
    return min(xs) if xs else None


def _max(xs: list[float | None]) -> float | None:
    xs = [x for x in xs if x is not None]
    return max(xs) if xs else None


def _stage_map(row: dict) -> dict[str, dict]:
    """stage name -> stage dict, for one raw row."""
    return {s["name"]: s for s in row.get("stages", [])}


def _stage_wall(row: dict, stage_name: str) -> float | None:
    s = _stage_map(row).get(stage_name)
    return s["wall_s"] if s else None


# --------------------------------------------------------------------------
# Per-question summary (requirement 3)
# --------------------------------------------------------------------------


def aggregate_by_question(rows: list[dict]) -> list[dict]:
    by_qid: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_qid[r["question_id"]].append(r)

    out = []
    for qid, qrows in by_qid.items():
        qrows_sorted = sorted(qrows, key=lambda r: r.get("repeat_idx", 0))
        totals = [r.get("total_wall_s") for r in qrows_sorted]
        outcomes = Counter(r.get("outcome") for r in qrows_sorted)
        if len(outcomes) == 1:
            outcome_str = next(iter(outcomes))
        else:
            outcome_str = ", ".join(f"{o}:{n}" for o, n in outcomes.most_common())

        stage_avgs = {
            stage: _mean([_stage_wall(r, stage) for r in qrows_sorted])
            for stage in STAGE_ORDER
        }

        out.append(
            {
                "question_id": qid,
                "category": qrows_sorted[0].get("category"),
                "question_text": qrows_sorted[0].get("question_text"),
                "n_runs": len(qrows_sorted),
                "outcome": outcome_str,
                "avg_total_s": _mean(totals),
                "min_total_s": _min(totals),
                "max_total_s": _max(totals),
                "avg_retrieval_s": stage_avgs["retrieval"],
                "avg_sql_generation_s": stage_avgs["sql_generation"],
                "avg_sql_execution_s": stage_avgs["sql_execution"],
                "avg_answer_generation_s": stage_avgs["answer_generation"],
            }
        )

    # stable, readable order: same order questions first appear in the file
    first_seen = {r["question_id"]: i for i, r in enumerate(rows)}
    out.sort(key=lambda q: first_seen[q["question_id"]])
    return out


# --------------------------------------------------------------------------
# Overall run summary (requirement 4)
# --------------------------------------------------------------------------


def overall_summary(rows: list[dict]) -> dict:
    n_questions = len({r["question_id"] for r in rows})
    n_runs = len(rows)
    n_ok = sum(1 for r in rows if r.get("outcome") == "ok")
    success_rate = (n_ok / n_runs) if n_runs else None

    totals = [r.get("total_wall_s") for r in rows]
    stage_avgs = {
        stage: _mean([_stage_wall(r, stage) for r in rows]) for stage in STAGE_ORDER
    }

    all_calls = []
    gpu_mem_samples = []
    ram_samples = []
    for r in rows:
        for s in r.get("stages", []):
            all_calls.extend(s.get("ollama_calls", []))
            res = s.get("resources") or {}
            if res.get("n_samples"):
                if res.get("gpu_mem_max_mb") is not None:
                    gpu_mem_samples.append(res["gpu_mem_max_mb"])
                if res.get("self_rss_max_mb") is not None:
                    ram_samples.append(res["self_rss_max_mb"])

    tok_per_sec = [
        c.get("tokens_per_sec") for c in all_calls if c.get("tokens_per_sec")
    ]

    return {
        "n_questions": n_questions,
        "n_runs": n_runs,
        "success_rate": success_rate,
        "avg_total_s": _mean(totals),
        "avg_sql_generation_s": stage_avgs["sql_generation"],
        "avg_answer_generation_s": stage_avgs["answer_generation"],
        "avg_sql_execution_s": stage_avgs["sql_execution"],
        "avg_tokens_per_sec": _mean(tok_per_sec),
        "avg_gpu_mem_mb": _mean(gpu_mem_samples),
        "max_gpu_mem_mb": _max(gpu_mem_samples),
        "avg_ram_mb": _mean(ram_samples),
        "max_ram_mb": _max(ram_samples),
    }


# --------------------------------------------------------------------------
# LLM performance table (requirement 5) -- one row per (question, stage),
# averaged across every repeat and every call within that stage (a repair
# loop in sql_generation contributes more than one call per repeat; all of
# them are folded into the same average on purpose, since a stage that
# repairs more often really is more expensive).
# --------------------------------------------------------------------------


def llm_performance_rows(rows: list[dict]) -> list[dict]:
    # (question_id, stage_name) -> list of ollama call dicts
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    meta: dict[tuple[str, str], dict] = {}

    first_seen_q = {}
    first_seen_stage = {}
    for i, r in enumerate(rows):
        first_seen_q.setdefault(r["question_id"], i)
        for j, s in enumerate(r.get("stages", [])):
            calls = s.get("ollama_calls", [])
            if not calls:
                continue
            key = (r["question_id"], s["name"])
            groups[key].extend(calls)
            first_seen_stage.setdefault(key, (first_seen_q[r["question_id"]], j))
            if key not in meta:
                meta[key] = {"category": r.get("category")}

    out = []
    for (qid, stage), calls in groups.items():
        model = next((c.get("model") for c in calls if c.get("model")), None)
        think_vals = {c.get("think") for c in calls if c.get("think") is not None}
        think = (
            "mixed"
            if len(think_vals) > 1
            else (next(iter(think_vals)) if think_vals else None)
        )

        prompt_tokens = _mean([c.get("prompt_eval_count") for c in calls])
        output_tokens = _mean([c.get("eval_count") for c in calls])
        gen_time_s = _mean(
            [
                c["eval_duration_ns"] / 1e9
                for c in calls
                if c.get("eval_duration_ns") is not None
            ]
        )
        tok_per_sec = _mean([c.get("tokens_per_sec") for c in calls])
        load_time_s = _mean(
            [
                c["load_duration_ns"] / 1e9
                for c in calls
                if c.get("load_duration_ns") is not None
            ]
        )

        out.append(
            {
                "question_id": qid,
                "stage": stage,
                "model": model,
                "thinking": think,
                "n_calls": len(calls),
                "avg_prompt_tokens": prompt_tokens,
                "avg_output_tokens": output_tokens,
                "avg_generation_time_s": gen_time_s,
                "avg_tokens_per_sec": tok_per_sec,
                "avg_load_time_s": load_time_s,
            }
        )

    out.sort(key=lambda x: first_seen_stage[(x["question_id"], x["stage"])])
    return out


# --------------------------------------------------------------------------
# Resource usage table (requirement 6) -- one row per (question, stage),
# averaged across repeats. Stages faster than the sampling interval (see
# resource_monitor.py) have n_samples == 0 on every repeat; those show "—"
# rather than a fake zero.
# --------------------------------------------------------------------------


def resource_usage_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    first_seen_q = {}
    first_seen_stage = {}
    for i, r in enumerate(rows):
        first_seen_q.setdefault(r["question_id"], i)
        for j, s in enumerate(r.get("stages", [])):
            res = s.get("resources") or {}
            key = (r["question_id"], s["name"])
            groups[key].append(res)
            first_seen_stage.setdefault(key, (first_seen_q[r["question_id"]], j))

    out = []
    for (qid, stage), res_list in groups.items():
        measured = [res for res in res_list if res.get("n_samples")]
        if not measured:
            out.append(
                {
                    "question_id": qid,
                    "stage": stage,
                    "measured": False,
                    "cpu_pct": None,
                    "llama_server_cpu_pct": None,
                    "gpu_pct": None,
                    "gpu_mem_mb": None,
                    "ram_mb": None,
                }
            )
            continue

        out.append(
            {
                "question_id": qid,
                "stage": stage,
                "measured": True,
                "cpu_pct": _mean([res.get("self_cpu_avg") for res in measured]),
                "llama_server_cpu_pct": _mean(
                    [
                        res.get("proc_cpu_avg", {}).get("llama-server")
                        for res in measured
                    ]
                ),
                "gpu_pct": _mean([res.get("gpu_util_avg") for res in measured]),
                "gpu_mem_mb": _max([res.get("gpu_mem_max_mb") for res in measured]),
                "ram_mb": _max([res.get("self_rss_max_mb") for res in measured]),
            }
        )

    out.sort(key=lambda x: first_seen_stage[(x["question_id"], x["stage"])])
    return out


def build_report(rows: list[dict]) -> dict[str, Any]:
    """Everything report.py needs, bundled -- also what gets written to
    benchmark_summary.json."""
    return {
        "overall": overall_summary(rows),
        "per_question": aggregate_by_question(rows),
        "llm_performance": llm_performance_rows(rows),
        "resource_usage": resource_usage_rows(rows),
    }
