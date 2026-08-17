import json
from pathlib import Path
from typing import Any


def load_table_metadata(meta_dir: Path) -> list[dict[str, Any]]:
    """Loads every <table>.json directly under meta_dir. Skips the manual
    descriptions glossary file if it happens to sit in the same directory
    (it isn't table metadata itself)."""
    tables = []
    for path in sorted(meta_dir.glob("*.json")):
        if path.stem == "descriptions":
            continue
        with open(path, "r", encoding="utf-8") as f:
            tables.append(json.load(f))
    return tables


def _column_text(table: dict[str, Any], col: dict[str, Any]) -> str:
    parts = [
        f"جدول: {table['table']}",
        f"توضیح جدول: {table['description']}",
        f"ستون: {col['name']}",
        f"نوع داده: {col['type']}",
        f"توضیح ستون: {col['description']}",
    ]
    if col.get("primary_key"):
        parts.append("این ستون کلید اصلی (primary key) جدول است.")
    if col.get("foreign_key"):
        parts.append(f"این ستون کلید خارجی است و به {col['foreign_key']} اشاره می‌کند.")
    return "\n".join(parts)


def table_schema_text(table: dict[str, Any]) -> str:
    col_lines = [
        f"- {c['name']} ({c['type']}): {c['description']}" for c in table["columns"]
    ]
    return (
        f"جدول: {table['table']}\n"
        f"توضیح: {table['description']}\n"
        f"ستون‌ها:\n" + "\n".join(col_lines)
    )


def build_documents(tables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Returns a flat list of documents:
    {id, level, table, column, text, type, primary_key, foreign_key}
    `text` is what gets embedded; the rest is metadata carried alongside
    the vector so retrieval results are directly usable (no re-parsing).
    """
    docs: list[dict[str, Any]] = []

    for table in tables:
        table_name = table["table"]

        docs.append(
            {
                "id": f"table::{table_name}",
                "level": "table",
                "table": table_name,
                "column": None,
                "text": table_schema_text(table),
            }
        )

        for col in table["columns"]:
            docs.append(
                {
                    "id": f"column::{table_name}.{col['name']}",
                    "level": "column",
                    "table": table_name,
                    "column": col["name"],
                    "type": col["type"],
                    "primary_key": col.get("primary_key", False),
                    "foreign_key": col.get("foreign_key"),
                    "text": _column_text(table, col),
                }
            )

    return docs
