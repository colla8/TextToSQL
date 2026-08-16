import argparse
import json
import os
import sys
from pathlib import Path
from app.core.config import Setting
import psycopg2

DESCRIPTIONS_FILE = Setting.DESCRIPTION
OUTPUT_DIR = Setting.META_DATA

TODO = "TODO"


def get_connection(args):
    return psycopg2.connect(
        host=args.host or os.environ.get("PGHOST", "localhost"),
        port=args.port or os.environ.get("PGPORT", "5432"),
        dbname=args.db or os.environ.get("PGDATABASE"),
        user=args.user or os.environ.get("PGUSER"),
        password=args.password or os.environ.get("PGPASSWORD"),
    )


def load_descriptions():
    if not DESCRIPTIONS_FILE.exists():
        print(
            f"warning: {DESCRIPTIONS_FILE} not found, all descriptions will be '{TODO}'."
        )
        return {"_tables": {}, "_columns": {}}
    with open(DESCRIPTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_tables(conn, schema="public"):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = %s AND table_type = 'BASE TABLE'
            ORDER BY table_name;
            """,
            (schema,),
        )
        return [r[0] for r in cur.fetchall()]


def fetch_columns(conn, table, schema="public"):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name, data_type, is_nullable, character_maximum_length
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position;
            """,
            (schema, table),
        )
        return cur.fetchall()


def fetch_primary_keys(conn, table, schema="public"):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_schema = %s AND tc.table_name = %s;
            """,
            (schema, table),
        )
        return {r[0] for r in cur.fetchall()}


def fetch_foreign_keys(conn, table, schema="public"):
    """Returns {column_name: 'ref_table.ref_column'}"""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT kcu.column_name, ccu.table_name AS ref_table, ccu.column_name AS ref_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage ccu
              ON tc.constraint_name = ccu.constraint_name AND tc.table_schema = ccu.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
              AND tc.table_schema = %s AND tc.table_name = %s;
            """,
            (schema, table),
        )
        return {row[0]: f"{row[1]}.{row[2]}" for row in cur.fetchall()}


def build_table_metadata(conn, table, descriptions, warnings):
    pk_cols = fetch_primary_keys(conn, table)
    fk_cols = fetch_foreign_keys(conn, table)
    columns_raw = fetch_columns(conn, table)

    table_desc = descriptions.get("_tables", {}).get(table)
    if table_desc is None:
        table_desc = TODO
        warnings.append(f"table '{table}': no description in descriptions.json")

    columns = []
    for col_name, data_type, is_nullable, max_len in columns_raw:
        key = f"{table}.{col_name}"
        col_desc = descriptions.get("_columns", {}).get(key)
        if col_desc is None:
            col_desc = TODO
            warnings.append(f"column '{key}': no description in descriptions.json")

        entry = {
            "name": col_name,
            "type": data_type,
            "nullable": is_nullable == "YES",
            "description": col_desc,
        }
        if col_name in pk_cols:
            entry["primary_key"] = True
        if col_name in fk_cols:
            entry["foreign_key"] = fk_cols[col_name]
        columns.append(entry)

    return {
        "table": table,
        "description": table_desc,
        "columns": columns,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build per-table metadata JSON for text-to-SQL"
    )
    parser.add_argument("--host")
    parser.add_argument("--port")
    parser.add_argument("--db")
    parser.add_argument("--user")
    parser.add_argument("--password")
    parser.add_argument("--schema", default="public")
    args = parser.parse_args()

    descriptions = load_descriptions()
    conn = get_connection(args)
    warnings = []

    try:
        tables = fetch_tables(conn, args.schema)
        if not tables:
            sys.exit(f"no tables found in schema '{args.schema}'.")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for table in tables:
            meta = build_table_metadata(conn, table, descriptions, warnings)
            out_path = OUTPUT_DIR / f"{table}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"{table}.json written ({len(meta['columns'])} columns).")

        if warnings:
            print(f"\n{len(warnings)} missing description(s) — filled with '{TODO}':")
            for w in warnings:
                print(f"  - {w}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
