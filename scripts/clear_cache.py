#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clear the pipeline's SQL/retrieval/result caches (app/cache/query_cache.py).

Usage:
    python scripts/clear_cache.py                  # clear everything
    python scripts/clear_cache.py --namespace result
    python scripts/clear_cache.py --namespace sql
    python scripts/clear_cache.py --namespace retrieval
    python scripts/clear_cache.py --purge-expired   # only drop stale rows
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cache.query_cache import QueryCache, NS_RESULT, NS_RETRIEVAL, NS_SQL


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--namespace",
        choices=["sql", "retrieval", "result"],
        default=None,
        help="Clear only this cache. Default: clear all three.",
    )
    ap.add_argument(
        "--purge-expired",
        action="store_true",
        help="Only delete already-expired rows across all namespaces, instead of a full clear.",
    )
    args = ap.parse_args()

    cache = QueryCache()

    if args.purge_expired:
        removed = cache.store.purge_expired()
        print(f"Purged {removed} expired entries.")
        return

    if args.namespace is None:
        removed = cache.invalidate_all()
        print(f"Cleared all caches ({removed} entries removed).")
        return

    ns = {"sql": NS_SQL, "retrieval": NS_RETRIEVAL, "result": NS_RESULT}[args.namespace]
    removed = cache.store.clear_namespace(ns)
    print(f"Cleared '{args.namespace}' cache ({removed} entries removed).")


if __name__ == "__main__":
    main()
