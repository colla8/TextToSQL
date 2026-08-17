#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline import TextToSQLPipeline


def main():
    question = " ".join(sys.argv[1:]).strip() or input("سؤال: ").strip()

    pipeline = TextToSQLPipeline()
    result = pipeline.ask(question)

    if result.get("not_found"):
        print("\n--- داده‌ای پیدا نشد ---")
        print(result["reason"])
        return

    print("\n--- SQL ---")
    print(result["sql"])

    if result.get("error"):
        print("\n--- خطا ---")
        print(result["error"])
        return

    print(f"\n--- نتیجه ({len(result['rows'])} ردیف) ---")
    print(result["columns"])
    for row in result["rows"][:10]:
        print(row)

    print("\n--- پاسخ ---")
    print(result["answer"])


if __name__ == "__main__":
    main()
