"""Score the analyst on eval/benchmark.json.

Examples:
  python eval.py                      # full pipeline (rich schema, 2 retries)
  python eval.py --provider claude    # compare models on the same benchmark
  python eval.py --retries 0          # ablation: no self-correction
  python eval.py --basic-schema       # ablation: dtypes only, no value hints
Results (including the generated code for failures) go to eval/results.json.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

from analyst import analyze
from llm import get_client


def flatten(res: dict) -> str:
    """Text of the answer. For tables, only the first row counts (top-1 convention)."""
    if res["kind"] == "table":
        t = res["table"]
        return " ".join(map(str, t.iloc[0].tolist())) if len(t) else ""
    return res["text"]


def numbers_in(text: str) -> list[float]:
    found = []
    for m in re.findall(r"-?\d[\d,]*\.?\d*", text):
        try:
            found.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return found


def is_correct(item: dict, res: dict) -> bool:
    if not res.get("ok"):
        return False
    text = flatten(res)
    if item["type"] == "number":
        tol = item.get("tolerance", 0.01)
        return any(
            math.isclose(x, item["expected"], rel_tol=tol, abs_tol=0.01)
            for x in numbers_in(text)
        )
    return any(a.lower() in text.lower() for a in item["expected"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sales.csv")
    ap.add_argument("--bench", default="eval/benchmark.json")
    ap.add_argument("--provider", choices=["gemini", "claude"], default=None,
                    help="Defaults to ANALYST_PROVIDER or whichever API key is set")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--basic-schema", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    items = json.loads(Path(args.bench).read_text(encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]
    client = get_client(args.provider)

    rows, correct, first_try = [], 0, 0
    for i, item in enumerate(items, 1):
        res = analyze(
            df, item["question"], client,
            max_retries=args.retries,
            rich_schema=not args.basic_schema,
            explain_result=False,
        )
        ok = is_correct(item, res)
        correct += ok
        first_try += ok and res["attempts"] == 1
        print(f"[{i:02d}/{len(items)}] {'PASS' if ok else 'FAIL'} (attempts={res['attempts']}) {item['question']}")
        rows.append({
            "question": item["question"],
            "expected": item["expected"],
            "got": flatten(res) if res.get("ok") else res.get("error", "")[-300:],
            "correct": bool(ok),
            "attempts": res["attempts"],
            "code": res["code"],
        })

    n = len(items)
    print(f"\nAccuracy: {correct}/{n} = {100 * correct / n:.0f}%")
    print(f"First-try accuracy (no retries needed): {first_try}/{n} = {100 * first_try / n:.0f}%")
    print(f"Config: provider={client.name}, model={client.model}, retries={args.retries}, "
          f"schema={'basic' if args.basic_schema else 'rich'}")
    Path("eval/results.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print("Details saved to eval/results.json")


if __name__ == "__main__":
    main()
