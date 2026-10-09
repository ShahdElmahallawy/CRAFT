"""
evaluation/oracle/report_solved_counts.py — the headline cross-system table:
how many queries each system+model solved CORRECTLY, i.e.

    feasible  AND  constraints_satisfied   (== constraint_satisfaction == 1.0)

counted over a FIXED denominator, which is the project's standing rule
(CLAUDE.md, "EVALUATION & SCORING"): never compare on the raw `feasible`
column (LLMAP's ignores per-leg distance) nor on the mean
`constraint_satisfaction` (it drops CRAFT's empty results from its own
denominator and inflates it).

Scoped to the 5 set1 templates x 200 queries = 1000 per model, so all
three systems are measured on exactly the same query set. The baseline
exporters default to all 7 names in cc.TEMPLATES and will happily pick up
leftover directories from retired template families, which is why the scope
is pinned here rather than trusted from the file.

Usage:
  conda run -n myenv python evaluation/oracle/report_solved_counts.py \
      --craft X.xlsx --llm Y.xlsx --llmap Z.xlsx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SET1_TEMPLATES = ["ev_routing", "ordered_poi", "poi_collection",
                  "ev+ordered_poi", "unordered_poi"]
EV_TEMPLATES = ["ev_routing", "ev+ordered_poi"]
MODELS = ["gemini-3_1-flash-lite", "gpt-4_1", "openai_gpt-oss-20b"]


def load(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Results", header=2)
    return df[df.template.isin(SET1_TEMPLATES) & df.model.isin(MODELS)].copy()


def solved(g: pd.DataFrame) -> int:
    return int(((g["feasible"] == True) & (g["constraints_satisfied"] == True)).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--craft", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--llmap", required=True)
    args = ap.parse_args()

    frames = {"CRAFT": load(args.craft), "LLM": load(args.llm), "LLMAP": load(args.llmap)}

    print("Scope: 5 set1 templates x 200 = 1000 queries per model\n")
    for name, df in frames.items():
        n = df.groupby("model").size().to_dict()
        print(f"  {name} rows per model: {n}")

    print("\n" + "=" * 78)
    print("SOLVED CORRECTLY  (feasible AND constraints_satisfied)  — out of 1000")
    print("=" * 78)
    hdr = f"{'model':<26}" + "".join(f"{s:>16}" for s in frames)
    print(hdr)
    print("-" * len(hdr))
    for m in MODELS:
        row = f"{m:<26}"
        for name, df in frames.items():
            g = df[df.model == m]
            row += f"{solved(g):>10} /{len(g):>5}"
        print(row)

    print("\n" + "=" * 78)
    print("EV TEMPLATES ONLY (ev_routing + ev+ordered_poi) — out of 400")
    print("the refuel-policy change bites here and nowhere else")
    print("=" * 78)
    hdr = f"{'model':<26}" + "".join(f"{s:>16}" for s in frames)
    print(hdr)
    print("-" * len(hdr))
    for m in MODELS:
        row = f"{m:<26}"
        for name, df in frames.items():
            g = df[(df.model == m) & (df.template.isin(EV_TEMPLATES))]
            row += f"{solved(g):>10} /{len(g):>5}"
        print(row)

    print("\n" + "=" * 78)
    print("PER-TEMPLATE breakdown of solved-correctly")
    print("=" * 78)
    for t in SET1_TEMPLATES:
        print(f"\n{t}")
        hdr = f"  {'model':<24}" + "".join(f"{s:>16}" for s in frames)
        print(hdr)
        for m in MODELS:
            row = f"  {m:<24}"
            for name, df in frames.items():
                g = df[(df.model == m) & (df.template == t)]
                row += f"{solved(g):>10} /{len(g):>5}"
            print(row)


if __name__ == "__main__":
    main()
