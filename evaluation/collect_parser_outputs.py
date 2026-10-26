#!/usr/bin/env python3
"""
collect_parser_outputs.py — snapshot every model's parser_output for all 1600
gold queries (8 gold sets x 200) into one JSON file per model:
  outputs/parser_outputs/{model}.json

This is step 1 of the parser evaluation (step 2 = evaluate_parser.py, which
scores these snapshots against benchmark/ground_truth/). Splitting collection
out means: (a) re-running after any craft rerun is a single explicit step,
(b) the exact parser outputs being scored are inspectable/diffable on disk,
(c) evaluate_parser.py doesn't need to re-glob thousands of raw record files
every time.

WHY THIS DOESN'T JUST READ THE FINAL CRAFT RECORD'S parser_output FIELD
-------------------------------------------------------------------------
scripts/run_craft.py has a bookkeeping bug: it captures parser_output by
re-reading a copy that the GENERATE subprocess writes, not the parser's own
Step-1 output. If generate crashes (bad codegen, LLM hiccup, etc.) that copy
never gets written, so the saved record shows parser_output=null even when
the parser itself succeeded. This collector therefore falls back to the
Step-1 staging file (outputs/eval_staging_craft_socal/{model}__{query_id}.json
— written directly by craft/orchestrator.py::run_parser(), before generate
ever runs) whenever the record's own field is null, recovering the true
parse result. This IS a real fallback, not a full fix — that staging dir is a
shared, mutable scratch space keyed only by (model, query_id), so a later
unrelated rerun of the same pair can overwrite it; a query whose staging file
is also gone/stale is reported as `parse_failed` with source "none", same as
before. Each record's `parser_output_source` field ("record" / "staging_fallback"
/ "none") makes this fully auditable per query.

Re-running is always safe — it just re-reads outputs/craft*/ and
outputs/eval_staging_craft_socal/, and overwrites the snapshot. Run this
whenever craft output records change (new run, gap fill, regenerated
queries) and BEFORE evaluate_parser.py.

Usage:
  python evaluation/collect_parser_outputs.py
  python evaluation/collect_parser_outputs.py --models gpt-4_1
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parser_eval_common import (
    COLLECTED_DIR, GOLD_DIR, GOLD_TO_RECORDS, MODELS, OUTPUTS_DIR, STAGING_DIR,
    collected_path, load_gold_sets, load_raw_records, load_staging_output,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--gold-dir", default=str(GOLD_DIR))
    ap.add_argument("--outputs-dir", default=str(OUTPUTS_DIR))
    ap.add_argument("--staging-dir", default=str(STAGING_DIR),
                    help="Step-1 pre-generator staging dir, used as a fallback "
                         "when a record's own parser_output is null")
    ap.add_argument("--out-dir", default=str(COLLECTED_DIR))
    args = ap.parse_args()

    gold_dir = Path(args.gold_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_sets = {}
    for name in GOLD_TO_RECORDS:
        p = gold_dir / f"{name}_gold.json"
        if p.exists():
            gold_sets[name] = json.loads(p.read_text())
    if not gold_sets:
        sys.exit(f"no gold files in {gold_dir} — expected benchmark/ground_truth/*_gold.json")

    total_expected = sum(len(v) for v in gold_sets.values())
    print(f"[collect] {len(gold_sets)} gold sets, {total_expected} queries/model\n")

    staging_dir = Path(args.staging_dir)
    any_gap = False
    for model in args.models:
        records = []
        n_missing = 0
        n_parse_failed = 0
        n_rescued = 0
        for set_name, recs in gold_sets.items():
            subdir, _is_socal = GOLD_TO_RECORDS[set_name]
            d = Path(args.outputs_dir) / subdir.format(model=model)
            valid = {r["query_id"] for r in recs}
            found = load_raw_records(d, valid)
            for r in recs:
                qid = r["query_id"]
                present = qid in found
                po = found.get(qid) if present else None
                source = "record" if (present and po is not None) else "none"
                if present and po is None:
                    # record's own parser_output is null — fall back to the
                    # Step-1 staging file (see module docstring for why).
                    staged = load_staging_output(model, qid, staging_dir)
                    if staged is not None:
                        po = staged
                        source = "staging_fallback"
                        n_rescued += 1
                if not present:
                    n_missing += 1
                elif po is None:
                    n_parse_failed += 1
                records.append({
                    "set": set_name,
                    "template": r.get("template"),
                    "query_id": qid,
                    "record_found": present,
                    "parse_failed": present and po is None,
                    "parser_output_source": source,
                    "parser_output": po,
                })
        n = len(records)
        n_ok = n - n_missing
        snapshot = {
            "model": model,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "n_expected": total_expected,
            "n_records_found": n_ok,
            "n_missing": n_missing,
            "n_parse_failed": n_parse_failed,
            "n_rescued_from_staging": n_rescued,
            "records": records,
        }
        out_path = collected_path(model, out_dir)
        out_path.write_text(json.dumps(snapshot, indent=2, default=str))
        flag = "  ⚠️  INCOMPLETE" if n_missing else ""
        any_gap = any_gap or bool(n_missing)
        print(f"[{model:22s}] n={n:4d}  found={n_ok:4d}  missing={n_missing:3d}  "
              f"parse_failed={n_parse_failed:3d}  (rescued_from_staging={n_rescued:3d})"
              f"  -> {out_path}{flag}")
        if n_missing:
            missing_ids = [r["query_id"] for r in records if not r["record_found"]]
            print(f"    missing ids: {missing_ids[:20]}"
                  f"{' ...' if len(missing_ids) > 20 else ''}")

    if any_gap:
        print("\n[collect] WARNING: at least one model is missing craft records for "
              "some gold queries — see above. Run craft on the missing ids, then "
              "re-run this collector before evaluate_parser.py.")
    else:
        print("\n[collect] all models have a record for every gold query. "
              "(remaining parse_failed queries have no staging file either — "
              "genuine failures or unrecoverable, not a gap in this collector.)")


if __name__ == "__main__":
    main()
