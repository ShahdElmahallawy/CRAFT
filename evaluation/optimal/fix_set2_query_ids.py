"""One-time-bug-class fix for results/craft_set2.xlsx: strips a leftover
literal "algo1_" substring from query_id (e.g. set2_algo1_ev_routing_001 ->
set2_ev_routing_001) left over from before the template was renamed, and
drops always-empty columns.

This does NOT re-score from raw data — the original scorer that produced
this workbook's constraints_satisfied/solved_correctly columns from
outputs/craft_set2_ca/{model}/socal/{query_id}_output/ was never committed
to the repo, so there is no from-scratch generator for this file today.
Verified (2026-10-25) that the existing scored values are current: for every
row, states_expanded matches the corresponding raw
outputs/craft_set2_ca/{model}/opt/v0_baseline/{new_id}.json exactly — the
bug was in the ID string only, not the underlying data.

Usage:
  python evaluation/optimal/fix_set2_query_ids.py --in results/craft_set2.xlsx --out results/craft_set2.xlsx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluation.evaluate_llm import _write_table

EXTRA_DESC = {
    "trip_dist_mi": "Trip distance in miles (haversine-derived)",
    "dist_group": "moderate (15-75mi) or long (75-300mi) distance band",
    "needs_charger": "True if the query was built so the trip cannot be completed on the starting charge alone",
    "chargers_in_corridor": "Number of charging stations within the src-dst corridor band",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    wb_in = openpyxl.load_workbook(args.in_path, data_only=True)
    ws_in = wb_in["Results"]
    headers = [c.value for c in ws_in[3]]
    rows_raw = list(ws_in.iter_rows(min_row=4, values_only=True))

    empty = [h for i, h in enumerate(headers) if all(r[i] is None for r in rows_raw)]
    keep_idx = [i for i, h in enumerate(headers) if h not in empty]
    kept_headers = [headers[i] for i in keep_idx]
    print("dropping empty columns:", empty)

    qi = kept_headers.index("query_id")
    rows = []
    for r in rows_raw:
        row = [r[i] for i in keep_idx]
        row[qi] = row[qi].replace("algo1_", "")
        rows.append(row)

    wb_d = wb_in["Column Descriptions"]
    desc = {r[0]: r[1] for r in wb_d.iter_rows(min_row=4, values_only=True) if r[0]}
    desc.update(EXTRA_DESC)
    missing = [h for h in kept_headers if not desc.get(h)]
    if missing:
        raise SystemExit(f"missing descriptions for: {missing}")

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    ws = wb.create_sheet("Results")
    _write_table(ws, kept_headers, rows, title="CRAFT Set 2 (California long EV trips) Results")
    ws_d = wb.create_sheet("Column Descriptions")
    _write_table(ws_d, ["Column", "Description"],
                 [[h, desc[h]] for h in kept_headers], title="Column Descriptions")
    wb.save(args.out)
    print(f"saved {args.out}: rows={len(rows)} cols={len(kept_headers)}")


if __name__ == "__main__":
    main()
