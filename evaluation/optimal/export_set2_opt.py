"""Builds results/craft_set2_opt.xlsx: CRAFT's baseline Dijkstra engine (Long,
v0_baseline) vs. the same generated code under an A* engine with a corridor
heuristic (Opt, v2_astar), on the 200 California long-EV-trip queries.

Source: outputs/craft_set2_ca/{model}/opt/{v0_baseline,v2_astar}/*.json —
raw per-query search-engine output (route, run_time_ms, states_expanded, ...).

Median is used for Search(s)/States, not mean — mean is dragged up hard by a
handful of very slow queries in v0_baseline; median matches what the Long/Opt
comparison is actually trying to show (typical-case speedup, not worst-case).
Precomputed in Python, not an Excel formula — MEDIAN(IF(...)) needs CSE array
entry that openpyxl can't write (bit the project before, see CLAUDE.md)."""
import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluation.evaluate_llm import _write_table

DEFAULT_OUT = ROOT / "results" / "craft_set2_opt.xlsx"

MODELS = [("gemini-3_1-flash-lite", "Flash Lite"), ("gpt-4_1", "GPT-4.1"), ("openai_gpt-oss-20b", "gpt-oss-20b")]
VARIANTS = [("v0_baseline", "Long"), ("v2_astar", "Opt")]

results_rows = []
paper_cells = {}
for model, mdisp in MODELS:
    for variant, vdisp in VARIANTS:
        files = sorted(glob.glob(str(ROOT / "outputs" / "craft_set2_ca" / model / "opt" / variant / "*.json")))
        recs = [json.loads(open(f).read()) for f in files]
        for r in recs:
            results_rows.append([
                r["query_id"], model, variant,
                r.get("route") is not None, r.get("run_time_ms"),
                r.get("states_expanded"), r.get("states_generated"),
                r.get("states_pruned"), r.get("peak_heap_size"),
                r.get("route_cost_m"), r.get("heuristic_calls"),
                r.get("corridor_build_ms"),
            ])
        n = len(recs)
        succ = sum(1 for r in recs if r.get("route") is not None)
        times_s = [r["run_time_ms"] / 1000.0 for r in recs if r.get("route") is not None]
        states = [r["states_expanded"] for r in recs if r.get("route") is not None]
        paper_cells[(model, variant)] = {
            "n": n, "success_pct": round(100.0 * succ / n, 1) if n else None,
            "median_search_s": round(statistics.median(times_s), 2) if times_s else None,
            "median_states": round(statistics.median(states)) if states else None,
        }

wb = openpyxl.Workbook()
wb.remove(wb.active)

ws = wb.create_sheet("Results")
_write_table(ws,
    ["query_id", "model", "variant", "feasible", "run_time_ms", "states_expanded",
     "states_generated", "states_pruned", "peak_heap_size", "route_cost_m",
     "heuristic_calls", "corridor_build_ms"],
    results_rows, title="CRAFT Set 2 (California long EV trips): Long (v0_baseline) vs Opt (v2_astar)")

ws_d = wb.create_sheet("Column Descriptions")
_write_table(ws_d, ["Column", "Description"], [
    ["query_id", "Unique identifier for the benchmark query"],
    ["model", "LLM used for code generation"],
    ["variant", "v0_baseline (Long) = plain Dijkstra; v2_astar (Opt) = A* with a corridor heuristic, same generated code"],
    ["feasible", "True if a route was found"],
    ["run_time_ms", "Search time in milliseconds"],
    ["states_expanded", "States popped from the open list and expanded"],
    ["states_generated", "Successor states produced by expand(), before dominance filtering"],
    ["states_pruned", "Successor states discarded by the dominance check"],
    ["peak_heap_size", "Largest open-list size seen during the search"],
    ["route_cost_m", "Realized route cost in meters"],
    ["heuristic_calls", "Number of heuristic evaluations (Opt only; 0 for Long)"],
    ["corridor_build_ms", "Time spent building the corridor heuristic (Opt only; 0 for Long)"],
], title="Column Descriptions")

ws_p = wb.create_sheet("Paper Tables")
ws_p.append(["Table — CRAFT on the 200 long EV trips: Long (v0_baseline) vs Opt (A* + corridor heuristic, v2_astar). Search(s)/States are MEDIAN over feasible queries."])
ws_p.cell(1, 1).font = openpyxl.styles.Font(bold=True, size=13)
ws_p.append([])
header = ["Model", "Success Long (%)", "Success Opt (%)", "Search(s) Long", "Search(s) Opt", "States Long", "States Opt"]
ws_p.append(header)
hrow = ws_p.max_row
for c in range(1, len(header) + 1):
    cell = ws_p.cell(hrow, c)
    cell.fill = openpyxl.styles.PatternFill("solid", fgColor="1F4E79")
    cell.font = openpyxl.styles.Font(bold=True, color="FFFFFF")
    cell.alignment = openpyxl.styles.Alignment(horizontal="center", wrap_text=True)
for model, mdisp in MODELS:
    long_c = paper_cells[(model, "v0_baseline")]
    opt_c = paper_cells[(model, "v2_astar")]
    ws_p.append([mdisp, long_c["success_pct"], opt_c["success_pct"],
                 long_c["median_search_s"], opt_c["median_search_s"],
                 long_c["median_states"], opt_c["median_states"]])
ws_p.append([])
ws_p.append(["Notes"])
ws_p.append(["- Precomputed in Python from outputs/craft_set2_ca/{model}/opt/{v0_baseline,v2_astar}/*.json — see evaluation/optimal/export_set2_opt.py."])
ws_p.append(["- Search(s)/States use the MEDIAN over feasible (route found) queries, not the mean — a handful of very slow v0_baseline queries would otherwise dominate the average."])

wb.move_sheet("Paper Tables", offset=-2)
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    wb.save(args.out)
    print("saved", args.out)
for model, mdisp in MODELS:
    for variant, vdisp in VARIANTS:
        c = paper_cells[(model, variant)]
        print(mdisp, vdisp, c)
