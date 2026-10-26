"""Adds a Paper Tables sheet (Table 1 — poi_road_pref correctness) to a
craft/llm/llmap poi_road_pref export, and drops always-empty columns.

Run order:
  1. python evaluation/evaluate_craft.py  --templates poi_road_pref --out <tmp>.xlsx
     (or evaluate_llm.py / evaluate_llmap.py for those systems)
  2. python evaluation/build_poi_road_pref_tables.py --system craft --in <tmp>.xlsx \\
       --out results/craft_poi_road_pref.xlsx

Formulas resolve columns by header name at build time, never a hard-coded
letter, so a future column reorder can't silently repoint them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import openpyxl
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from evaluation.evaluate_llm import _write_table

SYSTEM_CFG = {
    "craft": {
        "system_name": "CRAFT",
        "scored_col": "hwy_miles_realized", "trivial_col": "hwy_distance_share_neutral_realized",
        "correct_col": "matches_avoidsp_realized", "avoided_col": "hwy_fully_avoided_realized",
        "has_correct": True,
        "title": "Table 1 — poi_road_pref correctness. Measured on: CRAFT's OWN realized route (its generated code).",
    },
    "llm": {
        "system_name": "LLM",
        "scored_col": "hwy_miles", "trivial_col": "hwy_distance_share_neutral",
        "correct_col": None, "avoided_col": "hwy_fully_avoided",
        "has_correct": False,
        "title": "Table 1 — poi_road_pref correctness. Measured on: the reconstructed preference-weighted SP through the system's own chosen stops.",
    },
    "llmap": {
        "system_name": "LLMAP",
        "scored_col": "hwy_miles", "trivial_col": "hwy_distance_share_neutral",
        "correct_col": None, "avoided_col": "hwy_fully_avoided",
        "has_correct": False,
        "title": "Table 1 — poi_road_pref correctness. Measured on: the reconstructed preference-weighted SP through the system's own chosen stops.",
    },
}

GENERIC_DESC = {
    "query_id": "Unique identifier for the benchmark query",
    "template": "Benchmark template family",
    "model": "LLM used",
    "feasible": "True if a valid route was returned",
    "ran": "True if generated.py executed without crashing",
    "solved_correctly": "True iff feasible AND constraints_satisfied",
    "generate_ok": "True if the LLM code-generation step succeeded",
    "execute_ok": "True if generated.py ran without errors",
    "error_stage": "Which stage threw an exception: generate, execute, or none",
    "error": "Exception message if the API call itself failed; blank if the call completed",
    "search_time_ms": "Search time in milliseconds",
    "required_stops": "Number of POI stops the query requires",
    "n_stops": "Number of intended POI stops actually visited",
    "stops_count_ok": "True if n_stops == required_stops",
    "poi_type_match": "True if the visited stop types match what the query required",
    "road_preference_met": "True if the route uses less highway than the neutral (no-preference) baseline through the same stops — a weak bar; see matches_avoidsp for the strict version",
    "constraint_satisfaction": "Fraction of applicable constraint checks that passed (0-1)",
    "constraints_satisfied": "True iff every applicable hard constraint passed",
    "objective_type": "Whether the objective is distance or time, fixed per template",
    "objective_value": "Shortest-path cost through [src, visited stops, dst]",
    "route_segments": "Number of road segments in the scored route",
    "hwy_segments": "How many of those segments run on a highway",
    "hwy_segment_share": "hwy_segments / route_segments",
    "hwy_distance_share": "Share of the route's distance spent on highways",
    "hwy_distance_share_neutral": "Same share for the no-preference baseline through the same stops — the counterfactual",
    "hwy_fully_avoided": "True if the route uses zero highway segments",
    "hwy_miles": "Highway miles on the route",
    "nonhwy_miles": "Non-highway miles on the route",
    "total_miles": "Total route distance in miles",
    "hwy_miles_avoidsp": "Highway miles on the preference-weighted shortest path through the same stops (the reference an optimal solver would produce)",
    "hwy_distance_share_avoidsp": "Highway share of that reference route",
    "hwy_miles_excess_vs_avoidsp": "hwy_miles minus hwy_miles_avoidsp — how much extra highway the route used versus the reference",
    "matches_avoidsp": "True if the route used no more highway than the reference (within 0.1mi tolerance) — the strict correctness bar",
    "dijkstra_objective_value": "Cost of CRAFT's own realized route",
    "distance_limits_met_realized": "distance_limits_met scored on CRAFT's own realized route",
    "time_budget_met_realized": "time_budget_met scored on CRAFT's own realized route",
    "road_preference_met_realized": "road_preference_met scored on CRAFT's own realized route",
    "route_segments_realized": "route_segments on CRAFT's own realized route",
    "hwy_segments_realized": "hwy_segments on CRAFT's own realized route",
    "hwy_segment_share_realized": "hwy_segment_share on CRAFT's own realized route",
    "hwy_distance_share_realized": "hwy_distance_share on CRAFT's own realized route",
    "hwy_distance_share_neutral_realized": "hwy_distance_share_neutral on CRAFT's own realized route",
    "hwy_fully_avoided_realized": "hwy_fully_avoided on CRAFT's own realized route",
    "hwy_miles_realized": "hwy_miles on CRAFT's own realized route",
    "nonhwy_miles_realized": "nonhwy_miles on CRAFT's own realized route",
    "total_miles_realized": "total_miles on CRAFT's own realized route",
    "hwy_miles_avoidsp_realized": "hwy_miles_avoidsp on CRAFT's own realized route",
    "hwy_distance_share_avoidsp_realized": "hwy_distance_share_avoidsp on CRAFT's own realized route",
    "hwy_miles_excess_vs_avoidsp_realized": "hwy_miles_excess_vs_avoidsp on CRAFT's own realized route",
    "matches_avoidsp_realized": "matches_avoidsp on CRAFT's own realized route — the strict correctness bar for CRAFT specifically",
    "latency_s": "Wall-clock time for the LLM call (seconds)",
    "prompt_tokens": "Prompt tokens used",
    "completion_tokens": "Completion tokens generated",
    "parsed_pois": "Number of POI stops the LLM proposed",
    "n_proposed": "Number of stops the LLM proposed",
    "n_matched": "Number of proposed stops successfully matched to a real dataset POI",
    "n_dropped": "Number of proposed stops that could not be matched",
    "all_matched": "True if every proposed stop was matched",
    "match_rate": "n_matched / n_proposed",
    "match_methods": "Which matching tier each stop was matched by",
    "any_hallucinated": "True if at least one dropped stop was confirmed hallucinated by Google Places",
    "hallucination_reasons": "Per-stop hallucination classification",
    "status": "ok, direct_route_no_pois, parser_type_mismatch, or parser_failed",
    "parser_failed": "True if the LLM parsing step itself failed",
}

DROP_ALWAYS_EMPTY_CANDIDATES = {
    "states_expanded", "states_expanded_stale", "states_expanded_useful",
    "states_generated", "states_pruned", "peak_heap_size",
    "poi_order", "category_limits_met", "ev_battery_feasible", "time_budget_met",
    "distance_limits_met", "rating_ok", "clock_deadline_met", "poi_filters_respected",
    "distance_limits_met_realized", "time_budget_met_realized",
    "infeasible_reason", "filter_match_rate", "filter_passed", "filter_checked",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", required=True, choices=sorted(SYSTEM_CFG))
    ap.add_argument("--in", dest="in_path", required=True,
                    help="A Results export from evaluate_{craft,llm,llmap}.py --templates poi_road_pref")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = SYSTEM_CFG[args.system]

    wb_in = openpyxl.load_workbook(args.in_path, data_only=True)
    ws_in = wb_in["Results"]
    headers = [c.value for c in ws_in[3]]
    rows_raw = list(ws_in.iter_rows(min_row=4, values_only=True))

    empty_now = {h for i, h in enumerate(headers)
                 if h in DROP_ALWAYS_EMPTY_CANDIDATES and all(r[i] is None for r in rows_raw)}
    keep_idx = [i for i, h in enumerate(headers) if h not in empty_now]
    kept_headers = [headers[i] for i in keep_idx]
    rows = [[r[i] for i in keep_idx] for r in rows_raw]
    print(f"dropped {len(empty_now)} empty columns, kept {len(kept_headers)}")

    missing_desc = [h for h in kept_headers if h not in GENERIC_DESC]
    if missing_desc:
        raise SystemExit(f"missing descriptions for: {missing_desc}")

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    ws = wb.create_sheet("Results")
    _write_table(ws, kept_headers, rows, title=f"{cfg['system_name']} poi_road_pref Results")
    ws_legend = wb.create_sheet("Column Descriptions")
    _write_table(ws_legend, ["Column", "Description"],
                 [[h, GENERIC_DESC[h]] for h in kept_headers], title="Column Descriptions")

    col = {h: get_column_letter(i + 1) for i, h in enumerate(kept_headers)}

    def L(name):
        return f"Results!${col[name]}:${col[name]}"

    ws_p = wb.create_sheet("Paper Tables")
    ws_p.append([cfg["title"]])
    ws_p.cell(1, 1).font = openpyxl.styles.Font(bold=True, size=13)
    ws_p.append([])
    header_row = ["Model", "N", "Routes scored", "Trivial (neutral SP has 0 hwy)", "Non-trivial",
                  "Correct (<=ref)", "% of N", "Correct & non-trivial", "% of non-trivial",
                  "Fully avoided", "% of N", "road_preference_met (OLD weak bar)", "% of N"]
    ws_p.append(header_row)
    hrow = ws_p.max_row
    for c in range(1, len(header_row) + 1):
        cell = ws_p.cell(hrow, c)
        cell.fill = openpyxl.styles.PatternFill("solid", fgColor="1F4E79")
        cell.font = openpyxl.styles.Font(bold=True, color="FFFFFF")
        cell.alignment = openpyxl.styles.Alignment(horizontal="center", wrap_text=True)

    MODELS = ["gemini-3_1-flash-lite", "gpt-4_1", "openai_gpt-oss-20b"]
    for model in MODELS:
        r = ws_p.max_row + 1
        ws_p.cell(r, 1, model)
        ws_p.cell(r, 2, f'=COUNTIFS({L("model")},$A{r})')
        ws_p.cell(r, 3, f'=COUNTIFS({L("model")},$A{r},{L(cfg["scored_col"])},">=0")')
        ws_p.cell(r, 4, f'=COUNTIFS({L("model")},$A{r},{L(cfg["scored_col"])},">=0",{L(cfg["trivial_col"])},"<=0")')
        ws_p.cell(r, 5, f"=C{r}-D{r}")
        if cfg["has_correct"]:
            ws_p.cell(r, 6, f'=COUNTIFS({L("model")},$A{r},{L(cfg["correct_col"])},TRUE)')
            ws_p.cell(r, 7, f'=IF(B{r}=0,"",F{r}/B{r})')
            ws_p.cell(r, 8, f'=COUNTIFS({L("model")},$A{r},{L(cfg["correct_col"])},TRUE,{L(cfg["trivial_col"])},">0")')
            ws_p.cell(r, 9, f'=IF(E{r}=0,"",H{r}/E{r})')
        else:
            ws_p.cell(r, 6, "n/a — route IS the reference (self-comparison)")
        ws_p.cell(r, 10, f'=COUNTIFS({L("model")},$A{r},{L(cfg["avoided_col"])},TRUE)')
        ws_p.cell(r, 11, f'=IF(B{r}=0,"",J{r}/B{r})')
        ws_p.cell(r, 12, f'=COUNTIFS({L("model")},$A{r},{L("road_preference_met")},TRUE)')
        ws_p.cell(r, 13, f'=IF(B{r}=0,"",L{r}/B{r})')

    ws_p.append([])
    ws_p.append(["Notes"])
    ws_p.append(["- Formulas resolve columns by header name at build time — never hard-coded letters."])
    ws_p.append(["- Correct (<=ref) uses no more highway than the preference-weighted shortest path through the same stops (within 0.1mi) — the strict bar. road_preference_met (OLD weak bar) only requires beating the no-preference baseline, which nearly always succeeds."])

    wb.move_sheet("Paper Tables", offset=-2)
    wb.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
