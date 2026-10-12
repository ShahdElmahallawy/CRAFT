"""
evaluation/oracle/export_algo1_oracle_xlsx.py — package the ev_routing
optimality-gap analysis as results/algo1_oracle_gap.xlsx.

Compares each CRAFT run against the proven optimum from ev_oracle.py (which
solves the parser GROUND-TRUTH query, so the gap measures route quality, not
parsing). Uses the project's shared _write_table so the workbook matches the
other results/*.xlsx (title row 1, blank row 2, header row 3, data row 4+).

Sheets
  Summary          one row per model — the headline numbers
  Per-Query Gap    one row per (model, query) the oracle solved
  Oracle Solutions the 200 oracle records themselves
  Column Descriptions

Metric definitions are deliberate; see the Column Descriptions sheet and the
notes below, because each one changes the number:

* craft_objective_m = `dijkstra_objective_value` — the cost of CRAFT's REALIZED
  route. NOT the checker's `objective_value`, which for algo1 collapses to the
  raw unconstrained SP(src,dst) (the template has no POI stops, so its
  "intended stops" list is always empty) and would be identical for every route
  regardless of any charging detour.
* A row only counts toward the gap if CRAFT's answer is VALID
  (feasible AND constraints_satisfied). An invalid route is not "close to
  optimal", it is wrong — and it is typically SHORTER, since it skips the
  charging detour that makes the trip legal, so averaging it in would flatter
  the system. This is exactly how three battery-violating gpt-4.1 routes were
  caught appearing to beat the proven optimum.
* pct_optimal_of_solvable uses a FIXED denominator (every oracle-solvable
  query), so a model cannot score well by answering only the easy ones. MAPE is
  conditional on having produced a comparable answer and is therefore reported
  only alongside its n — never as the ranking metric. Same rule CLAUDE.md
  already mandates for constraints_satisfied.
* search-only vs end-to-end: rows where CRAFT's own parse diverges from gold are
  reported separately, so parser error is never mixed into a search-quality
  claim.

Usage:
  python evaluation/oracle/export_algo1_oracle_xlsx.py \
      --craft-xlsx <regenerated craft set1 xlsx>
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from craft.generator.tools.common_handlers import build_graph_from_gpkg
from evaluation.constraint_checker import NodeSnapper
from evaluation.evaluate_llm import _write_table
from evaluation.oracle.ev_oracle import GOLD_PATH, GPKG, battery_to_meters
from evaluation.oracle.validate_ev_oracle import parse_matches_gold

ORACLE_RESULTS = ROOT / "evaluation" / "oracle" / "results" / "ev_routing_oracle.json"
DEFAULT_OUT = ROOT / "results" / "algo1_oracle_gap.xlsx"
CRAFT_ROOT = ROOT / "outputs" / "craft"
TEMPLATE = "ev_routing"
MODELS = ["gemini-3_1-flash-lite", "gpt-4_1", "openai_gpt-oss-20b"]

AT_OPT_TOL_M = 0.5     # "hit the optimum": within half a metre
ZERO_TOL_M = 1e-6      # guard: never divide by a zero optimum


def _f(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--craft-xlsx", required=True,
                    help="regenerated craft set1 xlsx (current scoring)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()

    print("loading oracle, gold, craft, graph ...", flush=True)
    oracle_list = json.loads(ORACLE_RESULTS.read_text())
    oracle_by_id = {r["query_id"]: r for r in oracle_list}
    gold_by_id = {r["query_id"]: r for r in json.loads(GOLD_PATH.read_text())}

    craft = pd.read_excel(args.craft_xlsx, sheet_name="Results", header=2)
    craft = craft[(craft.template == TEMPLATE) & (craft.model.isin(args.models))]

    G = build_graph_from_gpkg(str(GPKG))
    snapper = NodeSnapper.from_gpkg(str(GPKG), G)

    n_solvable = sum(1 for r in oracle_list if r["status"] == "optimal")
    zero_opt = [r["query_id"] for r in oracle_list
                if r["status"] == "optimal" and abs(r["optimal_distance_m"] or 0.0) <= ZERO_TOL_M]
    print(f"  oracle: {len(oracle_list)} records, {n_solvable} solvable, "
          f"{len(zero_opt)} zero-optimum (excluded)")

    # cache parse-match per (model, qid) — needs the graph, so do it once
    per_query: List[list] = []
    stats: Dict[str, Dict[str, Any]] = {m: {
        "valid_comparable": 0, "exactly_optimal": 0, "apes": [],
        "parse_diverged": 0, "invalid": 0, "no_route": 0, "missing": 0,
        "worst": (None, 0.0),
    } for m in args.models}

    craft_by_key = {(r["model"], r["query_id"]): r for _, r in craft.iterrows()}

    for orec in oracle_list:
        qid = orec["query_id"]
        opt = _f(orec.get("optimal_distance_m"))
        for model in args.models:
            crow = craft_by_key.get((model, qid))
            if crow is None:
                if orec["status"] == "optimal":
                    stats[model]["missing"] += 1
                continue

            craft_obj = _f(crow.get("dijkstra_objective_value"))
            feasible = bool(crow.get("feasible") == True)
            cs = crow.get("constraints_satisfied")
            valid = feasible and (cs == True)

            parse_ok: Optional[bool] = None
            parse_reason = ""
            raw_path = CRAFT_ROOT / model / TEMPLATE / f"{qid}.json"
            if raw_path.exists():
                rec = json.loads(raw_path.read_text())
                # `result` is explicitly null (not merely absent) when CRAFT
                # produced no route, so `.get("result", {})` still yields None.
                if (rec.get("result") or {}).get("route"):
                    ok, reason = parse_matches_gold(rec, gold_by_id[qid], G, snapper)
                    parse_ok, parse_reason = ok, ("" if ok else reason)

            ratio = ape = None
            counted = ""
            if orec["status"] != "optimal":
                counted = "oracle infeasible — excluded"
            elif not feasible or craft_obj is None:
                stats[model]["no_route"] += 1
                counted = "no route — counts as failure"
            elif not valid:
                stats[model]["invalid"] += 1
                counted = "invalid (constraint violated) — counts as failure"
            elif opt is None or abs(opt) <= ZERO_TOL_M:
                counted = "zero/absent optimum — excluded"
            elif parse_ok is False:
                stats[model]["parse_diverged"] += 1
                ratio, ape = craft_obj / opt, abs(craft_obj - opt) / opt
                counted = "parse diverged — reported separately"
            else:
                ratio, ape = craft_obj / opt, abs(craft_obj - opt) / opt
                stats[model]["valid_comparable"] += 1
                stats[model]["apes"].append(ape)
                if craft_obj <= opt + AT_OPT_TOL_M:
                    stats[model]["exactly_optimal"] += 1
                counted = "counted in headline"
                if ape > stats[model]["worst"][1]:
                    stats[model]["worst"] = (qid, ape)

            per_query.append([
                qid, model, orec["status"], orec.get("infeasible_reason"),
                opt, orec.get("n_charging_stops"),
                craft_obj, feasible, (None if cs is None or pd.isna(cs) else bool(cs == True)),
                (None if crow.get("ev_battery_feasible") is None or pd.isna(crow.get("ev_battery_feasible"))
                 else bool(crow.get("ev_battery_feasible") == True)),
                parse_ok, parse_reason,
                (round(ratio, 6) if ratio is not None else None),
                (round(ape * 100, 6) if ape is not None else None),
                (None if ratio is None else bool(craft_obj <= opt + AT_OPT_TOL_M)),
                counted,
            ])

    # Summary
    summary_rows = []
    for model in args.models:
        s = stats[model]
        apes = s["apes"]
        n = len(apes)
        summary_rows.append([
            model, n_solvable, s["valid_comparable"], s["exactly_optimal"],
            round(s["exactly_optimal"] / n_solvable * 100, 2) if n_solvable else None,
            round(s["exactly_optimal"] / n * 100, 2) if n else None,
            round(sum(apes) / n * 100, 4) if n else None,
            round(statistics.median(apes) * 100, 4) if n else None,
            round(sorted(apes)[int(0.9 * (n - 1))] * 100, 4) if n else None,
            round(max(apes) * 100, 4) if n else None,
            s["worst"][0],
            s["no_route"], s["invalid"], s["parse_diverged"], s["missing"],
        ])

    # Oracle solutions
    oracle_rows = []
    for r in oracle_list:
        g = gold_by_id.get(r["query_id"], {})
        cur_m, max_m = battery_to_meters(g) if g else (None, None)
        oracle_rows.append([
            r["query_id"], r["status"], r.get("infeasible_reason"),
            r.get("optimal_distance_m"), r.get("n_charging_stops"),
            len(r["chain"]) if r.get("chain") else None,
            ";".join(str(c) for c in (r.get("charger_poi_ids") or [])) or None,
            round(cur_m, 1) if cur_m is not None else None,
            round(max_m, 1) if max_m is not None else None,
            r.get("n_candidates_final"),
        ])

    import openpyxl
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    _write_table(
        wb.create_sheet("Summary"),
        ["model", "n_solvable_queries", "valid_and_comparable", "exactly_optimal",
         "pct_optimal_of_solvable", "pct_optimal_of_comparable",
         "MAPE_pct", "median_APE_pct", "p90_APE_pct", "max_APE_pct", "worst_query",
         "craft_no_route", "craft_invalid", "craft_parse_diverged", "craft_missing"],
        summary_rows,
        title=f"ev_routing — optimality gap vs. proven optimum "
              f"(oracle: {n_solvable} solvable of {len(oracle_list)})")

    _write_table(
        wb.create_sheet("Per-Query Gap"),
        ["query_id", "model", "oracle_status", "oracle_infeasible_reason",
         "oracle_optimal_distance_m", "oracle_n_charging_stops",
         "craft_objective_m", "craft_feasible", "craft_constraints_satisfied",
         "craft_ev_battery_feasible", "parse_matches_gold", "parse_divergence_reason",
         "ratio_craft_over_optimal", "APE_pct", "is_exactly_optimal", "counted_as"],
        per_query,
        title="Per-query optimality gap (CRAFT realized route vs proven optimum)")

    _write_table(
        wb.create_sheet("Oracle Solutions"),
        ["query_id", "status", "infeasible_reason", "optimal_distance_m",
         "n_charging_stops", "chain_length_nodes", "charger_poi_ids",
         "current_battery_m", "max_battery_m", "n_candidates_final"],
        oracle_rows,
        title="Exact oracle solutions (evaluation/oracle/ev_oracle.py)")

    legend = [
        ["n_solvable_queries", f"Oracle-solvable queries ({n_solvable} of {len(oracle_list)}). "
                               "The FIXED denominator — the rest have no optimum to compare against."],
        ["valid_and_comparable", "CRAFT produced a VALID answer (feasible AND constraints_satisfied) "
                                 "whose parse matches gold. Only these enter the headline gap."],
        ["exactly_optimal", f"CRAFT's realized route cost is within {AT_OPT_TOL_M} m of the proven optimum."],
        ["pct_optimal_of_solvable", "exactly_optimal / n_solvable_queries. THE HEADLINE. Fixed denominator, "
                                    "so failures count as failures and a model cannot score well by "
                                    "answering only the easy queries."],
        ["pct_optimal_of_comparable", "exactly_optimal / valid_and_comparable. Conditional — read together "
                                      "with the failure columns, never on its own."],
        ["MAPE_pct", "mean(|craft-opt|/opt) x100 over valid_and_comparable ONLY. Conditional on producing "
                     "a comparable answer, so it silently drops failures — report with its n, never rank on it."],
        ["median_APE_pct / p90_APE_pct / max_APE_pct", "Distribution of the same error. For this template it "
                                                       "is bimodal (median 0), so '% optimal + worst case' "
                                                       "communicates better than the mean."],
        ["craft_objective_m", "dijkstra_objective_value = cost of CRAFT's REALIZED route. NOT the checker's "
                              "objective_value, which for algo1 collapses to the unconstrained SP(src,dst) "
                              "and is identical for every route regardless of any charging detour."],
        ["craft_invalid", "CRAFT returned a route that violates a hard constraint. Counted as a failure and "
                          "EXCLUDED from the gap: an invalid route is not 'close to optimal', and it is "
                          "usually shorter (it skips the required charging detour), so including it would "
                          "flatter the system."],
        ["craft_parse_diverged", "CRAFT's own parse differs from the gold parse, so it solved a different "
                                 "problem. Reported separately — never folded into a search-quality claim."],
        ["oracle_status", "'optimal' = provably optimal over ALL chargers (pruning is closed under a "
                          "necessary condition and re-solved until stable). 'infeasible' = no battery-feasible "
                          "chain exists even using every charger, or gold never states the max range."],
        ["ratio_craft_over_optimal", "craft_objective_m / oracle_optimal_distance_m. Must be >= 1.0 for a "
                                     "valid route; < 1.0 means the route is invalid (it beat the proven "
                                     "optimum) and is a bug signal."],
    ]
    _write_table(wb.create_sheet("Column Descriptions"), ["Column", "Description"],
                 legend, title="Column Descriptions")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f"\nwrote {out}")
    print(f"  Summary: {len(summary_rows)} models | Per-Query Gap: {len(per_query)} rows | "
          f"Oracle Solutions: {len(oracle_rows)} rows")
    for r in summary_rows:
        print(f"    {r[0]:<24} optimal {r[3]}/{r[1]} ({r[4]}%)  MAPE {r[6]}%  "
              f"[no_route {r[11]}, invalid {r[12]}, parse_div {r[13]}]")


if __name__ == "__main__":
    main()
