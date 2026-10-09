"""
evaluation/oracle/compare_craft_vs_oracle.py — optimality-gap report for
ev_routing: CRAFT's realized route cost vs the proven optimum.

Headline metric (as requested):
    MAPE = mean( |craft_obj - opt_obj| / opt_obj )

Guards and definitions this script is explicit about, because each one
changes the number:

1. DIVISION BY ZERO. Every optimum is checked for zero/near-zero before the
   ratio is formed; any such query is EXCLUDED and reported separately
   rather than silently producing inf/nan.

2. WHICH CRAFT NUMBER. craft_obj = compute_system_objective(route, G,
   "length") — the cost of CRAFT's REALIZED route. NOT the checker's
   `objective_value` column, which for this template collapses to the raw
   unconstrained SP(src,dst) (algo1 has no POI stops, so its "intended
   stops" list is always empty) and is therefore identical for every route
   regardless of any charging detour.

3. WHICH QUERIES. Only oracle status=="optimal" queries can contribute (an
   infeasible query has no optimum to divide by). Those are split into:
     search-only  : CRAFT parsed the same problem gold specifies
                    -> pure route-quality gap. THIS IS THE HEADLINE.
     parse-diverged: CRAFT solved a different problem -> the gap mixes
                    parser error with search error; reported separately and
                    never folded into the headline MAPE.

4. CRAFT-FEASIBLE-BUT-ORACLE-INFEASIBLE is counted and reported: if CRAFT
   returns a route for a query the oracle PROVED has no battery-feasible
   chain, that route must violate the battery constraint. That is a
   correctness signal, not a gap, so it is never averaged into MAPE.

Usage:
  conda run -n myenv python evaluation/oracle/compare_craft_vs_oracle.py
  conda run -n myenv python evaluation/oracle/compare_craft_vs_oracle.py --models gpt-4_1
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from craft.generator.tools.common_handlers import build_graph_from_gpkg
from evaluation.constraint_checker import NodeSnapper, compute_system_objective
from evaluation.constraint_checker import coerce as coerce_node
from evaluation.oracle.ev_oracle import GOLD_PATH, GPKG
from evaluation.oracle.validate_ev_oracle import craft_feasible, parse_matches_gold

ORACLE_RESULTS = ROOT / "evaluation" / "oracle" / "results" / "ev_routing_oracle.json"
OUT_PATH = ROOT / "evaluation" / "oracle" / "results" / "ev_routing_gap_report.json"
CRAFT_ROOT = ROOT / "outputs" / "craft"
TEMPLATE = "ev_routing"

ZERO_TOL_M = 1e-6      # an optimum at/below this is treated as zero -> excluded
AT_OPT_TOL_M = 0.5     # "CRAFT hit the optimum": within half a meter


def route_battery_feasible(route: List[Any], G: nx.MultiDiGraph, chargers: set,
                           cur_m: float, max_m: float) -> Tuple[bool, float]:
    """Does the battery stay >= 0 at EVERY point along the realized route
    (refilling to max at charging stations)?

    Kept as an INDEPENDENT re-implementation (not read off
    constraint_checker's `ev_battery_feasible`) so this report cross-checks
    the scorer rather than inheriting its verdict.

    History: until 2026-08-10 _system_checks tested only the FINAL battery
    value, so a route that ran the battery NEGATIVE and later passed a charger
    (resetting it to max) scored feasible. This function found 3 such gpt-4_1
    routes (-348m / -494m / -121m), which appeared *shorter than the proven
    optimum* because they skipped the charging detour. constraint_checker.py
    and evaluation/metrics.py have since been fixed to latch on any negative,
    matching the baseline path (_ev_feasible) -- all systems now agree. This
    stays as a redundant check; agreement between the two is the useful signal.
    """
    from craft.generator.tools.common_handlers import edge_weight

    battery = cur_m
    min_batt = battery
    for u, v in zip(route, route[1:]):
        data = G.get_edge_data(u, v)
        if not data:
            continue
        battery -= edge_weight(list(data.values())[0], "length")
        min_batt = min(min_batt, battery)
        if v in chargers:
            battery = max_m
    return min_batt >= -1e-6, min_batt


def pct(x: float) -> str:
    return f"{x * 100:.4f}%"


def summarize(name: str, errs: List[float]) -> Dict[str, Any]:
    """errs = |craft-opt|/opt per query."""
    if not errs:
        return {"n": 0}
    s = sorted(errs)
    return {
        "n": len(s),
        "MAPE": sum(s) / len(s),
        "median_APE": statistics.median(s),
        "p90_APE": s[int(0.9 * (len(s) - 1))],
        "max_APE": s[-1],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="craft model dirs under outputs/craft/ (default: all with this template)")
    args = ap.parse_args()

    print("loading graph + gold + oracle ...", flush=True)
    import geopandas as gpd
    from craft.generator.tools.common_handlers import resolve_graph_node_id
    from evaluation.oracle.ev_oracle import battery_to_meters

    G = build_graph_from_gpkg(str(GPKG))
    snapper = NodeSnapper.from_gpkg(str(GPKG), G)
    pois = gpd.read_file(str(GPKG), layer="pois")
    chargers = set()
    for val in pois[pois["amenity"].eq("charging_station")]["nearest_node_osmid"].dropna():
        nid = resolve_graph_node_id(G, val)
        if nid is not None:
            chargers.add(nid)
    gold_by_id = {r["query_id"]: r for r in json.loads(GOLD_PATH.read_text())}
    oracle_list = json.loads(ORACLE_RESULTS.read_text())
    oracle_by_id = {r["query_id"]: r for r in oracle_list}

    models = args.models
    if not models:
        models = sorted(d.name for d in CRAFT_ROOT.iterdir()
                        if d.is_dir() and (d / TEMPLATE).is_dir() and not d.name.startswith("_"))
    print(f"models: {models}")

    n_optimal = sum(1 for r in oracle_list if r["status"] == "optimal")
    n_infeasible = len(oracle_list) - n_optimal
    print(f"\noracle: {len(oracle_list)} records = {n_optimal} optimal + {n_infeasible} infeasible")

    # zero-optimum guard (asked for explicitly)
    zeros = [r["query_id"] for r in oracle_list
             if r["status"] == "optimal" and abs(r["optimal_distance_m"] or 0.0) <= ZERO_TOL_M]
    print(f"zero/near-zero optima (would divide by 0, EXCLUDED): {len(zeros)}"
          + (f" -> {zeros}" if zeros else ""))

    report: Dict[str, Any] = {
        "template": TEMPLATE,
        "oracle_total": len(oracle_list),
        "oracle_optimal": n_optimal,
        "oracle_infeasible": n_infeasible,
        "zero_optimum_excluded": zeros,
        "definitions": {
            "craft_objective": "compute_system_objective(route, G, 'length') = realized route cost",
            "MAPE": "mean(|craft-opt|/opt) over oracle-optimal queries CRAFT also solved",
            "headline_subset": "search-only (craft parse == gold parse)",
            "at_optimum_tolerance_m": AT_OPT_TOL_M,
        },
        "models": {},
    }

    for model in models:
        cdir = CRAFT_ROOT / model / TEMPLATE
        search_errs: List[float] = []
        diverged_errs: List[float] = []
        at_opt = 0
        worse: List[Tuple[str, float, float, float]] = []
        craft_no_route = 0
        craft_missing = 0
        craft_route_on_infeasible: List[str] = []
        below_opt: List[Tuple[str, float, float]] = []
        battery_violations: List[Tuple[str, float, float, Optional[float]]] = []

        for orec in oracle_list:
            qid = orec["query_id"]
            p = cdir / f"{qid}.json"
            if not p.exists():
                craft_missing += 1
                continue
            crec = json.loads(p.read_text())
            feasible = craft_feasible(crec)

            if orec["status"] != "optimal":
                # oracle PROVED no battery-feasible chain exists (or gold is
                # unparseable) -- a CRAFT route here cannot be valid.
                if feasible and orec["infeasible_reason"] == "no_feasible_chain_with_any_charger":
                    craft_route_on_infeasible.append(qid)
                continue

            if not feasible:
                craft_no_route += 1
                continue
            opt = orec["optimal_distance_m"]
            if abs(opt) <= ZERO_TOL_M:
                continue  # zero-guard, already reported
            craft_obj = compute_system_objective(crec["result"]["route"], G, "length")
            if craft_obj is None:
                craft_missing += 1
                continue

            ape = abs(craft_obj - opt) / opt
            ok, _reason = parse_matches_gold(crec, gold_by_id[qid], G, snapper)

            # A route that violates the battery constraint is an INVALID
            # answer, not a "gap". Averaging it into MAPE would actively
            # bias the result in CRAFT's favour, since invalid routes are
            # shorter (they skip the charging detour that makes the trip
            # legal). So they are excluded from MAPE and counted separately.
            g_cur, g_max = battery_to_meters(gold_by_id[qid])
            batt_ok = True
            min_batt = None
            if g_cur is not None and g_max is not None:
                route_nodes = [coerce_node(G, n) for n in crec["result"]["route"]]
                batt_ok, min_batt = route_battery_feasible(route_nodes, G, chargers, g_cur, g_max)
            if not batt_ok:
                battery_violations.append((qid, craft_obj, opt, min_batt))
                continue

            if ok:
                search_errs.append(ape)
                if craft_obj <= opt + AT_OPT_TOL_M:
                    at_opt += 1
                else:
                    worse.append((qid, craft_obj, opt, craft_obj / opt))
                if craft_obj < opt - AT_OPT_TOL_M:
                    below_opt.append((qid, craft_obj, opt))
            else:
                diverged_errs.append(ape)

        s_search = summarize("search-only", search_errs)
        s_div = summarize("parse-diverged", diverged_errs)

        print(f"\n{'=' * 70}\nMODEL: {model}\n{'=' * 70}")
        print(f"  craft had no route (oracle-optimal queries): {craft_no_route}")
        print(f"  craft output missing/uncomputable:           {craft_missing}")
        print(f"  craft returned a route where oracle PROVED infeasible: "
              f"{len(craft_route_on_infeasible)}"
              + (f" -> {craft_route_on_infeasible[:5]}" if craft_route_on_infeasible else ""))
        if battery_violations:
            print(f"  !! craft route VIOLATES the battery constraint (EXCLUDED from MAPE): "
                  f"{len(battery_violations)}")
            for qid, c, o, mb in battery_violations[:5]:
                print(f"       {qid}: craft={c:.1f}m opt={o:.1f}m min_battery={mb:.1f}m "
                      f"(negative = ran dry mid-route)")
            print(f"       (constraint_checker now agrees these are infeasible -- fixed 2026-08-10; "
                  f"before the fix it scored them ev_battery_feasible=True)")
        if below_opt:
            print(f"  !! craft BELOW proven optimum on a BATTERY-VALID route "
                  f"(would break oracle soundness): {len(below_opt)}")
            for b in below_opt[:5]:
                print(f"       {b}")

        if s_search["n"]:
            print(f"\n  --- HEADLINE: search-only (craft parse == gold parse), "
                  f"n={s_search['n']} ---")
            print(f"    exactly optimal:  {at_opt}/{s_search['n']} "
                  f"({at_opt / s_search['n'] * 100:.1f}%)")
            print(f"    MAPE:             {pct(s_search['MAPE'])}")
            print(f"    median APE:       {pct(s_search['median_APE'])}")
            print(f"    p90 APE:          {pct(s_search['p90_APE'])}")
            print(f"    max APE:          {pct(s_search['max_APE'])}")
            if worse:
                print(f"    worst offenders (craft / opt):")
                for qid, c, o, r in sorted(worse, key=lambda x: -x[3])[:5]:
                    print(f"      {qid}: craft={c:.1f}m opt={o:.1f}m ratio={r:.4f} "
                          f"(+{(r - 1) * 100:.2f}%)")
        if s_div["n"]:
            print(f"\n  --- parse-diverged (NOT in headline), n={s_div['n']} ---")
            print(f"    MAPE: {pct(s_div['MAPE'])}  max APE: {pct(s_div['max_APE'])}")

        report["models"][model] = {
            "search_only": {**s_search, "exactly_optimal": at_opt},
            "parse_diverged": s_div,
            "craft_no_route": craft_no_route,
            "craft_missing": craft_missing,
            "craft_route_on_proven_infeasible": craft_route_on_infeasible,
            "craft_below_optimum": [b[0] for b in below_opt],
            "craft_battery_violations": [
                {"query_id": q, "craft_m": c, "opt_m": o, "min_battery_m": mb}
                for q, c, o, mb in battery_violations],
        }

    OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
