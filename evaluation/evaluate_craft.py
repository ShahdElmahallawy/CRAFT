"""
evaluate_craft.py — Export Craft pipeline results to Excel.

All constraint checks and the comparable objective are computed by the shared
evaluation.constraint_checker, so columns match the LLM and LLMAP exports.

Two objective columns:
  objective_value          — leg-by-leg SP through the *intended* POI stops
                             (same computation as the baselines; comparable).
  dijkstra_objective_value — cost of Craft's realized Dijkstra route (same weight).

Usage:
  python evaluation/evaluate_craft.py
  python evaluation/evaluate_craft.py --models gemini-3_1-flash-lite
  python evaluation/evaluate_craft.py --no-sp
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluation.constraint_checker as cc
from evaluation.evaluate_llm import _write_table

CRAFT_DIR    = ROOT / "outputs" / "craft"
QUERIES_DIR  = ROOT / "benchmark" / "socal_queries" / "set1"
DEFAULT_GPKG = ROOT / "data" / "socal" / "road_network.gpkg"
DEFAULT_OUT  = ROOT / "outputs" / "craft_results.xlsx"

MODELS = [
    "gemini-3_1-flash-lite",
    "gemini-3_5-flash",
    "gemini-3_1-pro-preview",
    "Qwen_Qwen3-32B-AWQ",
]
# set1 active templates only (200 each = 1000 total)
TEMPLATES = ["ev_routing", "ordered_poi", "poi_collection", "ev+ordered_poi", "unordered_poi"]


def _exc_line(err) -> Optional[str]:
    """Last non-empty line of a traceback (the actual exception message)."""
    if not err:
        return None
    lines = [ln.strip() for ln in str(err).splitlines() if ln.strip()]
    return lines[-1][:300] if lines else None


def _craft_error(gen: dict, exe: dict):
    """(error_stage, error) — a processing failure, separate from feasibility.
    A clean run that simply found no route returns ('none', None)."""
    if gen and gen.get("ok") is False:
        return "generate", _exc_line(gen.get("error"))
    if exe and exe.get("ok") is False:
        return "execute", _exc_line(exe.get("error"))
    return "none", None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",          default=str(DEFAULT_OUT))
    ap.add_argument("--models",       nargs="+", default=MODELS)
    ap.add_argument("--templates",    nargs="+", default=TEMPLATES)
    ap.add_argument("--gpkg",         default=str(DEFAULT_GPKG))
    ap.add_argument("--baseline-dir", default=str(CRAFT_DIR))
    ap.add_argument("--queries-dir",  default=str(QUERIES_DIR))
    ap.add_argument("--no-sp",        action="store_true", help="Skip SP/graph computation")
    args = ap.parse_args()

    baseline_dir = Path(args.baseline_dir)
    queries_dir  = Path(args.queries_dir)

    G = None
    pois_gdf = None
    snapper = None
    caches = None
    if not args.no_sp:
        print("Loading graph + POIs for constraint/objective computation ...")
        import geopandas as gpd
        from craft.generator.tools.common_handlers import build_graph_from_gpkg
        G = build_graph_from_gpkg(args.gpkg)
        pois_gdf = gpd.read_file(args.gpkg, layer="pois")
        snapper = cc.NodeSnapper.from_gpkg(args.gpkg, G)
        print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(pois_gdf)} POIs")
        print("Precomputing POI-type + charger node caches (one pass) ...")
        caches = cc.build_poi_caches(G, pois_gdf)
        print(f"  cached {sum(len(v) for v in caches[0].values())} type-nodes, {len(caches[1])} chargers")

    nl_index: Dict[str, str] = {}
    valid_ids: set = set()
    for tmpl in args.templates:
        q_file = queries_dir / f"{tmpl}.json"
        if q_file.exists():
            for q in json.loads(q_file.read_text()):
                nl_index[q["query_id"]] = q["nl_query"]
                valid_ids.add(q["query_id"])

    rows = []
    for model in args.models:
        model_dir = baseline_dir / model
        if not model_dir.exists():
            print(f"[skip] {model} — no craft output dir")
            continue

        n_loaded = 0
        print(f"Loading {model} ...")
        for tmpl in args.templates:
            tmpl_dir = model_dir / tmpl
            if not tmpl_dir.exists():
                continue
            for fpath in sorted(tmpl_dir.glob("*.json")):
                try:
                    d = json.loads(fpath.read_text())
                except Exception as exc:
                    print(f"  [warn] {fpath.name}: {exc}")
                    continue

                qid      = d.get("query_id", fpath.stem)
                # Skip records not in the set1 ID set (stale rows from old runs)
                if valid_ids and qid not in valid_ids:
                    continue
                nl_query = d.get("nl_query") or nl_index.get(qid, "")
                metrics  = d.get("metrics") or {}
                result   = d.get("result") or {}
                llm_meta = d.get("llm_meta") or {}
                gen_info = d.get("generate") or {}
                exe_info = d.get("execute") or {}

                ran_ok   = bool(metrics.get("ran"))
                feasible = bool(metrics.get("feasible"))
                # Only use route from result when the run actually executed; a
                # non-empty route with ran=False is a partial/stale artifact.
                route    = result.get("route") if (isinstance(result, dict) and ran_ok) else None

                core = cc.evaluate_system(nl_query, tmpl, route or [], G, pois_gdf, snapper, caches=caches,
                                          collected_pois=result.get("collected_pois"))
                # Mask out constraint results for runs that never executed
                if not ran_ok:
                    core = {k: None for k in core}

                tok = llm_meta.get("tokens") or {}
                error_stage, error = _craft_error(gen_info, exe_info)
                solved_correctly = bool(feasible and core.get("constraints_satisfied") is True)
                row = {
                    "query_id":     qid,
                    "template":     tmpl,
                    "model":        model,
                    "feasible":     feasible,
                    "ran":          ran_ok,
                    "solved_correctly": solved_correctly,
                    "generate_ok":  bool(gen_info.get("ok")) if gen_info else False,
                    "execute_ok":   bool(exe_info.get("ok")) if exe_info else False,
                    "error_stage":  error_stage,
                    "error":        error,
                    "search_time_ms": metrics.get("run_time_ms"),
                    # Search-effort counters — only populated for runs made
                    # after 2026-08-05 (None for older records).
                    "states_expanded":        metrics.get("states_expanded"),
                    "states_expanded_stale":  metrics.get("states_expanded_stale"),
                    "states_expanded_useful": metrics.get("states_expanded_useful"),
                    "states_generated":       metrics.get("states_generated"),
                    "states_pruned":          metrics.get("states_pruned"),
                    "peak_heap_size":         metrics.get("peak_heap_size"),
                    **{k: core[k] for k in cc.CORE_FIELDS},
                    "dijkstra_objective_value": core["dijkstra_objective_value"],
                    "distance_limits_met_realized": core.get("distance_limits_met_realized"),
                    "time_budget_met_realized":     core.get("time_budget_met_realized"),
                    "road_preference_met_realized": core.get("road_preference_met_realized"),
                    **{k: core.get(k) for k in cc.ROAD_PREF_METRIC_FIELDS_REALIZED},
                    "latency_s":        gen_info.get("elapsed_s"),
                    "prompt_tokens":    tok.get("prompt_tokens"),
                    "completion_tokens": tok.get("completion_tokens"),
                }
                rows.append(row)
                n_loaded += 1
        print(f"  {n_loaded} records loaded")

    if not rows:
        sys.exit("No records found")

    out = Path(args.out)
    if out.suffix.lower() != ".xlsx":
        out = out.with_suffix(".xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    from openpyxl import Workbook
    wb = Workbook()
    ws_data = wb.active
    ws_data.title = "Results"
    _write_table(ws_data, fieldnames,
                 [[r[f] for f in fieldnames] for r in rows],
                 title="Craft Results (Schema-Driven Dijkstra + LLM Code Generation)")

    ws_legend = wb.create_sheet("Column Descriptions")
    legend = [
        ("query_id", "Unique identifier for the benchmark query"),
        ("template", "Benchmark template family"),
        ("model",    "LLM used for code generation"),
        ("feasible", "True if the Dijkstra search returned a valid route"),
        ("ran",      "True if generated.py executed without crashing"),
        ("solved_correctly", "True iff feasible AND constraints_satisfied is True — the project's headline "
                              "comparison metric (see CLAUDE.md EVALUATION & SCORING). Do not compare systems "
                              "on the raw feasible column or the constraint_satisfaction mean instead."),
        ("generate_ok", "True if the LLM code-generation step succeeded"),
        ("execute_ok",  "True if generated.py ran without errors"),
        ("error_stage", "Stage that threw an exception: 'generate', 'execute', or 'none'. Reported separately from feasibility — error_stage='none' with feasible=False means the search ran cleanly but found no route (a genuine infeasible, not an error)."),
        ("error",       "Exception message (last traceback line) from the failing stage; None if no exception occurred"),
        ("search_time_ms", "Dijkstra search time in milliseconds (search only — EXCLUDES graph loading)"),
        ("states_expanded", "States popped from the open list and expanded (standard search-effort measure). "
                            "Counts states, not graph nodes — one node recurs with different battery levels. "
                            "Blank for runs made before 2026-08-05."),
        ("states_expanded_stale", "Subset of states_expanded that had already been superseded by a dominating "
                                  "state when expanded (wasted work; dominance is checked at insertion, with no "
                                  "lazy deletion, so such states stay on the heap)"),
        ("states_expanded_useful", "states_expanded minus states_expanded_stale"),
        ("states_generated", "Successors produced by expand(), before dominance filtering"),
        ("states_pruned", "Successors discarded by the dominance check (never reached the open list)"),
        ("peak_heap_size", "Largest open-list (frontier) size observed during the search"),
        *cc.CORE_FIELD_DESCRIPTIONS,
        ("dijkstra_objective_value", "Cost of Craft's realized Dijkstra route with the fixed per-template weight (meters or seconds)"),
        ("route_segments_realized", "Craft-only: road segments in CRAFT's OWN realized route. The *_realized "
                                     "family measures what CRAFT's generated code actually did — prefer it "
                                     "whenever CRAFT is reported on its own; the unsuffixed columns measure a "
                                     "reconstructed SP and exist only for comparability with LLM/LLMAP."),
        ("hwy_segments_realized", "Craft-only: how many of CRAFT's OWN route segments run on a highway "
                                   "(motorway/trunk/primary, incl. _link). The per-route penalty count."),
        ("hwy_segment_share_realized", "Craft-only: hwy_segments_realized / route_segments_realized."),
        ("hwy_distance_share_realized", "Craft-only: share of CRAFT's OWN realized route distance on highways."),
        ("hwy_distance_share_neutral_realized", "Craft-only: highway distance share of the NEUTRAL "
                                                 "(no-preference) SP through the same intended stops — the "
                                                 "counterfactual CRAFT's realized route is judged against."),
        ("hwy_fully_avoided_realized", "Craft-only: True if CRAFT's OWN route uses zero highway segments."),
        ("road_preference_met_realized", "Craft-only: road_preference_met scored on CRAFT's OWN realized route "
                                          "(the real capability signal), vs. the shared road_preference_met "
                                          "column which now uses the same reconstructed-SP method as LLM/LLMAP "
                                          "for a fair cross-system comparison. None if not a road-preference query."),
        ("latency_s", "Wall-clock time for the LLM code-generation subprocess (seconds)"),
        ("prompt_tokens", "Prompt tokens used for code generation"),
        ("completion_tokens", "Completion tokens generated"),
    ]
    _write_table(ws_legend, ["Column", "Description"], legend, title="Column Descriptions")

    wb.save(out)
    print(f"Wrote {len(rows)} rows → {out}")

    # Console summary
    by_model: Dict[str, List] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    print(f"\n{'Model':<30} {'N':>5} {'Ran%':>6} {'Feas%':>7} {'AvgCS%':>8} {'SearchMs':>10}")
    print("-" * 70)
    for model in dict.fromkeys(r["model"] for r in rows):
        rs = by_model[model]
        n = len(rs)
        n_ran = sum(1 for r in rs if r["ran"])
        n_feas = sum(1 for r in rs if r["feasible"])
        cs_vals = [float(r["constraint_satisfaction"]) for r in rs
                   if r["constraint_satisfaction"] not in (None, "", "None")]
        ms_vals = [float(r["search_time_ms"]) for r in rs
                   if r["search_time_ms"] not in (None, "", "None")]
        avg_cs = f"{sum(cs_vals)/len(cs_vals)*100:.1f}%" if cs_vals else "N/A"
        avg_ms = f"{sum(ms_vals)/len(ms_vals):.1f}ms" if ms_vals else "N/A"
        print(f"{model:<30} {n:>5} {100*n_ran/n:>5.1f}% {100*n_feas/n:>6.1f}% {avg_cs:>8} {avg_ms:>10}")


if __name__ == "__main__":
    main()
