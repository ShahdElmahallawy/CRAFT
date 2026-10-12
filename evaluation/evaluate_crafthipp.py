"""
evaluate_crafthipp.py — Export craft_hipp pipeline results to Excel.

Sibling of evaluate_craft.py, kept as a SEPARATE script (not a flag on the
set1 evaluator) so the HIPP pilot never touches the production craft/
evaluation path. Reads outputs/crafthipp/{model}/{template}/{query_id}.json
(written by scripts/run_crafthipp.py) and scores each record against its
benchmark/socal_queries/llmap_queries/hipp_meta.json gold record via
evaluation.constraint_checker.evaluate_hipp_system — gold-driven, not
NL-regex-driven (see constraint_checker.py's HIPP section for why).

Usage:
  python evaluation/evaluate_crafthipp.py
  python evaluation/evaluate_crafthipp.py --models gemini-3_1-flash-lite
  python evaluation/evaluate_crafthipp.py --no-sp
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluation.constraint_checker as cc
from evaluation.evaluate_llm import _write_table

CRAFTHIPP_DIR = ROOT / "outputs" / "crafthipp"
DEFAULT_GPKG  = ROOT / "data" / "socal" / "road_network.gpkg"
DEFAULT_OUT   = ROOT / "outputs" / "craft_hipp_results.xlsx"

MODELS = [
    "gemini-3_1-flash-lite",
]
TEMPLATES = list(cc.HIPP_TEMPLATES)

# The 5 HIPP POI categories. build_poi_caches() defaults to
# _DISPLAY_TO_CATEGORY.values() (pharmacy/bank/supermarket/library) which is
# MISSING "mall" — passing this explicitly is required, or every HIPP query's
# mall stop silently fails to match (evaluate_hipp_system prefers a passed-in
# `caches` over building its own HIPP-aware one).
HIPP_POI_TYPES = ["bank", "library", "pharmacy", "supermarket", "mall"]


def _exc_line(err) -> Optional[str]:
    if not err:
        return None
    lines = [ln.strip() for ln in str(err).splitlines() if ln.strip()]
    return lines[-1][:300] if lines else None


def _craft_error(gen: dict, exe: dict):
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
    ap.add_argument("--baseline-dir", default=str(CRAFTHIPP_DIR))
    ap.add_argument("--no-sp",        action="store_true", help="Skip SP/graph computation")
    args = ap.parse_args()

    baseline_dir = Path(args.baseline_dir)

    print("Loading HIPP gold (hipp_meta.json) ...")
    gold_by_qid = cc.load_hipp_gold()
    print(f"  {len(gold_by_qid)} gold records")

    G = None
    pois_gdf = None
    snapper = None
    caches = None
    rating_cache: Dict = {}
    if not args.no_sp:
        print("Loading graph + POIs for constraint/objective computation ...")
        import geopandas as gpd
        from craft.generator.tools.common_handlers import build_graph_from_gpkg
        G = build_graph_from_gpkg(args.gpkg)
        pois_gdf = gpd.read_file(args.gpkg, layer="pois")
        snapper = cc.NodeSnapper.from_gpkg(args.gpkg, G)
        print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(pois_gdf)} POIs")
        print("Precomputing POI-type caches + rating cache (one pass each) ...")
        caches = cc.build_poi_caches(G, pois_gdf, poi_types=HIPP_POI_TYPES)
        rating_cache = cc.build_hipp_rating_cache(G, pois_gdf)
        print(f"  cached {sum(len(v) for v in caches[0].values())} type-nodes, "
              f"{len(rating_cache)} rated nodes")

    rows = []
    for model in args.models:
        model_dir = baseline_dir / model
        if not model_dir.exists():
            print(f"[skip] {model} — no crafthipp output dir")
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

                qid = d.get("query_id", fpath.stem)
                gold = gold_by_qid.get(qid)
                if gold is None:
                    print(f"  [warn] {qid}: no hipp_meta gold record — skipping")
                    continue

                metrics  = d.get("metrics") or {}
                result   = d.get("result") or {}
                gen_info = d.get("generate") or {}
                exe_info = d.get("execute") or {}

                ran_ok   = bool(metrics.get("ran"))
                feasible = bool(metrics.get("feasible"))
                route    = result.get("route") if (isinstance(result, dict) and ran_ok) else None

                core = cc.evaluate_hipp_system(
                    gold, route or [], G, pois_gdf, snapper,
                    caches=caches, rating_cache=rating_cache,
                )
                if not ran_ok:
                    core = {k: None for k in core}

                error_stage, error = _craft_error(gen_info, exe_info)
                row = {
                    "query_id":       qid,
                    "template":       tmpl,
                    "model":          model,
                    "hipp_index":     gold.get("hipp_index"),
                    "time_limit":     gold.get("time_limit"),
                    "quality_weight": gold.get("quality_weight"),
                    "distance_weight": gold.get("distance_weight"),
                    "feasible":       feasible,
                    "ran":            ran_ok,
                    "generate_ok":    bool(gen_info.get("ok")) if gen_info else False,
                    "execute_ok":     bool(exe_info.get("ok")) if exe_info else False,
                    "error_stage":    error_stage,
                    "error":          error,
                    "search_time_ms": metrics.get("run_time_ms"),
                    **{k: core[k] for k in cc.CORE_FIELDS},
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
                 title="craft_hipp Results (HIPP pilot — rating/deadline/order)")

    ws_legend = wb.create_sheet("Column Descriptions")
    legend = [
        ("query_id", "HIPP query id (hipp_NNNN)"),
        ("template", "hipp_ordered (has dependencies) or hipp_unordered"),
        ("model",    "LLM used for code generation"),
        ("hipp_index", "Index into the original LLMAP HIPP.json dataset"),
        ("time_limit", "Gold clock deadline string ('HH:00' or 'None')"),
        ("quality_weight", "Gold LLMAP quality_weight (rating emphasis, 0-1)"),
        ("distance_weight", "Gold LLMAP distance_weight (efficiency emphasis, 0-1)"),
        ("feasible", "True if the Dijkstra search returned a valid route"),
        ("ran",      "True if generated.py executed without crashing"),
        ("generate_ok", "True if the LLM code-generation step succeeded"),
        ("execute_ok",  "True if generated.py ran without errors"),
        ("error_stage", "Stage that threw an exception: 'generate', 'execute', or 'none'"),
        ("error",       "Exception message (last traceback line); None if no exception occurred"),
        ("search_time_ms", "Dijkstra search time in milliseconds"),
        *cc.CORE_FIELD_DESCRIPTIONS,
    ]
    _write_table(ws_legend, ["Column", "Description"], legend, title="Column Descriptions")

    wb.save(out)
    print(f"Wrote {len(rows)} rows → {out}")

    # Console summary
    by_model: Dict[str, List] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    print(f"\n{'Model':<24} {'N':>5} {'Ran%':>6} {'Feas%':>7} {'CS%':>6} {'RatingOK%':>10} {'DeadlineOK%':>12}")
    print("-" * 76)
    for model in dict.fromkeys(r["model"] for r in rows):
        rs = by_model[model]
        n = len(rs)
        n_ran  = sum(1 for r in rs if r["ran"])
        n_feas = sum(1 for r in rs if r["feasible"])
        cs_vals = [1 for r in rs if r.get("constraints_satisfied") is True]
        rating_vals = [r["rating_ok"] for r in rs if r.get("rating_ok") is not None]
        deadline_vals = [r["clock_deadline_met"] for r in rs if r.get("clock_deadline_met") is not None]
        rating_pct = f"{100*sum(1 for v in rating_vals if v)/len(rating_vals):.1f}%" if rating_vals else "n/a"
        deadline_pct = f"{100*sum(1 for v in deadline_vals if v)/len(deadline_vals):.1f}%" if deadline_vals else "n/a"
        print(f"{model:<24} {n:>5} {100*n_ran/n:>5.1f}% {100*n_feas/n:>6.1f}% "
              f"{100*len(cs_vals)/n:>5.1f}% {rating_pct:>10} {deadline_pct:>12}")


if __name__ == "__main__":
    main()
