"""
evaluate_llm_hipp.py — Export LLM-baseline-on-HIPP results to Excel.

Sibling of evaluate_llm.py, kept SEPARATE so the HIPP pilot never touches the
production set1 LLM-baseline evaluation path. Reads
outputs/llm/{model}/{hipp_ordered|hipp_unordered}/{query_id}.json (written by
scripts/run_llm.py --queries-dir benchmark/socal_queries/llmap_queries) and scores each
record's matched_stops against its hipp_meta.json gold record via
evaluation.constraint_checker.evaluate_hipp_baseline.

Hallucination columns (any_hallucinated / hallucination_reasons) are merged
in from outputs/evaluation/llm_hallucination/, reusing evaluate_llm.py's
index — populated for HIPP via
  evaluation/evaluate_llm_hallucination.py --baseline-dir outputs/llm
    --queries-dir benchmark/socal_queries/llmap_queries --templates hipp_ordered hipp_unordered
HIPP query_ids (hipp_NNNN) never collide with set1 ids, so the shared
index merges both sets safely.

Usage:
  conda run -n myenv python evaluation/evaluate_llm_hipp.py
  conda run -n myenv python evaluation/evaluate_llm_hipp.py --no-sp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluation.constraint_checker as cc
from evaluation.evaluate_llm import (
    _write_table, _match_methods, _load_hallucination_index, _hallucination_reasons,
)

BASELINE_DIR = ROOT / "outputs" / "llm"
DEFAULT_OUT  = ROOT / "outputs" / "llm_hipp_results.xlsx"
DEFAULT_GPKG = ROOT / "data" / "socal" / "road_network.gpkg"

MODELS = [
    "gemini-3_1-flash-lite",
]
TEMPLATES = list(cc.HIPP_TEMPLATES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",          default=str(DEFAULT_OUT))
    ap.add_argument("--gpkg",         default=str(DEFAULT_GPKG))
    ap.add_argument("--models",       nargs="+", default=MODELS)
    ap.add_argument("--templates",    nargs="+", default=TEMPLATES)
    ap.add_argument("--baseline-dir", default=str(BASELINE_DIR))
    ap.add_argument("--no-sp",        action="store_true", help="Skip SP/graph computation")
    args = ap.parse_args()

    baseline_dir = Path(args.baseline_dir)

    print("Loading HIPP gold (hipp_meta.json) ...")
    gold_by_qid = cc.load_hipp_gold()
    print(f"  {len(gold_by_qid)} gold records")

    print("Loading hallucination evaluation index ...")
    hall_index = _load_hallucination_index()
    print(f"  {sum(len(v) for v in hall_index.values())} records across {len(hall_index)} models")

    G = None
    snapper = None
    pois_gdf = None
    rating_cache: Dict = {}
    if not args.no_sp:
        print("Loading graph and POIs ...")
        import geopandas as gpd
        from craft.generator.tools.common_handlers import build_graph_from_gpkg
        G = build_graph_from_gpkg(args.gpkg)
        pois_gdf = gpd.read_file(args.gpkg, layer="pois")
        snapper = cc.NodeSnapper.from_gpkg(args.gpkg, G)
        rating_cache = cc.build_hipp_rating_cache(G, pois_gdf)
        print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
              f"{len(rating_cache)} rated nodes")

    rows = []
    for model in args.models:
        model_dir = baseline_dir / model
        if not model_dir.exists():
            print(f"[skip] {model} — no output dir")
            continue

        n_loaded = 0
        print(f"Loading {model} ...")
        for tmpl in args.templates:
            tmpl_dir = model_dir / tmpl
            if not tmpl_dir.exists():
                continue
            for fpath in sorted(tmpl_dir.glob("*.json")):
                d = json.loads(fpath.read_text())
                qid = d.get("query_id", fpath.stem)
                gold = gold_by_qid.get(qid)
                if gold is None:
                    print(f"  [warn] {qid}: no hipp_meta gold record — skipping")
                    continue

                llm_stops     = d.get("llm_stops", []) or []
                matched_stops = d.get("matched_stops", []) or []
                dropped_stops = d.get("dropped_stops", []) or []

                n_proposed = len(llm_stops)
                n_matched  = len(matched_stops)
                n_dropped  = len(dropped_stops)
                match_rate = round(n_matched / n_proposed, 4) if n_proposed > 0 else None

                core = cc.evaluate_hipp_baseline(
                    gold, matched_stops, G, snapper, rating_cache=rating_cache)

                llm_types = [cc.normalize_type(s.get("type") or s.get("poi_type"))
                             for s in llm_stops if s.get("type") or s.get("poi_type")]
                parsed_pois = ",".join(dict.fromkeys(t for t in llm_types if t))

                hall_rec   = hall_index.get(model, {}).get(qid, {})
                hall_drops = hall_rec.get("dropped_stops", [])
                any_hall   = bool(hall_rec.get("any_hallucinated")) if hall_rec else None
                hall_why   = _hallucination_reasons(hall_drops) if hall_rec else None

                row = {
                    "query_id": qid,
                    "template": tmpl,
                    "model":    model,
                    "time_limit":      gold.get("time_limit"),
                    "quality_weight":  gold.get("quality_weight"),
                    "distance_weight": gold.get("distance_weight"),
                    "feasible": d.get("llm_feasible"),
                    "error":    d.get("error"),
                    "infeasible_reason": d.get("llm_infeasible_reason"),
                    "latency_s": d.get("latency_s"),
                    "parsed_pois": parsed_pois,
                    **{k: core[k] for k in cc.CORE_FIELDS},
                    "n_proposed":    n_proposed,
                    "n_matched":     n_matched,
                    "n_dropped":     n_dropped,
                    "match_rate":    match_rate,
                    "match_methods": _match_methods(matched_stops),
                    "any_hallucinated":      any_hall,
                    "hallucination_reasons": hall_why,
                }
                rows.append(row)
                n_loaded += 1
        print(f"  {n_loaded} records loaded")

    if not rows:
        sys.exit("No LLM-baseline HIPP output files found in " + str(baseline_dir))

    out = Path(args.out)
    if out.suffix.lower() not in (".xlsx", ".xls"):
        out = out.with_suffix(".xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    from openpyxl import Workbook
    wb = Workbook()
    ws_data = wb.active
    ws_data.title = "Results"
    _write_table(ws_data, fieldnames,
                 [[r[f] for f in fieldnames] for r in rows],
                 title="LLM-Baseline-on-HIPP Results")

    ws_legend = wb.create_sheet("Column Descriptions")
    legend = [
        ("query_id", "HIPP query id (hipp_NNNN)"),
        ("template", "hipp_ordered (has dependencies) or hipp_unordered"),
        ("model",    "LLM used for the bare LLM baseline"),
        ("time_limit", "Gold clock deadline string ('HH:00' or 'None')"),
        ("quality_weight", "Gold LLMAP quality_weight (rating emphasis, 0-1)"),
        ("distance_weight", "Gold LLMAP distance_weight (efficiency emphasis, 0-1)"),
        ("feasible", "Whether the LLM declared its route feasible"),
        ("error", "Processing/pipeline error string; None if it ran cleanly"),
        ("infeasible_reason", "The reason text the LLM gave when declaring the route infeasible"),
        ("latency_s", "LLM API call wall time (seconds)"),
        ("parsed_pois", "Unique POI types the LLM proposed, in first-seen order"),
        *cc.CORE_FIELD_DESCRIPTIONS,
        ("n_proposed", "Total stops the LLM proposed"),
        ("n_matched",  "Proposed stops matched to a dataset POI"),
        ("n_dropped",  "Proposed stops that could not be matched"),
        ("match_rate", "n_matched / n_proposed"),
        ("match_methods", "Pipe-separated set of matching methods used across matched stops"),
        ("any_hallucinated", "True if ≥1 dropped stop was verified hallucinated via Google Places; None if not evaluated"),
        ("hallucination_reasons", "Unique hallucination types for dropped stops (address_not_found, place_not_found, wrong_location, type_mismatch)"),
    ]
    _write_table(ws_legend, ["Column", "Description"], legend, title="Column Descriptions")

    wb.save(out)
    print(f"Wrote {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
