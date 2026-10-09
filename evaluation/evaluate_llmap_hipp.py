"""
evaluate_llmap_hipp.py — Export LLMAP-on-HIPP results to Excel.

Sibling of evaluate_llmap.py, kept SEPARATE so the HIPP pilot never touches
the production set1 LLMAP evaluation path. Reads
outputs/llmap/{model}/{hipp_ordered|hipp_unordered}/{query_id}.json (written
by scripts/run_llmap.py --queries-dir benchmark/socal_queries/llmap_queries) and scores
each record's matched_stops against its hipp_meta.json gold record via
evaluation.constraint_checker.evaluate_hipp_baseline.

Usage:
  conda run -n myenv python evaluation/evaluate_llmap_hipp.py
  conda run -n myenv python evaluation/evaluate_llmap_hipp.py --no-sp
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

BASELINE_DIR = ROOT / "outputs" / "llmap"
DEFAULT_OUT  = ROOT / "outputs" / "llmap_hipp_results.xlsx"
DEFAULT_GPKG = ROOT / "data" / "socal" / "road_network.gpkg"

MODELS = [
    "gemini-3_1-flash-lite",
]
TEMPLATES = list(cc.HIPP_TEMPLATES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--gpkg", default=str(DEFAULT_GPKG))
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--templates", nargs="+", default=TEMPLATES)
    ap.add_argument("--baseline-dir", default=str(BASELINE_DIR))
    ap.add_argument("--no-sp", action="store_true", help="Skip SP/graph computation")
    args = ap.parse_args()

    baseline_dir = Path(args.baseline_dir)

    print("Loading HIPP gold (hipp_meta.json) ...")
    gold_by_qid = cc.load_hipp_gold()
    print(f"  {len(gold_by_qid)} gold records")

    G = None
    pois_gdf = None
    snapper = None
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
        for tmpl in args.templates:
            tmpl_dir = model_dir / tmpl
            if not tmpl_dir.exists():
                continue
            for fpath in sorted(tmpl_dir.glob("*.json")):
                d = json.loads(fpath.read_text())
                meta = d.get("llmap_meta") or {}

                status = meta.get("mode") or meta.get("status", "")
                if not status:
                    continue

                qid = d.get("query_id", fpath.stem)
                gold = gold_by_qid.get(qid)
                if gold is None:
                    print(f"  [warn] {qid}: no hipp_meta gold record — skipping")
                    continue

                matched_stops = d.get("matched_stops", []) or []
                core = cc.evaluate_hipp_baseline(
                    gold, matched_stops, G, snapper, rating_cache=rating_cache)

                parser_out = meta.get("parser_output") or {}
                parsed_pois = ",".join(parser_out.get("pois") or [])

                rows.append({
                    "query_id": qid,
                    "template": tmpl,
                    "model":    model,
                    "time_limit":      gold.get("time_limit"),
                    "quality_weight":  gold.get("quality_weight"),
                    "distance_weight": gold.get("distance_weight"),
                    "gold_weights_used": str(meta.get("gold_weights_used")),
                    "feasible": bool(meta.get("feasible")) if meta.get("feasible") is not None else False,
                    "status":         meta.get("status") or meta.get("mode"),
                    "parser_failed":  bool(meta.get("parser_failed")) if meta.get("parser_failed") is not None else None,
                    "search_time_ms": meta.get("msgs_runtime_ms"),
                    "parsed_pois":    parsed_pois,
                    **{k: core[k] for k in cc.CORE_FIELDS},
                })

    if not rows:
        sys.exit("No LLMAP HIPP output files found in " + str(baseline_dir))

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
                 title="LLMAP-on-HIPP Results")

    ws_legend = wb.create_sheet("Column Descriptions")
    legend = [
        ("query_id", "HIPP query id (hipp_NNNN)"),
        ("template", "hipp_ordered (has dependencies) or hipp_unordered"),
        ("model",    "LLM used as LLMAP's HIPP parser"),
        ("time_limit", "Gold clock deadline string ('HH:00' or 'None')"),
        ("quality_weight", "Gold LLMAP quality_weight (rating emphasis, 0-1)"),
        ("distance_weight", "Gold LLMAP distance_weight (efficiency emphasis, 0-1)"),
        ("gold_weights_used", "(quality_weight, distance_weight) actually fed into MSGS's alpha/beta — see scripts/run_llmap.py fairness fix"),
        ("feasible", "Whether MSGS found a valid route through candidate POIs"),
        ("status", "LLMAP processing status, separate from feasibility"),
        ("parser_failed", "True if LLMAP's HIPP parser failed for this query"),
        ("search_time_ms", "MSGS graph search time in milliseconds"),
        ("parsed_pois", "POI types LLMAP's own parser extracted from the NL query"),
        *cc.CORE_FIELD_DESCRIPTIONS,
    ]
    _write_table(ws_legend, ["Column", "Description"], legend, title="Column Descriptions")

    wb.save(out)
    print(f"Wrote {len(rows)} rows → {out}")

    by_model: Dict[str, List] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    print(f"\n{'Model':<24} {'N':>5} {'Feasible%':>10} {'ConstrSat%':>11}")
    print("-" * 55)
    for model in args.models:
        rs = by_model.get(model)
        if not rs:
            continue
        n = len(rs)
        feas = sum(1 for r in rs if str(r["feasible"]) == "True") / n * 100
        cs_vals = [1 for r in rs if r.get("constraints_satisfied") is True]
        print(f"{model:<24} {n:>5} {feas:>9.1f}% {100*len(cs_vals)/n:>10.1f}%")


if __name__ == "__main__":
    main()
