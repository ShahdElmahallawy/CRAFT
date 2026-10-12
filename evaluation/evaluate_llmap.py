"""
evaluate_llmap.py — Export LLMAP baseline results to Excel.

All constraint checks and the comparable objective are computed by the shared
evaluation.constraint_checker, so columns match the LLM and Craft exports.

Usage:
  python evaluation/evaluate_llmap.py
  python evaluation/evaluate_llmap.py --out outputs/llmap_results.xlsx
  python evaluation/evaluate_llmap.py --no-sp   # Tier-1 only
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
QUERIES_DIR  = ROOT / "benchmark" / "socal_queries" / "set1"
DEFAULT_OUT  = ROOT / "outputs" / "llmap_results.xlsx"
DEFAULT_GPKG = ROOT / "data" / "socal" / "road_network.gpkg"

MODELS = [
    "gemini-3_1-flash-lite",
    "gemini-3_5-flash",
    "gemini-3_1-pro-preview",
    "Qwen_Qwen3-32B-AWQ",
]
TEMPLATES = list(cc.TEMPLATES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--gpkg", default=str(DEFAULT_GPKG))
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--templates", nargs="+", default=TEMPLATES)
    ap.add_argument("--baseline-dir", default=str(BASELINE_DIR))
    ap.add_argument("--queries-dir",  default=str(QUERIES_DIR))
    ap.add_argument("--no-sp", action="store_true", help="Skip SP/graph computation")
    args = ap.parse_args()

    baseline_dir = Path(args.baseline_dir)
    queries_dir  = Path(args.queries_dir)

    nl_index: Dict[str, str] = {}
    for tmpl in args.templates:
        q_file = queries_dir / f"{tmpl}.json"
        if q_file.exists():
            for q in json.loads(q_file.read_text()):
                nl_index[q["query_id"]] = q["nl_query"]

    G = None
    snapper = None
    if not args.no_sp:
        print("Loading graph and node index ...")
        from craft.generator.tools.common_handlers import build_graph_from_gpkg
        G = build_graph_from_gpkg(args.gpkg)
        snapper = cc.NodeSnapper.from_gpkg(args.gpkg, G)
        print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

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

                # Skip records never attempted; keep "no_poi_types_found" (EV
                # queries LLMAP can't parse) as attempted failures.
                status = meta.get("mode") or meta.get("status", "")
                if not status:
                    continue

                qid      = d.get("query_id", fpath.stem)
                nl_query = d.get("nl_query") or nl_index.get(qid, "")
                matched_stops = d.get("matched_stops", []) or []

                core = cc.evaluate_baseline(nl_query, tmpl, matched_stops, G, snapper)

                parser_out  = meta.get("parser_output") or {}
                parsed_pois = ",".join(parser_out.get("pois") or [])

                rows.append({
                    "query_id":   qid,
                    "template":   tmpl,
                    "model":      model,
                    "feasible":   bool(meta.get("feasible")) if meta.get("feasible") is not None else False,
                    "status":         meta.get("status") or meta.get("mode"),
                    "parser_failed":  bool(meta.get("parser_failed")) if meta.get("parser_failed") is not None else None,
                    "search_time_ms": meta.get("msgs_runtime_ms"),
                    "parsed_pois":    parsed_pois,
                    **{k: core[k] for k in cc.CORE_FIELDS},
                })

    if not rows:
        sys.exit("No full_pipeline output files found.")

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
                 title="LLMAP Baseline Results (full_pipeline only)")

    ws_legend = wb.create_sheet("Column Descriptions")
    legend = [
        ("query_id", "Unique identifier for the benchmark query"),
        ("template", "Benchmark template family"),
        ("model",    "LLM used as LLMAP's HIPP parser"),
        ("feasible", "Whether MSGS found a valid route through candidate POIs"),
        ("status", "LLMAP processing status, separate from feasibility: 'ok' = parsed and attempted; 'no_poi_types_found' = parser could not extract any supported POI type (e.g. EV-only queries), so routing was never attempted"),
        ("parser_failed", "True if LLMAP's HIPP parser failed for this query; None if not recorded. Distinct from feasible — a parser failure is an error, not a clean 'no route found'"),
        ("search_time_ms", "MSGS graph search time in milliseconds (excludes LLM parse time)"),
        ("parsed_pois", "POI types LLMAP's parser extracted from the NL query"),
        *cc.CORE_FIELD_DESCRIPTIONS,
    ]
    _write_table(ws_legend, ["Column", "Description"], legend, title="Column Descriptions")

    wb.save(out)
    print(f"Wrote {len(rows)} rows → {out}")

    # Console summary
    by_model: Dict[str, List] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    print(f"\n{'Model':<32} {'N':>5} {'Feasible%':>10} {'ConstrSat%':>11} {'SearchMs':>10}")
    print("-" * 73)
    for model in args.models:
        rs = by_model.get(model)
        if not rs:
            continue
        n = len(rs)
        feas = sum(1 for r in rs if str(r["feasible"]) == "True") / n * 100
        cs_vals = [float(r["constraint_satisfaction"]) for r in rs
                   if r["constraint_satisfaction"] not in (None, "", "None")]
        ms_vals = [float(r["search_time_ms"]) for r in rs
                   if r["search_time_ms"] not in (None, "", "None")]
        cs = f"{sum(cs_vals)/len(cs_vals)*100:.1f}%" if cs_vals else "N/A"
        ms = f"{sum(ms_vals)/len(ms_vals):.1f}ms" if ms_vals else "N/A"
        print(f"{model:<32} {n:>5} {feas:>9.1f}% {cs:>11} {ms:>10}")


if __name__ == "__main__":
    main()
