"""
evaluate_llm.py — Export LLM baseline results to Excel.

All constraint checks and the comparable objective are computed by the shared
evaluation.constraint_checker so columns are identical across LLM / LLMAP / Craft.
LLM-specific columns (hallucination, matching detail, filter-match) are appended.

Usage:
  python evaluation/evaluate_llm.py
  python evaluation/evaluate_llm.py --out outputs/llm_results.xlsx
  python evaluation/evaluate_llm.py --no-sp   # Tier-1 only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluation.constraint_checker as cc
try:
    from evaluation.check_filters import per_query_scores, build_poi_index
except ImportError:
    per_query_scores = None  # type: ignore
    build_poi_index = None   # type: ignore

BASELINE_DIR = ROOT / "outputs" / "llm"
QUERIES_DIR  = ROOT / "benchmark" / "socal_queries" / "set1"
DEFAULT_OUT  = ROOT / "outputs" / "llm_results.xlsx"
DEFAULT_GPKG = ROOT / "data" / "socal" / "road_network.gpkg"

MODELS = [
    "gemini-3_1-flash-lite",
    "gemini-3_5-flash",
    "gemini-3_1-pro-preview",
    "Qwen_Qwen3-32B-AWQ",
]
TEMPLATES = list(cc.TEMPLATES)


# Hallucination index

HALL_DIR = ROOT / "outputs" / "evaluation" / "llm_hallucination"


def _load_hallucination_index() -> Dict[str, Dict[str, Dict]]:
    """Per model, merge ALL hallucination run dirs into one {query_id: record} map,
    later (lexicographically newer) runs overwriting earlier ones for the same
    query_id. Merging (rather than taking only the latest run dir) is deliberate:
    a model may have a full Set-1 run plus a later *partial* run for another set
    (e.g. set2_* ids) — taking just the newest dir would drop the Set-1 data."""
    index: Dict[str, Dict[str, Dict]] = {}
    if not HALL_DIR.exists():
        return index
    for model_dir in HALL_DIR.iterdir():
        if not model_dir.is_dir():
            continue
        run_dirs = sorted(d for d in model_dir.iterdir() if d.is_dir())
        qmap: Dict[str, Dict] = {}
        for run in run_dirs:  # ascending → newer runs overwrite older per query_id
            for fpath in run.glob("*.json"):
                if fpath.stem == "summary":
                    continue
                try:
                    d = json.loads(fpath.read_text())
                    qmap[d["query_id"]] = d
                except Exception:
                    pass
        if qmap:
            index[model_dir.name] = qmap
    return index


def _hallucination_reasons(dropped_stops: List[Dict]) -> str:
    order = ["address_not_found", "place_not_found", "wrong_location", "type_mismatch"]
    seen = {s.get("hallucination_type") for s in dropped_stops if s.get("hallucination_type")}
    return ",".join(t for t in order if t in seen)


def _match_methods(matched_stops: List[Dict]) -> str:
    methods = [s.get("match_method", "") for s in matched_stops]
    return "|".join(sorted(set(m for m in methods if m)))


# Excel writer

def _write_table(ws, headers: List[str], data_rows: List[list], title: str = "") -> None:
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
    HEADER_FONT = Font(bold=True, color="FFFFFF")
    ALT_FILL    = PatternFill("solid", fgColor="D9E1F2")

    if title:
        ws.append([title])
        ws.cell(ws.max_row, 1).font = Font(bold=True, size=13)
        ws.append([])
    ws.append(headers)
    hrow = ws.max_row
    for col, h in enumerate(headers, 1):
        cell = ws.cell(hrow, col)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
    for i, row in enumerate(data_rows):
        ws.append(row)
        if i % 2 == 1:
            for col in range(1, len(headers) + 1):
                ws.cell(ws.max_row, col).fill = ALT_FILL
    for col in range(1, len(headers) + 1):
        max_len = max(
            (len(str(ws.cell(r, col).value or "")) for r in range(hrow, ws.max_row + 1)),
            default=10,
        )
        ws.column_dimensions[get_column_letter(col)].width = min(max_len + 4, 50)


# Main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",          default=str(DEFAULT_OUT))
    ap.add_argument("--gpkg",         default=str(DEFAULT_GPKG))
    ap.add_argument("--models",       nargs="+", default=MODELS)
    ap.add_argument("--templates",    nargs="+", default=TEMPLATES)
    ap.add_argument("--baseline-dir", default=str(BASELINE_DIR))
    ap.add_argument("--queries-dir",  default=str(QUERIES_DIR))
    ap.add_argument("--no-sp",        action="store_true", help="Skip SP/graph computation")
    ap.add_argument("--filter-set",   choices=["set3", "set4"], default=None,
                    help="Compute description/attribute filter-match metric for this set")
    ap.add_argument("--filter-generators", default=None)
    args = ap.parse_args()

    baseline_dir = Path(args.baseline_dir)
    queries_dir  = Path(args.queries_dir)

    nl_index: Dict[str, str] = {}
    for tmpl in args.templates:
        q_file = queries_dir / f"{tmpl}.json"
        if q_file.exists():
            for q in json.loads(q_file.read_text()):
                nl_index[q["query_id"]] = q["nl_query"]

    print("Loading hallucination evaluation index ...")
    hall_index = _load_hallucination_index()
    print(f"  {sum(len(v) for v in hall_index.values())} records across {len(hall_index)} models")

    G = None
    snapper = None
    if not args.no_sp:
        print("Loading graph and node index ...")
        from craft.generator.tools.common_handlers import build_graph_from_gpkg
        G = build_graph_from_gpkg(args.gpkg)
        snapper = cc.NodeSnapper.from_gpkg(args.gpkg, G)
        print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    filter_gen = Path(args.filter_generators) if args.filter_generators else None
    poi_index = build_poi_index(args.gpkg) if args.filter_set else None

    rows = []
    for model in args.models:
        model_dir = baseline_dir / model
        if not model_dir.exists():
            print(f"[skip] {model} — no output dir")
            continue

        filter_scores = {}
        if args.filter_set:
            filter_scores = per_query_scores(
                args.filter_set, model_dir, args.gpkg, filter_gen, index=poi_index)

        n_loaded = 0
        print(f"Loading {model} ...")
        for tmpl in args.templates:
            tmpl_dir = model_dir / tmpl
            if not tmpl_dir.exists():
                continue
            for fpath in sorted(tmpl_dir.glob("*.json")):
                d = json.loads(fpath.read_text())
                qid      = d.get("query_id", fpath.stem)
                nl_query = d.get("nl_query") or nl_index.get(qid, "")

                llm_stops     = d.get("llm_stops", []) or []
                matched_stops = d.get("matched_stops", []) or []
                dropped_stops = d.get("dropped_stops", []) or []

                n_proposed = len(llm_stops)
                n_matched  = len(matched_stops)
                n_dropped  = len(dropped_stops)
                all_matched = (n_dropped == 0 and n_proposed > 0)
                match_rate  = round(n_matched / n_proposed, 4) if n_proposed > 0 else None

                # Hallucination classification must be looked up BEFORE evaluate_baseline
                # so a dropped-but-verified-real stop can be snapped into the objective.
                hall_rec  = hall_index.get(model, {}).get(qid, {})
                hall_drops = hall_rec.get("dropped_stops", [])
                any_hall  = bool(hall_rec.get("any_hallucinated")) if hall_rec else None
                hall_why  = _hallucination_reasons(hall_drops) if hall_rec else None

                obj_stops_wide = cc.objective_stops_incl_unmatched_real(
                    llm_stops, matched_stops, dropped_stops, hall_drops, G, snapper,
                )

                # Shared constraint + objective evaluation. EV battery falls back to
                # geocoded coords on matched + dropped stops when nodes aren't resolved.
                # objective_value_incl_unmatched_real additionally snaps in any dropped
                # stop the hallucination pass verified is real (just not in our dataset).
                core = cc.evaluate_baseline(
                    nl_query, tmpl, matched_stops, G, snapper,
                    all_stops=(matched_stops + dropped_stops) or matched_stops,
                    objective_stops_incl_unmatched_real=obj_stops_wide,
                )

                # parsed_pois = unique POI types the LLM proposed (its "parse")
                llm_types = [cc.normalize_type(s.get("type") or s.get("poi_type"))
                             for s in llm_stops if s.get("type") or s.get("poi_type")]
                parsed_pois = ",".join(dict.fromkeys(t for t in llm_types if t))

                tok       = d.get("tokens") or {}
                # PER-STOP magnitude. `any_hallucinated` is a query-level OR, so it
                # scores 1-of-3 bad stops the same as 3-of-3. The hallucination pass
                # is per stop, so carry its counts through and let the reader compute
                # a per-POI rate as well as a per-query one.
                n_hall    = hall_rec.get("n_hallucinated") if hall_rec else None
                hall_rate = (round(n_hall / n_proposed, 4)
                             if (n_hall is not None and n_proposed) else None)

                row = {
                    "query_id":   qid,
                    "template":   tmpl,
                    "model":      model,
                    "feasible":   d.get("llm_feasible"),
                    "error":      d.get("error"),
                    "infeasible_reason": d.get("llm_infeasible_reason"),
                    "latency_s":  d.get("latency_s"),
                    "parsed_pois": parsed_pois,
                    **{k: core[k] for k in cc.CORE_FIELDS},
                    # LLM-specific
                    "n_proposed":          n_proposed,
                    "n_matched":           n_matched,
                    "n_dropped":           n_dropped,
                    "all_matched":         all_matched,
                    "match_rate":          match_rate,
                    "match_methods":       _match_methods(matched_stops),
                    "any_hallucinated":    any_hall,
                    "n_hallucinated":      n_hall,
                    "hallucination_rate":  hall_rate,
                    "hallucination_reasons": hall_why,
                    "objective_value_incl_unmatched_real": core.get("objective_value_incl_unmatched_real"),
                    "n_unmatched_real_included": core.get("n_unmatched_real_included"),
                    "filter_match_rate":   filter_scores.get(qid, {}).get("rate"),
                    "filter_passed":       filter_scores.get(qid, {}).get("passed"),
                    "filter_checked":      filter_scores.get(qid, {}).get("checked"),
                    "prompt_tokens":       tok.get("prompt_tokens"),
                    "completion_tokens":   tok.get("completion_tokens"),
                }
                rows.append(row)
                n_loaded += 1
        print(f"  {n_loaded} records loaded")

    if not rows:
        sys.exit("No output files found in " + str(baseline_dir))

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
                 title="LLM Baseline Results")

    ws_legend = wb.create_sheet("Column Descriptions")
    legend = [
        ("query_id", "Unique identifier for the benchmark query"),
        ("template", "Benchmark template family"),
        ("model",    "LLM used for the bare LLM baseline"),
        ("feasible", "Whether the LLM declared its route feasible"),
        ("error",    "Processing/pipeline error string for this query (e.g. API or matching failure); None if it ran cleanly. Separate from feasibility — an infeasible-but-clean run has error=None"),
        ("infeasible_reason", "The reason text the LLM gave when declaring the route infeasible; None otherwise. This is a stated reason, not a processing error"),
        ("latency_s","LLM API call wall time (seconds)"),
        ("parsed_pois", "Unique POI types the LLM proposed (its implicit parse), in first-seen order"),
        *cc.CORE_FIELD_DESCRIPTIONS,
        ("n_proposed", "Total stops the LLM proposed (before matching to the dataset)"),
        ("n_matched",  "Proposed stops matched to a dataset POI"),
        ("n_dropped",  "Proposed stops that could not be matched and were discarded"),
        ("all_matched","True if every proposed stop matched (n_dropped == 0)"),
        ("match_rate", "n_matched / n_proposed"),
        ("match_methods", "Pipe-separated set of matching methods used across matched stops"),
        ("any_hallucinated", "True if ≥1 dropped stop was verified hallucinated via Google Places; None if not evaluated"),
        ("n_hallucinated", "COUNT of proposed stops verified hallucinated (per-POI, not per-query). "
                           "Use this for a per-POI rate; any_hallucinated is a query-level OR that "
                           "treats 1-of-3 bad stops the same as 3-of-3."),
        ("hallucination_rate", "n_hallucinated / n_proposed — the per-POI hallucination rate for this query"),
        ("hallucination_reasons", "Unique hallucination types for dropped stops (address_not_found, place_not_found, wrong_location, type_mismatch)"),
        ("filter_match_rate", "Set3/Set4: fraction of requirement-bearing stops whose POI satisfied the asked description/attribute"),
        ("filter_passed",  "Set3/Set4: count of requirement-bearing stops that passed"),
        ("filter_checked", "Set3/Set4: count of stops with a checkable requirement"),
        ("prompt_tokens",  "Input tokens sent to the LLM"),
        ("completion_tokens", "Output tokens generated by the LLM"),
    ]
    _write_table(ws_legend, ["Column", "Description"], legend, title="Column Descriptions")

    wb.save(out)
    print(f"Wrote {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()
