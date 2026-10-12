#!/usr/bin/env python3
"""
evaluate_parser.py — score craft/parser output against the deterministic gold
records in benchmark/ground_truth/.

Why this is not a simple dict diff
----------------------------------
The parser's output schema is deliberately open: craft/parser/prompt.md uses generic
placeholders ("some_resource", "some_time_limit"), so the parser names constraints
freely and the generator LLM interprets them downstream. Measured on real records,
the EV battery alone arrives under 11 different key names (battery_level,
battery_range, max_range, full_range, battery_capacity, max_battery_capacity,
max_battery_range, full_charge_range, total_battery_capacity, maximum_range,
total_capacity) and the time budget under 4. So a field can't be located by name.

This evaluator resolves parser keys semantically instead: match candidate keys by
name pattern, then disambiguate current-vs-max battery by magnitude — the same
heuristic normalize_constraints.py:147-166 itself uses when resolving a percentage.

Five possible outcomes per (query, field), so "failed" is never conflated with "wrong":
  correct   — parser value matches gold
  wrong     — parser emitted a value, it disagrees with gold
  missing   — gold expects a value, parser emitted none
  spurious  — gold expects nothing, parser invented a constraint
  n/a       — gold has nothing and parser emitted nothing
Plus a query-level `parse_failed` when there is no parser_output at all.

Metrics:
  precision/recall/F1 — only for the genuinely multi-valued fields (poi_types,
                        category_limits, poi_filters). For scalars P and R both
                        collapse to accuracy, so accuracy is reported instead.
  accuracy            — reported twice: over all gold records (failures count as
                        wrong) and over attempted-only (parser emitted something).

Units: gold is native (mi / % / hours); the parser emits base SI. Converted here
with mi x 1609.34, hr x 3600, and percent resolved against max range, matching
normalize_constraints.py.

Not scored: visit_time — stored records show a per-category dict injected by the
system rather than parsed from the query, so scoring it would blame the parser for
something it never produced.

Usage:
  python evaluation/evaluate_parser.py
  python evaluation/evaluate_parser.py --no-endpoints
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parser_eval_common import (
    COLLECTED_DIR, GOLD_DIR, GOLD_TO_RECORDS, MODELS, collected_path,
)

DEFAULT_GPKG = ROOT / "data" / "socal" / "road_network.gpkg"
DEFAULT_OUT = ROOT / "results" / "craft_parser_eval.xlsx"
DEFAULT_SUMMARY = ROOT / "results" / "craft_parser_eval_summary.json"

MI_TO_M = 1609.34
HR_TO_S = 3600.0
MIN_TO_S = 60.0
REL_TOL = 1e-3

STRUCTURAL_KEYS = {"src", "dst", "objective", "template_file", "poi_types", "poi_filters",
                   "ordered", "leg_distances", "category_limits", "visit_time",
                   "road_preference"}
BATTERY_PAT = re.compile(r"batter|range|capacit|charge|fuel", re.I)
TIME_PAT = re.compile(r"time|budget|duration", re.I)

CORRECT, WRONG, MISSING, SPURIOUS, NA = "correct", "wrong", "missing", "spurious", "n/a"

SCALAR_FIELDS = ["src", "dst", "objective", "ordered", "leg_distances",
                 "battery_current", "battery_max", "time_budget", "road_preference"]
MULTI_FIELDS = ["poi_types", "category_limits", "poi_filters"]


# helpers

def close(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return False
    return math.isclose(float(a), float(b), rel_tol=REL_TOL, abs_tol=1.0)


def numeric_constraint_keys(po: Dict[str, Any]) -> Dict[str, float]:
    return {k: float(v) for k, v in po.items()
            if k not in STRUCTURAL_KEYS and isinstance(v, (int, float))
            and not isinstance(v, bool)}


def resolve_battery(po: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """(current_m, max_m). The parser names these freely, so match by pattern and
    disambiguate by magnitude — max range is the larger. Mirrors the heuristic in
    normalize_constraints.py:147-166."""
    cands = {k: v for k, v in numeric_constraint_keys(po).items() if BATTERY_PAT.search(k)}
    if not cands:
        return None, None
    if len(cands) == 1:
        # a lone value: "level"/"current" reads as current, anything else as max
        k, v = next(iter(cands.items()))
        return (v, None) if re.search(r"level|current|remain", k, re.I) else (None, v)
    ordered = sorted(cands.values())
    return ordered[0], ordered[-1]


def resolve_time_budget(po: Dict[str, Any]) -> Optional[float]:
    cands = {k: v for k, v in numeric_constraint_keys(po).items()
             if TIME_PAT.search(k) and not BATTERY_PAT.search(k)}
    if not cands:
        return None
    return max(cands.values())


def outcome(gold_present: bool, pred_present: bool, ok: bool) -> str:
    if not gold_present and not pred_present:
        return NA
    if not gold_present and pred_present:
        return SPURIOUS
    if gold_present and not pred_present:
        return MISSING
    return CORRECT if ok else WRONG


def prf(pred: Set, gold: Set) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """P/R/F1, defined only where GOLD is non-empty.

    Returning a score when gold is empty would fold "parser invented a constraint on
    a template that has none" into the same average as "parser got set3's filters
    wrong" — two different failures. Spurious output is already captured by the
    `spurious` outcome state, so it is deliberately excluded here.
    """
    if not gold or not pred:
        # gold empty  -> a spurious invention, already counted as `spurious`
        # pred empty  -> an omission, already counted as `missing`
        # Folding either into P/R would mix omission with mis-extraction; P/R here
        # measures extraction quality AMONG ATTEMPTS, and the outcome columns carry
        # the omission/invention rates separately.
        return None, None, None
    inter = len(pred & gold)
    p = inter / len(pred) if pred else 0.0
    r = inter / len(gold) if gold else 0.0
    f = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f


# endpoint resolution (gold coords -> nearest graph node)

def build_node_table(gpkg: Path):
    import numpy as np
    import pyogrio
    sys.path.append(str(ROOT / "craft" / "generator" / "tools"))
    from common_handlers import _normalize_node_ids

    df = pyogrio.read_dataframe(gpkg, layer="nodes",
                                columns=["osmid_original", "lat", "lon"],
                                read_geometry=False)
    ids, lats, lons = [], [], []
    for raw, la, lo in zip(df["osmid_original"], df["lat"], df["lon"]):
        n = _normalize_node_ids(raw)
        if not n or la is None or lo is None:
            continue
        ids.append(n[0]); lats.append(float(la)); lons.append(float(lo))
    return ids, np.radians(np.array(lats)), np.radians(np.array(lons))


def nearest_nodes(coords: List[Tuple[float, float]], table) -> Dict[Tuple[float, float], Any]:
    """Exact haversine nearest node, matching common_handlers.nearest_node_to_coords.
    Chunked so 248k nodes x ~1.4k coords stays in memory."""
    import numpy as np
    ids, nlat, nlon = table
    out: Dict[Tuple[float, float], Any] = {}
    CH = 64
    for i in range(0, len(coords), CH):
        chunk = coords[i:i + CH]
        la = np.radians(np.array([c[0] for c in chunk]))[:, None]
        lo = np.radians(np.array([c[1] for c in chunk]))[:, None]
        dlat = nlat[None, :] - la
        dlon = nlon[None, :] - lo
        a = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(nlat[None, :]) * np.sin(dlon / 2) ** 2
        idx = np.argmin(a, axis=1)
        for c, j in zip(chunk, idx):
            out[c] = ids[int(j)]
    return out


# per-query scoring

def score_query(gold: Dict[str, Any], po: Optional[Dict[str, Any]],
                node_for: Optional[Dict[Tuple[float, float], Any]]) -> Dict[str, Any]:
    res: Dict[str, Any] = {"parse_failed": po is None}
    po = po or {}
    fields: Dict[str, str] = {}
    extra: Dict[str, Any] = {}

    # endpoints
    for key, gkey in (("src", "src_latlon"), ("dst", "dst_latlon")):
        gc = gold.get(gkey)
        pred = po.get(key)
        if node_for is None or gc is None:
            fields[key] = NA
            continue
        want = node_for.get(tuple(gc))
        ok = pred is not None and str(pred) == str(want)
        fields[key] = outcome(want is not None, pred is not None, ok)

    # plain scalars
    fields["objective"] = outcome(gold.get("objective") is not None,
                                  po.get("objective") is not None,
                                  po.get("objective") == gold.get("objective"))
    fields["ordered"] = outcome(gold.get("ordered") is not None,
                                po.get("ordered") is not None,
                                bool(po.get("ordered")) == bool(gold.get("ordered")))
    fields["road_preference"] = outcome(gold.get("road_preference") is not None,
                                        po.get("road_preference") is not None,
                                        po.get("road_preference") == gold.get("road_preference"))

    # leg distances (metres); gold may carry ONE shared per-leg limit
    g_legs = gold.get("leg_distance_limits_mi")
    shared = bool(gold.get("shared_leg_limit"))
    p_legs = po.get("leg_distances")
    if isinstance(p_legs, dict):
        p_legs = p_legs.get("values")
    if g_legs:
        want = [v * MI_TO_M for v in g_legs]
        if p_legs:
            if shared:
                # a single stated per-leg limit: every emitted leg must equal it,
                # and the leg COUNT is deliberately not prejudged by gold.
                ok = all(close(x, want[0]) for x in p_legs)
            else:
                ok = (len(p_legs) == len(want)
                      and all(close(x, y) for x, y in zip(p_legs, want)))
            fields["leg_distances"] = CORRECT if ok else WRONG
        else:
            fields["leg_distances"] = MISSING
    else:
        fields["leg_distances"] = SPURIOUS if p_legs else NA

    # battery (parser resolves percent away into metres)
    g_max = gold.get("max_battery_mi")
    g_max_m = g_max * MI_TO_M if g_max is not None else None
    g_cur_m = None
    # 21 queries state the charge as a PERCENTAGE and deliberately omit the max range
    # (e.g. algo1 variant 21, "a route that works with my available range"). With no
    # max to resolve the percentage against, the absolute value is not derivable from
    # the text by anyone — parser included — so the field is UNSCOREABLE, not wrong
    # and certainly not spurious.
    unscoreable_current = (gold.get("current_battery_pct") is not None
                           and g_max_m is None
                           and gold.get("current_battery_mi") is None)
    if g_max_m is not None and gold.get("current_battery_pct") is not None:
        g_cur_m = gold["current_battery_pct"] / 100.0 * g_max_m
    elif gold.get("current_battery_mi") is not None:
        g_cur_m = gold["current_battery_mi"] * MI_TO_M
    p_cur, p_max = resolve_battery(po)
    fields["battery_current"] = (NA if unscoreable_current else
                                 outcome(g_cur_m is not None, p_cur is not None,
                                         close(p_cur, g_cur_m)))
    fields["battery_max"] = outcome(g_max_m is not None, p_max is not None,
                                    close(p_max, g_max_m))

    # time budget (seconds)
    g_t = gold.get("time_budget_min")
    g_t_s = g_t * MIN_TO_S if g_t is not None else None
    p_t = resolve_time_budget(po)
    fields["time_budget"] = outcome(g_t_s is not None, p_t is not None, close(p_t, g_t_s))

    # poi_types: set P/R plus order
    g_types = [str(t).lower() for t in (gold.get("poi_types") or [])]
    p_types = [str(t).strip().lower() for t in (po.get("poi_types") or [])]
    p, r, f = prf(set(p_types), set(g_types))
    extra.update(poi_types_precision=p, poi_types_recall=r, poi_types_f1=f)
    fields["poi_types"] = outcome(bool(g_types), bool(p_types), set(p_types) == set(g_types))
    if gold.get("ordered") and g_types:
        extra["poi_order_correct"] = (p_types == g_types) if p_types else False
    else:
        extra["poi_order_correct"] = None

    # category limits
    g_lim = gold.get("category_limits") or {}
    p_lim = po.get("category_limits") or {}
    if isinstance(p_lim, dict):
        p_lim = {str(k).lower(): v for k, v in p_lim.items()}
    p_k, r_k, f_k = prf(set(p_lim), set(g_lim))
    extra.update(cat_limit_precision=p_k, cat_limit_recall=r_k, cat_limit_f1=f_k)
    fields["category_limits"] = outcome(
        bool(g_lim), bool(p_lim),
        set(p_lim) == set(g_lim) and all(close(p_lim.get(k), v) for k, v in g_lim.items()))

    # poi_filters (set3)
    g_f = gold.get("poi_filters")
    p_f = po.get("poi_filters")
    g_pairs = _filter_pairs(g_f)
    p_pairs = _filter_pairs(p_f)
    p_p, p_r, p_f1 = prf(p_pairs, g_pairs)
    extra.update(poi_filter_precision=p_p, poi_filter_recall=p_r, poi_filter_f1=p_f1)
    fields["poi_filters"] = outcome(bool(g_pairs), bool(p_pairs), p_pairs == g_pairs)

    res["fields"] = fields
    res["extra"] = extra
    res["all_correct"] = all(v in (CORRECT, NA) for v in fields.values())
    return res


def _filter_pairs(f: Any) -> Set[Tuple[int, str, str]]:
    """Flatten poi_filters into {(stop_index, column, value)} for set comparison."""
    out: Set[Tuple[int, str, str]] = set()
    if not isinstance(f, list):
        return out
    for i, entry in enumerate(f):
        if isinstance(entry, dict):
            for k, v in entry.items():
                out.add((i, str(k).lower(), str(v).lower()))
    return out


# main

def load_collected(model: str, collected_dir: Path) -> Dict[str, Dict[str, Optional[Dict[str, Any]]]]:
    """set_name -> {query_id: parser_output or None}, from a snapshot written by
    collect_parser_outputs.py (step 1). Only records the snapshot marked
    record_found are included — a genuinely missing craft record is different
    from a parse failure (parser_output=None) and must not be scored as one."""
    p = collected_path(model, collected_dir)
    if not p.exists():
        return {}
    snap = json.loads(p.read_text())
    by_set: Dict[str, Dict[str, Optional[Dict[str, Any]]]] = defaultdict(dict)
    for r in snap.get("records", []):
        if r.get("record_found"):
            by_set[r["set"]][r["query_id"]] = r.get("parser_output")
    return by_set


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--gold-dir", default=str(GOLD_DIR))
    ap.add_argument("--collected-dir", default=str(COLLECTED_DIR),
                    help="per-model parser-output snapshots written by "
                         "evaluation/collect_parser_outputs.py (run that first)")
    ap.add_argument("--gpkg", default=str(DEFAULT_GPKG))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--summary-out", default=str(DEFAULT_SUMMARY))
    ap.add_argument("--no-endpoints", action="store_true",
                    help="skip src/dst scoring (avoids loading the node table)")
    ap.add_argument("--include-set2", action="store_true",
                    help="also score set2 (California graph; endpoints not scored)")
    args = ap.parse_args()

    gold_dir = Path(args.gold_dir)
    gold_sets: Dict[str, List[Dict[str, Any]]] = {}
    for name in GOLD_TO_RECORDS:
        p = gold_dir / f"{name}_gold.json"
        if p.exists():
            gold_sets[name] = json.loads(p.read_text())
    if not gold_sets:
        sys.exit(f"no gold files in {gold_dir} — expected benchmark/ground_truth/*_gold.json")

    collected_dir = Path(args.collected_dir)
    missing_snapshots = [m for m in args.models if not collected_path(m, collected_dir).exists()]
    if missing_snapshots:
        sys.exit(f"missing parser-output snapshot(s) for {missing_snapshots} in "
                 f"{collected_dir} — run `python evaluation/collect_parser_outputs.py` first")
    collected = {model: load_collected(model, collected_dir) for model in args.models}

    # endpoint resolution: one exact nearest-node pass over the distinct SoCal coords
    node_for = None
    if not args.no_endpoints:
        coords: Set[Tuple[float, float]] = set()
        for name, recs in gold_sets.items():
            if not GOLD_TO_RECORDS[name][1]:
                continue
            for r in recs:
                for k in ("src_latlon", "dst_latlon"):
                    if r.get(k):
                        coords.add(tuple(r[k]))
        print(f"[endpoints] resolving {len(coords)} distinct coords against the node table ...")
        table = build_node_table(Path(args.gpkg))
        node_for = nearest_nodes(sorted(coords), table)
        print(f"[endpoints] done ({len(node_for)} resolved)")

    rows: List[Dict[str, Any]] = []
    for set_name, recs in gold_sets.items():
        _subdir, is_socal = GOLD_TO_RECORDS[set_name]
        if set_name.startswith("set2") and not args.include_set2:
            continue
        gold_by_id = {r["query_id"]: r for r in recs}
        for model in args.models:
            found = collected.get(model, {}).get(set_name, {})
            if not found:
                continue
            for qid, g in gold_by_id.items():
                if qid not in found:
                    continue
                nf = node_for if (is_socal and not args.no_endpoints) else None
                s = score_query(g, found[qid], nf)
                rows.append({"set": set_name, "model": model, "query_id": qid,
                             "template": g["template"], **s})
            print(f"[score] {set_name:26s} {model:22s} {len(found):4d} records")

    if not rows:
        sys.exit("no parser outputs matched the gold records")

    # aggregate
    summary: Dict[str, Any] = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                               "rel_tol": REL_TOL, "by_model": {}, "by_field": {}}
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        failed = sum(1 for r in sub if r["parse_failed"])
        summary["by_model"][model] = {
            "n": len(sub),
            "parse_failed": failed,
            "all_fields_correct": sum(1 for r in sub if r["all_correct"]),
            "all_fields_correct_pct_of_all": round(
                100 * sum(1 for r in sub if r["all_correct"]) / len(sub), 1),
            "all_fields_correct_pct_of_attempted": (round(
                100 * sum(1 for r in sub if r["all_correct"]) / (len(sub) - failed), 1)
                if len(sub) - failed else None),
        }

    field_rows: List[list] = []
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        for fld in SCALAR_FIELDS + MULTI_FIELDS:
            c = Counter(r["fields"].get(fld, NA) for r in sub)
            applicable = c[CORRECT] + c[WRONG] + c[MISSING]
            attempted = c[CORRECT] + c[WRONG]
            field_rows.append([
                model, fld, c[CORRECT], c[WRONG], c[MISSING], c[SPURIOUS], c[NA],
                round(100 * c[CORRECT] / applicable, 1) if applicable else None,
                round(100 * c[CORRECT] / attempted, 1) if attempted else None,
            ])
    summary["by_field"] = [dict(zip(
        ["model", "field", "correct", "wrong", "missing", "spurious", "na",
         "accuracy_all_pct", "accuracy_attempted_pct"], r)) for r in field_rows]

    # precision/recall only where genuinely multi-valued
    pr_rows: List[list] = []
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        for label, pk, rk, fk in (
                ("poi_types", "poi_types_precision", "poi_types_recall", "poi_types_f1"),
                ("category_limits", "cat_limit_precision", "cat_limit_recall", "cat_limit_f1"),
                ("poi_filters", "poi_filter_precision", "poi_filter_recall", "poi_filter_f1")):
            vals = [(r["extra"].get(pk), r["extra"].get(rk), r["extra"].get(fk))
                    for r in sub if r["extra"].get(pk) is not None]
            if not vals:
                continue
            pr_rows.append([model, label, len(vals),
                            round(sum(v[0] for v in vals) / len(vals), 4),
                            round(sum(v[1] for v in vals) / len(vals), 4),
                            round(sum(v[2] for v in vals) / len(vals), 4)])
        ordered_vals = [r["extra"]["poi_order_correct"] for r in sub
                        if r["extra"].get("poi_order_correct") is not None]
        if ordered_vals:
            pr_rows.append([model, "poi_order (ordered families)", len(ordered_vals),
                            None, None, round(sum(ordered_vals) / len(ordered_vals), 4)])
    summary["precision_recall"] = [dict(zip(
        ["model", "field", "n", "precision", "recall", "f1"], r)) for r in pr_rows]

    # field categories for the chart tab
    # groups the 12 scored fields into the buckets a reader actually asks
    # about ("did it get the endpoints right", "did it get the battery
    # right"), so the chart shows ~9 bars per model instead of 12 near-
    # identical scalar fields plus 3 multi-valued ones.
    FIELD_CATEGORIES = [
        ("Endpoints (src/dst)",   ["src", "dst"]),
        ("Objective",             ["objective"]),
        ("Ordering",              ["ordered"]),
        ("Distance limits",       ["leg_distances"]),
        ("Battery",               ["battery_current", "battery_max"]),
        ("Time budget",           ["time_budget"]),
        ("Road preference",       ["road_preference"]),
        ("POI types",             ["poi_types"]),
        ("Category limits",       ["category_limits"]),
        ("POI filters",           ["poi_filters"]),
    ]
    model_order = sorted({r["model"] for r in rows})
    category_rows: List[list] = []  # [category, acc_model1, acc_model2, ...]
    for cat_label, fields_in_cat in FIELD_CATEGORIES:
        row_out = [cat_label]
        for model in model_order:
            c = w = mi = 0
            for fr in field_rows:
                if fr[0] == model and fr[1] in fields_in_cat:
                    c += fr[2]; w += fr[3]; mi += fr[4]
            applicable = c + w + mi
            row_out.append(round(100 * c / applicable, 1) if applicable else None)
        category_rows.append(row_out)
    summary["by_category"] = [dict(zip(["category"] + model_order, r)) for r in category_rows]

    # write
    from openpyxl import Workbook
    from openpyxl.chart import BarChart, Reference
    from evaluation.evaluate_llm import _write_table

    wb = Workbook(); wb.remove(wb.active)
    _write_table(wb.create_sheet("by_field"),
                 ["model", "field", "correct", "wrong", "missing", "spurious", "n/a",
                  "accuracy_all_%", "accuracy_attempted_%"], field_rows,
                 "Per-field outcomes (failed vs wrong kept distinct)")
    _write_table(wb.create_sheet("precision_recall"),
                 ["model", "field", "n", "precision", "recall", "f1"], pr_rows,
                 "Precision/recall — multi-valued fields only")
    _write_table(wb.create_sheet("by_model"),
                 ["model", "n", "parse_failed", "all_fields_correct",
                  "% of all", "% of attempted"],
                 [[m, v["n"], v["parse_failed"], v["all_fields_correct"],
                   v["all_fields_correct_pct_of_all"], v["all_fields_correct_pct_of_attempted"]]
                  for m, v in summary["by_model"].items()],
                 "Whole-record accuracy")
    per_q = [[r["set"], r["model"], r["query_id"], r["parse_failed"], r["all_correct"]]
             + [r["fields"].get(f, NA) for f in SCALAR_FIELDS + MULTI_FIELDS] for r in rows]
    _write_table(wb.create_sheet("per_query"),
                 ["set", "model", "query_id", "parse_failed", "all_correct"]
                 + SCALAR_FIELDS + MULTI_FIELDS, per_q, "Per-query detail")
    _write_table(wb.create_sheet("legend"), ["term", "meaning"], [
        ["correct", "parser value matches gold"],
        ["wrong", "parser emitted a value and it disagrees with gold"],
        ["missing", "gold expects a value, parser emitted none"],
        ["spurious", "gold expects nothing, parser invented a constraint"],
        ["n/a", "field not applicable to this template and parser emitted nothing"],
        ["accuracy_all_%", "correct / (correct+wrong+missing) — omissions count against"],
        ["accuracy_attempted_%", "correct / (correct+wrong) — extraction quality only"],
        ["key resolution", "the parser names constraints freely (11 battery key names "
                           "observed), so battery/time are matched by name pattern then "
                           "disambiguated by magnitude, as normalize_constraints.py does"],
        ["visit_time", "NOT scored — system-injected, not parsed from the query"],
        ["set2", "excluded by default (California graph, flash-lite only)"],
    ], "Definitions")

    # charts tab
    ws_chart = wb.create_sheet("charts")
    overall_rows = [[m, v["all_fields_correct_pct_of_all"], v["all_fields_correct_pct_of_attempted"]]
                    for m, v in summary["by_model"].items()]
    _write_table(ws_chart, ["model", "% of all", "% of attempted"], overall_rows,
                "Overall accuracy — all fields correct (%)")
    overall_header_row = 3  # title(1) + blank(2) + header(3)
    overall_last_row = overall_header_row + len(overall_rows)

    ws_chart.append([]); ws_chart.append([])
    cat_header_row = ws_chart.max_row + 1
    _write_table(ws_chart, ["category"] + model_order, category_rows,
                "Accuracy by field category (% of all, per model)")
    cat_last_row = cat_header_row + len(category_rows)

    chart1 = BarChart()
    chart1.type, chart1.grouping = "col", "clustered"
    chart1.title = "Overall accuracy — all fields correct (% of all queries)"
    chart1.y_axis.title, chart1.x_axis.title = "% correct", "model"
    chart1.y_axis.scaling.min, chart1.y_axis.scaling.max = 0, 100
    data1 = Reference(ws_chart, min_col=2, max_col=2,
                      min_row=overall_header_row, max_row=overall_last_row)
    cats1 = Reference(ws_chart, min_col=1, max_col=1,
                      min_row=overall_header_row + 1, max_row=overall_last_row)
    chart1.add_data(data1, titles_from_data=True)
    chart1.set_categories(cats1)
    chart1.width, chart1.height = 16, 9
    ws_chart.add_chart(chart1, f"A{cat_last_row + 3}")

    chart2 = BarChart()
    chart2.type, chart2.grouping = "col", "clustered"
    chart2.title = "Accuracy by field category, per model (%)"
    chart2.y_axis.title, chart2.x_axis.title = "% correct (of all)", "field category"
    chart2.y_axis.scaling.min, chart2.y_axis.scaling.max = 0, 100
    data2 = Reference(ws_chart, min_col=2, max_col=1 + len(model_order),
                      min_row=cat_header_row, max_row=cat_last_row)
    cats2 = Reference(ws_chart, min_col=1, max_col=1,
                      min_row=cat_header_row + 1, max_row=cat_last_row)
    chart2.add_data(data2, titles_from_data=True)
    chart2.set_categories(cats2)
    chart2.width, chart2.height = 24, 12
    ws_chart.add_chart(chart2, f"H{cat_last_row + 3}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    Path(args.summary_out).write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n[done] {out_path}")
    for m, v in summary["by_model"].items():
        print(f"  {m:22s} n={v['n']:4d} parse_failed={v['parse_failed']:3d} "
              f"all-fields-correct={v['all_fields_correct']:4d} "
              f"({v['all_fields_correct_pct_of_all']}% of all)")


if __name__ == "__main__":
    main()
