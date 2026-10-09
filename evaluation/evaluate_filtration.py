#!/usr/bin/env python3
"""
evaluate_filtration.py — Phase 2: evaluate CRAFT's POI FILTER CASCADE (set3).

Phase 1 (evaluate_possible_targets.py) covered step 1 of the cascade: category ->
nodes. This covers steps 2-3, which only fire when poi_filters is non-empty, i.e.
on set3 (ordered+desc).

The two defects this is built to measure
----------------------------------------
1. SILENT BROADENING. When a filter matches nothing, filtration.py:122-129 keeps the
   BROADER (unfiltered) set and merely prints a warning — "a cafe with Wi-Fi" quietly
   becomes "any cafe". stdout is never captured, so this is invisible in every stored
   result. poi_embeddings.filter_nodes_by_description has a SECOND, undocumented
   fallback that returns the unfiltered set, in which case filtration prints nothing
   at all. Detection therefore cannot rely on log lines: we use set identity
   (sys == base while gt is strictly smaller than base).

2. NODE-LEVEL LEAKAGE. filtration.py:122 does `nodes & col_nodes`, intersecting at
   NODE granularity. ~7 POIs share a nearest node on average, so a node qualifies as
   "restaurant with cuisine=sushi" when ONE row is a restaurant and a DIFFERENT row
   has sushi. Ground truth here is therefore ROW-level (category AND column on the
   same row); the gap between them is the leakage, and it shows up as precision < 1.

Four node sets per stop
  base       category-only set (the real build_poi_type_nodes)
  gt         independent ROW-level truth, computed here from the dataframe
  sys_gold   cascade replay driven by the GOLD filter (set3_meta.json)
  sys_parser cascade replay driven by the PARSER's filter (what actually ran)
Running both sys_* separates "wrong candidate set because the PARSER mis-extracted
the filter" from "wrong because FILTRATION mishandled a correct filter" — worth
separating, since the parser evaluation measured poi_filters at only P/R ~0.89.

The cascade is a line-by-line PORT of filtration.py:93-158 (bugs included, on
purpose) so it can run in seconds instead of ~45s per query; --verify-replay checks
the port against the real build_possible_targets and ABORTS on any mismatch.

Description stops (29) have no deterministic ground truth and are reported
SEPARATELY, never mixed into the structured aggregates.

Usage:
  conda run -n myenv python evaluation/evaluate_filtration.py
  conda run -n myenv python evaluation/evaluate_filtration.py --verify-replay 5
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "craft" / "generator" / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.append(str(TOOLS_DIR))          # APPEND: filtration's bare inner imports

GOLD = ROOT / "benchmark" / "ground_truth" / "set3_ordered+desc_gold.json"
DEFAULT_GPKG = ROOT / "data" / "socal" / "road_network.gpkg"
DEFAULT_OUT = ROOT / "results" / "craft_filtration_set3.xlsx"
DEFAULT_SUMMARY = ROOT / "results" / "craft_filtration_set3_summary.json"
CACHE_DIR = ROOT / "outputs" / "evaluation" / "possible_targets_cache"
NODE_COL = "nearest_node_osmid"
MODELS = ["gemini-3_1-flash-lite", "gpt-4_1", "openai_gpt-oss-20b"]


# shared plumbing (mirrors evaluate_possible_targets.py)

class NodeIndex:
    def __init__(self, canonical: Set[Any], c2c: Dict[Any, Any]) -> None:
        self._c = canonical
        self.graph = {"comp_to_canonical": c2c}

    def __contains__(self, k: Any) -> bool:
        return k in self._c

    def __len__(self) -> int:
        return len(self._c)


def build_node_index(gpkg: Path) -> NodeIndex:
    import pyogrio
    from common_handlers import _normalize_node_ids
    df = pyogrio.read_dataframe(gpkg, layer="nodes", columns=["osmid_original"],
                                read_geometry=False)
    canon: Set[Any] = set()
    c2c: Dict[Any, Any] = {}
    for raw in df["osmid_original"]:
        ids = _normalize_node_ids(raw)
        if not ids:
            continue
        can = ids[0]
        for nid in ids:
            c2c[nid] = can
            c2c[str(nid)] = can
        canon.add(can)
    return NodeIndex(canon, c2c)


def load_pois(gpkg: Path, extra_cols: List[str]):
    import pyogrio
    from common_handlers import CATEGORY_COLS
    avail = set(pyogrio.read_info(gpkg, layer="pois")["fields"])
    cols = [c for c in CATEGORY_COLS if c in avail]
    for c in [NODE_COL, "name", "description"] + extra_cols:
        if c in avail and c not in cols:
            cols.append(c)
    return pyogrio.read_dataframe(gpkg, layer="pois", columns=cols,
                                  read_geometry=False), avail


def _enc(v): return f"i:{v}" if isinstance(v, int) else f"s:{v}"
def _dec(vs): return [int(v[2:]) if v.startswith("i:") else v[2:] for v in vs]


def base_type_nodes(node_index, pois, cats: List[str], gpkg: Path,
                    no_cache: bool) -> Dict[str, Set[Any]]:
    """Step 1 via the REAL build_poi_type_nodes (~45s), disk-cached."""
    import hashlib
    from common_handlers import build_poi_type_nodes
    st = gpkg.stat()
    key = hashlib.sha256(
        f"{st.st_size}:{int(st.st_mtime)}:{','.join(sorted(cats))}".encode()).hexdigest()[:16]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cf = CACHE_DIR / f"literal_type_nodes_{key}.json"
    if cf.exists() and not no_cache:
        return {k: set(_dec(v)) for k, v in json.loads(cf.read_text()).items()}
    t0 = time.perf_counter()
    tn = build_poi_type_nodes(node_index, pois, cats, NODE_COL)
    print(f"[base] build_poi_type_nodes {time.perf_counter()-t0:.1f}s")
    cf.write_text(json.dumps({k: sorted(_enc(x) for x in v) for k, v in tn.items()}))
    return tn


# the two candidate-set rules

def col_nodes_for(pois, col: str, val: str, canon: Dict[Any, Any]) -> Set[Any]:
    """THE SYSTEM'S rule (filtration.py:110-121): rows anywhere in the table whose
    `col` contains `val`, mapped to nodes. Deliberately ignores category — that is
    exactly what makes the later node-level intersection leak."""
    mask = (pois[col].astype(str).str.lower()
            .str.contains(str(val).lower(), na=False, regex=False))
    out: Set[Any] = set()
    for raw in pois.loc[mask, NODE_COL]:
        nid = canon.get(raw)
        if nid is not None:
            out.add(nid)
    return out


def row_level_gt(pois, cat: str, col: str, val: str, canon: Dict[Any, Any],
                 cat_mask_cache: Dict[str, Any]) -> Set[Any]:
    """GROUND TRUTH: the category AND the attribute must hold on the SAME ROW.
    Written here from scratch — importing filtration would make this circular."""
    m = cat_mask_cache[cat] & (
        pois[col].astype(str).str.lower()
        .str.contains(str(val).lower(), na=False, regex=False))
    out: Set[Any] = set()
    for raw in pois.loc[m, NODE_COL]:
        nid = canon.get(raw)
        if nid is not None:
            out.add(nid)
    return out


def independent_cat_mask(pois, cat: str):
    """Own category matcher (not build_poi_type_nodes) over the 17 category columns,
    replicating split_multi_value semantics."""
    from common_handlers import CATEGORY_COLS
    m = None
    for col in [c for c in CATEGORY_COLS if c in pois.columns]:
        norm = (pois[col].astype(str).str.lower()
                .str.replace("|", ";", regex=False).str.replace(",", ";", regex=False))
        hit = norm.str.contains(r"(?:^|;)\s*" + re.escape(cat) + r"\s*(?:;|$)",
                                regex=True, na=False)
        m = hit if m is None else (m | hit)
    return m


# cascade port (filtration.py:93-158)

def replay_cascade(base: Set[Any], filters: Dict[str, Any], available: Set[str],
                   col_cache) -> Tuple[Set[Any], List[str]]:
    """Faithful port of the structured branch, INCLUDING broaden-on-empty. Returns
    (nodes, events). Description filters are not applied here — they need the 3.7GB
    embedding cache and have no deterministic truth; they are flagged instead."""
    nodes = set(base)
    events: List[str] = []
    for col, val in (filters or {}).items():
        if col == "description":
            events.append("description_deferred")
            continue
        if col not in available:
            events.append(f"unknown_column:{col}")     # filtration.py:106-108
            continue
        narrowed = nodes & col_cache(col, val)
        if not narrowed:
            events.append(f"broaden_on_empty:{col}={val}")   # filtration.py:122-129
        else:
            nodes = narrowed
    return nodes, events


# metrics

def stop_metrics(sys_set: Set[Any], gt: Set[Any], base: Set[Any]) -> Dict[str, Any]:
    inter = sys_set & gt
    degenerate = not base
    return {
        "n_base": len(base), "n_gt": len(gt), "n_sys": len(sys_set),
        "exact_match": sys_set == gt,
        "precision": round(len(inter) / len(sys_set), 4) if sys_set else None,
        "recall": round(len(inter) / len(gt), 4) if gt else None,
        "leakage_fp": len(sys_set - gt),
        "missed_fn": len(gt - sys_set),
        # the headline: the filter had NO effect although a narrower truth existed
        "broadened": (not degenerate) and sys_set == base and len(gt) < len(base),
        "gt_empty": len(gt) == 0,
        "degenerate": degenerate,
        "narrowing": round(len(sys_set) / len(base), 4) if base else None,
    }


def parser_filter_for(po: Dict[str, Any], idx: int) -> Dict[str, Any]:
    pf = (po or {}).get("poi_filters")
    if isinstance(pf, list) and 0 <= idx < len(pf):
        v = pf[idx]
        if isinstance(v, dict):
            return dict(v)
        if isinstance(v, str):
            return {"description": v}
    return {}


# main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpkg", default=str(DEFAULT_GPKG))
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--records-dir", default=str(ROOT / "outputs" / "craft"))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--summary-out", default=str(DEFAULT_SUMMARY))
    ap.add_argument("--verify-replay", type=int, default=3)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    gold = json.loads(GOLD.read_text())
    gold_by_id = {r["query_id"]: r for r in gold}

    # distinct categories + filter columns
    cats: Set[str] = set()
    cols: Set[str] = set()
    for r in gold:
        for s in (r["_provenance"].get("stops") or []):
            cats.add(s["poi_type"])
            if s.get("filter_kind") == "structured":
                cols.add(s["column"])
    for m in args.models:
        d = Path(args.records_dir) / m / "ordered+desc"
        for f in d.glob("*.json"):
            try:
                po = (json.loads(f.read_text()).get("parser_output") or {})
            except Exception:
                continue
            for t in (po.get("poi_types") or []):
                cats.add(str(t).strip().lower())
            for e in (po.get("poi_filters") or []):
                if isinstance(e, dict):
                    cols.update(k for k in e if k != "description")
    cats_l = sorted(cats)
    print(f"[scope] categories={cats_l}")
    print(f"[scope] filter columns={sorted(cols)}")

    node_index = build_node_index(Path(args.gpkg))
    pois, available = load_pois(Path(args.gpkg), sorted(cols))
    print(f"[data] {len(node_index)} nodes, {len(pois)} POI rows")
    unknown_cols = sorted(c for c in cols if c not in available)
    if unknown_cols:
        print(f"[data] filter columns absent from the gpkg: {unknown_cols}")

    from common_handlers import resolve_graph_node_id
    canon = {raw: resolve_graph_node_id(node_index, raw)
             for raw in pois[NODE_COL].dropna().unique()}

    base = base_type_nodes(node_index, pois, cats_l, Path(args.gpkg), args.no_cache)
    print("[base] " + ", ".join(f"{c}={len(base.get(c,set()))}" for c in cats_l))

    cat_masks = {c: independent_cat_mask(pois, c) for c in cats_l}
    _col_cache: Dict[Tuple[str, str], Set[Any]] = {}

    def col_cache(col: str, val: str) -> Set[Any]:
        k = (col, str(val).lower())
        if k not in _col_cache:
            _col_cache[k] = col_nodes_for(pois, col, val, canon)
        return _col_cache[k]

    _gt_cache: Dict[Tuple[str, str, str], Set[Any]] = {}

    def gt_for(cat: str, col: str, val: str) -> Set[Any]:
        k = (cat, col, str(val).lower())
        if k not in _gt_cache:
            _gt_cache[k] = row_level_gt(pois, cat, col, val, canon, cat_masks)
        return _gt_cache[k]

    # verify the cascade port against the REAL build_possible_targets
    if args.verify_replay:
        from filtration import build_possible_targets
        modf = Path(sys.modules["filtration"].__file__).resolve()
        assert modf == (TOOLS_DIR / "filtration.py").resolve(), f"shadowed: {modf}"
        qids = sorted(gold_by_id)[:args.verify_replay]
        print(f"[verify] replaying {len(qids)} queries through the real cascade ...")
        bad = []
        for qid in qids:
            g = gold_by_id[qid]
            types = [t.lower() for t in (g.get("poi_types") or [])]
            gfilters = g.get("poi_filters") or []
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                real = build_possible_targets(G=node_index, pois_gdf=pois,
                                              poi_types=types, node_col=NODE_COL,
                                              poi_filters=gfilters,
                                              embedding_cache=None, api_key=None)
            for i, t in enumerate(types):
                mine, _ = replay_cascade(base.get(t, set()),
                                         gfilters[i] if i < len(gfilters) else {},
                                         available, col_cache)
                if mine != real.get(i, set()):
                    bad.append(f"{qid}[{i}]")
        if bad:
            sys.exit(f"PORT MISMATCH vs real build_possible_targets: {bad[:10]}")
        print(f"[verify] port matches the real cascade on {len(qids)} queries (0 mismatches)")

    # score
    rows: List[Dict[str, Any]] = []
    for model in args.models:
        d = Path(args.records_dir) / model / "ordered+desc"
        if not d.is_dir():
            continue
        n = 0
        for f in sorted(d.glob("*.json")):
            try:
                rec = json.loads(f.read_text())
            except Exception:
                continue
            qid = rec.get("query_id")
            g = gold_by_id.get(qid)
            if not g:
                continue
            n += 1
            po = rec.get("parser_output") or {}
            stops = g["_provenance"].get("stops") or []
            gfilters = g.get("poi_filters") or []
            for i, spec in enumerate(stops):
                cat = spec["poi_type"]
                kind = spec.get("filter_kind")
                b = base.get(cat, set())
                gf = gfilters[i] if i < len(gfilters) else {}
                pf = parser_filter_for(po, i)
                sys_gold, ev_g = replay_cascade(b, gf, available, col_cache)
                sys_par, ev_p = replay_cascade(b, pf, available, col_cache)
                row = {"model": model, "query_id": qid, "stop_index": i,
                       "category": cat, "filter_kind": kind,
                       "gold_filter": json.dumps(gf) if gf else None,
                       "parser_filter": json.dumps(pf) if pf else None,
                       "filter_match": (gf == pf),
                       "events_gold": ";".join(ev_g) or None,
                       "events_parser": ";".join(ev_p) or None}
                if kind == "structured":
                    gt = gt_for(cat, spec["column"], spec["value"])
                    row.update({f"gold_{k}": v for k, v in
                                stop_metrics(sys_gold, gt, b).items()})
                    row.update({f"parser_{k}": v for k, v in
                                stop_metrics(sys_par, gt, b).items()})
                else:
                    row.update({"gold_n_base": len(b), "gold_n_sys": len(sys_gold),
                                "parser_n_sys": len(sys_par)})
                rows.append(row)
        print(f"[score] {model:22s} {n:4d} queries")

    if not rows:
        sys.exit("no records scored")

    # aggregate (structured only)
    summary: Dict[str, Any] = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                               "by_model": {}, "by_filter": []}
    agg_rows: List[list] = []
    for model in args.models:
        st = [r for r in rows if r["model"] == model and r["filter_kind"] == "structured"]
        if not st:
            continue
        for src in ("gold", "parser"):
            ok = [r for r in st if r.get(f"{src}_exact_match") is not None]
            scored = [r for r in ok if not r.get(f"{src}_degenerate")]
            prec = [r[f"{src}_precision"] for r in scored if r[f"{src}_precision"] is not None]
            rec = [r[f"{src}_recall"] for r in scored if r[f"{src}_recall"] is not None]
            broad = [r for r in scored if r[f"{src}_broadened"]]
            leak = [r for r in scored if r[f"{src}_leakage_fp"] > 0]
            agg_rows.append([
                model, src, len(scored),
                sum(1 for r in scored if r[f"{src}_exact_match"]),
                round(100 * sum(1 for r in scored if r[f"{src}_exact_match"]) / len(scored), 1),
                round(sum(prec) / len(prec), 4) if prec else None,
                round(sum(rec) / len(rec), 4) if rec else None,
                len(broad), round(100 * len(broad) / len(scored), 1),
                len(leak), sum(r[f"{src}_leakage_fp"] for r in scored),
                sum(1 for r in scored if r[f"{src}_gt_empty"]),
            ])
        summary["by_model"][model] = {
            "structured_stops": len(st),
            "filter_extraction_match": sum(1 for r in st if r["filter_match"]),
        }

    # per distinct filter (gold-driven), to expose which filters silently broaden
    per_filter: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in rows:
        if r["filter_kind"] != "structured" or r.get("gold_exact_match") is None:
            continue
        k = (r["category"], r["gold_filter"] or "")
        d = per_filter.setdefault(k, {"n": 0, "broadened": 0, "leak": 0,
                                      "n_base": r["gold_n_base"], "n_gt": r["gold_n_gt"],
                                      "n_sys": r["gold_n_sys"]})
        d["n"] += 1
        d["broadened"] += int(bool(r["gold_broadened"]))
        d["leak"] += r["gold_leakage_fp"]
    filt_rows = [[c, f, v["n"], v["n_base"], v["n_gt"], v["n_sys"],
                  v["broadened"], v["leak"]]
                 for (c, f), v in sorted(per_filter.items(), key=lambda x: -x[1]["leak"])]
    summary["by_filter"] = [dict(zip(
        ["category", "gold_filter", "stops", "n_base", "n_gt", "n_sys",
         "broadened", "leakage_fp"], r)) for r in filt_rows]

    desc = [r for r in rows if r["filter_kind"] == "description"]
    none_stops = [r for r in rows if r["filter_kind"] == "none"]
    summary["description_stops"] = len(desc)
    summary["unfiltered_stops"] = len(none_stops)
    summary["aggregate"] = [dict(zip(
        ["model", "filter_source", "scored_stops", "exact_match", "exact_match_pct",
         "mean_precision", "mean_recall", "broadened", "broadened_pct",
         "stops_with_leakage", "total_leaked_nodes", "gt_empty"], r)) for r in agg_rows]

    # write
    from openpyxl import Workbook
    from evaluation.evaluate_llm import _write_table
    wb = Workbook(); wb.remove(wb.active)
    _write_table(wb.create_sheet("aggregate"),
                 ["model", "filter_source", "scored_stops", "exact_match", "exact_%",
                  "mean_precision", "mean_recall", "broadened", "broadened_%",
                  "stops_with_leakage", "leaked_nodes", "gt_empty"], agg_rows,
                 "Filter cascade vs row-level ground truth (structured stops)")
    _write_table(wb.create_sheet("by_filter"),
                 ["category", "gold_filter", "stops", "n_base", "n_gt", "n_sys",
                  "broadened", "leakage_fp"], filt_rows,
                 "Per distinct filter — where leakage and silent broadening happen")
    detail_cols = ["model", "query_id", "stop_index", "category", "filter_kind",
                   "gold_filter", "parser_filter", "filter_match",
                   "gold_n_base", "gold_n_gt", "gold_n_sys", "gold_exact_match",
                   "gold_precision", "gold_recall", "gold_leakage_fp", "gold_broadened",
                   "parser_n_sys", "parser_exact_match", "parser_precision",
                   "parser_recall", "parser_broadened", "events_gold", "events_parser"]
    _write_table(wb.create_sheet("per_stop"), detail_cols,
                 [[r.get(c) for c in detail_cols] for r in rows], "Per-stop detail")
    _write_table(wb.create_sheet("legend"), ["term", "meaning"], [
        ["base", "category-only candidate set (real build_poi_type_nodes)"],
        ["gt", "ROW-level truth: category AND column=value on the SAME POI row"],
        ["sys_gold", "cascade replay driven by the gold filter (set3_meta.json)"],
        ["sys_parser", "cascade replay driven by the parser's filter (what actually ran)"],
        ["broadened", "sys == base while gt is strictly smaller — the filter silently "
                      "had NO effect (filtration.py:122-129 keeps the broader set)"],
        ["leakage_fp", "nodes in sys but not gt — caused by filtration.py:122 "
                       "intersecting at NODE granularity, so one POI's category can "
                       "combine with a different POI's attribute at the same node"],
        ["gt_empty", "no POI anywhere satisfies the filter: broadening is CORRECT here, "
                     "so these are excluded from the broadened rate's meaning"],
        ["description stops", "reported separately — embedding-based, no deterministic "
                              "ground truth exists"],
    ], "Definitions")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    Path(args.summary_out).write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[done] {out}")
    for r in agg_rows:
        print(f"  {r[0]:22s} {r[1]:6s} stops={r[2]:4d} exact={r[4]:5}%  "
              f"P={r[5]} R={r[6]}  broadened={r[7]} ({r[8]}%)  leaked_nodes={r[10]}")


if __name__ == "__main__":
    main()
