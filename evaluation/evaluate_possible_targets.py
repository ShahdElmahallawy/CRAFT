#!/usr/bin/env python3
"""
evaluate_possible_targets.py — evaluate CRAFT's POI candidate-set component.

CRAFT builds `possible_targets[stop_index] -> set(graph node IDs)` ONCE per query,
before the Dijkstra search starts (craft/generator/tools/template.py:195 →
craft/generator/tools/filtration.py::build_possible_targets). That candidate set
bounds everything the search can ever find, yet it is never persisted and has never
been evaluated — so today a bad candidate set is indistinguishable from a codegen
failure in the results.

This evaluator measures PHASE 1: category resolution (no filter cascade). It answers
"given a POI category, does the component retrieve the right nodes, and what search-
space reduction does it buy?" under TWO explicit definitions of ground truth:

  D1 "literal"  — what CRAFT actually does: build_poi_type_nodes matches the LITERAL
                  OSM column value (common_handlers.py:294-301).
  D2 "catalog"  — what the benchmark generator assumed: data/socal/catalog_pois.json
                  groups categories far more broadly (supermarket also covers
                  shop=convenience / department_store / mall / variety_store, ...).

Neither is asserted to be correct; the GAP between them is the finding.

METHOD — replay, not re-run. build_possible_targets is deterministic given
(G, pois_gdf, poi_types, poi_filters), and the stored records already contain those
inputs under `parser_output`. So we replay the real component against the real data.
Nothing under outputs/ or results/ is modified; no locked file is touched.

NO GRAPH LOAD. build_possible_targets touches G only via resolve_graph_node_id, which
uses exactly `node_id in G` and `G.graph["comp_to_canonical"]`. A duck-typed NodeIndex
built from the `nodes` layer alone replaces build_graph_from_gpkg (0.1s vs minutes/GBs).
`--full-graph` validates the shim against the real graph.

NODES vs POIs: 627,220 POI rows collapse onto ~89k distinct nodes (many POIs share a
nearest node). Catalog counts are POI counts; candidate sets are deduplicated NODE
sets. Both are reported — never compare one against the other.

Usage:
  python evaluation/evaluate_possible_targets.py
  python evaluation/evaluate_possible_targets.py --verify-replay 10
  python evaluation/evaluate_possible_targets.py --full-graph
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "craft" / "generator" / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# APPEND (never insert): filtration.py does a bare `from common_handlers import ...`
# inside its function body, so this dir must be importable — but it also holds the
# generic names template.py / result.py, which must not shadow anything.
if str(TOOLS_DIR) not in sys.path:
    sys.path.append(str(TOOLS_DIR))

DEFAULT_GPKG     = ROOT / "data" / "socal" / "road_network.gpkg"
DEFAULT_CATALOG  = ROOT / "data" / "socal" / "catalog_pois.json"
DEFAULT_QUERIES  = ROOT / "benchmark" / "socal_queries" / "set1" / "ordered_poi.json"
DEFAULT_RECORDS  = ROOT / "outputs" / "craft"
DEFAULT_OUT      = ROOT / "results" / "craft_possible_targets_algo2.xlsx"
DEFAULT_SUMMARY  = ROOT / "results" / "craft_possible_targets_algo2_summary.json"
DEFAULT_CACHE    = ROOT / "outputs" / "evaluation" / "possible_targets_cache"

NODE_COL = "nearest_node_osmid"


# node index (duck-typed stand-in for the road graph)

class NodeIndex:
    """Minimal stand-in for G exposing only what resolve_graph_node_id needs:
    `node_id in G` and `G.graph["comp_to_canonical"]`. Replicates
    build_graph_from_gpkg:96-114 without building a MultiDiGraph."""

    def __init__(self, canonical: Set[Any], comp_to_canonical: Dict[Any, Any]) -> None:
        self._canonical = canonical
        self.graph: Dict[str, Any] = {"comp_to_canonical": comp_to_canonical}

    def __contains__(self, key: Any) -> bool:
        return key in self._canonical

    def __len__(self) -> int:
        return len(self._canonical)

    @property
    def canonical(self) -> Set[Any]:
        return self._canonical


def build_node_index(gpkg: Path) -> NodeIndex:
    import pyogrio
    from common_handlers import _normalize_node_ids

    df = pyogrio.read_dataframe(gpkg, layer="nodes", columns=["osmid_original"],
                                read_geometry=False)
    canonical: Set[Any] = set()
    c2c: Dict[Any, Any] = {}
    for raw in df["osmid_original"]:
        ids = _normalize_node_ids(raw)
        if not ids:
            continue
        can = ids[0]
        for nid in ids:
            c2c[nid] = can
            c2c[str(nid)] = can
        canonical.add(can)
    return NodeIndex(canonical, c2c)


# data loading

def load_pois(gpkg: Path, extra_cols: List[str]):
    """Load only the columns needed. Returns (dataframe, available_field_set)."""
    import pyogrio
    from common_handlers import CATEGORY_COLS

    available = set(pyogrio.read_info(gpkg, layer="pois")["fields"])
    cols = [c for c in CATEGORY_COLS if c in available]
    for c in [NODE_COL, "name", "lon", "lat"]:
        if c in available and c not in cols:
            cols.append(c)
    for c in extra_cols:
        if c in available and c not in cols:
            cols.append(c)
    df = pyogrio.read_dataframe(gpkg, layer="pois", columns=cols, read_geometry=False)
    return df, available


def raw_to_canonical(node_index: NodeIndex, raw_values) -> Dict[Any, Any]:
    """Resolve each DISTINCT raw nearest_node_osmid once (~89k) instead of per row."""
    from common_handlers import resolve_graph_node_id
    return {raw: resolve_graph_node_id(node_index, raw) for raw in raw_values}


# D1: literal (the real CRAFT function, cached)

def _cache_key(gpkg: Path, cats: List[str]) -> str:
    st = gpkg.stat()
    ch = (TOOLS_DIR / "common_handlers.py").read_bytes()
    h = hashlib.sha256()
    h.update(f"{st.st_size}:{int(st.st_mtime)}:{','.join(sorted(cats))}:".encode())
    h.update(hashlib.sha256(ch).hexdigest().encode())
    return h.hexdigest()[:16]


def literal_type_nodes(node_index: NodeIndex, pois, cats: List[str], gpkg: Path,
                       cache_dir: Path, no_cache: bool = False) -> Tuple[Dict[str, Set[Any]], float]:
    """D1 — call the REAL build_poi_type_nodes (this is CRAFT's actual behaviour,
    replayed rather than reimplemented). ~44s: one iterrows() over 627k rows."""
    from common_handlers import build_poi_type_nodes

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"literal_type_nodes_{_cache_key(gpkg, cats)}.json"
    if cache_file.exists() and not no_cache:
        raw = json.loads(cache_file.read_text())
        return {k: set(_decode(v)) for k, v in raw.items()}, 0.0

    t0 = time.perf_counter()
    tn = build_poi_type_nodes(node_index, pois, cats, NODE_COL)
    elapsed = time.perf_counter() - t0
    cache_file.write_text(json.dumps(
        {k: sorted(_encode(x) for x in v) for k, v in tn.items()}))
    return tn, elapsed


def _encode(v: Any) -> str:
    return f"i:{v}" if isinstance(v, int) else f"s:{v}"


def _decode(vals: List[str]) -> List[Any]:
    return [int(v[2:]) if v.startswith("i:") else v[2:] for v in vals]


# D1': independent reimplementation (cross-check)

def independent_type_nodes(pois, cats: List[str], canon_map: Dict[Any, Any]) -> Dict[str, Set[Any]]:
    """Independently recompute the literal rule with vectorised pandas string ops
    instead of reusing build_poi_type_nodes, so a genuine bug in the real code
    surfaces as a mismatch rather than agreeing with itself.

    Node RESOLUTION is deliberately shared (canon_map) so any divergence isolates
    to the CATEGORY-matching logic, which is what this check is for.
    """
    from common_handlers import CATEGORY_COLS

    out: Dict[str, Set[Any]] = {c: set() for c in cats}
    cat_cols = [c for c in CATEGORY_COLS if c in pois.columns]
    node_raw = pois[NODE_COL]

    for col in cat_cols:
        # Replicate split_multi_value: "|" and "," both act as ";" separators,
        # parts are stripped, empties dropped.
        norm = (pois[col].astype(str).str.lower()
                .str.replace("|", ";", regex=False)
                .str.replace(",", ";", regex=False))
        for cat in cats:
            pat = r"(?:^|;)\s*" + re.escape(cat) + r"\s*(?:;|$)"
            mask = norm.str.contains(pat, regex=True, na=False)
            if not mask.any():
                continue
            for raw in node_raw[mask]:
                nid = canon_map.get(raw)
                if nid is not None:
                    out[cat].add(nid)
    return out


# D2: catalog definition

def catalog_type_nodes(catalog_path: Path, cats: List[str], node_index: NodeIndex
                       ) -> Dict[str, Dict[str, Any]]:
    from common_handlers import resolve_graph_node_id

    catalog = json.loads(catalog_path.read_text())
    out: Dict[str, Dict[str, Any]] = {}
    for cat in cats:
        entries = catalog.get(cat, [])
        nodes: Set[Any] = set()
        for e in entries:
            nid = resolve_graph_node_id(node_index, e.get("node_id"))
            if nid is not None:
                nodes.add(nid)
        out[cat] = {
            "poi_rows": len(entries),
            "nodes": nodes,
            "subtypes": Counter(e.get("subtype") for e in entries),
        }
    return out


# records / queries

def load_queries(path: Path) -> Dict[str, str]:
    return {q["query_id"]: q["nl_query"] for q in json.loads(path.read_text())}


def load_records(records_dir: Path, model: str, template: str,
                 valid_ids: Set[str]) -> Dict[str, dict]:
    d = records_dir / model / template
    out: Dict[str, dict] = {}
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.json")):
        if f.stem == "summary":
            continue
        try:
            rec = json.loads(f.read_text())
        except Exception:
            continue
        qid = rec.get("query_id")
        if qid in valid_ids:
            out[qid] = rec
    return out


def filters_for_stop(poi_filters: Any, idx: int) -> Dict[str, Any]:
    """Mirror filtration.py::_filters_for_stop for the list/dict/str input shapes."""
    if isinstance(poi_filters, list):
        if 0 <= idx < len(poi_filters):
            v = poi_filters[idx]
            if v is None:
                return {}
            if isinstance(v, str):
                return {"description": v}
            if isinstance(v, dict):
                return dict(v)
        return {}
    if isinstance(poi_filters, dict):
        v = poi_filters.get(str(idx))
        if v is None:
            return {}
        if isinstance(v, str):
            return {"description": v}
        if isinstance(v, dict):
            return dict(v)
    return {}


# verification: replay against the real build_possible_targets

def verify_replay(node_index: NodeIndex, pois, records: Dict[str, dict],
                  literal: Dict[str, Set[Any]], k: int, seed: int = 42) -> Dict[str, Any]:
    """Run K queries through the REAL build_possible_targets and assert per-stop set
    equality against our replayed candidate sets. Captures stdout so [filtration]
    warnings can be cross-checked against the predicted unknown_column flags."""
    import random
    from filtration import build_possible_targets

    mod = sys.modules["filtration"].__file__
    expected = str(TOOLS_DIR / "filtration.py")
    if Path(mod).resolve() != Path(expected).resolve():
        raise RuntimeError(f"filtration.py shadowed: imported {mod}, expected {expected}")

    qids = sorted(records)
    rng = random.Random(seed)
    sample = qids if k >= len(qids) else rng.sample(qids, k)

    mismatches: List[str] = []
    warnings: Dict[str, List[str]] = {}
    for qid in sample:
        po = records[qid].get("parser_output") or {}
        ptypes = [str(t).strip().lower() for t in (po.get("poi_types") or [])]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got = build_possible_targets(
                G=node_index, pois_gdf=pois, poi_types=ptypes,
                node_col=NODE_COL, poi_filters=po.get("poi_filters"),
                embedding_cache=None, api_key=None,
            )
        lines = [l for l in buf.getvalue().splitlines()
                 if l.startswith("[filtration]") or l.startswith("[poi_embeddings]")]
        if lines:
            warnings[qid] = lines
        for i, t in enumerate(ptypes):
            if got.get(i, set()) != literal.get(t, set()):
                mismatches.append(f"{qid}[{i}]:{t}")
    return {
        "sampled": len(sample),
        "mismatches": mismatches,
        "warning_queries": len(warnings),
        "warning_examples": dict(list(warnings.items())[:5]),
    }


def verify_full_graph(gpkg: Path, node_index: NodeIndex, pois,
                      cats: List[str], literal: Dict[str, Set[Any]]) -> Dict[str, Any]:
    """One-off: build the REAL graph and assert the shim is equivalent."""
    from common_handlers import build_graph_from_gpkg, build_poi_type_nodes

    t0 = time.perf_counter()
    G = build_graph_from_gpkg(str(gpkg))
    load_s = time.perf_counter() - t0
    same_nodes = set(G.nodes()) == node_index.canonical
    same_c2c = G.graph.get("comp_to_canonical") == node_index.graph["comp_to_canonical"]
    real = build_poi_type_nodes(G, pois, cats, NODE_COL)
    same_types = {c: (real.get(c, set()) == literal.get(c, set())) for c in cats}
    return {
        "graph_load_s": round(load_s, 1),
        "canonical_node_sets_identical": same_nodes,
        "comp_to_canonical_identical": same_c2c,
        "type_nodes_identical": same_types,
        "all_ok": bool(same_nodes and same_c2c and all(same_types.values())),
    }


# export

def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:
        return "unknown"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def write_xlsx(out: Path, sheets: List[Tuple[str, List[str], List[list], str]]) -> None:
    from openpyxl import Workbook
    from evaluation.evaluate_llm import _write_table

    wb = Workbook()
    wb.remove(wb.active)
    for name, headers, rows, title in sheets:
        ws = wb.create_sheet(name[:31])
        _write_table(ws, headers, rows, title)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)


# main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", default="ordered_poi")
    ap.add_argument("--model", default="gemini-3_1-flash-lite")
    ap.add_argument("--gpkg", default=str(DEFAULT_GPKG))
    ap.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    ap.add_argument("--queries", default=str(DEFAULT_QUERIES))
    ap.add_argument("--records-dir", default=str(DEFAULT_RECORDS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--summary-out", default=str(DEFAULT_SUMMARY))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--verify-replay", type=int, default=10,
                    help="replay K queries through the real build_possible_targets (0=skip)")
    ap.add_argument("--full-graph", action="store_true",
                    help="one-off: validate the NodeIndex shim against the real graph")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="overwrite --out/--summary-out (backs up first)")
    args = ap.parse_args()

    gpkg = Path(args.gpkg)
    out_path = Path(args.out)
    summary_path = Path(args.summary_out)

    for p in (out_path, summary_path):
        if p.exists() and not args.force:
            sys.exit(f"ERROR: {p} exists. Use --force (it will be backed up first).")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for p in (out_path, summary_path):
        if p.exists():
            shutil.copy2(p, p.with_suffix(p.suffix + f".bak.{stamp}"))

    t_start = time.perf_counter()

    # 1. queries + records
    queries = load_queries(Path(args.queries))
    records = load_records(Path(args.records_dir), args.model, args.template, set(queries))
    if args.limit:
        records = {k: records[k] for k in sorted(records)[:args.limit]}
    print(f"[load] {len(queries)} benchmark queries, {len(records)} records "
          f"({args.model}/{args.template})")

    # 2. categories in play: union of gold (NL) and parser types
    from evaluation.constraint_checker import required_poi_types
    gold_by_q = {qid: required_poi_types(nl) for qid, nl in queries.items()}
    cats: Set[str] = set()
    for g in gold_by_q.values():
        cats.update(g)
    filter_cols: Set[str] = set()
    for rec in records.values():
        po = rec.get("parser_output") or {}
        cats.update(str(t).strip().lower() for t in (po.get("poi_types") or []))
        pf = po.get("poi_filters")
        for i in range(len(po.get("poi_types") or [])):
            filter_cols.update(k for k in filters_for_stop(pf, i) if k != "description")
    cats_l = sorted(cats)
    print(f"[cats] {cats_l}   filter columns seen: {sorted(filter_cols)}")

    # 3. data
    node_index = build_node_index(gpkg)
    print(f"[nodes] {len(node_index)} canonical nodes")
    pois, available = load_pois(gpkg, sorted(filter_cols))
    print(f"[pois] {len(pois)} rows x {len(pois.columns)} cols")
    unknown_cols = sorted(c for c in filter_cols if c not in available)
    if unknown_cols:
        print(f"[pois] filter columns NOT in gpkg (skipped by filtration): {unknown_cols}")

    canon_map = raw_to_canonical(node_index, pois[NODE_COL].dropna().unique())

    # 4. D1 literal (real function), D1' independent, D2 catalog
    literal, secs = literal_type_nodes(node_index, pois, cats_l, gpkg,
                                       Path(args.cache_dir), args.no_cache)
    print(f"[D1] build_poi_type_nodes {'(cached)' if secs == 0 else f'{secs:.1f}s'}: "
          + ", ".join(f"{c}={len(literal.get(c, set()))}" for c in cats_l))

    indep = independent_type_nodes(pois, cats_l, canon_map)
    indep_agree = {c: (indep.get(c, set()) == literal.get(c, set())) for c in cats_l}
    print(f"[D1'] independent mask agreement: {indep_agree}")

    catalog = catalog_type_nodes(Path(args.catalog), cats_l, node_index)
    print("[D2] catalog: " + ", ".join(
        f"{c}={catalog[c]['poi_rows']}rows/{len(catalog[c]['nodes'])}nodes" for c in cats_l))

    # 5. POI-row counts per literal category (for the nodes-vs-POIs column)
    from common_handlers import CATEGORY_COLS
    cat_cols = [c for c in CATEGORY_COLS if c in pois.columns]
    literal_rows: Dict[str, int] = {}
    for cat in cats_l:
        m = None
        for col in cat_cols:
            norm = (pois[col].astype(str).str.lower()
                    .str.replace("|", ";", regex=False)
                    .str.replace(",", ";", regex=False))
            hit = norm.str.contains(r"(?:^|;)\s*" + re.escape(cat) + r"\s*(?:;|$)",
                                    regex=True, na=False)
            m = hit if m is None else (m | hit)
        literal_rows[cat] = int(m.sum()) if m is not None else 0

    n_nodes = len(node_index)

    # sheet: category
    cat_headers = ["category", "catalog_poi_rows", "catalog_nodes", "literal_poi_rows",
                   "literal_nodes", "node_recall_vs_catalog", "node_precision_vs_catalog",
                   "prune_ratio", "columns_spanned", "subtypes_matched_by_literal",
                   "subtypes_missed_by_literal"]
    cat_rows = []
    for cat in cats_l:
        lit = literal.get(cat, set())
        cnodes = catalog[cat]["nodes"]
        inter = lit & cnodes
        subs = catalog[cat]["subtypes"]
        matched = {s: n for s, n in subs.items() if s and s.split("=", 1)[-1] == cat}
        missed = {s: n for s, n in subs.items() if s and s.split("=", 1)[-1] != cat}
        cat_rows.append([
            cat, catalog[cat]["poi_rows"], len(cnodes), literal_rows.get(cat, 0), len(lit),
            round(len(inter) / len(cnodes), 4) if cnodes else None,
            round(len(inter) / len(lit), 4) if lit else None,
            round(len(lit) / n_nodes, 6),
            ";".join(sorted({s.split("=", 1)[0] for s in subs if s})),
            ";".join(f"{s}:{n}" for s, n in sorted(matched.items())),
            ";".join(f"{s}:{n}" for s, n in sorted(missed.items())),
        ])

    # sheets: stops + queries
    stop_headers = ["query_id", "stop_index", "gold_type", "parser_type", "type_correct",
                    "candidate_nodes_literal", "candidate_poi_rows_literal",
                    "catalog_nodes", "node_recall_vs_catalog", "empty_set",
                    "filter_present", "filter_cols", "unknown_column", "prune_ratio"]
    q_headers = ["query_id", "gold_types", "parser_types", "exact_ordered_match",
                 "n_stops", "any_empty_set", "any_unknown_column",
                 "min_candidate_nodes", "union_candidate_nodes",
                 "parser_degenerate", "ran", "feasible", "constraint_satisfaction"]
    stop_rows, q_rows = [], []

    for qid in sorted(records):
        rec = records[qid]
        po = rec.get("parser_output") or {}
        ptypes = [str(t).strip().lower() for t in (po.get("poi_types") or [])]
        pf = po.get("poi_filters")
        gold = gold_by_q.get(qid, [])
        degenerate = not ptypes
        union: Set[Any] = set()
        mins: List[int] = []
        any_empty = any_unknown = False

        for i, pt in enumerate(ptypes):
            gt = gold[i] if i < len(gold) else None
            lit = literal.get(pt, set())
            cnodes = catalog.get(pt, {}).get("nodes", set())
            flt = filters_for_stop(pf, i)
            fcols = [c for c in flt if c != "description"]
            unknown = [c for c in fcols if c not in available]
            union |= lit
            mins.append(len(lit))
            any_empty |= (len(lit) == 0)
            any_unknown |= bool(unknown)
            stop_rows.append([
                qid, i, gt, pt, (gt == pt) if gt is not None else None,
                len(lit), literal_rows.get(pt, 0), len(cnodes),
                round(len(lit & cnodes) / len(cnodes), 4) if cnodes else None,
                len(lit) == 0, bool(flt), ";".join(sorted(fcols)) or None,
                ";".join(sorted(unknown)) or None,
                round(len(lit) / n_nodes, 6),
            ])

        m = rec.get("metrics") or {}
        q_rows.append([
            qid, ";".join(gold), ";".join(ptypes), gold == ptypes, len(ptypes),
            any_empty, any_unknown, min(mins) if mins else None, len(union),
            degenerate, m.get("ran"), m.get("feasible"), m.get("constraint_satisfaction"),
        ])

    # verification
    sanity: List[list] = [
        ["canonical_nodes", n_nodes, ""],
        ["poi_rows", len(pois), ""],
        ["distinct_nearest_nodes", len(set(v for v in canon_map.values() if v is not None)),
         "POI->node collapse"],
        ["independent_mask_agrees", all(indep_agree.values()), json.dumps(indep_agree)],
        ["filtration_py_sha", _sha(TOOLS_DIR / "filtration.py"), "replay validity is tied to this"],
        ["common_handlers_py_sha", _sha(TOOLS_DIR / "common_handlers.py"), ""],
    ]
    for cat in cats_l:
        sanity.append([f"catalog_poi_rows[{cat}]", catalog[cat]["poi_rows"],
                       "cross-check vs data/socal/catalog_count.json"])

    replay = None
    if args.verify_replay:
        print(f"[verify] replaying {args.verify_replay} queries through the real "
              f"build_possible_targets ...")
        replay = verify_replay(node_index, pois, records, literal, args.verify_replay)
        sanity.append(["verify_replay_sampled", replay["sampled"], ""])
        sanity.append(["verify_replay_mismatches", len(replay["mismatches"]),
                       ";".join(replay["mismatches"][:5])])
        print(f"[verify] mismatches: {len(replay['mismatches'])}  "
              f"queries emitting warnings: {replay['warning_queries']}")

    full_graph = None
    if args.full_graph:
        print("[verify] building the REAL graph to validate the NodeIndex shim ...")
        full_graph = verify_full_graph(gpkg, node_index, pois, cats_l, literal)
        sanity.append(["full_graph_shim_ok", full_graph["all_ok"], json.dumps(full_graph)])
        print(f"[verify] shim equivalence: {full_graph['all_ok']}")

    # write
    legend = [
        ["D1 literal", "build_poi_type_nodes: matches the LITERAL OSM column value. What CRAFT does."],
        ["D2 catalog", "catalog_pois.json grouping, used by generate_set1.py to build the benchmark."],
        ["nodes vs POIs", "Catalog counts are POI rows; candidate sets are deduplicated NODE sets. "
                          "Never compare a POI count against a node count."],
        ["node_recall_vs_catalog", "|D1 & D2| / |D2| over NODE sets."],
        ["prune_ratio", "|candidate nodes| / canonical graph nodes — the search-space reduction."],
        ["empty_set", "Candidate set is empty => the query is guaranteed infeasible before search."],
        ["unknown_column", "Parser named a filter column absent from the gpkg; filtration.py:106-108 "
                           "skips it (the filter silently has no effect)."],
        ["type_correct", "parser poi_type == NL-derived gold (constraint_checker.required_poi_types)."],
        ["Interpretation", "The literal rule is arguably the more precise reading of 'a supermarket'; "
                           "the catalog grouping is the benchmark generator's notion. The GAP is the "
                           "finding, not a verdict that either side is wrong."],
    ]
    write_xlsx(out_path, [
        ("category", cat_headers, cat_rows,
         f"Category resolution: literal (CRAFT) vs catalog (benchmark) — {args.model}"),
        ("stops", stop_headers, stop_rows, f"Per-stop candidate sets — {args.template}"),
        ("queries", q_headers, q_rows, f"Per-query rollup — {args.template}"),
        ("sanity", ["check", "value", "note"], sanity, "Verification"),
        ("legend", ["term", "definition"], legend, "Column definitions & caveats"),
    ])

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_manifest": {
            "git_rev": _git_rev(),
            "argv": sys.argv[1:],
            "gpkg": str(gpkg),
            "gpkg_size": gpkg.stat().st_size,
            "gpkg_mtime": int(gpkg.stat().st_mtime),
            "filtration_py_sha256_16": _sha(TOOLS_DIR / "filtration.py"),
            "common_handlers_py_sha256_16": _sha(TOOLS_DIR / "common_handlers.py"),
            "elapsed_s": round(time.perf_counter() - t_start, 1),
        },
        "scope": {"model": args.model, "template": args.template,
                  "queries": len(queries), "records": len(records),
                  "stops": len(stop_rows), "categories": cats_l},
        "category": {
            cat: {
                "catalog_poi_rows": catalog[cat]["poi_rows"],
                "catalog_nodes": len(catalog[cat]["nodes"]),
                "literal_poi_rows": literal_rows.get(cat, 0),
                "literal_nodes": len(literal.get(cat, set())),
                "node_recall_vs_catalog": (
                    round(len(literal.get(cat, set()) & catalog[cat]["nodes"])
                          / len(catalog[cat]["nodes"]), 4) if catalog[cat]["nodes"] else None),
                "prune_ratio": round(len(literal.get(cat, set())) / n_nodes, 6),
            } for cat in cats_l
        },
        "headline": {
            "type_correct_stops": sum(1 for r in stop_rows if r[4] is True),
            "total_stops": len(stop_rows),
            "empty_set_stops": sum(1 for r in stop_rows if r[9]),
            "unknown_column_stops": sum(1 for r in stop_rows if r[12]),
            "queries_exact_type_match": sum(1 for r in q_rows if r[3]),
        },
        "verification": {
            "independent_mask_agrees": indep_agree,
            "verify_replay": replay,
            "full_graph": full_graph,
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n[done] {out_path}")
    print(f"[done] {summary_path}")
    print(f"[done] {summary['headline']}")


if __name__ == "__main__":
    main()
