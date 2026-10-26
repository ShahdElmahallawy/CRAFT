"""
evaluation/optimal/poi_collection_dfs.py — exact maximum-POI DFS solver for
poi_collection (orienteering: visit as many POIs as possible under a time
budget and per-category caps), with two entry points sharing one algorithm:

  oracle  Gold-driven, leg-matrix-backed. Removes the parser from the loop
          entirely (see cfg_from_gold) and writes the authoritative
          evaluation/optimal/results/poi_collection_oracle.json.
  batch   CRAFT-parsed-input-driven, Dijkstra-cache-backed. Solves using the
          same (possibly parser-flawed) inputs CRAFT's own algo3 search saw,
          to isolate "is CRAFT's search suboptimal?" from "did the parser
          get the query right?". Writes outputs/bruteforce_dfs/ — see
          evaluation/optimal/poi_collection_dfs_RESULTS.md for the algo3-vs-
          optimum comparison this produced.

Both modes run the IDENTICAL exact search: size-descending DFS with
branch-and-bound (a count-aware ellipse + a prefix bound, both necessary
conditions, so nothing that could be optimal is ever discarded). They differ
only in where POI->POI travel times come from:
  - `oracle` reads them from the prebuilt leg matrix (evaluation/optimal/
    leg_matrix/legs.npz) — flat memory, no Dijkstra runs inside the search.
  - `batch` caches one single_source_dijkstra_path_length per POI node the
    DFS actually steps through. This is what made an earlier from-scratch
    oracle attempt run to 20GB RSS and get killed (a few hundred POIs at a
    2-hour budget on the 248k-node graph, each cache entry ~1e5 entries) —
    fine for individual/batch runs capped by the RAM watchdog below, wrong
    for regenerating the oracle output. `--cross-validate` in oracle mode
    runs both engines on the same queries and requires identical counts, so
    the equivalence between the two backends is checked, not just asserted.

Consolidated 2026-09-30 (user request — one file instead of three sharing one
algorithm) from baselines/bruteforce/{brute_force.py,run_batch.py} and
evaluation/optimal/algo3_oracle.py.

Usage:
  python evaluation/optimal/poi_collection_dfs.py oracle [--limit N] [--cross-validate N]
  python evaluation/optimal/poi_collection_dfs.py batch [--limit N] [--overwrite]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import networkx as nx
import pandas as pd

from craft.generator.tools.common_handlers import (
    build_graph_from_gpkg,
    build_poi_type_nodes,
    coerce_node_id,
    edge_weight,
)
from evaluation.constraint_checker import NodeSnapper
from evaluation.optimal.ev_oracle import GPKG, LEG_MATRIX_DIR

INF = float("inf")

GOLD_PATH = ROOT / "benchmark" / "ground_truth" / "set1_poi_collection_gold.json"
ORACLE_OUT_PATH = ROOT / "evaluation" / "optimal" / "results" / "poi_collection_oracle.json"
BATCH_DEFAULT_INPUTS_DIR = ROOT / "outputs" / "eval_staging_craft_socal"
BATCH_DEFAULT_IDS_FROM = ROOT / "benchmark" / "socal_queries" / "set1" / "poi_collection.json"
BATCH_DEFAULT_OUT_ROOT = ROOT / "outputs" / "bruteforce_dfs"
BATCH_DEFAULT_GPKG = ROOT / "data" / "socal" / "road_network.gpkg"

# ---------------------------------------------------------------------------
# Shared: LLMAP per-category visit durations + parser-tolerant config readers.
# The many key spellings in read_caps/read_poi_types cope with the parser's
# historical inconsistency across templates; batch mode is the only caller
# that ever sees parser-shaped input, but both modes call these readers.
# ---------------------------------------------------------------------------

LLMAP_VISIT_SEC = {
    "shopping_mall": 7200, "shopping mall": 7200,
    "supermarket": 1800, "pharmacy": 900, "bank": 1200, "library": 3600,
}


def _visit_of(t) -> float:
    return float(LLMAP_VISIT_SEC.get(str(t).lower(), 900))


def read_budget_seconds(cfg: Dict[str, Any]) -> float:
    for k in ("total_time_limit", "time_limit", "total_time", "max_time", "time_budget"):
        if cfg.get(k) is not None:
            return float(cfg[k])
    return 3600.0


def read_visit_seconds(cfg: Dict[str, Any]) -> float:
    for k in ("visit_time", "stop_time"):
        if cfg.get(k) is not None:
            return float(cfg[k])
    return 900.0


def _cap_value(n) -> int:
    return int(n["value"]) if isinstance(n, dict) else int(n)


def read_caps(cfg: Dict[str, Any]) -> Dict[str, int]:
    """The parser used many spellings. Try each dict-valued cap key, then a
    scalar total count, then fall back to repeat-counts in poi_types."""
    for k in ("category_limits", "poi_counts", "poi_limits", "stop_limits", "poi_visit_limits"):
        v = cfg.get(k)
        if isinstance(v, dict) and v:
            return {str(kk).lower(): _cap_value(vv) for kk, vv in v.items()}
    raw = [str(t).lower() for t in (cfg.get("poi_types") or cfg.get("stop_types") or [])]
    if cfg.get("poi_count_limit") is not None:     # scalar total, e.g. {value:3}
        n = _cap_value(cfg["poi_count_limit"])
        u = list(dict.fromkeys(raw))
        return {u[0]: n} if len(u) == 1 else {t: n for t in u}
    if raw:                                          # repeats encode the caps
        return dict(Counter(raw))
    return {}


def read_poi_types(cfg: Dict[str, Any]) -> List[str]:
    """poi_types / stop_types, deduped; some files omit it (types live only in
    the caps), so fall back to the capped categories."""
    t = [str(x).lower() for x in (cfg.get("poi_types") or cfg.get("stop_types") or [])]
    t = list(dict.fromkeys(t))
    return t or list(read_caps(cfg).keys())


def build_time_graph(G: nx.MultiDiGraph, default_speed_kph: float) -> nx.DiGraph:
    """Simple DiGraph with one weight per (u,v): the min travel time across
    parallel edges (uses the same edge_weight/fallback as algo3.py)."""
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())
    for u, v, data in G.edges(data=True):
        w = edge_weight(data, "travel_time", default_speed_kph)
        if w is None or w <= 0:
            continue
        if H.has_edge(u, v):
            if w < H[u][v]["w"]:
                H[u][v]["w"] = w
        else:
            H.add_edge(u, v, w=w)
    return H


# BATCH-mode engine — Dijkstra-cache-backed exact DFS over a CRAFT-parsed cfg.

def solve_query_dfs(
    G: nx.MultiDiGraph, pois_gdf, cfg: Dict[str, Any],
    *, default_speed_kph: float = 30.0, H: Optional[nx.DiGraph] = None,
    max_steps: int = 5_000_000, tol: float = 1e-6, verbose: bool = False,
    prefix_bound: bool = True,
) -> Dict[str, Any]:
    """Exact max-count via SIZE-DESCENDING DFS with branch-and-bound.

    #1 count-aware ellipse: when targeting `m` stops, a POI must satisfy
       SP(src->p) + SP(p->dst) + m*visit <= budget  (tighter than the 1-visit ellipse).
    #3 prefix bound: prune a partial route [.. -> q] once
       travel + SP(q->dst) + m*visit > budget  (a valid lower bound on any completion).

    Lazy collapse: only Dijkstra from POIs the DFS actually steps through, and try
    size m=L first — so full-cap queries touch only the small L-ellipse + few Dijkstras.
    Returns count=None with status 'search_limit' if the step budget is exceeded
    (never a wrong/suboptimal count)."""
    src = coerce_node_id(G, str(cfg["src"]))
    dst = coerce_node_id(G, str(cfg["dst"]))
    budget = read_budget_seconds(cfg)
    visit = read_visit_seconds(cfg)
    caps = read_caps(cfg)
    poi_types = read_poi_types(cfg)
    L = sum(caps.values())
    if H is None:
        H = build_time_graph(G, default_speed_kph)

    # setup: 2 Dijkstras (forward from src, reverse from dst)
    dist_src = nx.single_source_dijkstra_path_length(H, src, cutoff=budget, weight="w")
    dist_dst = nx.single_source_dijkstra_path_length(H.reverse(copy=False), dst, cutoff=budget, weight="w")
    d_src_dst = dist_src.get(dst, INF)

    type_nodes = build_poi_type_nodes(G, pois_gdf, poi_types, "nearest_node_osmid")
    poi_nodes = [(p, t) for t in poi_types for p in type_nodes.get(t, set()) if p not in (src, dst)]

    # cache: single-source SP from a POI node (only computed when DFS steps there)
    sp_cache: Dict[Any, Dict] = {}
    def sp_from(p):
        d = sp_cache.get(p)
        if d is None:
            d = nx.single_source_dijkstra_path_length(H, p, cutoff=budget, weight="w")
            sp_cache[p] = d
        return d

    steps = [0]
    best_route_order = None
    best_count = 0

    # per-category visit durations (LLMAP), and the cheapest possible visit — used
    # as a lower bound on the still-unchosen stops' visit time.
    min_visit = min((_visit_of(t) for t in poi_types), default=900.0)

    for target in range(L, 0, -1):
        #1 count-aware ellipse for this target size: a candidate's own per-category
        # visit + a lower bound on the other (target-1) visits must still fit.
        cand = [(p, t) for (p, t) in poi_nodes
                if dist_src.get(p, INF) + dist_dst.get(p, INF)
                   + _visit_of(t) + (target - 1) * min_visit <= budget + tol]
        avail = Counter(t for _, t in cand)
        if sum(min(avail[c], caps.get(c, 0)) for c in avail) < target:
            continue  # not enough eligible POIs (under caps) to reach this size

        cap_left = dict(caps)
        chosen: List[Any] = []

        def rec(last, travel, vsum):
            steps[0] += 1
            if steps[0] > max_steps:
                return "LIMIT"
            if len(chosen) == target:
                total = travel + dist_dst.get(last, INF) + vsum   # vsum = all chosen visits
                return list(chosen) if total <= budget + tol else None
            for (p, t) in cand:
                if p in chosen or cap_left.get(t, 0) <= 0:
                    continue
                step = dist_src.get(p, INF) if last == src else sp_from(last).get(p, INF)
                if step >= INF:
                    continue
                ntravel = travel + step
                nvsum = vsum + _visit_of(t)
                #3 prefix bound: closing from p, with p's per-category visit plus a
                # lower bound on the remaining stops' visits, must fit the budget.
                remaining = target - len(chosen) - 1
                if prefix_bound and (ntravel + dist_dst.get(p, INF)
                                     + nvsum + remaining * min_visit > budget + tol):
                    continue
                chosen.append(p); cap_left[t] -= 1
                r = rec(p, ntravel, nvsum)
                chosen.pop(); cap_left[t] += 1
                if r == "LIMIT":
                    return "LIMIT"
                if r is not None:
                    return r
            return None

        found = rec(src, 0.0, 0.0)
        if found == "LIMIT":
            return {"query_id": cfg.get("query_id"), "status": "search_limit",
                    "count": None, "n_candidates": len(cand), "steps": steps[0], "route": None}
        if found is not None:
            best_count, best_route_order = target, found
            break

    # stitch node route src -> stops -> dst
    route: List[Any] = []
    ok = True
    stops = best_route_order or []
    for a, b in zip([src] + stops, stops + [dst]):
        try:
            seg = nx.dijkstra_path(H, a, b, weight="w")
        except nx.NetworkXNoPath:
            ok = False; break
        route.extend(seg if not route else seg[1:])
    return {
        "query_id": cfg.get("query_id"),
        "status": "solved",
        "count": best_count,
        "L": L,
        "n_poi_nodes": len(poi_nodes),
        "steps": steps[0],
        "stops": [str(n) for n in stops],          # the final SELECTED POI nodes (in visit order)
        "route": [str(n) for n in route] if ok else None,
    }


# ---------------------------------------------------------------------------
# ORACLE-mode engine — leg-matrix-backed exact DFS over a GOLD-derived cfg.
# Same pruning as solve_query_dfs above (checked, not just asserted, via
# --cross-validate); only the POI->POI lookup source differs.
# ---------------------------------------------------------------------------

def solve_dfs_matrix(
    src: Any, dst: Any, caps: Dict[str, int], budget_s: float,
    d_src: Dict[Any, float], d_dst: Dict[Any, float],
    time_s: "np.ndarray", row_of: Dict[Any, int],
    cand_nodes: Dict[str, List[Any]], visit_of,
    max_steps: int = 5_000_000, tol: float = 1e-6,
) -> Dict[str, Any]:
    L = sum(caps.values())
    poi = [(p, t) for t, ns in cand_nodes.items() for p in ns if p not in (src, dst)]
    min_visit = min((visit_of(t) for t in caps), default=900.0)

    steps = [0]
    steps_by_target: Dict[int, int] = {}
    for target in range(L, 0, -1):
        _steps_at_target_start = steps[0]
        # count-aware ellipse: p must leave room for its own visit plus a lower
        # bound on the other target-1 visits
        cand = [(p, t) for (p, t) in poi
                if d_src.get(p, INF) + d_dst.get(p, INF)
                + visit_of(t) + (target - 1) * min_visit <= budget_s + tol
                and p in row_of]
        avail: Dict[str, int] = {}
        for _, t in cand:
            avail[t] = avail.get(t, 0) + 1
        if sum(min(avail.get(c, 0), caps.get(c, 0)) for c in avail) < target:
            continue

        cap_left = dict(caps)
        chosen: List[Any] = []

        def rec(last, travel, vsum):
            steps[0] += 1
            if steps[0] > max_steps:
                return "LIMIT"
            if len(chosen) == target:
                return list(chosen) if travel + d_dst.get(last, INF) + vsum <= budget_s + tol else None
            for (p, t) in cand:
                if p in chosen or cap_left.get(t, 0) <= 0:
                    continue
                step = (d_src.get(p, INF) if last == src
                        else float(time_s[row_of[last], row_of[p]]))
                if not (step < INF):
                    continue
                ntravel, nvsum = travel + step, vsum + visit_of(t)
                remaining = target - len(chosen) - 1
                if (ntravel + d_dst.get(p, INF) + nvsum
                        + remaining * min_visit > budget_s + tol):
                    continue
                chosen.append(p); cap_left[t] -= 1
                r = rec(p, ntravel, nvsum)
                chosen.pop(); cap_left[t] += 1
                if r == "LIMIT":
                    return "LIMIT"
                if r is not None:
                    return r
            return None

        found = rec(src, 0.0, 0.0)
        steps_by_target[target] = steps[0] - _steps_at_target_start
        if found == "LIMIT":
            return {"status": "search_limit", "count": None, "stops": None,
                    "steps": steps[0], "n_candidates": len(cand),
                    "steps_by_target": steps_by_target}
        if found is not None:
            return {"status": "solved", "count": target,
                    "stops": [str(n) for n in found], "steps": steps[0],
                    "n_candidates": len(cand), "steps_by_target": steps_by_target}

    return {"status": "solved", "count": 0, "stops": [], "steps": steps[0],
            "n_candidates": 0, "steps_by_target": steps_by_target}


def cfg_from_gold(gold: Dict[str, Any], snapper: NodeSnapper) -> Dict[str, Any]:
    """Build the solver's config straight from gold — no parser anywhere.

    Keys chosen to hit read_caps/read_budget_seconds/read_poi_types' FIRST
    lookup each, so none of their parser-tolerant fallbacks can fire:
      read_caps          -> "category_limits"
      read_budget_seconds-> "total_time_limit"
      read_poi_types     -> "poi_types"
    """
    caps = {str(k).lower(): int(v) for k, v in (gold.get("category_limits") or {}).items()}
    budget_min = gold.get("time_budget_min")
    return {
        "query_id": gold["query_id"],
        "src": str(snapper.snap(*gold["src_latlon"])),
        "dst": str(snapper.snap(*gold["dst_latlon"])),
        "category_limits": caps,
        "poi_types": sorted(caps),
        # this module reads seconds; gold stores minutes
        "total_time_limit": float(budget_min) * 60.0 if budget_min is not None else None,
        # visit_time is deliberately ABSENT: solve_query_dfs/solve_dfs_matrix use
        # the per-category LLMAP table (_visit_of) regardless, matching the scorer.
    }


# CLI — two subcommands sharing the engine above.

def _avail_mb() -> int:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) // 1024
    return 1 << 30


def main_batch(args: argparse.Namespace) -> None:
    """Run the exact staged-DFS solver over CRAFT-parsed poi_collection
    queries. Resume-safe (skips queries that already have output) and
    RAM-safe (stops before starting a query if system-available memory drops
    below --min-avail-mb)."""
    import geopandas as gpd

    ids = [q["query_id"] for q in json.loads(Path(args.ids_from).read_text())]
    if args.limit:
        ids = ids[: args.limit]
    out_dir = Path(args.out_root) / args.template
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading graph ...", flush=True); t0 = time.time()
    G = build_graph_from_gpkg(args.gpkg)
    pois = gpd.read_file(args.gpkg, layer="pois")
    H = build_time_graph(G, 30.0)
    print(f"  graph+time-graph ready in {time.time()-t0:.0f}s ({G.number_of_nodes()} nodes)", flush=True)

    n_solved = n_limit = n_err = n_skip = 0
    for i, qid in enumerate(ids, 1):
        out_f = out_dir / f"{qid}.json"
        if out_f.exists() and not args.overwrite:
            n_skip += 1; continue
        inp = Path(args.inputs_dir) / f"{qid}.json"
        if not inp.exists():
            continue
        am = _avail_mb()
        if am < args.min_avail_mb:
            print(f"[{i}/{len(ids)}] STOP: system RAM low ({am}MB < {args.min_avail_mb}MB). Resume later.", flush=True)
            break
        cfg = json.loads(inp.read_text())
        t = time.time()
        try:
            res = solve_query_dfs(G, pois, cfg, H=H, max_steps=args.max_steps)
        except Exception as exc:
            res = {"query_id": qid, "status": f"error: {exc}", "count": None, "route": None}
        res["elapsed_s"] = round(time.time() - t, 1)
        out_f.write_text(json.dumps(res, indent=2))
        st = res.get("status", "")
        if st == "solved":
            n_solved += 1
            print(f"[{i}/{len(ids)}] {qid}: OPT count={res['count']} "
                  f"(steps={res.get('steps')}, poi_nodes={res.get('n_poi_nodes')}, {res['elapsed_s']}s, avail={am}MB)", flush=True)
        elif st == "search_limit":
            n_limit += 1
            print(f"[{i}/{len(ids)}] {qid}: SEARCH_LIMIT (steps={res.get('steps')}, {res['elapsed_s']}s)", flush=True)
        else:
            n_err += 1
            print(f"[{i}/{len(ids)}] {qid}: {st}", flush=True)

    summary = {"template": args.template, "n": len(ids),
               "solved": n_solved, "search_limit": n_limit, "error": n_err, "skipped_existing": n_skip}
    (Path(args.out_root) / f"_summary_{args.template}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDONE: solved={n_solved}  search_limit={n_limit}  error={n_err}  skipped={n_skip}  of {len(ids)}")
    print(f"outputs -> {out_dir}")


def main_oracle(args: argparse.Namespace) -> None:
    """Gold-driven, leg-matrix-backed exact solve -> poi_collection_oracle.json.
    Unproven queries are reported as status "search_limit" with count=None —
    never guessed. Those are the genuine orienteering blow-ups at loose time
    budgets."""
    print("loading graph, POIs, leg matrix ...", flush=True)
    t0 = time.time()
    G = build_graph_from_gpkg(str(GPKG))
    # the 627k-row POI table is used ONLY during cross-validation (the batch
    # engine needs it); the matrix path gets its candidates from the leg-matrix
    # index, so skip the load otherwise (keeps parallel workers small and
    # startup short)
    pois = None
    if args.cross_validate:
        import geopandas as gpd
        pois = gpd.read_file(str(GPKG), layer="pois")
    snapper = NodeSnapper.from_gpkg(str(GPKG), G)
    npz = np.load(LEG_MATRIX_DIR / "legs.npz")
    time_s = npz["time_s"]
    idx = pd.read_parquet(LEG_MATRIX_DIR / "index.parquet")
    row_of = {n: int(r) for n, r in
              idx[["node_id", "row_idx"]].drop_duplicates("node_id").itertuples(index=False)}
    cand_by_type = {t: g.drop_duplicates("node_id")["node_id"].tolist()
                    for t, g in idx.groupby("poi_type")}
    print(f"  ready in {time.time() - t0:.1f}s", flush=True)

    gold_list = json.loads(GOLD_PATH.read_text())
    if args.only:
        gold_list = [g for g in gold_list if g["query_id"] in set(args.only)]
    if args.limit:
        gold_list = gold_list[: args.limit]

    Grev = G.reverse(copy=False)
    results: List[Dict[str, Any]] = []
    xval_ok = xval_bad = 0
    t0 = time.time()

    for i, gold in enumerate(gold_list):
        cfg = cfg_from_gold(gold, snapper)
        caps = cfg["category_limits"]
        budget = cfg["total_time_limit"]
        gold_L = sum(caps.values())
        src = coerce_node_id(G, cfg["src"])
        dst = coerce_node_id(G, cfg["dst"])

        # the only two graph searches: src and dst are arbitrary coordinates,
        # everything POI-to-POI comes from the matrix
        d_src = nx.single_source_dijkstra_path_length(G, src, cutoff=budget, weight="travel_time")
        d_dst = nx.single_source_dijkstra_path_length(Grev, dst, cutoff=budget, weight="travel_time")
        cand_nodes = {c: cand_by_type.get(c, []) for c in caps}

        _q0 = time.time()
        r = solve_dfs_matrix(src, dst, caps, budget, d_src, d_dst, time_s, row_of,
                             cand_nodes, _visit_of, max_steps=args.max_steps)
        r["wall_s"] = round(time.time() - _q0, 1)
        r["max_steps"] = args.max_steps
        r["query_id"] = cfg["query_id"]
        r["gold_category_limits"] = caps
        r["gold_total_cap"] = gold_L
        r["time_budget_s"] = budget
        r["objective"] = "maximize_poi"
        r["optimum_is_full_cap"] = (r["count"] == gold_L) if r["count"] is not None else None

        if i < args.cross_validate:
            ref = solve_query_dfs(G, pois, cfg, max_steps=args.max_steps)
            same = (ref.get("count") == r["count"])
            r["xval_bruteforce_count"] = ref.get("count")
            r["xval_match"] = same
            xval_ok += same; xval_bad += (not same)
            print(f"    xval {cfg['query_id']}: matrix={r['count']} dfs_cache={ref.get('count')} "
                  f"{'OK' if same else 'MISMATCH!!'}", flush=True)

        results.append(r)
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  {i + 1}/{len(gold_list)} ({el:.0f}s, {el/(i+1):.1f}s/query)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print(f"\ndone: {len(results)} queries in {time.time() - t0:.0f}s -> {out}")
    print("status:", dict(Counter(str(r["status"]) for r in results)))
    if args.cross_validate:
        print(f"cross-validation vs solve_query_dfs: {xval_ok} match, {xval_bad} MISMATCH")
    solved = [r for r in results if r["status"] == "solved"]
    if solved:
        print("optimal count distribution:",
              dict(sorted(Counter(r["count"] for r in solved).items())))
        full = sum(1 for r in solved if r["optimum_is_full_cap"])
        print(f"optimum == full cap on {full}/{len(solved)} "
              f"(the rest are where the time budget genuinely bites)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    op = sub.add_parser("oracle", help="gold-driven, leg-matrix-backed -> poi_collection_oracle.json")
    op.add_argument("--out", default=str(ORACLE_OUT_PATH))
    op.add_argument("--limit", type=int, default=None)
    # 100M, not 5M: four queries legitimately need 12-40M steps. With the 5M
    # default a plain re-run silently overwrote their proven optima with
    # search_limit (200/200 -> 196/200, no error raised). 100M clears the 40M
    # worst case now that the pathological 6_v03 has been replaced.
    op.add_argument("--max-steps", type=int, default=100_000_000)
    op.add_argument("--only", nargs="*", default=None, help="specific query_ids")
    op.add_argument("--cross-validate", type=int, default=0,
                    help="also run the Dijkstra-cache engine on the first N "
                         "queries and require identical counts")

    bp = sub.add_parser("batch", help="CRAFT-parsed-input-driven -> outputs/bruteforce_dfs/")
    bp.add_argument("--template", default="poi_collection")
    bp.add_argument("--inputs-dir", default=str(BATCH_DEFAULT_INPUTS_DIR))
    bp.add_argument("--ids-from", default=str(BATCH_DEFAULT_IDS_FROM))
    bp.add_argument("--out-root", default=str(BATCH_DEFAULT_OUT_ROOT))
    bp.add_argument("--max-steps", type=int, default=5_000_000)
    bp.add_argument("--min-avail-mb", type=int, default=6000, help="stop before a query if system RAM below this")
    bp.add_argument("--limit", type=int, default=None)
    bp.add_argument("--overwrite", action="store_true")
    bp.add_argument("--gpkg", default=str(BATCH_DEFAULT_GPKG))

    args = ap.parse_args()
    if args.mode == "oracle":
        main_oracle(args)
    else:
        main_batch(args)


if __name__ == "__main__":
    main()
