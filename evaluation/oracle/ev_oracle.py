"""
evaluation/oracle/ev_oracle.py — Step 2 of the CRAFT optimal-oracle plan:
exact EV-routing oracle for ev_routing.

Input: the PARSER GROUND-TRUTH gold records (benchmark/ground_truth/
set1_ev_routing_gold.json), NOT CRAFT's own parser output — so the
gap this measures is route-search quality, not parsing quality.

Objective: minimize total distance. Constraint: no leg's shortest-path
distance may exceed the available battery range; battery resets to full at
any charging station; the FIRST leg is capped by CURRENT range, every
subsequent leg by MAX range.

Method (meta-graph over {src, dst, candidate chargers}):
  1. One Dijkstra from src (forward) and one from dst (on the REVERSED
     graph), both weight="length" -> d(src,x) and d(x,dst) for every x in
     one shot each.
  2. Candidate chargers = {c : d(src,c) + d(c,dst) <= 1.5 * d(src,dst)} (a
     generous initial filter, not the final answer). Meta-graph edges:
     src->c and c->dst from step 1; c->c' from the prebuilt leg matrix
     (evaluation/oracle/leg_matrix/, built by build_leg_matrix.py).
  3. Exact solve: Dijkstra on the meta-graph, where edge (a,b) exists only
     if dist(a,b) <= current_range_m (a==src) or <= max_range_m (a is a
     charger, since it just recharged to full).
  4. Optimality closure: once a solution with cost B is found, re-check
     every EXCLUDED charger c against the proven necessary condition
     d(src,c)+d(c,dst) <= B (any chain through c costs at least that much);
     add any that now qualify and re-solve. Repeat until no new charger
     qualifies -- at that point B is optimal over all chargers, not just
     the pruned set. If the initial window yields no feasible chain at all,
     skip straight to solving with the full charger set (only ~400 nodes,
     cheap) before declaring infeasible.

Does NOT touch dijkstra_core.py / template.py / prompt files / State class.

Output: evaluation/oracle/results/ev_routing_oracle.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from craft.generator.tools.common_handlers import build_graph_from_gpkg
from evaluation.constraint_checker import MILES_TO_M, NodeSnapper

GPKG = ROOT / "data" / "socal" / "road_network.gpkg"
LEG_MATRIX_DIR = ROOT / "evaluation" / "oracle" / "leg_matrix"
GOLD_PATH = ROOT / "benchmark" / "ground_truth" / "set1_ev_routing_gold.json"
OUT_PATH = ROOT / "evaluation" / "oracle" / "results" / "ev_routing_oracle.json"

INF = float("inf")
INITIAL_WINDOW_MULT = 1.5
EPS = 1e-6

SRC_KEY = "__SRC__"
DST_KEY = "__DST__"


def load_leg_matrix() -> Tuple[np.ndarray, pd.DataFrame, Dict[Any, int]]:
    npz = np.load(LEG_MATRIX_DIR / "legs.npz")
    dist_m = npz["dist_m"]
    idx_df = pd.read_parquet(LEG_MATRIX_DIR / "index.parquet")
    node_to_row: Dict[Any, int] = {}
    for row_idx, node_id in idx_df[["row_idx", "node_id"]].drop_duplicates().itertuples(index=False):
        node_to_row[node_id] = int(row_idx)
    return dist_m, idx_df, node_to_row


def battery_to_meters(rec: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """(current_m, max_m), or (None, None if max is missing -- 8/200 gold
    records state battery as a % but never state the max range at all, a
    template gap found in Step 0, not recoverable from any parse."""
    max_mi = rec.get("max_battery_mi")
    if max_mi is None:
        return None, None
    max_m = max_mi * MILES_TO_M
    cur_mi = rec.get("current_battery_mi")
    if cur_mi is not None:
        return cur_mi * MILES_TO_M, max_m
    pct = rec.get("current_battery_pct")
    if pct is not None:
        return (pct / 100.0) * max_m, max_m
    return None, None


def solve_meta(
    candidates: Set[Any], src: Any, dst: Any,
    current_m: float, max_m: float,
    d_src: Dict[Any, float], d_dst: Dict[Any, float],
    dist_m: np.ndarray, node_to_row: Dict[Any, int],
) -> Tuple[Optional[float], Optional[List[Any]]]:
    """One exact solve over a FIXED candidate charger set. Returns
    (total_distance_m, [src, c1, ..., ck, dst]) or (None, None) if dst is
    unreachable within this candidate set under the range constraints.

    Lexicographic (distance, hop_count) Dijkstra, NOT plain single-criterion
    Dijkstra: a charger that sits exactly ON the unconstrained shortest path
    adds zero extra distance, so a charger chain routing through it is tied
    with the direct edge on total distance. Plain Dijkstra breaks such ties
    arbitrarily and can report spurious "needed" charging stops that add
    nothing (caught empirically: a real query where a 2-charger chain tied
    the direct distance exactly, 13810.226 == 13810.226). Minimizing hops as
    a secondary key picks the fewest-stops representative among all paths
    achieving the true minimum distance -- distance optimality is unaffected,
    this only disambiguates ties in the reported chain."""
    import heapq

    edges: Dict[Any, List[Tuple[Any, float]]] = {SRC_KEY: [], DST_KEY: []}

    direct = d_src.get(dst, INF)
    if direct < INF and direct <= current_m + EPS:
        edges[SRC_KEY].append((DST_KEY, direct))

    cand_list = list(candidates)
    for c in cand_list:
        edges.setdefault(c, [])
        ds = d_src.get(c, INF)
        if ds < INF and ds <= current_m + EPS:
            edges[SRC_KEY].append((c, ds))
        dd = d_dst.get(c, INF)
        if dd < INF and dd <= max_m + EPS:
            edges[c].append((DST_KEY, dd))

    for c in cand_list:
        rc = node_to_row.get(c)
        if rc is None:
            continue
        for c2 in cand_list:
            if c2 == c:
                continue
            rc2 = node_to_row.get(c2)
            if rc2 is None:
                continue
            w = float(dist_m[rc, rc2])
            if np.isfinite(w) and w <= max_m + EPS:
                edges[c].append((c2, w))

    # Manual lexicographic Dijkstra: priority = (total_distance ROUNDED to
    # 0.1m, hops). The rounding matters: dist_m is stored float32 (~cm-level
    # noise at these magnitudes), so a charger sitting exactly on the
    # unconstrained shortest path can come out a fraction of a millimeter
    # "shorter" than the direct edge by raw float value alone, even though
    # they're physically identical -- caught empirically (13810.225783 true
    # direct vs 13810.225738 via a 2-charger chain that adds zero real
    # detour). 0.1m is far below any real routing distinction this data can
    # support, so comparing at that resolution recovers the true tie and
    # lets hop-count correctly prefer the fewest-stops path. The TRUE
    # (unrounded) distance is still what gets accumulated and reported.
    TIE_TOL = 0.1
    best_key: Dict[Any, Tuple[float, int]] = {SRC_KEY: (0.0, 0)}
    best_dist: Dict[Any, float] = {SRC_KEY: 0.0}
    prev: Dict[Any, Any] = {}
    pq: List[Tuple[float, int, Any]] = [(0.0, 0, SRC_KEY)]
    visited: Set[Any] = set()
    while pq:
        _, h, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == DST_KEY:
            break
        ud = best_dist[u]
        for v, w in edges.get(u, []):
            nd, nh = ud + w, h + 1
            nkey = (round(nd, 1), nh)
            if v not in best_key or nkey < best_key[v]:
                best_key[v] = nkey
                best_dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nkey[0], nh, v))

    if DST_KEY not in visited:
        return None, None

    length = best_dist[DST_KEY]
    path: List[Any] = [DST_KEY]
    while path[-1] != SRC_KEY:
        path.append(prev[path[-1]])
    path.reverse()

    chain = [src] + [n for n in path[1:-1]] + [dst]
    return length, chain


def solve_query(
    src: Any, dst: Any, current_m: float, max_m: float,
    G: nx.MultiDiGraph, all_chargers: Set[Any],
    dist_m: np.ndarray, node_to_row: Dict[Any, int],
) -> Dict[str, Any]:
    d_src = nx.single_source_dijkstra_path_length(G, src, weight="length")
    d_dst = nx.single_source_dijkstra_path_length(G.reverse(copy=False), dst, weight="length")

    direct = d_src.get(dst, INF)
    if direct == INF:
        return {"status": "infeasible", "infeasible_reason": "no_road_path_src_to_dst",
               "chain": None, "n_charging_stops": None, "optimal_distance_m": None,
               "n_candidates_final": 0}

    chargers_excl = all_chargers - {src, dst}
    candidates = {c for c in chargers_excl
                 if d_src.get(c, INF) + d_dst.get(c, INF) <= INITIAL_WINDOW_MULT * direct + EPS}

    B, chain = solve_meta(candidates, src, dst, current_m, max_m, d_src, d_dst, dist_m, node_to_row)

    if B is None:
        # initial window found nothing at all -- verify with the FULL charger
        # set (cheap: ~400 nodes) before declaring infeasible.
        B, chain = solve_meta(chargers_excl, src, dst, current_m, max_m,
                              d_src, d_dst, dist_m, node_to_row)
        if B is None:
            return {"status": "infeasible", "infeasible_reason": "no_feasible_chain_with_any_charger",
                   "chain": None, "n_charging_stops": None, "optimal_distance_m": None,
                   "n_candidates_final": len(chargers_excl)}
        candidates = chargers_excl  # already exhaustive
    else:
        # optimality closure: expand while any EXCLUDED charger could still
        # beat the current best (necessary condition), re-solving each time.
        while True:
            new = {c for c in chargers_excl - candidates
                  if d_src.get(c, INF) + d_dst.get(c, INF) <= B + EPS}
            if not new:
                break
            candidates |= new
            B2, chain2 = solve_meta(candidates, src, dst, current_m, max_m,
                                    d_src, d_dst, dist_m, node_to_row)
            if B2 is None:
                break  # shouldn't happen (candidates only grew), but never regress
            B, chain = B2, chain2

    n_stops = len(chain) - 2
    return {"status": "optimal", "infeasible_reason": None, "chain": chain,
           "n_charging_stops": n_stops, "optimal_distance_m": round(B, 2),
           "n_candidates_final": len(candidates)}


def main() -> None:
    print("loading leg matrix + graph + gold ...", flush=True)
    t0 = time.time()
    dist_m, idx_df, node_to_row = load_leg_matrix()
    G = build_graph_from_gpkg(str(GPKG))
    snapper = NodeSnapper.from_gpkg(str(GPKG), G)
    gold = json.loads(GOLD_PATH.read_text())
    print(f"  done in {time.time() - t0:.1f}s ({len(gold)} gold queries)")

    charger_idx = idx_df[idx_df.poi_type == "charging_station"]
    all_chargers: Set[Any] = set(charger_idx["node_id"].unique())
    # representative poi_id per node (smallest, deterministic)
    node_to_poi_id: Dict[Any, Any] = (
        charger_idx.sort_values("poi_id").drop_duplicates("node_id", keep="first")
        .set_index("node_id")["poi_id"].to_dict()
    )
    print(f"  {len(all_chargers)} unique charger nodes")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, rec in enumerate(gold):
        qid = rec["query_id"]
        current_m, max_m = battery_to_meters(rec)
        if current_m is None or max_m is None:
            results.append({
                "query_id": qid, "status": "infeasible",
                "infeasible_reason": "max_battery_mi missing from gold record "
                                     "(this template variant never states the max range)",
                "chain": None, "charger_poi_ids": None, "n_charging_stops": None,
                "optimal_distance_m": None, "current_battery_m": current_m, "max_battery_m": max_m,
            })
            continue

        src = snapper.snap(*rec["src_latlon"])
        dst = snapper.snap(*rec["dst_latlon"])
        if src is None or dst is None:
            results.append({
                "query_id": qid, "status": "infeasible",
                "infeasible_reason": "src/dst snap failed",
                "chain": None, "charger_poi_ids": None, "n_charging_stops": None,
                "optimal_distance_m": None, "current_battery_m": current_m, "max_battery_m": max_m,
            })
            continue

        res = solve_query(src, dst, current_m, max_m, G, all_chargers, dist_m, node_to_row)
        charger_poi_ids = None
        if res["chain"] is not None:
            charger_poi_ids = [node_to_poi_id.get(n) for n in res["chain"][1:-1]]
        results.append({
            "query_id": qid, "status": res["status"],
            "infeasible_reason": res["infeasible_reason"],
            "chain": [str(n) for n in res["chain"]] if res["chain"] is not None else None,
            "charger_poi_ids": charger_poi_ids,
            "n_charging_stops": res["n_charging_stops"],
            "optimal_distance_m": res["optimal_distance_m"],
            "current_battery_m": round(current_m, 2), "max_battery_m": round(max_m, 2),
            "n_candidates_final": res.get("n_candidates_final"),
        })

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(gold)} ({time.time() - t0:.0f}s)", flush=True)

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\ndone: {len(results)} queries in {time.time() - t0:.0f}s -> {OUT_PATH}")

    statuses: Dict[str, int] = {}
    for r in results:
        key = r["status"] if r["status"] == "optimal" else f"infeasible ({r['infeasible_reason']})"
        statuses[key] = statuses.get(key, 0) + 1
    print("status breakdown:", json.dumps(statuses, indent=2))


if __name__ == "__main__":
    main()
