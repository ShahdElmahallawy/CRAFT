"""
evaluation/optimal/poi_dist_time_oracle.py — exact optimal solver for
unordered_poi.

Problem: src -> [required POI types in any order] -> dst, every leg within its
stated mile limit, minimising TOTAL TRAVEL TIME.

Input is the parser GROUND TRUTH only (set1_unordered_poi_gold.json). No NL
text is parsed here.

TWO THINGS DIFFER FROM algo2
1. The order is free, so here permutations DO matter — but only over the 2-3
   required TYPES (2 or 6 orderings), never over the thousands of candidate
   nodes. For each ordering the problem is again a fixed-order layered DAG, so
   we run algo2's DP once per ordering and keep the best. Total work stays
   ~6 x the algo2 cost, not factorial in the candidates.
   The per-leg limits are POSITIONAL, not attached to a type: whichever type
   the solver puts in slot i must satisfy limit i. That matches the scorer,
   which for unordered templates scans the route for any remaining required
   type and measures legs by position.
2. The objective is travel TIME while the limits are in MILES, so the two use
   different edge weights: `time_s` accumulates the objective, `dist_m` decides
   which legs are legal. Both come from the same prebuilt matrix; src/dst need
   a Dijkstra per weight (4 per query rather than 2).

Optimality: within one ordering the min-plus DP is exact over every candidate
node (no pruning); across orderings we enumerate all of them. So the reported
value is the true optimum.

Output: evaluation/optimal/results/poi_dist_time_oracle.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from craft.generator.tools.common_handlers import build_graph_from_gpkg
from evaluation.constraint_checker import NodeSnapper
from evaluation.optimal.algo2_oracle import leg_limits_m
from evaluation.optimal.ev_oracle import GPKG, LEG_MATRIX_DIR

GOLD_PATH = ROOT / "benchmark" / "ground_truth" / "set1_unordered_poi_gold.json"
OUT_PATH = ROOT / "evaluation" / "optimal" / "results" / "poi_dist_time_oracle.json"

INF = float("inf")


def solve_order(
    order: Tuple[str, ...], limits: List[Optional[float]],
    d_src_t: Dict[Any, float], d_dst_t: Dict[Any, float],
    d_src_l: Dict[Any, float], d_dst_l: Dict[Any, float],
    time_s: np.ndarray, dist_m: np.ndarray,
    rows_by_type: Dict[str, np.ndarray], nodes_by_type: Dict[str, np.ndarray],
) -> Tuple[float, Optional[List[Any]]]:
    """Best total TIME for one fixed type ordering, or (inf, None)."""
    nodes0 = nodes_by_type[order[0]]
    cost = np.array([d_src_t.get(n, INF) for n in nodes0], dtype=np.float64)
    legal = np.array([d_src_l.get(n, INF) for n in nodes0], dtype=np.float64)
    if limits[0] is not None:
        cost = np.where(legal <= limits[0] + 1e-6, cost, INF)

    back: List[np.ndarray] = []
    for i in range(1, len(order)):
        pr, cr = rows_by_type[order[i - 1]], rows_by_type[order[i]]
        T = time_s[np.ix_(pr, cr)].astype(np.float64)   # objective
        D = dist_m[np.ix_(pr, cr)].astype(np.float64)   # legality
        if limits[i] is not None:
            T = np.where(D <= limits[i] + 1e-6, T, INF)
        tot = cost[:, None] + T
        arg = np.argmin(tot, axis=0)
        cost = tot[arg, np.arange(tot.shape[1])]
        back.append(arg)

    nodes_last = nodes_by_type[order[-1]]
    tail_t = np.array([d_dst_t.get(n, INF) for n in nodes_last], dtype=np.float64)
    tail_l = np.array([d_dst_l.get(n, INF) for n in nodes_last], dtype=np.float64)
    if limits[-1] is not None:
        tail_t = np.where(tail_l <= limits[-1] + 1e-6, tail_t, INF)
    total = cost + tail_t
    if not np.isfinite(total).any():
        return INF, None

    j = int(np.argmin(total))
    picks = [j]
    for arg in reversed(back):
        picks.append(int(arg[picks[-1]]))
    picks.reverse()
    stops = [nodes_by_type[order[s]][picks[s]] for s in range(len(order))]
    return float(total[j]), stops


def solve_query(
    gold: Dict[str, Any], G: nx.MultiDiGraph, snapper: NodeSnapper,
    time_s: np.ndarray, dist_m: np.ndarray,
    rows_by_type: Dict[str, np.ndarray], nodes_by_type: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    types: List[str] = list(gold.get("poi_types") or [])
    if not types:
        return {"status": "no_poi_types", "optimal_time_s": None}
    src = snapper.snap(*gold["src_latlon"])
    dst = snapper.snap(*gold["dst_latlon"])
    if src is None or dst is None:
        return {"status": "snap_failed", "optimal_time_s": None}
    for t in types:
        if t not in rows_by_type or len(rows_by_type[t]) == 0:
            return {"status": "no_candidates_of_type", "missing_type": t,
                    "optimal_time_s": None}

    limits = leg_limits_m(gold, len(types) + 1)

    d_src_t = nx.single_source_dijkstra_path_length(G, src, weight="travel_time")
    d_src_l = nx.single_source_dijkstra_path_length(G, src, weight="length")
    Grev = G.reverse(copy=False)
    d_dst_t = nx.single_source_dijkstra_path_length(Grev, dst, weight="travel_time")
    d_dst_l = nx.single_source_dijkstra_path_length(Grev, dst, weight="length")

    best_t, best_stops, best_order = INF, None, None
    tried = 0
    for order in itertools.permutations(types):
        tried += 1
        t, stops = solve_order(order, limits, d_src_t, d_dst_t, d_src_l, d_dst_l,
                               time_s, dist_m, rows_by_type, nodes_by_type)
        if t < best_t:
            best_t, best_stops, best_order = t, stops, order

    if best_stops is None:
        return {"status": "infeasible", "optimal_time_s": None,
                "orders_tried": tried, "poi_types": types,
                "leg_limits_m": [None if l is None else round(l, 2) for l in limits]}

    chain = [src] + list(best_stops) + [dst]
    legs_t, legs_l = [], []
    for a, b in zip(chain, chain[1:]):
        try:
            legs_t.append(float(nx.shortest_path_length(G, a, b, weight="travel_time")))
            legs_l.append(float(nx.shortest_path_length(G, a, b, weight="length")))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            legs_t.append(INF); legs_l.append(INF)

    return {
        "status": "optimal",
        "optimal_time_s": round(best_t, 2),
        "optimal_order": list(best_order),
        "stops": [str(n) for n in best_stops],
        "chain": [str(n) for n in chain],
        "leg_times_s": [round(x, 2) for x in legs_t],
        "leg_distances_m": [round(x, 2) for x in legs_l],
        "leg_limits_m": [None if l is None else round(l, 2) for l in limits],
        "orders_tried": tried,
        "poi_types": types,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    print("loading graph, leg matrix, gold ...", flush=True)
    t0 = time.time()
    G = build_graph_from_gpkg(str(GPKG))
    snapper = NodeSnapper.from_gpkg(str(GPKG), G)
    npz = np.load(LEG_MATRIX_DIR / "legs.npz")
    dist_m, time_s = npz["dist_m"], npz["time_s"]
    idx = pd.read_parquet(LEG_MATRIX_DIR / "index.parquet")
    gold_list = json.loads(GOLD_PATH.read_text())
    if args.limit:
        gold_list = gold_list[: args.limit]

    rows_by_type, nodes_by_type = {}, {}
    for t, g in idx.groupby("poi_type"):
        u = g.drop_duplicates("node_id").sort_values("row_idx")
        rows_by_type[t] = u["row_idx"].to_numpy()
        nodes_by_type[t] = u["node_id"].to_numpy()
    print(f"  ready in {time.time() - t0:.1f}s")

    results = []
    t0 = time.time()
    for i, gold in enumerate(gold_list):
        r = solve_query(gold, G, snapper, time_s, dist_m, rows_by_type, nodes_by_type)
        r["query_id"] = gold["query_id"]
        results.append(r)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(gold_list)} ({time.time() - t0:.0f}s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    from collections import Counter
    print(f"\ndone: {len(results)} queries in {time.time() - t0:.0f}s -> {out}")
    print("status:", dict(Counter(r["status"] for r in results)))
    solved = [r for r in results if r["status"] == "optimal"]
    if solved:
        print("stops per solved query:", dict(Counter(len(r["stops"]) for r in solved)))
        print("orderings tried:", dict(Counter(r["orders_tried"] for r in solved)))


if __name__ == "__main__":
    main()
