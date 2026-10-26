"""
evaluation/optimal/algo2_oracle.py — exact optimal solver for ordered_poi.

Problem: src -> [POI types in the order the query states] -> dst, every leg
within its stated mile limit, minimising TOTAL DISTANCE.

Input is the parser GROUND TRUTH only (benchmark/ground_truth/
set1_ordered_poi_gold.json). Nothing here re-parses the NL text: gold
recovers src/dst, the ordered POI types and the leg limits by template
inversion with a round-trip proof, which is strictly more reliable than a
regex over prose. (The NL regex in constraint_checker had exactly this class
of bug — it read "9, 2, and 4 miles" as [4].)

Why not permutations
The order is FIXED by the query, so there is nothing to permute; the only
choice is WHICH node fills each slot. Enumerating that product is up to
1255 x 1214 x 569 = 867M combinations for a 3-stop query. But a fixed order
makes this a layered DAG:

    src -> [layer 1: every node of type_1] -> [layer 2: ...] -> dst

so a min-plus DP across layers returns the exact same optimum in
sum(|layer_i| x |layer_i+1|) ~ 2.2M lookups. No pruning, no sampling: every
candidate node is considered, which is why the result is optimal by
construction and needs no closure step (unlike the EV oracle, which prunes
chargers and then proves the pruning safe).

Per-leg limits are enforced by masking edges that exceed them BEFORE each
min. That stays exact because the caps are independent per edge — feasibility
is an AND over edges, with no budget to carry between them.

Costs per query: 2 Dijkstras on the road graph (src forward, dst reverse) for
the first and last legs; every slot-to-slot distance is a lookup in the
prebuilt leg matrix.

Output: evaluation/optimal/results/ordered_poi_oracle.json
"""
from __future__ import annotations

import argparse
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
from evaluation.constraint_checker import MILES_TO_M, NodeSnapper
from evaluation.optimal.ev_oracle import GPKG, LEG_MATRIX_DIR

GOLD_PATH = ROOT / "benchmark" / "ground_truth" / "set1_ordered_poi_gold.json"
OUT_PATH = ROOT / "evaluation" / "optimal" / "results" / "ordered_poi_oracle.json"

INF = float("inf")


def leg_limits_m(gold: Dict[str, Any], n_legs: int) -> List[Optional[float]]:
    """Per-leg caps in metres, length n_legs, None = that leg is unconstrained.

    Mirrors the scorer's broadcast rule exactly (constraint_checker
    ._road_network_checks), because an oracle that enforced a different set of
    limits would be solving a different problem than the evaluation measures:
      * one shared limit  -> applies to EVERY leg, including the final one to dst
      * several limits    -> applied left-to-right; any leg beyond the stated
                             count is UNCONSTRAINED (the benchmark usually
                             leaves the final leg to dst free)
    """
    vals = [float(v) * MILES_TO_M for v in (gold.get("leg_distance_limits_mi") or [])]
    if not vals:
        return [None] * n_legs
    if len(vals) == 1 and n_legs > 1:
        vals = vals * n_legs
    return [vals[i] if i < len(vals) else None for i in range(n_legs)]


def solve_query(
    gold: Dict[str, Any], G: nx.MultiDiGraph, snapper: NodeSnapper,
    dist_m: np.ndarray, rows_by_type: Dict[str, np.ndarray],
    nodes_by_type: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    types: List[str] = list(gold.get("poi_types") or [])
    if not types:
        return {"status": "no_poi_types", "optimal_distance_m": None}

    src = snapper.snap(*gold["src_latlon"])
    dst = snapper.snap(*gold["dst_latlon"])
    if src is None or dst is None:
        return {"status": "snap_failed", "optimal_distance_m": None}

    n_legs = len(types) + 1
    limits = leg_limits_m(gold, n_legs)

    for t in types:
        if t not in rows_by_type or len(rows_by_type[t]) == 0:
            return {"status": "no_candidates_of_type", "missing_type": t,
                    "optimal_distance_m": None}

    d_src = nx.single_source_dijkstra_path_length(G, src, weight="length")
    d_dst = nx.single_source_dijkstra_path_length(
        G.reverse(copy=False), dst, weight="length")

    # layer 0: src -> slot 1
    nodes0 = nodes_by_type[types[0]]
    cost = np.array([d_src.get(n, INF) for n in nodes0], dtype=np.float64)
    if limits[0] is not None:
        cost = np.where(cost <= limits[0] + 1e-6, cost, INF)
    back: List[np.ndarray] = []

    # slot i-1 -> slot i, via the prebuilt matrix
    for i in range(1, len(types)):
        prev_rows, cur_rows = rows_by_type[types[i - 1]], rows_by_type[types[i]]
        M = dist_m[np.ix_(prev_rows, cur_rows)].astype(np.float64)
        if limits[i] is not None:
            M = np.where(M <= limits[i] + 1e-6, M, INF)
        # One node CAN fill two consecutive slots, and that is deliberate: 127
        # nodes are tagged both bank and supermarket, a supermarket with an
        # in-store pharmacy is genuinely both, and — decisively — the scorer
        # credits it (its per-position scan matched node 3872277409 as both the
        # bank and the pharmacy of ordered_poi_2_v03, giving a 0 m middle
        # leg). An earlier version masked these out on "you cannot satisfy
        # 'then a pharmacy' by standing still" grounds; that made the oracle
        # miss achievable solutions, so it reported 51 infeasible queries of
        # which several had real answers, and CRAFT/LLMAP appeared to BEAT the
        # optimum on 9 rows. The oracle must optimise over exactly what the
        # evaluation credits, not over a stricter rule of our own.
        tot = cost[:, None] + M
        arg = np.argmin(tot, axis=0)
        cost = tot[arg, np.arange(tot.shape[1])]
        back.append(arg)

    # last slot -> dst
    nodes_last = nodes_by_type[types[-1]]
    tail = np.array([d_dst.get(n, INF) for n in nodes_last], dtype=np.float64)
    if limits[-1] is not None:
        tail = np.where(tail <= limits[-1] + 1e-6, tail, INF)
    total = cost + tail

    if not np.isfinite(total).any():
        return {"status": "infeasible", "optimal_distance_m": None,
                "n_candidates_per_slot": [int(len(rows_by_type[t])) for t in types],
                "leg_limits_m": [None if l is None else round(l, 2) for l in limits]}

    j = int(np.argmin(total))
    best = float(total[j])

    # reconstruct the chosen node per slot, walking the argmin pointers back
    picks = [j]
    for arg in reversed(back):
        picks.append(int(arg[picks[-1]]))
    picks.reverse()
    stops = [nodes_by_type[types[s]][picks[s]] for s in range(len(types))]

    chain = [src] + list(stops) + [dst]
    legs: List[float] = []
    for a, b in zip(chain, chain[1:]):
        try:
            legs.append(float(nx.shortest_path_length(G, a, b, weight="length")))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            legs.append(INF)

    return {
        "status": "optimal",
        "optimal_distance_m": round(best, 2),
        "stops": [str(n) for n in stops],
        "chain": [str(n) for n in chain],
        "leg_distances_m": [round(x, 2) for x in legs],
        "leg_limits_m": [None if l is None else round(l, 2) for l in limits],
        "n_candidates_per_slot": [int(len(rows_by_type[t])) for t in types],
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
    dist_m = npz["dist_m"]
    idx = pd.read_parquet(LEG_MATRIX_DIR / "index.parquet")
    gold_list = json.loads(GOLD_PATH.read_text())
    if args.limit:
        gold_list = gold_list[: args.limit]

    rows_by_type: Dict[str, np.ndarray] = {}
    nodes_by_type: Dict[str, np.ndarray] = {}
    for t, g in idx.groupby("poi_type"):
        u = g.drop_duplicates("node_id").sort_values("row_idx")
        rows_by_type[t] = u["row_idx"].to_numpy()
        nodes_by_type[t] = u["node_id"].to_numpy()
    print(f"  ready in {time.time() - t0:.1f}s | candidates: "
          f"{ {k: len(v) for k, v in rows_by_type.items()} }")

    results: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, gold in enumerate(gold_list):
        res = solve_query(gold, G, snapper, dist_m, rows_by_type, nodes_by_type)
        res["query_id"] = gold["query_id"]
        results.append(res)
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
        print(f"stops per solved query: "
              f"{dict(Counter(len(r['stops']) for r in solved))}")


if __name__ == "__main__":
    main()
