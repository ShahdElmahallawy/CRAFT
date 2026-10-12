"""
evaluation/oracle/verify_ev_ordered_v2.py — verification gate for
ev_ordered_oracle_v2. Its results are not to be used until this passes.

1. REGRESSION vs v1 on the battery-SLACK queries — STRICT EQUALITY.
   Where the battery never binds, the Pareto front collapses to a single label
   per node and v2 degenerates to exactly v1's layered DP, so any difference is
   a v2 bug, not a modelling choice.

2. v1's BINDING-BUT-STUMBLED queries — v2 must be <= v1's cost, AND carry its
   own proof (status "optimal"). v1 only ever found a charger that happened to
   sit on its min-distance route; v2 searches for the best one, so v2 should
   strictly beat v1 on some of these. The count is reported, because "v2 never
   improves" would suggest the detour search is not actually running.

3. BRUTE FORCE on binding queries — an independent enumeration that shares no
   code with v2: every ordered POI combination over the leg-limit-legal
   candidates, times every insertion of 0, 1 or 2 chargers between consecutive
   waypoints, simulating the charge directly. Exponential, so it runs only on
   queries whose candidate layers are small enough to enumerate exhaustively.

Usage:
  python evaluation/oracle/verify_ev_ordered_v2.py \
      --v1 <v1 results json> --v2 <v2 results json> [--brute 5]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
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
from evaluation.oracle.algo2_oracle import leg_limits_m
from evaluation.oracle.ev_oracle import GPKG, LEG_MATRIX_DIR, battery_to_meters

GOLD = ROOT / "benchmark" / "ground_truth" / "set1_ev+ordered_poi_regen_gold.json"
TOL = 1.0          # metres; below this is float noise, not a real difference
MAX_LAYER = 14     # brute force only where a layer is this small
MAX_COMBOS = 300_000


def brute_force(gold, G, snapper, dist, rows_by_type, nodes_by_type,
                charger_rows, max_chargers=2) -> Optional[float]:
    """Independent min-cost battery-feasible ordered route. Enumerates POI
    combinations and charger insertions explicitly; no DP, no Pareto logic,
    none of v2's code. Returns None if the instance is too big to enumerate."""
    types = gold["poi_types"]
    cur_m, max_m = battery_to_meters(gold)
    src = snapper.snap(*gold["src_latlon"])
    dst = snapper.snap(*gold["dst_latlon"])
    limits = leg_limits_m(gold, len(types) + 1)

    d_src = nx.single_source_dijkstra_path_length(G, src, weight="length")
    d_dst = nx.single_source_dijkstra_path_length(G.reverse(copy=False), dst, weight="length")

    # candidates per slot, filtered by the DIRECT leg limit (scorer semantics)
    layers = []
    for i, t in enumerate(types):
        nodes, rows = nodes_by_type[t], rows_by_type[t]
        if i == 0:
            d = np.array([d_src.get(n, np.inf) for n in nodes])
            keep = np.isfinite(d) & ((d <= limits[0] + 1e-6) if limits[0] is not None else True)
        else:
            keep = np.ones(len(nodes), dtype=bool)   # filtered pairwise below
        idxs = np.nonzero(keep)[0]
        if len(idxs) == 0 or len(idxs) > MAX_LAYER:
            return None
        layers.append(idxs)

    n_comb = 1
    for l in layers:
        n_comb *= len(l)
    if n_comb > MAX_COMBOS:
        return None

    cr = charger_rows
    best = np.inf
    for combo in itertools.product(*layers):
        rows = [int(rows_by_type[types[i]][j]) for i, j in enumerate(combo)]
        nodes = [nodes_by_type[types[i]][j] for i, j in enumerate(combo)]
        # leg limits on DIRECT distances
        ok = True
        for i in range(1, len(rows)):
            d = float(dist[rows[i-1], rows[i]])
            if not np.isfinite(d) or (limits[i] is not None and d > limits[i] + 1e-6):
                ok = False
                break
        if not ok:
            continue
        d_last = d_dst.get(nodes[-1], np.inf)
        if not np.isfinite(d_last) or (limits[-1] is not None and d_last > limits[-1] + 1e-6):
            continue

        # hop distance getters for the waypoint chain src -> stops -> dst
        n_hops = len(rows) + 1
        def hop(u, v):
            if u == -1:
                return d_src.get(nodes[v], np.inf)
            if v == len(rows):
                return d_dst.get(nodes[u], np.inf)
            return float(dist[rows[u], rows[v]])
        def hop_via(u, v, c_row):
            a = (d_src.get(charger_nodes_by_row[c_row], np.inf) if u == -1
                 else float(dist[rows[u], c_row]))
            b = (d_dst.get(charger_nodes_by_row[c_row], np.inf) if v == len(rows)
                 else float(dist[c_row, rows[v]]))
            return a, b

        # try every assignment of <= max_chargers charger-insertions to hops
        for n_ins in range(0, max_chargers + 1):
            for hops in itertools.combinations(range(n_hops), n_ins):
                cand = [cr] * n_ins if n_ins else [[]]
                iters = itertools.product(*[cr for _ in range(n_ins)]) if n_ins else [()]
                for cs in iters:
                    batt, cost, feasible = cur_m, 0.0, True
                    ci = 0
                    for h in range(n_hops):
                        u, v = h - 1, h
                        if h in hops:
                            c_row = int(cs[ci]); ci += 1
                            a, b = hop_via(u, v, c_row)
                            if not (np.isfinite(a) and np.isfinite(b)) or a > batt + 1e-6 or b > max_m + 1e-6:
                                feasible = False; break
                            cost += a + b
                            batt = max_m - b
                        else:
                            d = hop(u, v)
                            if not np.isfinite(d) or d > batt + 1e-6:
                                feasible = False; break
                            cost += d
                            batt -= d
                    if feasible and cost < best:
                        best = cost
    return None if not np.isfinite(best) else best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True)
    ap.add_argument("--v2", required=True)
    ap.add_argument("--brute", type=int, default=5)
    args = ap.parse_args()

    v1 = {r["query_id"]: r for r in json.loads(Path(args.v1).read_text())}
    v2 = {r["query_id"]: r for r in json.loads(Path(args.v2).read_text())}
    gold = {g["query_id"]: g for g in json.loads(GOLD.read_text())}

    from collections import Counter
    print("v2 status:", dict(Counter(r["status"] for r in v2.values())))
    print("v1 status:", dict(Counter(r["status"] for r in v1.values())))

    # 1. strict regression on slack queries
    slack = [q for q, r in v1.items()
             if r["status"] == "optimal" and not r.get("battery_binding")]
    mism = []
    for q in slack:
        a, b = v1[q].get("optimal_distance_m"), v2[q].get("optimal_distance_m")
        if v2[q]["status"] != "optimal" or a is None or b is None or abs(a - b) > TOL:
            mism.append((q, a, b, v2[q]["status"]))
    print(f"\n1. REGRESSION on battery-slack queries: {len(slack)-len(mism)}/{len(slack)} exact")
    for m in mism[:6]:
        print(f"     MISMATCH {m[0]}: v1={m[1]} v2={m[2]} status={m[3]}")

    # 2. binding-but-stumbled: v2 <= v1, with its own proof
    stumbled = [q for q, r in v1.items()
                if r["status"] == "optimal" and r.get("battery_binding")]
    worse, improved, unproven = [], 0, []
    for q in stumbled:
        a, b = v1[q].get("optimal_distance_m"), v2[q].get("optimal_distance_m")
        if v2[q]["status"] != "optimal":
            unproven.append(q); continue
        if b > a + TOL:
            worse.append((q, a, b))
        elif b < a - TOL:
            improved += 1
    print(f"\n2. v1 BINDING-BUT-STUMBLED ({len(stumbled)} queries):")
    print(f"     v2 worse than v1 (must be 0): {len(worse)}")
    print(f"     v2 without its own proof (must be 0): {len(unproven)}")
    print(f"     v2 STRICTLY better than v1: {improved}  "
          f"(expected > 0 — v1 took an accidental charger, v2 searches for the best)")
    for w in worse[:5]:
        print(f"     WORSE {w[0]}: v1={w[1]} v2={w[2]}")

    # 3. independent brute force
    print(f"\n3. BRUTE FORCE cross-check (independent enumeration, <=2 chargers):")
    print("   loading graph ...", flush=True)
    G = build_graph_from_gpkg(str(GPKG))
    snapper = NodeSnapper.from_gpkg(str(GPKG), G)
    npz = np.load(LEG_MATRIX_DIR / "legs.npz")
    dist = npz["dist_m"]
    idx = pd.read_parquet(LEG_MATRIX_DIR / "index.parquet")
    rows_by_type, nodes_by_type = {}, {}
    for t, g in idx.groupby("poi_type"):
        u = g.drop_duplicates("node_id").sort_values("row_idx")
        rows_by_type[t] = u["row_idx"].to_numpy()
        nodes_by_type[t] = u["node_id"].to_numpy()
    cg = idx[idx.poi_type == "charging_station"].drop_duplicates("node_id")
    charger_rows = cg["row_idx"].to_numpy()
    global charger_nodes_by_row
    charger_nodes_by_row = dict(zip(cg["row_idx"].to_numpy(), cg["node_id"].to_numpy()))

    binding = [q for q, r in v2.items()
               if r["status"] == "optimal" and q in v1
               and v1[q]["status"] == "requires_charging_detour_not_modeled"]
    checked = matched = skipped = 0
    for q in binding:
        if checked >= args.brute:
            break
        bf = brute_force(gold[q], G, snapper, dist, rows_by_type, nodes_by_type, charger_rows)
        if bf is None:
            skipped += 1
            continue
        checked += 1
        got = v2[q]["optimal_distance_m"]
        ok = abs(bf - got) < TOL
        matched += ok
        print(f"     {q}: brute={bf:.2f} v2={got:.2f} {'MATCH' if ok else 'MISMATCH!!'}")
    print(f"   {matched}/{checked} match ({skipped} instances too large to enumerate)")

    fails = len(mism) + len(worse) + len(unproven) + (checked - matched)
    print(f"\n{'PASS — v2 verified' if fails == 0 else f'FAIL — {fails} problem(s)'}")


if __name__ == "__main__":
    main()
