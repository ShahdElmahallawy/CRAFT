"""
evaluation/oracle/brute_force_ev_ordered.py — INDEPENDENT brute-force check on
ev_ordered_oracle_v2.

Shares no code with v2: no layered DP, no Pareto fronts, no _leg_outcomes. It
enumerates POI combinations explicitly, tries every charger insertion point,
and simulates the charge hop by hop. Slow by design — the point is that it
arrives at the answer a completely different way.

Why one charger insertion is enough for a COST check: any chain
a -> c1 -> ... -> ck -> b with k >= 2 costs at least d(a,c2) + d(c2,b) by the
triangle inequality, which is the single-charger option through c2, and arrives
with the same charge (chargers reset to full). Chains can only extend REACH,
never reduce cost — so for confirming a cost optimum, single insertions suffice.
(v2 relies on the same argument; here it is applied to a solver written
independently, so agreement is still meaningful.)

The earlier version of this check inside verify_ev_ordered_v2.py silently did
nothing — it left POI layers 2+ unfiltered at their full 389-1255 candidates,
so every instance tripped the size guard and it reported "0/0 match" as a pass.
Layers are now filtered by the leg limits exactly as the scorer measures them
(DIRECT POI->POI distance), which is what makes the enumeration tractable.

Usage:
  conda run -n myenv python evaluation/oracle/brute_force_ev_ordered.py \
      --v2 evaluation/oracle/results/ev_ordered_poi_v2_oracle.json --n 5
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
from evaluation.oracle.algo2_oracle import leg_limits_m
from evaluation.oracle.ev_oracle import GPKG, LEG_MATRIX_DIR, battery_to_meters

GOLD = ROOT / "benchmark" / "ground_truth" / "set1_ev+ordered_poi_regen_gold.json"
TOL = 1.0
INF = float("inf")


def legal_layers(gold, d_src, d_dst, dist, rows_by_type, nodes_by_type, limits):
    """Candidate indices per slot after applying the leg limits to the DIRECT
    POI->POI distances (the scorer's basis). Returns None if a slot is empty."""
    types = gold["poi_types"]
    nodes0, rows0 = nodes_by_type[types[0]], rows_by_type[types[0]]
    d0 = np.array([d_src.get(n, INF) for n in nodes0])
    keep0 = np.isfinite(d0)
    if limits[0] is not None:
        keep0 &= d0 <= limits[0] + 1e-6
    layers = [np.nonzero(keep0)[0]]
    if layers[0].size == 0:
        return None
    for i in range(1, len(types)):
        prev_rows = rows_by_type[types[i - 1]][layers[i - 1]]
        cur_rows = rows_by_type[types[i]]
        M = dist[np.ix_(prev_rows, cur_rows)].astype(np.float64)
        ok = np.isfinite(M)
        if limits[i] is not None:
            ok &= M <= limits[i] + 1e-6
        keep = ok.any(axis=0)
        # the final slot must also be able to reach dst within its limit
        if i == len(types) - 1:
            dl = np.array([d_dst.get(n, INF) for n in nodes_by_type[types[i]]])
            keep &= np.isfinite(dl)
            if limits[-1] is not None:
                keep &= dl <= limits[-1] + 1e-6
        idx = np.nonzero(keep)[0]
        if idx.size == 0:
            return None
        layers.append(idx)
    return layers


def brute(gold, G, snapper, dist, rows_by_type, nodes_by_type,
          charger_rows, charger_nodes, max_combos=200_000):
    types = gold["poi_types"]
    cur_m, max_m = battery_to_meters(gold)
    src = snapper.snap(*gold["src_latlon"])
    dst = snapper.snap(*gold["dst_latlon"])
    limits = leg_limits_m(gold, len(types) + 1)

    d_src = nx.single_source_dijkstra_path_length(G, src, weight="length")
    d_dst = nx.single_source_dijkstra_path_length(G.reverse(copy=False), dst, weight="length")

    layers = legal_layers(gold, d_src, d_dst, dist, rows_by_type, nodes_by_type, limits)
    if layers is None:
        return None, 0, "no legal candidates"
    n_comb = int(np.prod([len(l) for l in layers]))
    if n_comb > max_combos:
        return None, n_comb, f"too large ({n_comb:,} combos)"

    d_src_c = np.array([d_src.get(n, INF) for n in charger_nodes])
    d_c_dst = np.array([d_dst.get(n, INF) for n in charger_nodes])
    best = INF

    for combo in itertools.product(*layers):
        rows = [int(rows_by_type[types[i]][j]) for i, j in enumerate(combo)]
        nodes = [nodes_by_type[types[i]][j] for i, j in enumerate(combo)]
        ok = True
        for i in range(1, len(rows)):
            d = float(dist[rows[i - 1], rows[i]])
            if not np.isfinite(d) or (limits[i] is not None and d > limits[i] + 1e-6):
                ok = False
                break
        if not ok:
            continue

        # waypoint chain: src -> stops -> dst
        hop_d = [d_src.get(nodes[0], INF)]
        for i in range(1, len(rows)):
            hop_d.append(float(dist[rows[i - 1], rows[i]]))
        hop_d.append(d_dst.get(nodes[-1], INF))
        if not all(np.isfinite(x) for x in hop_d):
            continue
        n_hops = len(hop_d)

        # 0 chargers
        b, c, feas = cur_m, 0.0, True
        for d in hop_d:
            if d > b + 1e-6:
                feas = False
                break
            b -= d
            c += d
        if feas:
            best = min(best, c)

        # exactly 1 charger, inserted into any hop, over ALL chargers
        for h in range(n_hops):
            # charge on arrival at the start of hop h
            b = cur_m
            pre = 0.0
            ok2 = True
            for k in range(h):
                if hop_d[k] > b + 1e-6:
                    ok2 = False
                    break
                b -= hop_d[k]
                pre += hop_d[k]
            if not ok2:
                continue
            a_node = src if h == 0 else nodes[h - 1]
            b_node = dst if h == n_hops - 1 else nodes[h]
            a_row = None if h == 0 else rows[h - 1]
            b_row = None if h == n_hops - 1 else rows[h]
            da = d_src_c if h == 0 else dist[a_row, charger_rows].astype(np.float64)
            db = d_c_dst if h == n_hops - 1 else dist[charger_rows, b_row].astype(np.float64)
            adm = np.isfinite(da) & np.isfinite(db) & (da <= b + 1e-6) & (db <= max_m + 1e-6)
            if not adm.any():
                continue
            for ci in np.nonzero(adm)[0]:
                cost = pre + float(da[ci]) + float(db[ci])
                bb = max_m - float(db[ci])
                feas2 = True
                for k in range(h + 1, n_hops):
                    if hop_d[k] > bb + 1e-6:
                        feas2 = False
                        break
                    bb -= hop_d[k]
                    cost += hop_d[k]
                if feas2:
                    best = min(best, cost)

    return (None if not np.isfinite(best) else best), n_comb, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--max-combos", type=int, default=200_000)
    args = ap.parse_args()

    v2 = {r["query_id"]: r for r in json.loads(Path(args.v2).read_text())}
    gold = {g["query_id"]: g for g in json.loads(GOLD.read_text())}

    print("loading graph + leg matrix ...", flush=True)
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
    charger_rows, charger_nodes = cg["row_idx"].to_numpy(), cg["node_id"].to_numpy()

    # prefer queries whose optimum actually needed a charger (the interesting ones)
    binding = [q for q, r in v2.items() if r["status"] == "optimal"]
    checked = matched = 0
    skipped: List[Tuple[str, str]] = []
    print(f"\n{'query':<28}{'brute':>12}{'v2':>12}{'combos':>10}  verdict")
    for q in binding:
        if checked >= args.n:
            break
        t0 = time.time()
        bf, ncomb, note = brute(gold[q], G, snapper, dist, rows_by_type, nodes_by_type,
                                charger_rows, charger_nodes, args.max_combos)
        if note != "ok":
            skipped.append((q, note))
            continue
        checked += 1
        got = v2[q]["optimal_distance_m"]
        ok = bf is not None and abs(bf - got) < TOL
        matched += ok
        print(f"{q:<28}{bf if bf is None else round(bf,1):>12}{got:>12.1f}{ncomb:>10,}"
              f"  {'MATCH' if ok else 'MISMATCH!!'} ({time.time()-t0:.1f}s)")

    print(f"\n{matched}/{checked} match")
    if skipped:
        print(f"skipped {len(skipped)} (enumeration too large): "
              f"{[s[0] for s in skipped[:4]]}")
    if checked == 0:
        print("FAIL — the check ran on ZERO queries; that is not evidence of anything")
        raise SystemExit(1)
    print("PASS" if matched == checked else "FAIL — v2 disagrees with brute force")
    raise SystemExit(0 if matched == checked else 1)


if __name__ == "__main__":
    main()
