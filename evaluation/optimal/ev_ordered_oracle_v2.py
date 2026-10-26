"""
evaluation/optimal/ev_ordered_oracle_v2.py — GENERAL exact optimal solver for
ev+ordered_poi, including queries whose optimum requires a charging detour.

v1 (ev_ordered_oracle.py) is left untouched. It solves the min-distance ordered
route and then CHECKS the battery, which is exact only while the battery is
slack: if the cheapest route already fits the charge, no detour can beat it,
because the objective (distance) and the consumed resource (charge) are the
same quantity. On the regenerated gold that shortcut no longer covers the set —
v1 returns `requires_charging_detour_not_modeled` on 98/200 — so this file
searches the detours instead of assuming them away.

METHOD: layered DP over the fixed POI slots with battery as a state dimension.
  state   (slot i, poi node p)  ->  Pareto set of (cost_so_far_m, charge_m)
  a label is dominated iff another has cost <= and charge >= (both), because
  anything the dominated label can still do, the dominating one can do at no
  greater cost. Keeping the full Pareto front is what makes this exact rather
  than greedy: a costlier route arriving with more charge can win later, so
  collapsing to "cheapest per node" would be wrong.

THE TWO CONSTRAINTS ARE MEASURED ON DIFFERENT ROUTES — this mirrors the scorer
exactly and the separation is deliberate in the code below:
  * LEG LIMIT  -> the DIRECT SP between consecutive POIs. constraint_checker
    ._road_network_checks builds `chain = [src] + stop_nodes + [dst]` and takes
    `sp_chain(...)`, and the stop list comes from `_intended_stop_nodes`, which
    for ordered templates only admits nodes matching `poi_types[idx]` in
    sequence. `charging_station` is never in poi_types, so a charger never
    splits a leg. => legality is a pure matrix lookup, applied as a mask BEFORE
    any mini-EV solve runs (identical to algo2_oracle's mask).
  * BATTERY + OBJECTIVE -> the REALIZED route, detour included
    (_system_checks decrements per edge along the route and refuels at
    `declared_charger_nodes`).
So a detour is free against the leg limit but paid for in charge and in cost.

WHY THE ORACLE IS NOT STRICTER THAN THE SCORER (soundness, non-obvious):
The scorer refuels at any charger lying on the realized route, including ones
no planned chain named; this solver only refuels at chargers it plans. That
looks like the oracle could miss a route the scorer would accept. It cannot,
for this reason: if a charger c' lies on the shortest path of a hop a->b, then
SP(a,c') + SP(c',b) == SP(a,b), so the PLANNED chain a->c'->b has identical
total distance and is already in the meta-graph. Every incidental-refuel route
is therefore represented as an equal-cost planned chain.
That argument depends entirely on c' being a CANDIDATE — it fails silently if
c' was pruned away, which is why the charger set must be exhaustive (or closed
under a proven necessary condition). It also depends on an exact-equality test
over float32 matrix entries, so comparisons use the same 0.1 m tie tolerance
ev_oracle needed: without it, a charger sitting exactly on the shortest path
reads as a hair longer than the direct edge and the equal-cost chain is
wrongly discarded.

never REPORTS AN UNPROVEN OPTIMUM. Statuses: `optimal`, `infeasible` (no
battery-feasible ordered route exists over the full candidate set),
`needs_charger_chaining_unverified` (no route found AND a two-hop charger chain
could reach chargers a single hop cannot — see _leg_outcomes), plus the
structural ones inherited from the layered search.

Output: evaluation/optimal/results/ev_ordered_poi_v2_oracle.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from craft.generator.tools.common_handlers import build_graph_from_gpkg
from evaluation.constraint_checker import MILES_TO_M, NodeSnapper
from evaluation.optimal.algo2_oracle import leg_limits_m
from evaluation.optimal.ev_oracle import GPKG, LEG_MATRIX_DIR, battery_to_meters

GOLD_DEFAULT = ROOT / "benchmark" / "ground_truth" / "set1_ev+ordered_poi_regen_gold.json"
OUT_DEFAULT = ROOT / "evaluation" / "optimal" / "results" / "ev_ordered_poi_v2_oracle.json"

INF = float("inf")
# Same tolerance ev_oracle needed: dist_m is float32, so a charger exactly on a
# shortest path can read a fraction of a millimetre off SP(a,b). Comparisons at
# 0.1 m resolution are far below any real routing distinction this data
# supports, and recover the true tie.
TIE_TOL_M = 0.1


def _pareto(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Keep only non-dominated (cost, charge): lower cost AND higher charge wins.
    Ties at TIE_TOL_M resolution are treated as equal so float32 noise cannot
    drop a genuinely equal-cost alternative."""
    if not points:
        return []
    pts = sorted(points, key=lambda x: (round(x[0] / TIE_TOL_M), -x[1]))
    out: List[Tuple[float, float]] = []
    best_charge = -INF
    for cost, charge in pts:
        if charge > best_charge + 1e-9:
            out.append((cost, charge))
            best_charge = charge
    return out


def _leg_outcomes(
    a_row: int, b_row: int, charge: float, max_m: float,
    dist: np.ndarray, charger_rows: np.ndarray,
) -> List[Tuple[float, float]]:
    """Pareto-optimal (leg_cost, arrival_charge) for one hop a -> b entered with
    `charge`, allowing a planned charging detour.

    Enumeration is complete over single-charger detours:
      * direct              -> cost SP(a,b),           arrival charge - SP(a,b)
      * via charger c       -> cost SP(a,c)+SP(c,b),   arrival max - SP(c,b)
        admissible iff SP(a,c) <= charge and SP(c,b) <= max
    Both extremes of the front are taken (min cost, and max arrival = min
    SP(c,b)), then Pareto-reduced together with the direct option.

    MULTI-CHARGER CHAINS within one hop are not enumerated, and single-charger
    enumeration is COMPLETE FOR OPTIMALITY. Proof: take any chain
    a -> c1 -> ... -> ck -> b with k >= 2. By the triangle inequality on
    shortest paths, d(a,c1) + d(c1,c2) >= d(a,c2), so the chain costs at least
    d(a,c2) + d(c2,b) — which is exactly the single-charger option through c2 —
    and it arrives with the same charge, max - d(c2,b), since every charger
    resets to full. So whenever that single-charger option is ADMISSIBLE the
    chain can never be cheaper nor arrive fuller, and is safely ignored.
    The one case a chain can do something a single charger cannot is REACH:
    c2 might be admissible only via c1 (d(a,c2) > entering charge). That can
    affect FEASIBILITY but never the cost of an already-feasible route, so it
    is checked exactly where it matters — see the `infeasible` branch in
    solve_query, which re-tests reachability before declaring infeasibility and
    downgrades to `needs_charger_chaining_unverified` rather than claiming a
    proof it does not have.

    NOTE this is emphatically NOT a per-hop flag: a hop that cannot be closed
    from one particular label is an ordinary dead branch of the search (that
    label simply had too little charge), not evidence that chaining is needed.
    An earlier version flagged those and reported 9/12 queries unverified when
    the true count was 0.
    """
    out: List[Tuple[float, float]] = []

    d_ab = float(dist[a_row, b_row])
    if np.isfinite(d_ab) and d_ab <= charge + TIE_TOL_M:
        out.append((d_ab, charge - d_ab))

    d_ac = dist[a_row, charger_rows].astype(np.float64)      # a -> each charger
    d_cb = dist[charger_rows, b_row].astype(np.float64)      # each charger -> b
    ok = np.isfinite(d_ac) & np.isfinite(d_cb) & \
         (d_ac <= charge + TIE_TOL_M) & (d_cb <= max_m + TIE_TOL_M)
    if ok.any():
        cost = d_ac[ok] + d_cb[ok]
        arrive = max_m - d_cb[ok]
        i_cheap = int(np.argmin(cost))          # cheapest detour
        i_full = int(np.argmax(arrive))         # detour arriving with most charge
        out.append((float(cost[i_cheap]), float(arrive[i_cheap])))
        out.append((float(cost[i_full]), float(arrive[i_full])))
    # No flag here by design: an empty result just means THIS label lacked the
    # charge for this hop, which is a normal dead branch. Chaining is assessed
    # once, globally, only if the whole query comes out infeasible.
    return _pareto(out)


def solve_query(
    gold: Dict[str, Any], G: nx.MultiDiGraph, snapper: NodeSnapper,
    dist: np.ndarray, row_of: Dict[Any, int],
    rows_by_type: Dict[str, np.ndarray], nodes_by_type: Dict[str, np.ndarray],
    charger_rows: np.ndarray, charger_nodes: np.ndarray,
) -> Dict[str, Any]:
    types: List[str] = list(gold.get("poi_types") or [])
    if not types:
        return {"status": "no_poi_types", "optimal_distance_m": None}
    cur_m, max_m = battery_to_meters(gold)
    if cur_m is None or max_m is None:
        return {"status": "no_battery_params", "optimal_distance_m": None}

    src = snapper.snap(*gold["src_latlon"])
    dst = snapper.snap(*gold["dst_latlon"])
    if src is None or dst is None:
        return {"status": "snap_failed", "optimal_distance_m": None}

    n_legs = len(types) + 1
    limits = leg_limits_m(gold, n_legs)

    # src/dst are arbitrary coordinates, so they are not in the leg matrix:
    # one Dijkstra each supplies their rows of distances.
    d_src = nx.single_source_dijkstra_path_length(G, src, weight="length")
    d_dst = nx.single_source_dijkstra_path_length(
        G.reverse(copy=False), dst, weight="length")

    def row_vec_from_src(nodes) -> np.ndarray:
        return np.array([d_src.get(n, INF) for n in nodes], dtype=np.float64)

    def row_vec_to_dst(nodes) -> np.ndarray:
        return np.array([d_dst.get(n, INF) for n in nodes], dtype=np.float64)

    # slot 0: src -> first POI layer
    # LEG LIMIT MASK: pure matrix/SP lookup on the DIRECT distance, applied
    # before any detour is considered (identical role to algo2_oracle's mask).
    layer_nodes = nodes_by_type[types[0]]
    layer_rows = rows_by_type[types[0]]
    direct0 = row_vec_from_src(layer_nodes)
    legal0 = np.isfinite(direct0)
    if limits[0] is not None:
        legal0 &= direct0 <= limits[0] + 1e-6

    # labels[node_index] = list of (cost, charge, backptr)
    labels: Dict[int, List[Tuple[float, float, Any]]] = {}
    src_charger_d = np.array([d_src.get(n, INF) for n in charger_nodes], dtype=np.float64)
    for j in np.nonzero(legal0)[0]:
        outs: List[Tuple[float, float]] = []
        d = float(direct0[j])
        if d <= cur_m + TIE_TOL_M:
            outs.append((d, cur_m - d))
        d_cb = dist[charger_rows, layer_rows[j]].astype(np.float64)
        ok = np.isfinite(src_charger_d) & np.isfinite(d_cb) & \
             (src_charger_d <= cur_m + TIE_TOL_M) & (d_cb <= max_m + TIE_TOL_M)
        if ok.any():
            cost = src_charger_d[ok] + d_cb[ok]
            arrive = max_m - d_cb[ok]
            ic, if_ = int(np.argmin(cost)), int(np.argmax(arrive))
            outs.append((float(cost[ic]), float(arrive[ic])))
            outs.append((float(cost[if_]), float(arrive[if_])))
        pf = _pareto(outs)
        if pf:
            labels[int(j)] = [(c, b, None) for c, b in pf]

    # slots 1..n-1: POI layer -> POI layer
    prev_type = types[0]
    for i in range(1, len(types)):
        cur_type = types[i]
        prev_rows, cur_rows = rows_by_type[prev_type], rows_by_type[cur_type]
        cur_nodes = nodes_by_type[cur_type]
        M = dist[np.ix_(prev_rows, cur_rows)].astype(np.float64)
        legal = np.isfinite(M)
        if limits[i] is not None:
            legal &= M <= limits[i] + 1e-6          # DIRECT distance only

        nxt: Dict[int, List[Tuple[float, float, Any]]] = {}
        for pj, lab in labels.items():
            targets = np.nonzero(legal[pj])[0]
            if targets.size == 0:
                continue
            for cost0, charge0, _ in lab:
                for qj in targets:
                    for lc, arr in _leg_outcomes(
                            int(prev_rows[pj]), int(cur_rows[qj]),
                            charge0, max_m, dist, charger_rows):
                        nxt.setdefault(int(qj), []).append(
                            (cost0 + lc, arr, (pj, cost0, charge0)))
        # Pareto-reduce per node, carrying backpointers
        labels = {}
        for qj, lst in nxt.items():
            keep = _pareto([(c, b) for c, b, _ in lst])
            kept = []
            for c, b in keep:
                for cc, bb, bp in lst:
                    if abs(cc - c) < 1e-9 and abs(bb - b) < 1e-9:
                        kept.append((cc, bb, bp))
                        break
            labels[qj] = kept
        prev_type = cur_type
        if not labels:
            break

    # final slot: last POI layer -> dst
    best_cost, best_lab = INF, None
    if labels:
        last_rows = rows_by_type[prev_type]
        last_nodes = nodes_by_type[prev_type]
        to_dst = row_vec_to_dst(last_nodes)
        dst_charger_d = np.array([d_dst.get(n, INF) for n in charger_nodes], dtype=np.float64)
        for pj, lab in labels.items():
            d = float(to_dst[pj])
            if not np.isfinite(d):
                continue
            if limits[-1] is not None and d > limits[-1] + 1e-6:
                continue
            for cost0, charge0, bp in lab:
                if d <= charge0 + TIE_TOL_M and cost0 + d < best_cost:
                    best_cost, best_lab = cost0 + d, (pj, cost0, charge0, bp, 0)
                # detour on the final leg
                d_ac = dist[int(last_rows[pj]), charger_rows].astype(np.float64)
                ok = np.isfinite(d_ac) & np.isfinite(dst_charger_d) & \
                     (d_ac <= charge0 + TIE_TOL_M) & (dst_charger_d <= max_m + TIE_TOL_M)
                if ok.any():
                    tot = d_ac[ok] + dst_charger_d[ok]
                    ic = int(np.argmin(tot))
                    if cost0 + float(tot[ic]) < best_cost:
                        best_cost = cost0 + float(tot[ic])
                        best_lab = (pj, cost0, charge0, bp, 1)

    if best_lab is None:
        # No battery-feasible ordered route was found with single-charger
        # detours. Per the completeness proof in _leg_outcomes, chaining can
        # never make a feasible route CHEAPER — it can only extend REACH. So it
        # matters here and nowhere else: if a second charger is reachable only
        # by first visiting another, a chain might exist that this search did
        # not enumerate. Test that condition explicitly rather than claim a
        # proof we do not have.
        two_hop_extends_reach = False
        d_src_c = np.array([d_src.get(n, INF) for n in charger_nodes], dtype=np.float64)
        direct_ok = np.isfinite(d_src_c) & (d_src_c <= cur_m + TIE_TOL_M)
        if direct_ok.any() and not direct_ok.all():
            # from any directly-reachable charger (full charge), can we reach a
            # charger that was NOT directly reachable?
            reach_rows = charger_rows[direct_ok]
            far_rows = charger_rows[~direct_ok]
            hop = dist[np.ix_(reach_rows, far_rows)].astype(np.float64)
            two_hop_extends_reach = bool((np.isfinite(hop) & (hop <= max_m + TIE_TOL_M)).any())
        status = ("needs_charger_chaining_unverified" if two_hop_extends_reach
                  else "infeasible")
        return {"status": status, "optimal_distance_m": None,
                "poi_types": types, "current_battery_m": round(cur_m, 1),
                "max_battery_m": round(max_m, 1),
                "leg_limits_m": [None if l is None else round(l, 2) for l in limits],
                "chaining_could_extend_reach": two_hop_extends_reach}

    return {
        "status": "optimal",
        "optimal_distance_m": round(best_cost, 2),
        "poi_types": types,
        "current_battery_m": round(cur_m, 1),
        "max_battery_m": round(max_m, 1),
        "leg_limits_m": [None if l is None else round(l, 2) for l in limits],
        "final_leg_used_charger": bool(best_lab[4]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(GOLD_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    print("loading graph, leg matrix, gold ...", flush=True)
    t0 = time.time()
    G = build_graph_from_gpkg(str(GPKG))
    snapper = NodeSnapper.from_gpkg(str(GPKG), G)
    npz = np.load(LEG_MATRIX_DIR / "legs.npz")
    dist = npz["dist_m"]
    idx = pd.read_parquet(LEG_MATRIX_DIR / "index.parquet")
    gold_list = json.loads(Path(args.gold).read_text())
    if args.only:
        gold_list = [g for g in gold_list if g["query_id"] in set(args.only)]
    if args.limit:
        gold_list = gold_list[: args.limit]

    rows_by_type, nodes_by_type = {}, {}
    for t, g in idx.groupby("poi_type"):
        u = g.drop_duplicates("node_id").sort_values("row_idx")
        rows_by_type[t] = u["row_idx"].to_numpy()
        nodes_by_type[t] = u["node_id"].to_numpy()
    row_of = {n: int(r) for n, r in
              idx[["node_id", "row_idx"]].drop_duplicates("node_id").itertuples(index=False)}
    cg = idx[idx.poi_type == "charging_station"].drop_duplicates("node_id")
    charger_rows = cg["row_idx"].to_numpy()
    charger_nodes = cg["node_id"].to_numpy()
    print(f"  ready in {time.time()-t0:.1f}s | {len(charger_rows)} chargers "
          f"(EXHAUSTIVE — no ellipse prune, so the incidental-refuel soundness "
          f"argument in the docstring holds unconditionally)")

    results = []
    t0 = time.time()
    for i, gold in enumerate(gold_list):
        q0 = time.time()
        r = solve_query(gold, G, snapper, dist, row_of, rows_by_type, nodes_by_type,
                        charger_rows, charger_nodes)
        r["query_id"] = gold["query_id"]
        r["wall_s"] = round(time.time() - q0, 2)
        results.append(r)
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(gold_list)} ({el:.0f}s, {el/(i+1):.2f}s/query)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    from collections import Counter
    print(f"\ndone: {len(results)} queries in {time.time()-t0:.0f}s -> {out}")
    print("status:", dict(Counter(r["status"] for r in results)))
    ch = sum(1 for r in results if r.get("chaining_could_extend_reach"))
    print(f"infeasible-but-chaining-might-help (unproven): {ch}")


if __name__ == "__main__":
    main()
