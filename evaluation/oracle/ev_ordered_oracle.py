"""
evaluation/oracle/ev_ordered_oracle.py — exact optimal solver for
ev+ordered_poi.

Problem: src -> [POI types in the stated order] -> dst, every leg within its
mile limit, minimising TOTAL DISTANCE, and the battery must never go negative.

Input is the parser GROUND TRUTH only (set1_ev+ordered_poi_gold.json).

WHY THIS IS algo2's DP PLUS A CHECK, NOT A HARDER SEARCH
Battery is a resource that carries ACROSS legs, unlike the per-leg mile caps
(which are independent and can simply be masked out), so in general it cannot
be folded into a layered DP without adding a state dimension. But it does not
need to be here, for a reason that was MEASURED rather than assumed:

    battery slack (min-distance route fits the starting charge) : 147
    battery binds (a charging detour would be required)         :   0
    spare charge at the optimum: min 4.1 km, median 106 km
    (only ONE query comes within 5 km of its limit)

Because the objective (total distance) and the binding resource (charge
consumed = distance travelled) are THE same QUANTITY, minimising distance also
minimises consumption. So if the unconstrained-by-battery optimum satisfies the
battery, it is optimal for the battery-constrained problem too — no charging
detour can beat a route that already fits. That makes this exact, not an
approximation.

The `requires_charging_detour` branch below therefore never fires on the
current benchmark. It is kept, and reported honestly, so that a future query
set where battery DOES bind fails loudly instead of silently returning a route
that only looks optimal.

Output: evaluation/oracle/results/ev_ordered_poi_oracle.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from craft.generator.tools.common_handlers import (
    build_graph_from_gpkg, edge_weight, resolve_graph_node_id,
)
from evaluation.constraint_checker import NodeSnapper, coerce
from evaluation.oracle.algo2_oracle import solve_query as solve_ordered
from evaluation.oracle.ev_oracle import GPKG, LEG_MATRIX_DIR, battery_to_meters

GOLD_PATH = ROOT / "benchmark" / "ground_truth" / "set1_ev+ordered_poi_gold.json"
OUT_PATH = ROOT / "evaluation" / "oracle" / "results" / "ev_ordered_poi_oracle.json"

INF = float("inf")


def battery_ok_along(chain: List[Any], G: nx.MultiDiGraph, chargers: Set[Any],
                     cur_m: float, max_m: float) -> Dict[str, Any]:
    """Simulate the charge edge by edge along the realized SP path through the
    chain, refilling to max at any charging station the path crosses.

    Matches how the scorer treats CRAFT (constraint_checker._system_checks):
    non-negative at EVERY point, refuel at chargers on the route.
    """
    battery, min_b, refuels = cur_m, cur_m, 0
    for a, b in zip(chain, chain[1:]):
        try:
            seg = nx.shortest_path(G, a, b, weight="length")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return {"ok": False, "min_battery_m": None, "refuels": refuels}
        for u, v in zip(seg, seg[1:]):
            data = G.get_edge_data(u, v)
            if not data:
                continue
            battery -= edge_weight(list(data.values())[0], "length")
            min_b = min(min_b, battery)
            if v in chargers:
                battery = max_m
                refuels += 1
    return {"ok": min_b >= -1e-6, "min_battery_m": round(min_b, 2), "refuels": refuels}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--gold", default=str(GOLD_PATH),
                    help="gold file to solve; point at the regen gold to score the new set")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    print("loading graph, leg matrix, gold ...", flush=True)
    t0 = time.time()
    G = build_graph_from_gpkg(str(GPKG))
    snapper = NodeSnapper.from_gpkg(str(GPKG), G)
    npz = np.load(LEG_MATRIX_DIR / "legs.npz")
    dist_m = npz["dist_m"]
    idx = pd.read_parquet(LEG_MATRIX_DIR / "index.parquet")
    gold_list = json.loads(Path(args.gold).read_text())
    if args.limit:
        gold_list = gold_list[: args.limit]

    rows_by_type, nodes_by_type = {}, {}
    for t, g in idx.groupby("poi_type"):
        u = g.drop_duplicates("node_id").sort_values("row_idx")
        rows_by_type[t] = u["row_idx"].to_numpy()
        nodes_by_type[t] = u["node_id"].to_numpy()
    chargers: Set[Any] = set(
        idx[idx.poi_type == "charging_station"]["node_id"].to_numpy())
    print(f"  ready in {time.time() - t0:.1f}s | {len(chargers)} charger nodes")

    results: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, gold in enumerate(gold_list):
        r = solve_ordered(gold, G, snapper, dist_m, rows_by_type, nodes_by_type)
        r["query_id"] = gold["query_id"]
        cur_m, max_m = battery_to_meters(gold)
        r["current_battery_m"] = None if cur_m is None else round(cur_m, 1)
        r["max_battery_m"] = None if max_m is None else round(max_m, 1)

        if r["status"] == "optimal":
            if cur_m is None:
                r["status"] = "no_battery_params"
            elif r["optimal_distance_m"] <= cur_m + 1e-6:
                # cheapest route already fits the starting charge -> optimal for
                # the battery-constrained problem too (same quantity minimised)
                r["battery_feasible"] = True
                r["battery_binding"] = False
                r["battery_spare_m"] = round(cur_m - r["optimal_distance_m"], 1)
            else:
                sim = battery_ok_along([coerce(G, n) for n in r["chain"]], G,
                                       chargers, cur_m, max_m)
                r["battery_binding"] = True
                r["battery_feasible"] = sim["ok"]
                r["min_battery_m"] = sim["min_battery_m"]
                r["refuels_on_route"] = sim["refuels"]
                if not sim["ok"]:
                    # a charging detour would be needed; this solver does not
                    # search those, so say so rather than report a wrong optimum
                    r["status"] = "requires_charging_detour_not_modeled"
        results.append(r)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(gold_list)} ({time.time() - t0:.0f}s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    from collections import Counter
    print(f"\ndone: {len(results)} queries in {time.time() - t0:.0f}s -> {out}")
    print("status:", dict(Counter(r["status"] for r in results)))
    binding = sum(1 for r in results if r.get("battery_binding"))
    print(f"battery binding on: {binding} queries "
          f"(0 expected — see the module docstring's measurement)")
    solved = [r for r in results if r["status"] == "optimal"]
    if solved:
        spare = [r["battery_spare_m"] for r in solved if r.get("battery_spare_m") is not None]
        print(f"stops per solved query: {dict(Counter(len(r['stops']) for r in solved))}")
        if spare:
            print(f"spare charge at the optimum (m): min={min(spare):,.0f} "
                  f"median={sorted(spare)[len(spare)//2]:,.0f}")


if __name__ == "__main__":
    main()
