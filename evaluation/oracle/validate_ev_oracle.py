"""
evaluation/oracle/validate_ev_oracle.py — Step 3 of the CRAFT optimal-oracle
plan: validate ev_oracle.py's results.

1. For 20 random queries: verify oracle_distance <= CRAFT's dijkstra_objective_value
   (CRAFT's REALIZED route cost -- see note below on why not the plain
   "objective_value" column) for every query where CRAFT was feasible AND the
   gold parse matches what CRAFT's own parser produced. Skip + report queries
   where CRAFT's parse diverged from gold (those gaps confound parser error
   with search error).
2. For 5 short queries: cross-check via an INDEPENDENT brute-force enumeration
   of every chain with <= 2 charging stops (does not call ev_oracle.py's own
   solver code at all).
3. Report: per-query gap distribution (craft_objective / oracle_objective),
   infeasible-query count, runtime per query, matrix build time.

NOTE on "checker objective_value" vs "dijkstra_objective_value": for
ev_routing, constraint_checker.evaluate_system's `objective_value` is
computed from `_intended_stop_nodes`, which is driven by `poi_types` -- and
ev_routing has NO required POI types (TEMPLATE_CHECKS only lists
"ev_battery_feasible"), so `intended` is always empty and `objective_value`
collapses to the RAW UNCONSTRAINED shortest_path(src, dst) -- i.e. it is
IDENTICAL for every route regardless of whether a charging detour actually
happened. Comparing the oracle against that field would be comparing against
a number that ignores the battery constraint entirely, and would show
spurious "violations" on every query that genuinely needs a charger. The
field that reflects CRAFT's *actual, realized* route cost (charging detour
included) is `dijkstra_objective_value` = compute_system_objective(route, G,
weight), so that is what this script compares against. Flagging this
substitution explicitly since it isn't what the plan's Step 3 wording named.
"""
from __future__ import annotations

import itertools
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from craft.generator.tools.common_handlers import build_graph_from_gpkg, coerce_node_id, edge_weight
from evaluation.constraint_checker import NodeSnapper, coerce, compute_system_objective
from evaluation.oracle.ev_oracle import (
    GOLD_PATH, GPKG, battery_to_meters, load_leg_matrix, solve_query,
)

ORACLE_RESULTS_PATH = ROOT / "evaluation" / "oracle" / "results" / "ev_routing_oracle.json"
CRAFT_MODEL = "gemini-3_1-flash-lite"  # default; flag if a different model was wanted
CRAFT_DIR = ROOT / "outputs" / "craft" / CRAFT_MODEL / "ev_routing"

N_RANDOM = 20
N_BRUTE_FORCE = 5
MATCH_TOL_M = 10.0  # generous enough to absorb independent unit-conversion float noise
SEED = 0


def load_craft_record(qid: str) -> Optional[Dict[str, Any]]:
    p = CRAFT_DIR / f"{qid}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def craft_feasible(rec: Dict[str, Any]) -> bool:
    return bool(rec.get("execute", {}).get("ok")) and bool(rec.get("result", {}).get("route"))


# CRAFT's parser_output is LLM-generated JSON with NO enforced schema, and
# matching by key NAME is provably unreliable here: across the 200
# ev_routing records the same key "battery_range" holds the CURRENT
# battery in some records ({'battery_range': 8046.7, 'battery_capacity':
# 349226.8}) and the MAX battery in others ({'battery_level': 5986.7,
# 'battery_range': 149668.6}). At least 10 distinct key names appear for the
# max value alone. A name-priority heuristic silently mislabels one group or
# the other.
#
# So we match by VALUE instead, which is what the question actually is: "did
# CRAFT solve the SAME problem gold specifies?" -- i.e. do gold's current and
# max battery both appear among the numbers CRAFT parsed? This is
# name-agnostic and therefore robust to the schema drift. (It is deliberately
# permissive about WHICH key held which value; that is a parser-hygiene
# question, not a route-quality one, and this script measures route quality.)
_NON_BATTERY_KEYS = {"src", "dst", "objective", "template_file"}


def _numeric_values(po: Dict[str, Any]) -> List[float]:
    out: List[float] = []
    for k, v in po.items():
        if k in _NON_BATTERY_KEYS:
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(float(v))
    return out


def parse_matches_gold(rec: Dict[str, Any], gold: Dict[str, Any], G: nx.MultiDiGraph,
                       snapper: NodeSnapper) -> Tuple[bool, str]:
    po = rec.get("parser_output", {})
    gold_cur_m, gold_max_m = battery_to_meters(gold)
    if gold_cur_m is None or gold_max_m is None:
        return False, "gold has no battery params"

    vals = _numeric_values(po)
    if not vals:
        return False, f"craft parser_output has no numeric battery values (keys: {sorted(po.keys())})"

    cur_ok = any(abs(v - gold_cur_m) <= MATCH_TOL_M for v in vals)
    max_ok = any(abs(v - gold_max_m) <= MATCH_TOL_M for v in vals)
    if not cur_ok:
        return False, (f"current battery not found among craft's parsed values: "
                       f"gold={gold_cur_m:.1f}m craft_values={[round(v,1) for v in vals]}")
    if not max_ok:
        return False, (f"max battery not found among craft's parsed values: "
                       f"gold={gold_max_m:.1f}m craft_values={[round(v,1) for v in vals]}")

    gold_src = snapper.snap(*gold["src_latlon"])
    gold_dst = snapper.snap(*gold["dst_latlon"])
    craft_src = coerce(G, po.get("src"))
    craft_dst = coerce(G, po.get("dst"))
    if craft_src != gold_src:
        return False, f"src diverges: craft={craft_src} gold={gold_src}"
    if craft_dst != gold_dst:
        return False, f"dst diverges: craft={craft_dst} gold={gold_dst}"

    return True, "match"


def brute_force_reference(
    src: Any, dst: Any, current_m: float, max_m: float,
    G: nx.MultiDiGraph, all_chargers: List[Any], max_stops: int = 2,
) -> Tuple[Optional[float], Optional[List[Any]]]:
    """Fully independent of ev_oracle.py's own code -- recomputes everything
    from scratch via direct networkx calls, enumerating every 0/1/2-charger
    chain explicitly (no meta-graph, no pruning)."""
    d_src = nx.single_source_dijkstra_path_length(G, src, weight="length")
    d_dst = nx.single_source_dijkstra_path_length(G.reverse(copy=False), dst, weight="length")

    best_dist, best_chain = None, None

    direct = d_src.get(dst)
    if direct is not None and direct <= current_m + 1e-6:
        best_dist, best_chain = direct, [src, dst]

    for c in all_chargers:
        if c in (src, dst):
            continue
        ds = d_src.get(c)
        dd = d_dst.get(c)
        if ds is None or dd is None:
            continue
        if ds <= current_m + 1e-6 and dd <= max_m + 1e-6:
            total = ds + dd
            if best_dist is None or total < best_dist - 1e-6:
                best_dist, best_chain = total, [src, c, dst]

    if max_stops >= 2:
        for c1 in all_chargers:
            if c1 in (src, dst):
                continue
            ds1 = d_src.get(c1)
            if ds1 is None or ds1 > current_m + 1e-6:
                continue
            d_c1 = nx.single_source_dijkstra_path_length(G, c1, weight="length")
            for c2 in all_chargers:
                if c2 in (src, dst, c1):
                    continue
                d12 = d_c1.get(c2)
                dd2 = d_dst.get(c2)
                if d12 is None or dd2 is None:
                    continue
                if d12 <= max_m + 1e-6 and dd2 <= max_m + 1e-6:
                    total = ds1 + d12 + dd2
                    if best_dist is None or total < best_dist - 1e-6:
                        best_dist, best_chain = total, [src, c1, c2, dst]

    return best_dist, best_chain


def main() -> None:
    print("loading graph, leg matrix, gold, oracle results ...")
    t0 = time.time()
    G = build_graph_from_gpkg(str(GPKG))
    snapper = NodeSnapper.from_gpkg(str(GPKG), G)
    dist_m, idx_df, node_to_row = load_leg_matrix()
    gold_list = json.loads(GOLD_PATH.read_text())
    gold_by_id = {r["query_id"]: r for r in gold_list}
    oracle_list = json.loads(ORACLE_RESULTS_PATH.read_text())
    oracle_by_id = {r["query_id"]: r for r in oracle_list}
    all_chargers = list(set(idx_df[idx_df.poi_type == "charging_station"]["node_id"].unique()))
    print(f"  loaded in {time.time() - t0:.1f}s")

    # infeasible-query count (full 200)
    status_counts: Dict[str, int] = {}
    for r in oracle_list:
        key = r["status"] if r["status"] == "optimal" else f"infeasible ({r['infeasible_reason']})"
        status_counts[key] = status_counts.get(key, 0) + 1
    print("\n=== oracle status breakdown (full 200) ===")
    print(json.dumps(status_counts, indent=2))

    # Part 1: 20 random queries vs CRAFT
    print(f"\n=== Part 1: {N_RANDOM} random queries vs CRAFT ({CRAFT_MODEL}) ===")
    rng = random.Random(SEED)
    optimal_ids = [r["query_id"] for r in oracle_list if r["status"] == "optimal"]
    sample_ids = rng.sample(optimal_ids, min(N_RANDOM, len(optimal_ids)))

    gaps: List[Tuple[str, float, float, float]] = []  # qid, craft_obj, oracle_obj, ratio
    skipped: List[Tuple[str, str]] = []
    violations: List[Tuple[str, float, float]] = []

    for qid in sample_ids:
        craft_rec = load_craft_record(qid)
        if craft_rec is None:
            skipped.append((qid, "no craft output found"))
            continue
        if not craft_feasible(craft_rec):
            skipped.append((qid, "craft not feasible (no route)"))
            continue
        ok, reason = parse_matches_gold(craft_rec, gold_by_id[qid], G, snapper)
        if not ok:
            skipped.append((qid, f"parse diverged: {reason}"))
            continue

        oracle_obj = oracle_by_id[qid]["optimal_distance_m"]
        craft_dijkstra_obj = compute_system_objective(
            craft_rec["result"]["route"], G, "length")
        if craft_dijkstra_obj is None:
            skipped.append((qid, "could not compute craft's realized route cost"))
            continue

        ratio = craft_dijkstra_obj / oracle_obj if oracle_obj > 0 else float("inf")
        gaps.append((qid, craft_dijkstra_obj, oracle_obj, ratio))
        if oracle_obj > craft_dijkstra_obj + 1.0:  # 1m tolerance
            violations.append((qid, oracle_obj, craft_dijkstra_obj))

    print(f"\ncompared: {len(gaps)}/{len(sample_ids)}  skipped: {len(skipped)}")
    for qid, reason in skipped:
        print(f"  SKIPPED {qid}: {reason}")
    if gaps:
        ratios = [g[3] for g in gaps]
        print(f"\nper-query gap (craft_dijkstra_objective_value / oracle_optimal_distance_m):")
        for qid, c, o, r in gaps:
            print(f"  {qid}: craft={c:.1f}m oracle={o:.1f}m ratio={r:.4f}")
        print(f"\nratio stats: min={min(ratios):.4f} max={max(ratios):.4f} "
              f"mean={sum(ratios)/len(ratios):.4f}")
    print(f"\nCORRECTNESS VIOLATIONS (oracle > craft's realized cost by >1m -- should be ZERO): "
          f"{len(violations)}")
    for v in violations:
        print("  ", v)

    # Part 2: 5 short queries, independent brute-force cross-check
    print(f"\n=== Part 2: {N_BRUTE_FORCE} short queries, independent brute-force (<=2 stops) ===")
    scored = []
    for r in gold_list:
        cur_m, max_m = battery_to_meters(r)
        if cur_m is None or max_m is None:
            continue
        scored.append((cur_m, r))
    scored.sort(key=lambda x: x[0])  # shortest current range first (tends to be short trips)
    short_records = [r for _, r in scored[:N_BRUTE_FORCE]]

    bf_mismatches = []
    for r in short_records:
        qid = r["query_id"]
        cur_m, max_m = battery_to_meters(r)
        src = snapper.snap(*r["src_latlon"])
        dst = snapper.snap(*r["dst_latlon"])
        t0 = time.time()
        bf_dist, bf_chain = brute_force_reference(src, dst, cur_m, max_m, G, all_chargers, max_stops=2)
        dt = time.time() - t0
        oracle_rec = oracle_by_id[qid]
        oracle_dist = oracle_rec["optimal_distance_m"] if oracle_rec["status"] == "optimal" else None
        match = (bf_dist is None and oracle_dist is None) or \
                (bf_dist is not None and oracle_dist is not None and abs(bf_dist - oracle_dist) < 1.0)
        print(f"  {qid}: brute_force={bf_dist} oracle={oracle_dist} "
              f"{'MATCH' if match else 'MISMATCH!!'} ({dt:.1f}s brute-force)")
        if not match:
            bf_mismatches.append((qid, bf_dist, oracle_dist))

    print(f"\nbrute-force mismatches (should be ZERO): {len(bf_mismatches)}")
    for m in bf_mismatches:
        print("  ", m)

    # Part 3: FULL-SET comparison (every query, not just the 20 sample)
    # The plan only asked for 20, but the oracle solved all 200, so the full
    # gap distribution is free and far more informative. Reported in two
    # buckets so parser error and search error stay separable:
    #   search-only : gold parse == craft parse -> pure route-quality gap
    #   parse-diverged: craft solved a different problem -> gap is confounded,
    #                   reported separately, never mixed into the headline
    print("\n=== Part 3: FULL-SET comparison (all 200) ===")
    full_search: List[Tuple[str, float, float, float]] = []
    full_diverged: List[Tuple[str, float, float, float, str]] = []
    craft_missing = craft_infeasible = oracle_infeasible = 0

    for orec in oracle_list:
        qid = orec["query_id"]
        if orec["status"] != "optimal":
            oracle_infeasible += 1
            continue
        craft_rec = load_craft_record(qid)
        if craft_rec is None:
            craft_missing += 1
            continue
        if not craft_feasible(craft_rec):
            craft_infeasible += 1
            continue
        craft_obj = compute_system_objective(craft_rec["result"]["route"], G, "length")
        if craft_obj is None:
            craft_missing += 1
            continue
        oracle_obj = orec["optimal_distance_m"]
        ratio = craft_obj / oracle_obj if oracle_obj > 0 else float("inf")
        ok, reason = parse_matches_gold(craft_rec, gold_by_id[qid], G, snapper)
        if ok:
            full_search.append((qid, craft_obj, oracle_obj, ratio))
        else:
            full_diverged.append((qid, craft_obj, oracle_obj, ratio, reason))

    def _dist(rs: List[float]) -> str:
        if not rs:
            return "n/a"
        s = sorted(rs)
        at_opt = sum(1 for r in s if r <= 1.0 + 1e-6)
        return (f"n={len(s)} min={s[0]:.4f} median={s[len(s)//2]:.4f} max={s[-1]:.4f} "
                f"mean={sum(s)/len(s):.4f} | at optimum (ratio<=1.0): {at_opt}/{len(s)} "
                f"({at_opt/len(s)*100:.1f}%)")

    print(f"  oracle infeasible (excluded):        {oracle_infeasible}")
    print(f"  craft infeasible / no route:         {craft_infeasible}")
    print(f"  craft output missing / uncomputable: {craft_missing}")
    print(f"\n  SEARCH-ONLY gap (gold parse == craft parse) — the headline number:")
    print(f"    {_dist([g[3] for g in full_search])}")
    print(f"\n  PARSE-DIVERGED (craft solved a different problem; gap confounded):")
    print(f"    {_dist([g[3] for g in full_diverged])}")
    worst = sorted(full_search, key=lambda g: -g[3])[:5]
    if worst and worst[0][3] > 1.0 + 1e-6:
        print("\n  worst search-only gaps:")
        for qid, c, o, r in worst:
            if r > 1.0 + 1e-6:
                print(f"    {qid}: craft={c:.1f}m oracle={o:.1f}m ratio={r:.4f}")
    full_violations = [g for g in full_search + [d[:4] for d in full_diverged]
                       if g[2] > g[1] + 1.0]
    print(f"\n  CORRECTNESS VIOLATIONS across ALL compared (oracle > craft): {len(full_violations)}")
    for v in full_violations[:10]:
        print("   ", v)

    print("\n=== summary ===")
    print(f"matrix build time: 900s (from Step 1 log)")
    print(f"oracle full-200 status breakdown: {json.dumps(status_counts)}")
    print(f"Part 1 (20 random): {len(gaps)} compared, {len(violations)} correctness violations, "
          f"{len(skipped)} skipped")
    print(f"Part 2: {len(short_records)} brute-force cross-checks, {len(bf_mismatches)} mismatches")
    print(f"Part 3 (all 200): {len(full_search)} search-only comparable, "
          f"{len(full_diverged)} parse-diverged, {len(full_violations)} violations")


if __name__ == "__main__":
    main()
