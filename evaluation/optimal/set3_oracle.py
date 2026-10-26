"""
evaluation/optimal/set3_oracle.py — exact optimal solver for set3 (ordered+desc).

Problem: src -> [stop_1 ... stop_k, each a POI of a given category that ALSO
satisfies its GOLD filter, in the stated order] -> dst; leg i (src->stop_1,
stop_1->stop_2, ...) within distance_limits_mi[i]; the final leg stop_k->dst is
unconstrained (set3 gold carries one limit per stop, matching
constraint_checker._set3_distance_limits_met). Minimise TOTAL road distance.

Input is the GOLD (set3_meta.json), never CRAFT's parser output, so the answer
separates "no such route exists" from "CRAFT failed to find it".

Candidate sets use EXACTLY the predicates the scorer uses, so oracle and
evaluation describe the same problem:
  structured  -> node in type_nodes[cat] AND node in filter_cache['structured'][(col,val)]
  description -> node in type_nodes[cat] AND some POI row at the node carries a
                 definite True frozen LLM label for the phrase
                 (outputs/evaluation/description_gt/description_labels.json).
                 The labels cover only 534 candidate rows, so a description stop
                 can only be satisfied by a labelled row.
  none        -> node in type_nodes[cat]

Exactness: layered DAG min-plus DP; every candidate is considered, per-leg caps
applied per edge BEFORE the min (they are independent per leg). Pairwise leg
distances come from scipy multi-source Dijkstra with `limit=cap`, so only pairs
within the cap are ever finite — no pruning that could remove an optimum.
A node may fill consecutive slots (scorer credits it; see algo2_oracle).

Output: evaluation/optimal/results/set3_oracle.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import constraint_checker as cc  # noqa: E402

GPKG = ROOT / "data" / "socal" / "road_network.gpkg"
OUT_PATH = ROOT / "evaluation" / "optimal" / "results" / "set3_oracle.json"
INF = float("inf")
CHUNK = 64


def build_csr(G):
    nodes = list(G.nodes())
    ix = {n: i for i, n in enumerate(nodes)}
    best: Dict[tuple, float] = {}
    for u, v, d in G.edges(data=True):
        w = d.get("length")
        if w is None:
            continue
        k = (ix[u], ix[v])
        w = float(w)
        if k not in best or w < best[k]:
            best[k] = w
    r = np.fromiter((k[0] for k in best), dtype=np.int64, count=len(best))
    c = np.fromiter((k[1] for k in best), dtype=np.int64, count=len(best))
    w = np.fromiter(best.values(), dtype=np.float64, count=len(best))
    # zero-length edges would be dropped as "no edge" by csgraph; nudge
    w = np.maximum(w, 1e-6)
    n = len(nodes)
    return nodes, ix, sp.csr_matrix((w, (r, c)), shape=(n, n))


def description_nodes(G, pois_gdf, cat_nodes: Dict[str, Set[Any]]) -> Dict[tuple, Set[Any]]:
    """(category, phrase) -> nodes with a definite True frozen label."""
    labels = cc.load_set3_description_labels()
    out: Dict[tuple, Set[Any]] = {}
    for row_id, per_phrase in labels.items():
        try:
            raw = pois_gdf.loc[int(row_id), "nearest_node_osmid"]
        except (KeyError, ValueError):
            continue
        nid = cc.resolve_graph_node_id(G, raw)
        if nid is None:
            continue
        for phrase, rec in per_phrase.items():
            if isinstance(rec, dict) and rec.get("label") and not rec.get("uncertain"):
                out.setdefault(phrase, set()).add(nid)
    return out


def stop_candidates(stop, cat_nodes, filter_cache, desc_nodes) -> Set[Any]:
    cat = cc.normalize_type(stop["poi_type"])
    base = set(cat_nodes.get(cat, set()))
    kind = stop.get("filter_kind")
    if kind == "structured":
        return base & filter_cache["structured"].get(
            (str(stop["column"]), str(stop.get("value"))), set())
    if kind == "description":
        phrase = stop.get("description") or stop.get("nl_phrase")
        return base & desc_nodes.get(phrase, set())
    return base


def solve_query(gold, cands: List[np.ndarray], base_cands: List[np.ndarray],
                A, AT, ix, nodes, snapper) -> Dict[str, Any]:
    src = snapper.snap(float(gold["src"][0]), float(gold["src"][1]))
    dst = snapper.snap(float(gold["dst"][0]), float(gold["dst"][1]))
    k = len(cands)
    out: Dict[str, Any] = {
        "query_id": gold["query_id"],
        "n_candidates_per_slot": [int(len(c)) for c in cands],
        "n_base_candidates_per_slot": [int(len(c)) for c in base_cands],
    }
    if src is None or dst is None:
        return {**out, "status": "snap_failed"}
    if any(len(c) == 0 for c in cands):
        empty = [i for i, c in enumerate(cands) if len(c) == 0]
        return {**out, "status": "no_candidates", "empty_slots": empty}

    limits = [float(x) * cc.MILES_TO_M for x in (gold.get("distance_limits_mi") or [])]
    limits = (limits + [None] * k)[:k]  # one per stop; final leg free

    s, d = ix[src], ix[dst]
    d_src = dijkstra(A, indices=s, limit=limits[0] if limits[0] else np.inf)
    cost = d_src[cands[0]]
    if limits[0] is not None:
        cost = np.where(cost <= limits[0] + 1e-6, cost, INF)
    alive = np.isfinite(cost)
    # keep only reachable layer-1 candidates (cuts the Dijkstra fan-out)
    cur = cands[0][alive]
    cost = cost[alive]
    if len(cur) == 0:
        return {**out, "status": "infeasible", "dead_at_leg": 0}
    layers = [cur]
    back: List[np.ndarray] = []

    for i in range(1, k):
        nxt = cands[i]
        lim = limits[i]
        newcost = np.full(len(nxt), INF)
        arg = np.zeros(len(nxt), dtype=np.int64)
        for a in range(0, len(cur), CHUNK):
            sub = cur[a:a + CHUNK]
            D = dijkstra(A, indices=sub, limit=lim if lim else np.inf)[:, nxt]
            if lim is not None:
                D = np.where(D <= lim + 1e-6, D, INF)
            tot = cost[a:a + CHUNK, None] + D
            j = np.argmin(tot, axis=0)
            v = tot[j, np.arange(tot.shape[1])]
            better = v < newcost
            newcost[better] = v[better]
            arg[better] = j[better] + a
        alive = np.isfinite(newcost)
        if not alive.any():
            return {**out, "status": "infeasible", "dead_at_leg": i}
        back.append(arg[alive])
        cur, cost = nxt[alive], newcost[alive]
        layers.append(cur)

    d_dst = dijkstra(AT, indices=d)[cur]
    total = cost + d_dst
    if not np.isfinite(total).any():
        return {**out, "status": "infeasible", "dead_at_leg": k}
    j = int(np.argmin(total))
    picks = [j]
    for bk in reversed(back):
        picks.append(int(bk[picks[-1]]))
    picks.reverse()
    stops = [nodes[int(layers[i][picks[i]])] for i in range(k)]
    return {**out, "status": "optimal", "optimal_distance_m": round(float(total[j]), 2),
            "stops": [str(n) for n in stops]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import geopandas as gpd
    from craft.generator.tools.common_handlers import build_graph_from_gpkg

    t0 = time.time()
    gold_by_qid = cc.load_set3_gold()
    gold_list = list(gold_by_qid.values())
    if args.limit:
        gold_list = gold_list[: args.limit]
    print("loading graph/POIs ...", flush=True)
    G = build_graph_from_gpkg(str(GPKG))
    pois = gpd.read_file(str(GPKG), layer="pois")
    snapper = cc.NodeSnapper.from_gpkg(str(GPKG), G)
    cat_nodes, _ = cc.build_poi_caches(G, pois, poi_types=cc.SET3_POI_TYPES)
    fcache = cc.build_set3_filter_cache(G, pois, list(gold_by_qid.values()))
    dnodes = description_nodes(G, pois, cat_nodes)
    nodes, ix, A = build_csr(G)
    AT = A.T.tocsr()
    print(f"ready in {time.time() - t0:.0f}s", flush=True)

    results = []
    t0 = time.time()
    for n, gold in enumerate(gold_list):
        cands, bases = [], []
        for stop in gold["stops"]:
            c = stop_candidates(stop, cat_nodes, fcache, dnodes)
            cands.append(np.array(sorted(ix[x] for x in c if x in ix), dtype=np.int64))
            b = cat_nodes.get(cc.normalize_type(stop["poi_type"]), set())
            bases.append(np.array(sorted(ix[x] for x in b if x in ix), dtype=np.int64))
        res = solve_query(gold, cands, bases, A, AT, ix, nodes, snapper)
        results.append(res)
        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{len(gold_list)} ({time.time() - t0:.0f}s)", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print("status:", dict(Counter(r["status"] for r in results)), "->", out)


if __name__ == "__main__":
    main()
