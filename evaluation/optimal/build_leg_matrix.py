"""
evaluation/optimal/build_leg_matrix.py — Step 1 of the CRAFT optimal-oracle
plan: a one-time prebuilt pairwise leg matrix (distance in meters + travel
time in seconds) between every POI/charger candidate node the oracle solvers
need, computed via full single-source Dijkstra from each candidate node on
the real SoCal road network.

Does NOT touch craft/generator/tools/dijkstra_core.py, template.py, any
prompt file, or the State class — this is a standalone precompute script
unrelated to CRAFT's own search engine.

Candidate nodes: every road node that is the nearest-node snap target of at
least one POI in {pharmacy, bank, library, supermarket, charging_station}
(charging_station identified the same way the rest of the pipeline already
does it: evaluation/constraint_checker.py's amenity=="charging_station").

Output (evaluation/optimal/leg_matrix/):
  legs.npz       — dist_m, time_s: float32 (N, N) matrices, row/col order =
                   the row_idx in index.parquet. Unreachable = np.inf.
  index.parquet  — one row per (POI record, matched type) pair — a single
                   road node can host more than one POI/type, so this has
                   >= N rows; `row_idx` is the shared matrix row for that
                   node. Columns: [row_idx, node_id, poi_id, poi_type, lat, lon].
                   Sorted by (poi_type, node_id) so a later solver can slice
                   "all row_idx for type X" directly off this table without
                   touching the matrix. NOTE: the matrix itself has exactly
                   one row per UNIQUE node (Dijkstra computed once per node,
                   not once per type) — see the design note printed at the
                   end of main() for why, and flag if you wanted literal
                   duplicate per-type matrix rows instead.

Usage:
  python evaluation/optimal/build_leg_matrix.py [--workers 4]

Resumable: checkpoints dist_m/time_s + the set of completed row indices to
evaluation/optimal/leg_matrix/_checkpoints/ every --checkpoint-every rows.
Re-running the same command picks up where it left off.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from craft.generator.tools.common_handlers import (
    CATEGORY_COLS,
    build_graph_from_gpkg,
    poi_categories,
    resolve_graph_node_id,
)

GPKG = ROOT / "data" / "socal" / "road_network.gpkg"
OUT_DIR = ROOT / "evaluation" / "optimal" / "leg_matrix"

POI_TYPES = ["pharmacy", "bank", "library", "supermarket"]
CHARGER_TYPE = "charging_station"
ALL_TYPES = POI_TYPES + [CHARGER_TYPE]

INF = float("inf")

# Set once in the parent process (before the fork-based Pool is created) so
# every worker inherits them via copy-on-write — avoids re-loading the ~460MB
# gpkg / rebuilding the 248k-node graph once per worker.
_G: nx.MultiDiGraph = None
_UNIQUE_NODES: List[Any] = None


def _compute_row(args: Tuple[int, Any]) -> Tuple[int, np.ndarray, np.ndarray]:
    i, node = args
    dist_full = nx.single_source_dijkstra_path_length(_G, node, weight="length")
    time_full = nx.single_source_dijkstra_path_length(_G, node, weight="travel_time")
    dist_row = np.array([dist_full.get(n, INF) for n in _UNIQUE_NODES], dtype=np.float32)
    time_row = np.array([time_full.get(n, INF) for n in _UNIQUE_NODES], dtype=np.float32)
    return i, dist_row, time_row


def build_index(G: nx.MultiDiGraph, pois) -> pd.DataFrame:
    entries = []
    for idx, row in pois.iterrows():
        node_id = resolve_graph_node_id(G, row.get("nearest_node_osmid"))
        if node_id is None:
            continue
        cats = set(poi_categories(row, CATEGORY_COLS))
        matched = [t for t in POI_TYPES if t in cats]
        if row.get("amenity") == "charging_station":
            matched.append(CHARGER_TYPE)
        for t in matched:
            entries.append((idx, node_id, t, row.get("lat"), row.get("lon")))
    return pd.DataFrame(entries, columns=["poi_id", "node_id", "poi_type", "lat", "lon"])


def sanity_checks(G: nx.MultiDiGraph, dist_m: np.ndarray, time_s: np.ndarray,
                  unique_nodes: List[Any]) -> None:
    N = len(unique_nodes)
    print("\n=== sanity checks ===")

    diag_d = np.diag(dist_m)
    diag_t = np.diag(time_s)
    print(f"diagonal(dist_m) max abs: {np.max(np.abs(diag_d)):.6f} (should be 0)")
    print(f"diagonal(time_s) max abs: {np.max(np.abs(diag_t)):.6f} (should be 0)")

    finite = np.isfinite(dist_m)
    both_finite = finite & finite.T
    if both_finite.sum() > 0:
        asym = np.abs(dist_m - dist_m.T)
        asym_both_finite = asym[both_finite]
        print(f"graph is DIRECTED (verified in Step 0) — NOT assuming symmetry.")
        print(f"  dist_m asymmetry where both directions are finite: "
              f"max={np.max(asym_both_finite):.1f}m, mean={np.mean(asym_both_finite):.1f}m, "
              f"fraction >1m different: {(asym_both_finite > 1.0).mean() * 100:.1f}%")

    rng = np.random.default_rng(0)
    print("spot-check vs networkx.shortest_path_length (5 random pairs):")
    for _ in range(5):
        i, j = rng.integers(0, N, size=2)
        a, b = unique_nodes[i], unique_nodes[j]
        try:
            nx_d = nx.shortest_path_length(G, a, b, weight="length")
        except nx.NetworkXNoPath:
            nx_d = INF
        mat_d = dist_m[i, j]
        ok = (nx_d == INF and not np.isfinite(mat_d)) or abs(nx_d - mat_d) < 1e-3
        print(f"  ({i},{j}) node {a}->{b}: matrix={mat_d:.2f} networkx={nx_d if nx_d==INF else round(nx_d,2)} "
              f"{'OK' if ok else 'MISMATCH!!'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4,
                    help="conservative default — see module docstring / chat log for why")
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "_checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    import geopandas as gpd

    print("loading graph + POIs ...", flush=True)
    t0 = time.time()
    global _G, _UNIQUE_NODES
    _G = build_graph_from_gpkg(str(GPKG))
    pois = gpd.read_file(str(GPKG), layer="pois")
    print(f"  done in {time.time() - t0:.1f}s "
          f"({_G.number_of_nodes()} nodes, {_G.number_of_edges()} edges, {len(pois)} POI rows)")

    idx_df = build_index(_G, pois)
    _UNIQUE_NODES = sorted(idx_df["node_id"].unique())
    N = len(_UNIQUE_NODES)
    node_to_row = {n: i for i, n in enumerate(_UNIQUE_NODES)}
    idx_df["row_idx"] = idx_df["node_id"].map(node_to_row)
    idx_df = idx_df.sort_values(["poi_type", "node_id"]).reset_index(drop=True)

    print(f"N unique nodes = {N}, index rows (poi_id x type pairs) = {len(idx_df)}")
    for t in ALL_TYPES:
        print(f"  {t}: {len(idx_df[idx_df.poi_type == t])} index rows, "
              f"{idx_df[idx_df.poi_type == t]['node_id'].nunique()} unique nodes")

    dist_m = np.full((N, N), np.nan, dtype=np.float32)
    time_s = np.full((N, N), np.nan, dtype=np.float32)

    completed_path = ckpt_dir / "completed_rows.json"
    completed = set(json.loads(completed_path.read_text())) if completed_path.exists() else set()
    dist_ckpt, time_ckpt = ckpt_dir / "dist_m_partial.npy", ckpt_dir / "time_s_partial.npy"
    if completed and dist_ckpt.exists() and time_ckpt.exists():
        print(f"resuming: {len(completed)}/{N} rows already done")
        dist_m = np.load(dist_ckpt)
        time_s = np.load(time_ckpt)

    todo = [(i, _UNIQUE_NODES[i]) for i in range(N) if i not in completed]
    print(f"{len(todo)} rows to compute, {args.workers} workers (fork, conservative "
          f"count — see chat log: another CRAFT run is active on this machine)", flush=True)

    t0 = time.time()
    if todo:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=args.workers) as pool:
            since_ckpt = 0
            for i, dist_row, time_row in pool.imap_unordered(_compute_row, todo, chunksize=4):
                dist_m[i] = dist_row
                time_s[i] = time_row
                completed.add(i)
                since_ckpt += 1
                if since_ckpt >= args.checkpoint_every:
                    since_ckpt = 0
                    np.save(dist_ckpt, dist_m)
                    np.save(time_ckpt, time_s)
                    completed_path.write_text(json.dumps(sorted(completed)))
                    print(f"  checkpoint: {len(completed)}/{N} rows done "
                          f"({time.time() - t0:.0f}s elapsed)", flush=True)
            np.save(dist_ckpt, dist_m)
            np.save(time_ckpt, time_s)
            completed_path.write_text(json.dumps(sorted(completed)))

    print(f"\nall {N} rows computed in {time.time() - t0:.0f}s")

    sanity_checks(_G, dist_m, time_s, _UNIQUE_NODES)

    np.savez(out_dir / "legs.npz", dist_m=dist_m, time_s=time_s)
    idx_df.to_parquet(out_dir / "index.parquet")
    print(f"\nsaved {out_dir / 'legs.npz'} and {out_dir / 'index.parquet'}")
    print("\nDESIGN NOTE: the matrix has one row per UNIQUE node (Dijkstra computed once "
          "per node even when a node hosts multiple POI types, e.g. a bank+supermarket "
          "at the same node) — NOT one physically duplicated row per type. index.parquet "
          "is still sorted by (poi_type, node_id) as specified, so a later solver slices "
          "candidates via idx_df[idx_df.poi_type=='X'].row_idx.values then fancy-indexes "
          "into the matrix. Flag if you actually wanted literal duplicate contiguous rows.")


if __name__ == "__main__":
    main()
