"""
General Query Algorithm
Supports any combination of: EV battery, ordered POI sequence, time budget, per-leg distance limits,
category limits, corridor constraints, or any other resource tracked in state.constraints.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import math
import re

import geopandas as gpd
import networkx as nx
import pandas as pd


def _find_project_root(start: Path) -> Path:
    """Walk up from start until we find the project root (contains .git or README.md)."""
    for p in [start] + list(start.parents):
        if (p / ".git").exists() or (p / "README.md").exists():
            return p
    return start.parents[2]  # fallback


ROOT = _find_project_root(Path(__file__).resolve().parent)
_TOOLS_DIR = ROOT / "craft" / "generator" / "tools"
for _p in [str(ROOT), str(_TOOLS_DIR)]:
    if _p not in sys.path:
        sys.path.append(_p)

from common_handlers import (
    build_graph_from_gpkg,
    build_poi_type_nodes,
    coerce_node_id,
    edge_weight,
    ensure_nodes_in_graph,
    min_time_to_dest,
    read_input_json,
    resolve_graph_node_id,
    save_route_gpkg,
)
from result import save_output, save_route_visualization
from dijkstra_core import dijkstra_search, reconstruct_path


class State:
    def __init__(
        self,
        node: Any,
        prev: Optional[int],
        constraints: Dict[str, Any],
        objective: float,
    ) -> None:
        self.node = node
        self.prev = prev
        self.constraints = constraints
        self._objective = float(objective)

    @property
    def objective(self) -> float:
        return float(self._objective)

    def update_prev(self, prev_id: Optional[int]) -> "State":
        return State(self.node, prev_id, dict(self.constraints), self._objective)

    @classmethod
    def start(
        cls,
        origin: Any,
        init_context: Dict[str, Any],
    ) -> "State":
        __STATE_START_INIT_BODY__


class Algorithm:
    def __init__(self, *, G: nx.MultiDiGraph, **context) -> None:
        self.G = G
        for key, val in context.items():
            setattr(self, key, val)

    def bound_prune(self, successor: State, optimistic_objective: float) -> bool:
        if successor.node == self.destination:        # a complete, feasible route
            cur = getattr(self, "_best", None)
            real = successor.objective                # achieved -count, not the estimate
            if cur is None or real < cur:
                self._best = real
            return False                              # never prune a complete route
        best = getattr(self, "_best", None)
        return best is not None and optimistic_objective >= best

    def dominates(self, a: State, b: State) -> bool:
        __DOMINATES_BODY__

    def is_goal(self, state: State) -> bool:
        __IS_GOAL_BODY__

    def expand(self, state: State) -> Iterable[State]:
        __EXPAND_BODY__


def master(
    G: nx.MultiDiGraph,
    origin: Any,
    context: Dict[str, Any],
) -> Tuple[Optional[List[Any]], Optional[List[State]]]:
    ensure_nodes_in_graph(G, origin, context["destination"])
    start = State.start(origin=origin, init_context=context)
    logic = Algorithm(G=G, **context)
    result = dijkstra_search(
        start_state=start,
        expand=logic.expand,
        dominates=logic.dominates,
        is_goal=logic.is_goal,
    )
    if str(context.get("objective", "")).strip().lower() == "maximize_poi":
        dst = context["destination"]
        if result.goal_state_id is not None:
            best_id = result.goal_state_id
        else:
            best_id, best = None, None
            for i, s in enumerate(result.states):
                if s.node == dst:
                    if best is None or s.objective < best:
                        best, best_id = s.objective, i
        if best_id is None:
            return None, None
        return reconstruct_path(result.states, best_id, node_getter=lambda s: s.node)
    return reconstruct_path(result.states, result.goal_state_id, node_getter=lambda s: s.node)


def _derive_collected_pois(
    path: List[Any],
    states: List[State],
    context: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """For maximize_poi: figure out which (node, category) pairs the winning path
    actually collected. path/states is the reconstructed winning path from master()
    (states[i].node == path[i]), so each entry is a real decision, not a guess.

    A collection step is any point where objective drops (prompt.md fixes
    objective = -count, so this works regardless of what key name the LLM used
    for its own counters). The category at that node comes from
    context['possible_targets'] rather than the LLM's state — if a node could
    belong to more than one category we record all of them instead of guessing.
    """
    if str(context.get("objective", "")).strip().lower() != "maximize_poi":
        return None
    possible_targets: Dict[int, Any] = context.get("possible_targets") or {}
    poi_types: List[str] = context.get("poi_types") or []
    collected: List[Dict[str, Any]] = []
    for i in range(1, len(states)):
        if states[i].objective >= states[i - 1].objective:
            continue  # objective only decreases on a genuine collection step
        node = path[i]
        candidates = [
            poi_types[idx] for idx, targets in possible_targets.items()
            if idx < len(poi_types) and node in targets
        ]
        entry: Dict[str, Any] = {"node_id": str(node), "step_index": i}
        if len(candidates) == 1:
            entry["category"] = candidates[0]
        else:
            entry["category"] = None
            entry["ambiguous_candidates"] = candidates
        collected.append(entry)
    return collected


def main() -> None:
    import time as _time

    here = Path(__file__).resolve().parent
    cfg = read_input_json(str(here / "input.json"))
    gpkg = cfg.get("gpkg") or str(ROOT / "data" / "riverside" / "road_network.gpkg")

    G = build_graph_from_gpkg(gpkg)
    pois_gdf = gpd.read_file(gpkg, layer="pois")

    src_raw = cfg.get("src", cfg.get("origin"))
    dst_raw = cfg.get("dst", cfg.get("destination"))
    if src_raw is None or dst_raw is None:
        raise ValueError("input.json must include source/destination fields: src|origin and dst|destination")
    src = coerce_node_id(G, str(src_raw))
    dst = coerce_node_id(G, str(dst_raw))

    context: Dict[str, Any] = dict(cfg)
    context["destination"] = dst
    # overwrite the raw string src/dst from the parser with coerced node ids,
    # so generated bodies comparing self.dst/self.src/self.origin compare node-to-node
    context["dst"] = dst
    context["src"] = src
    context["origin"] = src
    context["pois_gdf"] = pois_gdf
    _obj = str(cfg.get("objective", "")).strip().lower()
    if _obj == "minimize_distance":
        context["weight_key"] = "length"
    elif _obj == "minimize_travel_time":
        context["weight_key"] = "travel_time"
    else:
        context["weight_key"] = cfg.get("weight_key", "travel_time")
    context["default_speed_kph"] = float(cfg.get("default_speed_kph", 30.0))

    poi_types = list(context.get("poi_types", []))
    has_battery = any("battery" in str(k).lower() for k in cfg)
    if has_battery and "charging_station" not in poi_types:
        _types_for_nodes = poi_types + ["charging_station"]
    else:
        _types_for_nodes = poi_types

    if _types_for_nodes:
        context["type_nodes"] = build_poi_type_nodes(G, pois_gdf, _types_for_nodes, "nearest_node_osmid")
        try:
            from filtration import build_possible_targets
            from poi_embeddings import embedding_cache_path, load_embedding_cache

            poi_filters = context.get("poi_description_filters") or context.get("poi_filters")
            gpkg_dir = Path(gpkg).parent
            cache_path = embedding_cache_path(gpkg_dir)
            # only load the (~3.5GB) embedding cache when a query has a description filter
            def _needs_embeddings(pf):
                items = pf if isinstance(pf, list) else (list(pf.values()) if isinstance(pf, dict) else [])
                return any((isinstance(v, str) and v.strip()) or
                           (isinstance(v, dict) and v.get("description")) for v in items)
            embedding_cache = (load_embedding_cache(str(cache_path))
                               if (cache_path.exists() and _needs_embeddings(poi_filters)) else {})

            _api_key = ""
            _base_url = None
            _env = ROOT / ".env"
            if _env.exists():
                for _line in _env.read_text().splitlines():
                    _line = _line.strip()
                    if _line.startswith("GEMINI_API_KEY="):
                        _api_key = _line.split("=", 1)[1].strip().strip("'\"")
                    if _line.startswith("GEMINI_BASE_URL="):
                        _base_url = _line.split("=", 1)[1].strip().strip("'\"")

            context["possible_targets"] = build_possible_targets(
                G=G,
                pois_gdf=pois_gdf,
                poi_types=poi_types,
                node_col="nearest_node_osmid",
                poi_filters=poi_filters,
                embedding_cache=embedding_cache if embedding_cache else None,
                api_key=_api_key or None,
                embedding_base_url=_base_url,
            )
        except Exception as _e:
            print(f"[template] possible_targets unavailable ({_e}) — using type_nodes fallback.")
            context["possible_targets"] = {
                i: context["type_nodes"].get(cat, set())
                for i, cat in enumerate(poi_types)
            }

    query_id = cfg.get("query_id", here.parent.name.removesuffix("_output"))

    t0 = _time.perf_counter()
    path, states = master(G, src, context)
    search_ms = (_time.perf_counter() - t0) * 1000

    if path is None or states is None:
        print("No feasible route found.")
        save_output(query_id, None, search_ms, here)
        print(f"Output saved: {here / 'output.json'}")
        return

    collected_pois = _derive_collected_pois(path, states, context)

    print(f"Feasible route found — {len(path)} nodes  ({search_ms:.1f} ms)")
    save_output(query_id, path, search_ms, here, collected_pois=collected_pois)
    print(f"Output saved: {here / 'output.json'}")

    save_route_visualization(
        out_path=here / "output.gpkg",
        G=G,
        route=path,
        context=context,
        pois_gdf=pois_gdf,
    )


if __name__ == "__main__":
    main()
