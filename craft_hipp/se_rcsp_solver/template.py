"""SE-RCSP instance template (paper Section 7).

The LLM fills the four bodies in Algorithm; everything else here is fixed.
Any combination of EV battery, ordered POI stops, time budget, per-leg distance
limits or category limits is expressed as resources in state.resources.
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
_TOOLS_DIR = ROOT / "craft_hipp" / "orcstrator" / "tools"
_SOLVER_DIR = ROOT / "craft_hipp" / "se_rcsp_solver"
for _p in [str(ROOT), str(_TOOLS_DIR), str(_SOLVER_DIR)]:
    if _p not in sys.path:
        sys.path.append(_p)

from common_handlers import (
    build_graph_from_gpkg,
    build_poi_type_nodes,
    coerce_node_id,
    edge_weight,
    ensure_nodes_in_graph,
    is_open_at,
    min_time_to_dest,
    parse_opening_hours,
    read_input_json,
    resolve_graph_node_id,
    save_route_gpkg,
    visit_seconds_for_poi_type as _visit_seconds_for_poi_type,
)

# Rebound in master() to close over the spec's own visit_time, so a generated
# body can call visit_seconds_for_poi_type(category) with one argument and still
# get the dwell this query actually asked for.
visit_seconds_for_poi_type = _visit_seconds_for_poi_type
from result import save_output, save_route_visualization
from se_rcsp_search import se_rcsp_search, NO_BOUND


class State:
    """A search state (v, R, f) — paper Section 3."""

    def __init__(
        self,
        node: Any,                          # v
        prev: Optional[int],                # pi: parent state id
        resources: Dict[str, Any],          # R
        objective: float,                   # f
        bound: float = NO_BOUND,            # f-hat
    ) -> None:
        self.node = node
        self.prev = prev
        self.resources = resources
        self.objective = float(objective)
        self.bound = float(bound)

    def with_prev(self, prev_id: Optional[int]) -> "State":
        """A copy of this state with its parent recorded."""
        return State(self.node, prev_id, dict(self.resources),
                     self.objective, self.bound)


class Algorithm:
    """The four query-specific functions — paper Section 6.1.

    Everything the query needs (graph, POI candidates, limits) is set as an
    attribute here, so the generated bodies reach it as self.<name>.
    """

    def __init__(self, *, G: nx.MultiDiGraph, **context) -> None:
        self.G = G
        for key, val in context.items():
            setattr(self, key, val)

    def start(self) -> State:
        __START_BODY__

    def expand(self, state: State) -> Iterable[State]:
        __EXPAND_BODY__

    def is_goal(self, state: State) -> bool:
        __IS_GOAL_BODY__

    def dominates(self, a: State, b: State) -> bool:
        __DOMINATES_BODY__


def master(
    G: nx.MultiDiGraph,
    origin: Any,
    context: Dict[str, Any],
) -> Tuple[Optional[List[Any]], Optional[List[State]]]:
    """Solve one SE-RCSP instance (paper Section 6)."""
    ensure_nodes_in_graph(G, origin, context["destination"])

    # Charge the dwell this query specifies. Bound here, in the fixed scaffold,
    # so generated bodies call visit_seconds_for_poi_type(category) and get the
    # spec's value without having to thread it through themselves.
    global visit_seconds_for_poi_type
    _spec_visit_time = context.get("visit_time")
    visit_seconds_for_poi_type = (
        lambda poi_type: _visit_seconds_for_poi_type(poi_type, _spec_visit_time)
    )

    # A generated body sometimes reads self.departure_time_sec /
    # self.arrival_deadline_sec unconditionally — for an opening-hours test, say
    # — even on a query with no clock deadline, where the spec builder correctly
    # omits both. Defaulting them here turns that stray read into a deadline
    # that is always satisfied instead of an AttributeError that kills the whole
    # search. 36000s = 10:00, the departure assumed whenever a deadline IS set.
    context.setdefault("departure_time_sec", 36000.0)
    context.setdefault("arrival_deadline_sec", float("inf"))

    algorithm = Algorithm(G=G, **context)

    # A maximized objective is stored negated, so f only shrinks and the first
    # goal popped is not optimal (Algorithm 2, lines 6-8).
    objective = str(context.get("objective", "")).strip().lower()
    result = se_rcsp_search(algorithm, monotonic=not objective.startswith("maximize"))

    return result.path, result.path_states


def _progress_value(v: Any) -> Optional[int]:
    """Read a resource as "how many stops done", or None if it isn't that shape.
    int -> itself; tuple/list of bools -> count of True; set/sequence -> len;
    dict of ints -> sum. A float is never progress (that is elapsed time)."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, (tuple, list)):
        if v and all(isinstance(x, bool) for x in v):
            return sum(v)
        return len(v)
    if isinstance(v, (set, frozenset)):
        return len(v)
    if isinstance(v, dict) and v and all(isinstance(x, int) and not isinstance(x, bool)
                                         for x in v.values()):
        return sum(v.values())
    return None


def _stop_pois(context: Dict[str, Any], stop_idx: int, node: Any) -> List[Dict[str, Any]]:
    """The POI rows that qualified `node` for this stop — right category AND
    every filter satisfied by the SAME POI (filtration's target_pois). An empty
    list means no single POI at this node satisfies the stop. Recording only;
    the search never reads it."""
    rows = ((context.get("target_pois") or {}).get(stop_idx) or {}).get(node) or []
    pois_gdf = context.get("pois_gdf")
    out: List[Dict[str, Any]] = []
    for r in rows:
        name = None
        if pois_gdf is not None and "name" in pois_gdf.columns:
            v = pois_gdf.at[r, "name"]
            name = None if v is None or v != v else str(v)
        out.append({"row_id": int(r), "name": name})
    return out


def _derive_stops(
    path: List[Any],
    states: List[State],
    context: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Which stops the winning route actually committed to.

    Read off the reconstructed winning path (states[i].node == path[i]), so every
    entry is the search's own decision rather than a guess from the geometry of
    the route. The generated code picks its own resource names, so the stop is
    found by SHAPE, using whichever signal that query provides:

      1. a resource that counts progress and rises at this step, or
      2. the objective dropping at this step (a maximized objective is stored
         negated, so a drop is a collect).

    `expected` (number of required stops, or total category caps) guards the
    search for a counter: elapsed time also rises, but it rises far more often
    than there are stops, so a candidate that rises too many times is rejected.

    Categories come from possible_targets, not the generated state. A node that
    could serve more than one stop records every candidate rather than guessing.
    Returns [] when the route made no stops; that is a real answer, not a gap.
    """
    if not states or len(states) != len(path):
        return []

    poi_types: List[str] = list(context.get("poi_types") or [])
    cat_limits: Dict[str, Any] = context.get("category_limits") or {}
    possible_targets: Dict[int, Any] = context.get("possible_targets") or {}

    expected: Optional[int] = None
    if poi_types:
        expected = len(poi_types)
    elif cat_limits:
        try:
            expected = sum(int(v) for v in cat_limits.values())
        except (TypeError, ValueError):
            expected = None

    # Signal 1: a progress-shaped resource that only ever rises.
    steps: List[int] = []
    progress_at: Dict[int, int] = {}        # step index -> stop index reached
    for key in (states[0].resources or {}):
        series = [_progress_value(s.resources.get(key)) for s in states]
        if any(v is None for v in series):
            continue
        if any(b < a for a, b in zip(series, series[1:])):
            continue                                    # must be non-decreasing
        rises = [i for i in range(1, len(series)) if series[i] > series[i - 1]]
        if not rises:
            continue
        if expected is not None and len(rises) > expected:
            continue                 # rises too often to be a stop counter
        steps = rises
        progress_at = {i: series[i - 1] for i in rises}
        break

    # Signal 2: the objective dropping.
    if not steps:
        steps = [i for i in range(1, len(states))
                 if states[i].objective < states[i - 1].objective]

    stops: List[Dict[str, Any]] = []
    for i in steps:
        node = path[i]
        candidates = [poi_types[idx] for idx, targets in possible_targets.items()
                      if idx < len(poi_types) and node in targets]
        entry: Dict[str, Any] = {"step_index": i, "node_id": str(node)}
        if i in progress_at:
            entry["stop_index"] = progress_at[i]
        if len(candidates) == 1:
            entry["category"] = candidates[0]
        else:
            entry["category"] = None
            if candidates:
                entry["ambiguous_candidates"] = candidates
        # The specific POIs at this node that satisfy the stop, so the filters
        # can be checked per POI rather than per node.
        if i in progress_at:
            stop_idx = progress_at[i]
        else:
            matched = [idx for idx, targets in possible_targets.items() if node in targets]
            stop_idx = matched[0] if len(matched) == 1 else None
        entry["pois"] = _stop_pois(context, stop_idx, node) if stop_idx is not None else []
        # A filter that matched nothing is abandoned and the broader candidate
        # set kept, so this stop satisfies its category but NOT that filter.
        dropped = (context.get("dropped_filters") or {}).get(stop_idx)
        if dropped:
            entry["filters_dropped"] = list(dropped)
        stops.append(entry)
    return stops


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

            # target_pois records WHICH POI rows qualified each node for each
            # stop (right category AND every filter on the SAME POI), so the
            # recorded stops can be checked per POI, not just per node.
            _target_pois: Dict[int, Dict[Any, List[int]]] = {}
            # Stops whose filter matched nothing and was abandoned, so the
            # output can say the constraint was not actually applied.
            _dropped: Dict[int, List[str]] = {}
            context["possible_targets"] = build_possible_targets(
                G=G,
                pois_gdf=pois_gdf,
                poi_types=poi_types,
                node_col="nearest_node_osmid",
                poi_filters=poi_filters,
                # Hard rating floor, index-aligned with poi_types. Applied here
                # rather than in a generated body so the candidate set is
                # already filtered when the search starts — the generated code
                # needs no rating logic of its own.
                rating_filters=context.get("rating_min"),
                embedding_cache=embedding_cache if embedding_cache else None,
                api_key=_api_key or None,
                embedding_base_url=_base_url,
                target_pois=_target_pois,
                dropped_filters=_dropped,
            )
            context["target_pois"] = _target_pois
            context["dropped_filters"] = _dropped
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

    # The stops the search itself committed to, for every template.
    stops = _derive_stops(path, states, context)

    print(f"Feasible route found — {len(path)} nodes, {len(stops)} stops "
          f"({search_ms:.1f} ms)")
    save_output(query_id, path, search_ms, here, stops=stops)
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
