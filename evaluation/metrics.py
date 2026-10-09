"""
metrics.py — Constraint checking and metric computation.

Works directly from output.json (route) + input.json (params) + graph.

Per-query metrics:
  ran                   bool   — output.json was written without crash
  feasible              bool   — route is not null
  constraint_satisfaction float — mean of all constraint check values (0–1)
  run_time_ms           float  — search time from output.json

Aggregate metrics:
  run_rate, feasibility_rate, avg_constraint_satisfaction, avg_run_time_ms
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import networkx as nx
import geopandas as gpd

from craft.generator.tools.common_handlers import (
    build_poi_type_nodes,
    edge_weight,
    resolve_graph_node_id,
)


# Edge cost helper

def _edge_cost(G: nx.MultiDiGraph, u: Any, v: Any, weight_key: str) -> float:
    data = G.get_edge_data(u, v)
    if not data:
        return 0.0
    return edge_weight(data[min(data)], weight_key, 30.0)


# Charger node loader

def _charger_nodes(G: nx.MultiDiGraph, pois_gdf: gpd.GeoDataFrame) -> Set[Any]:
    mask = None
    if "amenity" in pois_gdf.columns:
        mask = pois_gdf["amenity"].eq("charging_station")
    if "power" in pois_gdf.columns:
        pm = pois_gdf["power"].eq("charging_station")
        mask = pm if mask is None else (mask | pm)
    if mask is None or "nearest_node_osmid" not in pois_gdf.columns:
        return set()
    nodes: Set[Any] = set()
    for val in pois_gdf[mask]["nearest_node_osmid"].dropna():
        nid = resolve_graph_node_id(G, val)
        if nid is not None:
            nodes.add(nid)
    return nodes


# Constraint checks

def check_constraints(
    route: List[Any],
    input_params: Dict[str, Any],
    G: nx.MultiDiGraph,
    pois_gdf: gpd.GeoDataFrame,
    collected_pois: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Evaluate all applicable constraints for a route given the query input.

    Returns a dict of {check_name: bool | float}.
    Which checks are included depends on what fields are in input_params.

    collected_pois (maximize_poi only, optional): the search's own (node,
    category) collection decisions along its winning path -- see
    craft/generator/tools/template.py::_derive_collected_pois. When present,
    the Collection section below uses THIS directly instead of re-deriving
    counts from route/POI-location overlap. That re-derivation can't tell "the
    search deliberately stopped here" from "the route happens to drive past
    another same-category POI it never intended to collect" -- collected_pois
    is the search's actual decision, not a geometric guess.
    """
    checks: Dict[str, Any] = {}
    # Coerce all route node IDs to the canonical graph type so set membership
    # works regardless of whether nodes are stored as str or int.
    from craft.generator.tools.common_handlers import coerce_node_id as _coerce
    route = [_coerce(G, str(n)) or n for n in route]
    route_set = set(route)

    # EV battery
    if "current_battery" in input_params or "max_battery" in input_params:
        max_battery = float(input_params.get("max_battery", float("inf")))
        current_battery = float(input_params.get("current_battery", max_battery))
        chargers = _charger_nodes(G, pois_gdf)

        # Non-negative at EVERY point, not just at the end — a route that runs
        # flat and only *then* reaches a charger (which resets to max) leaves the
        # car stranded before it arrives. Kept identical to
        # constraint_checker._system_checks, of which this is the live mirror.
        #
        # Refuelling honours the shared EV_REFUEL_POLICY so this live checker and
        # the scorer can never disagree. Under "declared_stops_only" only the
        # chargers the system itself proposed count (input_params
        # ["declared_charger_nodes"]); CRAFT declares none today, so it gets no
        # mid-route refills — matching how the LLM/LLMAP baselines have always
        # been scored. See the constant's comment in constraint_checker.py.
        from evaluation.constraint_checker import EV_REFUEL_POLICY

        if EV_REFUEL_POLICY == "declared_stops_only":
            refill_at = set(input_params.get("declared_charger_nodes") or ())
        else:
            refill_at = chargers
        battery = current_battery
        nonneg = True
        for u, v in zip(route, route[1:]):
            battery -= _edge_cost(G, u, v, "length")
            if battery < 0.0:
                nonneg = False
            if v in refill_at:
                battery = max_battery
        checks["ev_battery_nonnegative"] = nonneg

    # Ordered POI
    poi_types: List[str] = input_params.get("poi_types", [])
    # An unordered collection query (maximize_poi / poi_collection) never
    # asked for a specific visiting sequence — "ordered" is explicit False there.
    # Templates that don't set the flag at all (ordered_poi, ev+ordered_poi,
    # ...) default to True, preserving their existing behaviour unchanged.
    ordered = bool(input_params.get("ordered", True))
    poi_idx = 0
    if poi_types:
        type_nodes = build_poi_type_nodes(G, pois_gdf, poi_types, "nearest_node_osmid")

        # Ordered: all types visited in sequence
        poi_stop_positions: List[int] = []
        for pos, node in enumerate(route):
            if poi_idx < len(poi_types) and node in type_nodes.get(poi_types[poi_idx], set()):
                poi_stop_positions.append(pos)
                poi_idx += 1

        if ordered:
            checks["ordered_all_pois_visited"] = poi_idx == len(poi_types)

            # Unordered type coverage
            types_found = sum(1 for t in poi_types if type_nodes.get(t, set()) & route_set)
            checks["ordered_type_coverage"] = round(types_found / len(poi_types), 4)

            # Per-leg distance limits — accept leg_distances or distance_limits;
            # each element may be a plain float (meters) or {"value": n, "unit": "mi"/"km"/"m"}.
            raw_limits = input_params.get("leg_distances") or input_params.get("distance_limits", [])
            if not isinstance(raw_limits, list):
                raw_limits = [raw_limits] if raw_limits is not None else []
            distance_limits_m: List[float] = []
            for raw in raw_limits:
                if isinstance(raw, dict):
                    val  = float(raw.get("value", 0))
                    unit = str(raw.get("unit", "m")).lower()
                    if unit in ("mi", "mile", "miles"):
                        val *= 1609.34
                    elif unit in ("km", "kilometer", "kilometers"):
                        val *= 1000.0
                    distance_limits_m.append(val)
                else:
                    distance_limits_m.append(float(raw))
            if distance_limits_m:
                breakpoints = [0] + poi_stop_positions + [len(route) - 1]
                n_legs = len(breakpoints) - 1
                # Broadcast single limit to all legs ("each leg within X miles").
                if len(distance_limits_m) == 1 and n_legs > 1:
                    distance_limits_m = distance_limits_m * n_legs
                all_limits_met = True
                for i in range(n_legs):
                    leg_dist = sum(
                        _edge_cost(G, route[j], route[j + 1], "length")
                        for j in range(breakpoints[i], breakpoints[i + 1])
                    )
                    if i < len(distance_limits_m) and leg_dist > distance_limits_m[i]:
                        all_limits_met = False
                        break
                checks["ordered_distance_limits_met"] = all_limits_met

    # Collection (unordered maximize_poi)
    # Category caps are upper limits, not requirements — the query never asked
    # to fill every cap, only to (1) stay within the time budget, (2) never
    # count a stop outside a requested category, (3) never exceed a cap.
    # score_collection() in constraint_checker.py is the single source of
    # truth for this — both this live scorer and the offline/export scorer
    # (constraint_checker._system_checks) call it, so they can't drift apart
    # the way they did before (confirmed disagreeing on 17/200 real records).
    from evaluation.constraint_checker import score_collection

    category_limits: Dict[str, int] = input_params.get("category_limits", {})
    num_collection_visits = 0
    if category_limits:
        type_nodes_for_collection = (
            {} if collected_pois is not None else
            build_poi_type_nodes(G, pois_gdf, list(category_limits.keys()), "nearest_node_osmid")
        )
        sc = score_collection(route, category_limits, type_nodes_for_collection,
                              collected_pois=collected_pois)
        counts = sc["counts"]
        visited = sc["visited"]
        num_collection_visits = sc["num_collection_visits"]

        checks["category_membership_valid"] = sc["category_membership_valid"]
        checks["category_caps_respected"] = sc["category_caps_respected"]

        visit_time = input_params.get("visit_time", 1800)

        def _dwell_s(cat: str) -> float:
            if isinstance(visit_time, dict):
                return float(visit_time.get(cat, 1800))
            return float(visit_time)

        dwell_s = sum(counts[cat] * _dwell_s(cat) for cat in category_limits)
        travel_s = sum(_edge_cost(G, u, v, "travel_time") for u, v in zip(route, route[1:]))
        total_s = travel_s + dwell_s

        budget_s = None
        if "time_limit" in input_params:
            budget_s = float(input_params["time_limit"])
        elif "time_budget_s" in input_params:
            budget_s = float(input_params["time_budget_s"])
        elif "time_budget_min" in input_params:
            budget_s = float(input_params["time_budget_min"]) * 60.0
        if budget_s is not None:
            checks["time_budget_met"] = total_s <= budget_s

    # ── Time budget (ordered templates carrying time_budget_min/s but no
    #    category_limits, e.g. ordered+time_budget) ──────────────────────────
    if not category_limits and ("time_budget_min" in input_params or "time_budget_s" in input_params):
        budget_s = (
            float(input_params["time_budget_s"])
            if "time_budget_s" in input_params
            else float(input_params["time_budget_min"]) * 60.0
        )
        visit_s = (
            float(input_params.get("visit_time_s", input_params.get("visit_min", 30) * 60))
        )
        travel_s = sum(_edge_cost(G, u, v, "travel_time") for u, v in zip(route, route[1:]))
        num_visits = num_collection_visits or poi_idx if poi_types else num_collection_visits
        total_s = travel_s + num_visits * visit_s
        checks["time_budget_met"] = total_s <= budget_s

    return checks


# Per-query metrics

def compute_metrics(
    output: Dict[str, Any],
    input_params: Dict[str, Any],
    G: nx.MultiDiGraph,
    pois_gdf: gpd.GeoDataFrame,
) -> Dict[str, Any]:
    """
    Compute metrics for one query from output.json + input.json + graph.

    Args:
        output:       loaded output.json dict
        input_params: loaded input.json dict
        G:            road network graph
        pois_gdf:     POIs GeoDataFrame

    Returns:
        {ran, feasible, constraint_satisfaction, run_time_ms, constraint_checks}
    """
    route = output.get("route")
    ran = True  # if we got here, the file loaded — code ran
    feasible = route is not None and len(route) > 0
    run_time_ms = float(output.get("run_time_ms", 0.0))

    # Search-effort counters (added 2026-08-05). Present only for runs made
    # after that date — older output.json files have no counters, so these
    # come back None rather than 0 (0 would be a real measurement).
    search_effort = {
        k: output.get(k)
        for k in ("states_expanded", "states_expanded_stale",
                  "states_expanded_useful", "states_generated", "states_pruned",
                  "peak_heap_size", "states_total")
    }

    if not feasible:
        return {
            "ran": ran,
            "feasible": False,
            "constraint_satisfaction": 0.0,
            "run_time_ms": run_time_ms,
            "constraint_checks": {},
            **search_effort,
        }

    checks = check_constraints(route, input_params, G, pois_gdf,
                                collected_pois=output.get("collected_pois"))

    if checks:
        cs = sum(float(v) for v in checks.values()) / len(checks)
    else:
        cs = 1.0  # feasible with no tracked constraints

    return {
        "ran": ran,
        "feasible": True,
        "constraint_satisfaction": round(cs, 4),
        "run_time_ms": run_time_ms,
        "constraint_checks": checks,
        **search_effort,
    }


def compute_metrics_error() -> Dict[str, Any]:
    """Return metrics for a query that failed to run (no output.json)."""
    return {
        "ran": False,
        "feasible": False,
        "constraint_satisfaction": 0.0,
        "run_time_ms": None,
        "constraint_checks": {},
        "states_expanded": None,
        "states_expanded_stale": None,
        "states_expanded_useful": None,
        "states_generated": None,
        "states_pruned": None,
        "peak_heap_size": None,
        "states_total": None,
    }


# Aggregate across many runs

def aggregate_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not metrics_list:
        return {"n_total": 0}

    n_total    = len(metrics_list)
    n_runs     = sum(1 for m in metrics_list if m["ran"])
    n_feasible = sum(1 for m in metrics_list if m["feasible"])

    cs_values  = [m["constraint_satisfaction"] for m in metrics_list]
    avg_cs     = sum(cs_values) / n_total

    rt_values  = [m["run_time_ms"] for m in metrics_list if m.get("run_time_ms") is not None]
    avg_rt     = sum(rt_values) / len(rt_values) if rt_values else None
    total_rt   = sum(rt_values) if rt_values else None

    per_check: Dict[str, List[float]] = {}
    for m in metrics_list:
        for key, val in m.get("constraint_checks", {}).items():
            per_check.setdefault(key, []).append(float(val))
    per_check_rate = {k: round(sum(v) / len(v), 4) for k, v in per_check.items()}

    return {
        "n_total":                    n_total,
        "n_runs":                     n_runs,
        "run_rate":                   round(n_runs / n_total, 4),
        "n_feasible":                 n_feasible,
        "feasibility_rate":           round(n_feasible / n_total, 4),
        "avg_constraint_satisfaction": round(avg_cs, 4),
        "avg_run_time_ms":            round(avg_rt, 2) if avg_rt is not None else None,
        "total_run_time_ms":          round(total_rt, 2) if total_rt is not None else None,
        "per_check_pass_rate":        per_check_rate,
    }


def aggregate_by_template(run_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Group records by template_file and aggregate metrics per template."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in run_records:
        tpl = record.get("template_file") or "unknown"
        groups.setdefault(tpl, []).append(record["metrics"])
    return {tpl: aggregate_metrics(mlist) for tpl, mlist in groups.items()}
