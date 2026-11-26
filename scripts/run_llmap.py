"""
run_llmap_baseline.py — Run LLMAP (LLM parser + MSGS) on benchmark queries.

Always runs the full pipeline: LLMAP's LLM parser (HIPP) then MSGS search.
LLMAP code is imported read-only from baselines/LLMAP/ (do NOT modify those files).
Known issues recorded in metadata (not fixed):
  - stay_time_mapping hardcoded for 5 types — wrong stay times for our types
  - search_type_mapping only covers their 5 types — ordering not enforced

Usage:
  # Default model (from OPENAI_MODEL env var):
  conda run -n myenv python scripts/run_llmap_baseline.py

  # Override model:
  conda run -n myenv python scripts/run_llmap_baseline.py --model gemini-2.5-flash

  # Local endpoint (shorthand for --base-url http://127.0.0.1:8000/v1):
  conda run -n myenv python scripts/run_llmap_baseline.py --local --no-thinking

  # Explicit base URL:
  conda run -n myenv python scripts/run_llmap_baseline.py \\
    --base-url http://myhost:8000/v1 --model mymodel

  # Different dataset:
  conda run -n myenv python scripts/run_llmap_baseline.py \\
    --queries-dir benchmark/socal_queries/set1 \\
    --gpkg data/riverside/road_network.gpkg \\
    --out-dir outputs/llmap_riverside

  # Smoke test:
  conda run -n myenv python scripts/run_llmap_baseline.py --limit 3
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
from dotenv import load_dotenv

ROOT     = Path(__file__).resolve().parents[1]
LLMAP_SRC = ROOT / "baselines" / "LLMAP"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from craft.orcstrator.tools.common_handlers import CATEGORY_COLS, poi_categories
from benchmark.socal_queries.set1.generate_set1 import (
    _extract_cat_limits,
    _extract_poi_types,
)
from evaluation.opening_hours_hipp import to_llmap_day_string

DEFAULT_GPKG        = str(ROOT / "data" / "socal" / "road_network.gpkg")
DEFAULT_OUT_DIR     = ROOT / "outputs" / "llmap"
DEFAULT_QUERIES_DIR = ROOT / "benchmark" / "socal_queries" / "set1"
# set1 active templates (200 each = 1000 total); ev+time and ordered+time_budget dropped from v2
DEFAULT_TEMPLATES   = ["ev_routing", "ordered_poi", "poi_collection", "ev+ordered_poi", "unordered_poi"]

# HIPP gold (poi_types_mapped/dependencies/time_limit/quality_weight/distance_weight),
# loaded once and joined by query_id when --queries-dir points at benchmark/socal_queries/llmap_queries.
HIPP_META_PATH = ROOT / "benchmark" / "socal_queries" / "llmap_queries" / "hipp_meta.json"


def _sanitize_model_name(model: str) -> str:
    return re.sub(r"[/\\. ]+", "_", model)


_OSM_ALIASES: Dict[str, List[str]] = {
    "fitness": ["fitness_centre", "sports_centre", "gym"],
    "shopping_mall": ["mall", "department_store"],
    # HIPP's LLM parser emits "shopping mall" (with a space, per HIPP.py's
    # all_pois list) — alias it the same as "shopping_mall" so the
    # full_pipeline (parser-driven) path resolves to our real "mall" category
    # instead of silently falling back to msgs_only mode on every mall query.
    "shopping mall": ["mall", "department_store"],
}

_DEPARTURE_HOUR = 10  # 10 AM


# ── LLMAP module loading ───────────────────────────────────────────────────────

def _load_utils_module():
    """Load LLMAP utils.py (MSGS algorithm) via importlib. Read-only."""
    import networkx as nx

    spec = importlib.util.spec_from_file_location(
        "llmap_utils", LLMAP_SRC / "main" / "utils.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Known LLMAP bug: stay_time_mapping unit mismatch causes negative edge
    # weights → NetworkXUnfeasible. Monkey-patch nx inside the module to fall
    # back to Bellman-Ford, which handles negatives. Documented limitation only.
    _orig_ssd = nx.single_source_dijkstra

    def _ssd_bf_fallback(G, source, target=None, cutoff=None, weight="weight"):
        try:
            return _orig_ssd(G, source, target=target, cutoff=cutoff, weight=weight)
        except Exception:
            return nx.single_source_bellman_ford(G, source, target=target, weight=weight)

    mod.nx.single_source_dijkstra = _ssd_bf_fallback
    return mod


def _load_hipp_module(api_key: str, base_url: str, model: str, no_thinking: bool = False):
    """
    Load LLMAP HIPP.py (LLM parser) via importlib.
    Monkey-patches module-level `client` to use our endpoint.
    No files in baselines/LLMAP/ are modified.
    """
    import openai
    spec = importlib.util.spec_from_file_location(
        "llmap_hipp", LLMAP_SRC / "dataset" / "HIPP.py"
    )
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        print(f"[warn] HIPP.py exec warning: {exc}")
    mod.client      = openai.OpenAI(api_key=api_key, base_url=base_url)
    mod._our_model  = model
    if no_thinking:
        _orig_create = mod.client.chat.completions.create
        def _create_no_thinking(*a, **kw):
            eb = kw.setdefault("extra_body", {})
            eb.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
            return _orig_create(*a, **kw)
        mod.client.chat.completions.create = _create_no_thinking
    return mod


# ── POI lookup ────────────────────────────────────────────────────────────────

def _to_float_or_none(v) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        return None if f != f else f  # NaN check without importing math/pandas here
    except (TypeError, ValueError):
        return None


def _build_full_poi_lookup(pois_gdf: gpd.GeoDataFrame) -> Dict[str, List[Dict]]:
    """Carries rating/num_ratings/opening_hours through so LLMAP's quality
    objective and availability check can read real (synthetic) per-POI values
    instead of the hardcoded stub — see _build_msgs_graph."""
    lookup: Dict[str, List[Dict]] = {}
    for _, row in pois_gdf.iterrows():
        cats    = [c for c in poi_categories(row, CATEGORY_COLS) if c and c != "nan"]
        node_raw = str(row.get("nearest_node_osmid") or "")
        node_id  = node_raw.split(";")[0].strip()
        lat      = float(row.get("lat") or 0)
        lon      = float(row.get("lon") or 0)
        name     = str(row.get("name") or "")
        rating       = _to_float_or_none(row.get("rating"))
        num_ratings  = _to_float_or_none(row.get("num_ratings"))
        opening_hours = row.get("opening_hours")
        opening_hours = opening_hours if isinstance(opening_hours, str) and opening_hours else None
        for cat in cats:
            lookup.setdefault(cat, []).append({
                "node_id": node_id, "name": name, "lat": lat, "lon": lon,
                "rating": rating, "num_ratings": num_ratings,
                "opening_hours": opening_hours,
            })
    return lookup


def _get_osm_keys(our_type: str) -> List[str]:
    if our_type in _OSM_ALIASES:
        return _OSM_ALIASES[our_type]
    return [our_type]


def _lookup_type(full_lookup: Dict, our_type: str) -> List[Dict]:
    result: List[Dict] = []
    for osm_key in _get_osm_keys(our_type):
        result.extend(full_lookup.get(osm_key, []))
    return result


# ── Corridor filtering ─────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R    = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = (math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(min(1.0, a)))


def _filter_corridor(
    poi_list: List[Dict],
    src: Tuple[float, float],
    dst: Tuple[float, float],
    expand_factor: float = 0.5,
    min_expand_km: float = 5.0,
    max_per_type: int = 30,
) -> List[Dict]:
    if not poi_list:
        return []
    src_lat, src_lon = src
    dst_lat, dst_lon = dst
    d_km    = _haversine_km(src_lat, src_lon, dst_lat, dst_lon)
    mid_lat = (src_lat + dst_lat) / 2

    def _bbox_filter(expand_km: float) -> List[Dict]:
        lat_exp = expand_km / 110.574
        lon_exp = expand_km / (111.320 * math.cos(math.radians(mid_lat)) + 1e-9)
        lat_min = min(src_lat, dst_lat) - lat_exp
        lat_max = max(src_lat, dst_lat) + lat_exp
        lon_min = min(src_lon, dst_lon) - lon_exp
        lon_max = max(src_lon, dst_lon) + lon_exp
        return [p for p in poi_list if lat_min <= p["lat"] <= lat_max and lon_min <= p["lon"] <= lon_max]

    filtered = _bbox_filter(max(d_km * expand_factor, min_expand_km))
    if len(filtered) < 2:
        filtered = _bbox_filter(max(d_km, 15.0))
    return filtered[:max_per_type]


# ── NL query parsing ──────────────────────────────────────────────────────────

def _parse_nl(nl_query: str) -> Dict:
    coords = re.findall(r"\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)", nl_query)
    if len(coords) < 2:
        raise ValueError("Cannot find src/dst coordinates in query")
    src = (float(coords[0][0]), float(coords[0][1]))
    dst = (float(coords[1][0]), float(coords[1][1]))

    poi_types  = _extract_poi_types(nl_query)
    # generate_set1.py's LLMAP_DISPLAY_TO_CATEGORY (shared with the frozen
    # set1 benchmark generator, deliberately not touched here) has no
    # "shopping mall" entry, so mall-only HIPP queries would otherwise get an
    # empty poi_types and bail out of run_query before reaching the parser.
    if re.search(r"\bshopping\s+malls?\b|\bmalls?\b", nl_query, re.I) and "mall" not in poi_types:
        poi_types.append("mall")
    cat_limits = _extract_cat_limits(nl_query)
    has_order  = bool(re.search(r"in that order|in order|first.{0,60}then", nl_query, re.I))

    time_budget_min: Optional[float] = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*hours?\b", nl_query, re.I)
    if m:
        time_budget_min = float(m.group(1)) * 60
    else:
        for m in re.finditer(r"(\d+)\s*minutes?\b", nl_query, re.I):
            val = int(m.group(1))
            if val > 60:
                time_budget_min = float(val)
                break

    visit_min = 30
    m = re.search(r"(\d+)\s*minutes?\s*(?:per stop|at each|per visit)", nl_query, re.I)
    if not m:
        m = re.search(r"(?:takes?|taking)\s+(?:about\s+)?(\d+)\s*minutes?", nl_query, re.I)
    if m:
        visit_min = int(m.group(1))

    return {
        "src": src, "dst": dst, "poi_types": poi_types, "cat_limits": cat_limits,
        "has_order": has_order, "time_budget_min": time_budget_min, "visit_min": visit_min,
    }


# ── Graph construction ────────────────────────────────────────────────────────

def _build_msgs_graph(
    src: Tuple[float, float],
    dst: Tuple[float, float],
    pois_by_type: Dict[str, List[Dict]],
    type_id_map: Dict[str, int],
    time_budget_min: Optional[float],
    parser_output: Optional[Dict] = None,
    gold_weights: Optional[Tuple[float, float]] = None,
) -> Tuple[Dict, Dict[int, Dict]]:
    """
    Build the graph dict expected by MSGS() in utils.py.

    Known issues (documented, not fixed):
      - stay_time_mapping wrong for our types (LLMAP uses their 5 hardcoded types)
      - ordering silently skipped (our types not in search_type_mapping)
    """
    from haversine import haversine

    src_lat, src_lon = src
    dst_lat, dst_lon = dst

    poi_entries: List[Tuple[Dict, int, str]] = []
    for poi_type, pois in pois_by_type.items():
        tid = type_id_map[poi_type]
        for p in pois:
            poi_entries.append((p, tid, poi_type))
        if pois:
            poi_entries.append((pois[0], tid, poi_type))  # anchor duplicate

    n_pois    = len(poi_entries)
    num_nodes = n_pois + 2
    start_node = 0
    goal_node  = num_nodes - 1

    positions        = ([(src_lat, src_lon)]
                        + [(e[0]["lat"], e[0]["lon"]) for e in poi_entries]
                        + [(dst_lat, dst_lon)])
    groups           = [-1] + [e[1] for e in poi_entries] + [-1]
    # Real per-POI synthetic rating/num_ratings/opening_hours from the gpkg
    # (data/socal/enrich_synthetic_attrs.py), not a constant stub — otherwise
    # LLMAP's quality objective and availability check are inert (fairness fix).
    # Fall back to LLMAP's original stub values only for POIs missing data
    # (non-HIPP types looked up via a fallback path never hit by HIPP queries).
    ratings          = [-1.0] + [
        e[0]["rating"] if e[0].get("rating") is not None else 4.0 for e in poi_entries
    ] + [-1.0]
    num_ratings_list = [-1] + [
        int(e[0]["num_ratings"]) if e[0].get("num_ratings") is not None else 100
        for e in poi_entries
    ] + [-1]
    openings         = [""] + [
        [to_llmap_day_string(e[0].get("opening_hours"))] for e in poi_entries
    ] + [""]

    nodes         = list(range(num_nodes))
    edge_indices  = []
    edge_weights  = []
    for i in nodes:
        for j in nodes:
            if i != j and groups[i] != groups[j]:
                edge_indices.append([i, j])
                edge_weights.append(haversine(positions[i], positions[j]))

    if [start_node, goal_node] not in edge_indices:
        edge_indices.append([start_node, goal_node])
        edge_weights.append(haversine(positions[start_node], positions[goal_node]))

    time_constraint: Optional[str] = None
    if parser_output and parser_output.get("time_limit") and parser_output["time_limit"] != "None":
        time_constraint = parser_output["time_limit"]
    elif time_budget_min is not None:
        abs_hour        = int(_DEPARTURE_HOUR + time_budget_min / 60)
        time_constraint = f"{abs_hour}:00"

    dependencies = []
    if parser_output and parser_output.get("dependencies"):
        dependencies = parser_output["dependencies"]

    # Prefer gold HIPP weights over the parser's own extraction — quality_weight
    # is a noisy 0-1 float pulled from vague language ("prioritize good ratings")
    # and we want the rating-vs-distance objective mix to reflect the query's
    # true intent, not compound parser error on top of the rating-data fairness
    # fix above. Falls back to parser_output, then the LLMAP default (distance-only).
    if gold_weights is not None:
        quality_weight, distance_weight = gold_weights
    elif parser_output:
        quality_weight  = parser_output.get("quality_weight", 0.0)
        distance_weight = parser_output.get("distance_weight", 1.0)
    else:
        quality_weight, distance_weight = 0.0, 1.0

    alpha = quality_weight / 2
    beta  = distance_weight

    graph = {
        "positions": positions, "edge_indices": edge_indices, "edge_weights": edge_weights,
        "start_node": start_node, "goal_node": goal_node, "groups": groups,
        "time_constraint": time_constraint, "ratings": ratings, "num_ratings": num_ratings_list,
        "dependencies": dependencies, "openings": openings, "alpha": alpha, "beta": beta,
    }

    node_to_poi: Dict[int, Dict] = {}
    for i, (poi_dict, _, poi_type) in enumerate(poi_entries):
        node_to_poi[i + 1] = {**poi_dict, "poi_type": poi_type}

    return graph, node_to_poi


# ── MSGS execution ────────────────────────────────────────────────────────────

def _run_msgs(utils_mod, graph: Dict, timeout_s: int = 60) -> Tuple:
    def _call():
        return utils_mod.MSGS([graph])

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_call)
        try:
            paths, grps    = fut.result(timeout=timeout_s)
            runtime_ms     = (time.time() - t0) * 1000
            return paths[0], grps[0], runtime_ms, "ok"
        except concurrent.futures.TimeoutError:
            return None, None, (time.time() - t0) * 1000, "timeout"
        except Exception as exc:
            return None, None, (time.time() - t0) * 1000, f"error:{exc}"


def _extract_stops(
    path: List[int], node_to_poi: Dict[int, Dict], start_node: int, goal_node: int
) -> List[Dict]:
    stops = []
    for idx in path:
        if idx in (start_node, goal_node):
            continue
        if idx in node_to_poi:
            p = node_to_poi[idx]
            stops.append({
                "name": p.get("name", ""), "poi_type": p.get("poi_type", ""),
                "lat": p["lat"], "lng": p["lon"], "dataset_node": p.get("node_id", ""),
                "match_method": "name",
            })
    return stops


# ── LLM parser ────────────────────────────────────────────────────────────────

def _run_lm_parser(hipp_mod, nl_query: str) -> Tuple[Optional[Dict], bool]:
    try:
        result = hipp_mod.gpt_estimate_from_instruction(
            nl_query, model_name=hipp_mod._our_model, CoT=False
        )
        if not isinstance(result, dict) or not result.get("pois"):
            return result, True
        return result, False
    except Exception as exc:
        return {"error": str(exc)}, True


# ── Per-query runner ───────────────────────────────────────────────────────────

def _save_result(result: Dict, out_dir: Path, query_id: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{query_id}.json").write_text(json.dumps(result, indent=2))


def run_query(
    query: Dict,
    full_poi_lookup: Dict,
    utils_mod,
    hipp_mod,
    out_dir: Path,
    run_parser: bool = True,
    timeout_s: int = 60,
    gold: Optional[Dict] = None,
) -> Dict:
    """`gold` (optional): the matching hipp_meta.json record, when running the
    HIPP query set. Supplies quality_weight/distance_weight so LLMAP's
    rating-vs-distance objective reflects the query's true intent rather than
    the HIPP parser's own (noisy) weight extraction — see _build_msgs_graph."""
    query_id = query["query_id"]
    nl_query = query["nl_query"]

    result: Dict[str, Any] = {
        "query_id": query_id, "nl_query": nl_query,
        "llm_stops": [], "matched_stops": [], "dropped_stops": [],
        "classification": None, "llmap_meta": {},
    }

    try:
        parsed = _parse_nl(nl_query)
    except Exception as exc:
        result["llmap_meta"] = {"status": "parse_error", "error": str(exc)}
        _save_result(result, out_dir, query_id)
        return result

    src, dst, poi_types = parsed["src"], parsed["dst"], parsed["poi_types"]
    time_budget_min     = parsed["time_budget_min"]

    if not poi_types:
        # No POI stops to plan (e.g. EV point-to-point routing). LLMAP runs Dijkstra
        # on a graph, so the trivial answer for a POI-less query is the direct src->dst
        # route with zero stops. The scorer reconstructs the SP through [src, dst] and
        # applies the template's constraint checks (e.g. ev_battery_feasible), so
        # non-binding EV trips pass and binding ones (needing a charger LLMAP can't
        # plan for) correctly fail. Previously this bailed out with feasible absent,
        # forcing LLMAP to 0 on every EV query. Consistent with the feasible rule at
        # the MSGS path (len(effective_types) == 0 -> feasible).
        result["matched_stops"] = []
        result["llm_stops"]     = []
        result["llmap_meta"]    = {"status": "direct_route_no_pois",
                                   "feasible": True, "mode": "direct_no_pois"}
        _save_result(result, out_dir, query_id)
        return result

    parser_output: Optional[Dict] = None
    parser_failed  = True
    if run_parser and hipp_mod is not None:
        parser_output, parser_failed = _run_lm_parser(hipp_mod, nl_query)
        if parser_failed:
            print(f"    [parser] failed: {parser_output}")

    # No silent fallback to our own regex-derived `poi_types` on a parser
    # failure or a type-vocabulary mismatch (removed 2026-10-21). That
    # fallback quietly substituted a DIFFERENT, possibly smaller category set
    # than the model's own parse requested -- for poi_collection specifically,
    # our regex fallback has its own irregular-plural bug ("pharmacies"/
    # "libraries" don't match the naive `display + 's?'` pattern, "banks"
    # does), so a parser output like {"pois": ["pharmacies","libraries",
    # "banks"]} silently collapsed to just {"bank"}, turning a 3-category
    # "visit as many as you can" query into a trivial 1-category search.
    # That inflated whichever model's parser happened to fail/mismatch more
    # often (measured: 28/200 for gpt-oss-20b vs 4/200 for gpt-4.1 on
    # poi_collection), rewarding a broken parse with an easier problem rather
    # than measuring "what this model's own parse lets LLMAP do." A parser
    # failure or vocabulary mismatch is now recorded as a genuine failure.
    if parser_failed or not parser_output or not parser_output.get("pois"):
        result["llmap_meta"] = {
            "status": "parser_failed", "mode": "parser_failed",
            "parser_output": parser_output, "parser_failed": True, "feasible": False,
        }
        _save_result(result, out_dir, query_id)
        return result

    parser_types = [t.lower().strip() for t in parser_output["pois"]]
    if not all(bool(_lookup_type(full_poi_lookup, t)) for t in parser_types):
        print(f"    [parser] types not in dataset ({parser_types}) → parser failure")
        result["llmap_meta"] = {
            "status": "parser_type_mismatch", "mode": "parser_failed",
            "parser_output": parser_output, "parser_failed": True, "feasible": False,
            "parsed_types": parser_types,
        }
        _save_result(result, out_dir, query_id)
        return result

    effective_types = parser_types
    mode = "full_pipeline"

    # HIPP queries (gold is not None) use max_per_type=10 to match the LLMAP
    # paper's own stated cap (Sec 3.4/4.1: "limited to 10 POIs per type" / "we
    # sample 10 specific POIs for each POI type") — a fidelity argument that
    # only applies when reproducing their own dataset. Non-HIPP query sets
    # (e.g. set1) keep the original 30/type default, which was never
    # dictated by their paper and has established, already-documented results.
    corridor_max_per_type = 10 if gold is not None else 30

    pois_by_type: Dict[str, List[Dict]] = {}
    for pt in effective_types:
        candidates = _lookup_type(full_poi_lookup, pt)
        corridor   = _filter_corridor(candidates, src, dst, max_per_type=corridor_max_per_type)
        if len(corridor) == 1:
            corridor = corridor * 2
        pois_by_type[pt] = corridor

    empty_types = [t for t, pois in pois_by_type.items() if not pois]
    if empty_types:
        result["llmap_meta"] = {
            "status": "no_pois_for_type", "empty_types": empty_types,
            "parser_output": parser_output, "parser_failed": parser_failed, "mode": mode,
        }
        _save_result(result, out_dir, query_id)
        return result

    type_id_map = {t: i for i, t in enumerate(effective_types)}

    gold_weights: Optional[Tuple[float, float]] = None
    if gold and gold.get("quality_weight") is not None and gold.get("distance_weight") is not None:
        gold_weights = (float(gold["quality_weight"]), float(gold["distance_weight"]))

    graph, node_to_poi = _build_msgs_graph(
        src, dst, pois_by_type, type_id_map, time_budget_min,
        parser_output if not parser_failed else None,
        gold_weights=gold_weights,
    )

    n_cands                   = {t: len(pois) for t, pois in pois_by_type.items()}
    path, grps_chosen, runtime_ms, status = _run_msgs(utils_mod, graph, timeout_s)

    stops: List[Dict] = []
    if path is not None and status == "ok":
        stops = _extract_stops(path, node_to_poi, graph["start_node"], graph["goal_node"])

    feasible = status == "ok" and (len(stops) > 0 or len(effective_types) == 0)

    result["llm_stops"]     = [
        {"name": s["name"], "poi_type": s["poi_type"], "lat": s["lat"], "lng": s["lng"]}
        for s in stops
    ]
    result["matched_stops"] = stops
    result["llmap_meta"]    = {
        "mode": mode, "parser_output": parser_output, "parser_failed": parser_failed,
        "n_candidate_pois": n_cands, "type_id_map": type_id_map,
        "msgs_runtime_ms": round(runtime_ms, 1), "feasible": feasible, "status": status,
        "time_constraint_passed": graph["time_constraint"],
        "gold_weights_used": gold_weights,
        "known_issues": [
            "stay_time_mapping_uses_llmap_hardcoded_types_times_are_wrong",
            "search_type_mapping_lacks_our_types_ordering_not_enforced",
            "time_constraint_integer_hours_only_up_to_59min_budget_loss",
            "stay_time_unit_bug_causes_negative_edge_weights_bellman_ford_fallback_used",
        ],
    }

    _save_result(result, out_dir, query_id)
    return result


# ── Main batch runner ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Run LLMAP MSGS on benchmark queries.")
    ap.add_argument("--templates", nargs="+", default=DEFAULT_TEMPLATES,
                    help="Template names to run")
    ap.add_argument("--gpkg", default=DEFAULT_GPKG)
    ap.add_argument("--queries-dir", default=str(DEFAULT_QUERIES_DIR),
                    help=f"Directory containing template JSON files (default: {DEFAULT_QUERIES_DIR})")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help=f"Root output directory (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max queries per template (smoke test)")
    ap.add_argument("--timeout", type=int, default=60,
                    help="MSGS timeout per query (seconds)")
    ap.add_argument("--model", default=None,
                    help="LLM model for HIPP parser (overrides OPENAI_MODEL / GEMINI_MODEL)")
    ap.add_argument("--base-url", default=None,
                    help="API base URL (overrides OPENAI_BASE_URL / GEMINI_BASE_URL)")
    ap.add_argument("--local", action="store_true",
                    help="Shorthand: sets --base-url to http://127.0.0.1:8000/v1")
    ap.add_argument("--openai", action="store_true",
                    help="Use OpenAI: OPENAI_BASE_URL + OPENAI_API_KEY from .env (for gpt-4.1)")
    ap.add_argument("--no-thinking", action="store_true",
                    help="Inject enable_thinking=false into HIPP requests (Qwen3)")
    args = ap.parse_args()

    queries_dir = Path(args.queries_dir)

    model   = (
        args.model
        or os.environ.get("OPENAI_MODEL", "")
        or os.environ.get("GEMINI_MODEL", "")
        or "gemini-2.0-flash"
    )
    is_gemini = "gemini" in model.lower()

    # Resolve base URL — prefer the Gemini endpoint when the model is a Gemini model,
    # otherwise prefer the OpenAI endpoint.
    if args.local:
        base_url = "http://127.0.0.1:8000/v1"
    elif args.openai:
        base_url = os.environ.get("OPENAI_BASE_URL", "") or "https://api.openai.com/v1"
    elif args.base_url:
        base_url = args.base_url
    elif is_gemini:
        base_url = (
            os.environ.get("GEMINI_BASE_URL", "")
            or os.environ.get("OPENAI_BASE_URL", "")
            or "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    else:
        base_url = (
            os.environ.get("OPENAI_BASE_URL", "")
            or os.environ.get("GEMINI_BASE_URL", "")
            or "https://generativelanguage.googleapis.com/v1beta/openai/"
        )

    if args.openai:
        api_key = os.environ.get("OPENAI_API_KEY", "")
    else:
        api_key = (
            (os.environ.get("GEMINI_API_KEY", "") if is_gemini else "")
            or os.environ.get("OPENAI_API_KEY", "")
            or os.environ.get("GEMINI_API_KEY", "")
        )

    if not api_key:
        sys.exit("No LLM API key found. Set OPENAI_API_KEY or GEMINI_API_KEY in .env")

    model_slug = _sanitize_model_name(model)
    out_root   = Path(args.out_dir) / model_slug

    print("Loading LLMAP modules...")
    utils_mod = _load_utils_module()
    print("  utils.py (MSGS) loaded.")

    hipp_mod = _load_hipp_module(api_key, base_url, model, no_thinking=args.no_thinking)
    print(f"  HIPP.py (parser) loaded. model={model}  thinking={'off' if args.no_thinking else 'default'}")

    print(f"\nModel slug  : {model_slug}")
    print(f"Output dir  : {out_root}")
    print(f"Templates   : {args.templates}")
    print(f"Limit       : {args.limit or 'all'}")
    print(f"Queries dir : {queries_dir}")

    print(f"\nLoading POIs from {args.gpkg}...")
    pois_gdf        = gpd.read_file(args.gpkg, layer="pois")
    full_poi_lookup = _build_full_poi_lookup(pois_gdf)
    print(f"  Built lookup: {len(full_poi_lookup)} POI types, "
          f"{sum(len(v) for v in full_poi_lookup.values())} total entries")

    # HIPP gold (quality_weight/distance_weight etc.) — joined by query_id when
    # present, so LLMAP's rating-vs-distance objective uses ground truth
    # instead of compounding its own parser's weight-extraction noise on top
    # of the rating-data fairness fix above. No-op for non-HIPP query sets.
    hipp_gold: Dict[str, Dict] = {}
    if HIPP_META_PATH.exists():
        hipp_gold = {r["query_id"]: r for r in json.loads(HIPP_META_PATH.read_text())}
        print(f"  Loaded HIPP gold: {len(hipp_gold)} records from {HIPP_META_PATH}")

    total_ok = total_fail = total_skip = 0
    for template in args.templates:
        q_file = queries_dir / f"{template}.json"
        if not q_file.exists():
            print(f"[skip] {q_file} not found")
            continue
        queries = json.loads(q_file.read_text())
        if args.limit:
            queries = queries[: args.limit]

        tmpl_out = out_root / template
        tmpl_out.mkdir(parents=True, exist_ok=True)
        print(f"\n── {template}: {len(queries)} queries ──")

        n_ok = n_fail = n_skip = 0
        for i, q in enumerate(queries, 1):
            qid      = q["query_id"]
            out_path = tmpl_out / f"{qid}.json"
            if out_path.exists():
                print(f"  [{i}/{len(queries)}] {qid} ... skipped (exists)")
                n_skip += 1
                continue

            print(f"  [{i}/{len(queries)}] {qid} ...", end=" ", flush=True)
            t0  = time.perf_counter()
            res = run_query(q, full_poi_lookup, utils_mod, hipp_mod, tmpl_out,
                            run_parser=True, timeout_s=args.timeout,
                            gold=hipp_gold.get(qid))
            elapsed = time.perf_counter() - t0

            status   = res["llmap_meta"].get("status", "?")
            feasible = res["llmap_meta"].get("feasible", False)
            n_stops  = len(res.get("matched_stops", []))
            rt       = res["llmap_meta"].get("msgs_runtime_ms", "?")
            print(f"status={status}  feasible={feasible}  stops={n_stops}  "
                  f"mode={res['llmap_meta'].get('mode', 'msgs_only')}  "
                  f"msgs={rt}ms  ({elapsed:.1f}s)")
            if status in ("ok", "no_poi_types_found", "no_pois_for_type", "parse_error"):
                n_ok += 1
            else:
                n_fail += 1

        print(f"  Template done: ok={n_ok}  fail={n_fail}  skipped={n_skip}")
        total_ok   += n_ok
        total_fail += n_fail
        total_skip += n_skip

    print(f"\n{'='*60}")
    print(f"Model: {model_slug}")
    print(f"Total: ok={total_ok}  fail={total_fail}  skipped={total_skip}")
    print(f"Results saved to: {out_root}")


if __name__ == "__main__":
    main()
