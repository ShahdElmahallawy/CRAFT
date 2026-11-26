from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import geopandas as gpd
except Exception:
    gpd = None
try:
    import networkx as nx
except Exception:
    nx = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


CATEGORY_COLS = [
    "amenity",
    "tourism",
    "leisure",
    "shop",
    "historic",
    "natural",
    "man_made",
    "aeroway",
    "railway",
    "public_transport",
    "highway",
    "healthcare",
    "emergency",
    "sport",
    "office",
    "craft",
    "building",
]


# Opening hours. Shared with scripts/run_llmap.py and
# evaluation/constraint_checker.py's opening_hours_met check, so the arrival-time
# feasibility test the generated expand() enforces during search is the same one
# the scorer applies afterwards. The fallbacks below keep a generated body
# runnable if the evaluation package isn't importable (a bare execute, say);
# they are permissive on purpose, so a missing import can never silently turn
# into "every stop is closed".
try:
    from evaluation.opening_hours_hipp import parse_opening_hours, is_open_at
except Exception:  # pragma: no cover - import fallback only
    def parse_opening_hours(osm_str):
        return None

    def is_open_at(rules, clock_sec, day_abbr="Mo"):
        return True


# Dwell time per POI category, in seconds. These are LLMAP's own
# stay_time_mapping values (baselines/LLMAP/main/utils.py); HIPP is LLMAP's
# benchmark, and the clock-deadline check only means anything if generation,
# LLMAP and scoring all charge the same dwell. evaluation/constraint_checker.py
# ::LLMAP_VISIT_SEC holds the same table for the scoring side — keep them in step.
LLMAP_VISIT_SEC: Dict[str, int] = {
    "mall": 7200,
    "shopping_mall": 7200,
    "shopping mall": 7200,
    "supermarket": 1800,
    "pharmacy": 900,
    "bank": 1200,
    "library": 3600,
}
LLMAP_VISIT_DEFAULT_SEC = 900


def visit_seconds_for_poi_type(poi_type: str, visit_time: Any = None) -> float:
    """Dwell seconds to charge for one stop of `poi_type`.

    `visit_time` is the spec's own value, passed through from the query context,
    and takes precedence when given: either a per-category mapping
    ({'pharmacy': 900, ...}) or a single number applied to every category. Only
    when the spec says nothing does this fall back to the LLMAP table above.

    The spec is consulted first so the dwell assumption is a property of the
    query being solved rather than a constant compiled into the solver. On the
    HIPP benchmark the spec builder fills `visit_time` from that same LLMAP
    table, so the fair-comparison numbers are unchanged — but a query that
    states its own dwell ("each stop takes 25 minutes") is now honoured instead
    of silently overridden.
    """
    cat = str(poi_type).strip().lower()
    if isinstance(visit_time, dict):
        if cat in visit_time:
            return float(visit_time[cat])
    elif isinstance(visit_time, (int, float)):
        return float(visit_time)
    return float(LLMAP_VISIT_SEC.get(cat, LLMAP_VISIT_DEFAULT_SEC))


def read_input_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _normalize_node_ids(val: Any) -> List[Any]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    if isinstance(val, int):
        return [val]
    if isinstance(val, float):
        if float(val).is_integer():
            return [int(val)]
        return [val]
    text = str(val).strip()
    if not text:
        return []
    parts = [p.strip() for p in text.split(";") if p.strip()]
    ids: List[Any] = []
    for p in parts:
        if p.isdigit():
            ids.append(int(p))
        else:
            ids.append(p)
    return ids


def node_coords(node_data: Dict) -> Optional[Tuple[float, float]]:
    if node_data.get("lon") is not None and node_data.get("lat") is not None:
        return float(node_data["lon"]), float(node_data["lat"])
    if node_data.get("x") is not None and node_data.get("y") is not None:
        return float(node_data["x"]), float(node_data["y"])
    return None


def min_time_to_dest(G, node, dst, max_speed_kph: float = 130.0) -> float:
    """Optimistic (lower-bound) travel time in SECONDS from `node` to `dst`:
    straight-line great-circle (haversine) distance / fastest road speed.
    Admissible — never larger than the real shortest-path travel time — so it is
    safe to subtract inside an optimistic pruning bound."""
    a = node_coords(G.nodes[node])
    b = node_coords(G.nodes[dst])
    if a is None or b is None:
        return 0.0
    lon1, lat1 = a
    lon2, lat2 = b
    h = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    meters = 6371000.0 * 2 * math.asin(math.sqrt(min(1.0, h)))
    return meters / (max_speed_kph * 1000.0 / 3600.0)


# bound_prune() was removed: the solver now owns the bound test (Algorithm 2,
# lines 16-17) and Expand only attaches the estimate as state.bound (f-hat).


def build_graph_from_gpkg(
    gpkg_path: str,
    nodes_layer: str = "nodes",
    edges_layer: str = "edges",
) -> nx.MultiDiGraph:
    if nx is None:
        raise RuntimeError("networkx is required to load/build the graph")
    if gpd is None:
        raise RuntimeError("geopandas is required to load the gpkg graph data")

    nodes_gdf = gpd.read_file(gpkg_path, layer=nodes_layer)
    edges_gdf = gpd.read_file(gpkg_path, layer=edges_layer)

    G = nx.MultiDiGraph()

    comp_to_canonical: Dict[Any, Any] = {}

    for _, row in nodes_gdf.iterrows():
        node_ids = _normalize_node_ids(row.get("osmid_original", row.get("osmid")))
        if not node_ids:
            continue
        canonical = node_ids[0]
        for nid in node_ids:
            comp_to_canonical[nid] = canonical
            comp_to_canonical[str(nid)] = canonical
        if canonical not in G:
            G.add_node(
                canonical,
                lon=row.get("lon"),
                lat=row.get("lat"),
                x=row.get("x"),
                y=row.get("y"),
                geometry=row.get("geometry"),
            )

    G.graph["comp_to_canonical"] = comp_to_canonical

    for idx, row in edges_gdf.iterrows():
        u = row.get("u_original", row.get("u"))
        v = row.get("v_original", row.get("v"))
        u_ids = _normalize_node_ids(u)
        v_ids = _normalize_node_ids(v)
        if not u_ids or not v_ids:
            continue
        u_c = comp_to_canonical.get(u_ids[0], comp_to_canonical.get(str(u_ids[0])))
        v_c = comp_to_canonical.get(v_ids[0], comp_to_canonical.get(str(v_ids[0])))
        if u_c is None or v_c is None or u_c not in G or v_c not in G:
            continue
        if u_c == v_c:
            continue  
        G.add_edge(
            u_c,
            v_c,
            key=idx,
            length=row.get("length"),
            travel_time=row.get("travel_time"),
            speed_kph=row.get("speed_kph"),
            geometry=row.get("geometry"),
        )

    return G


def parse_lat_lon(raw: str) -> Optional[Tuple[float, float]]:
    """Parse "(lat, lon)" or "lat, lon" into a (lat, lon) float tuple, or None."""
    if raw is None:
        return None
    s = str(raw).strip().strip("()")
    parts = s.split(",")
    if len(parts) != 2:
        return None
    try:
        lat = float(parts[0].strip())
        lon = float(parts[1].strip())
        if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
            return lat, lon
    except ValueError:
        pass
    return None


def nearest_node_to_coords(G: nx.MultiDiGraph, lat: float, lon: float) -> Optional[Any]:
    best_node = None
    best_dist = float("inf")
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    for node, data in G.nodes(data=True):
        coords = node_coords(data)
        if coords is None:
            continue
        nlon, nlat = coords  # node_coords returns (lon, lat)
        dlat = math.radians(nlat) - lat_r
        dlon = math.radians(nlon) - lon_r
        a = math.sin(dlat / 2) ** 2 + math.cos(lat_r) * math.cos(math.radians(nlat)) * math.sin(dlon / 2) ** 2
        d = 2 * math.asin(math.sqrt(min(1.0, a)))
        if d < best_dist:
            best_dist = d
            best_node = node
    return best_node


def coerce_node_id(G: nx.MultiDiGraph, node_raw: str) -> Any:
    if ";" in node_raw:
        node_raw = node_raw.split(";", 1)[0].strip()
    if node_raw in G:
        return node_raw
    try:
        as_int = int(node_raw)
        if as_int in G:
            return as_int
    except ValueError:
        pass
    if str(node_raw) in G:
        return str(node_raw)
    c2c = G.graph.get("comp_to_canonical", {})
    canonical = c2c.get(node_raw) or c2c.get(str(node_raw))
    if canonical is not None and canonical in G:
        return canonical
    return node_raw


def ensure_nodes_in_graph(G: nx.MultiDiGraph, origin: Any, destination: Any) -> None:
    if origin not in G or destination not in G:
        raise ValueError("origin/destination must exist in graph nodes")


def resolve_graph_node_id(G: nx.MultiDiGraph, node_id: Any) -> Optional[Any]:
    if node_id is None or (isinstance(node_id, float) and math.isnan(node_id)):
        return None
    if isinstance(node_id, str) and ";" in node_id:
        node_id = node_id.split(";", 1)[0].strip()
    if isinstance(node_id, float) and float(node_id).is_integer():
        node_id = int(node_id)
    if node_id in G:
        return node_id
    try:
        as_int = int(node_id)
        if as_int in G:
            return as_int
    except (TypeError, ValueError):
        pass
    node_str = str(node_id)
    if node_str in G:
        return node_str
    c2c = G.graph.get("comp_to_canonical", {})
    canonical = c2c.get(node_id) or c2c.get(node_str)
    if canonical is not None and canonical in G:
        return canonical
    return None


def edge_weight(edge_data: Dict, weight_key: str, default_speed_kph: float = 30.0) -> float:
    val = edge_data.get(weight_key)
    if val is not None and not (isinstance(val, float) and math.isnan(val)):
        return float(val)
    if weight_key == "travel_time":
        length = edge_data.get("length")
        speed_kph = edge_data.get("speed_kph") or default_speed_kph
        if length is None or (isinstance(length, float) and math.isnan(length)):
            return 1.0
        speed_mps = max(float(speed_kph), 1e-6) * 1000.0 / 3600.0
        return float(length) / speed_mps
    if weight_key == "length":
        length = edge_data.get("length")
        if length is None or (isinstance(length, float) and math.isnan(length)):
            return 1.0
        return float(length)
    return 1.0


def compute_shortest_path_length(
    G: nx.MultiDiGraph,
    origin: Any,
    destination: Any,
    weight_key: str,
) -> Optional[float]:
    try:
        return float(nx.shortest_path_length(G, origin, destination, weight=weight_key))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def split_multi_value(raw: str) -> List[str]:
    parts = [p.strip() for p in raw.replace("|", ";").replace(",", ";").split(";") if p.strip()]
    return parts


def poi_categories(row: "gpd.GeoSeries", category_cols: List[str]) -> List[str]:
    categories: List[str] = []
    for col in category_cols:
        if col not in row or row[col] is None:
            continue
        for value in split_multi_value(str(row[col]).lower()):
            if value not in categories:
                categories.append(value)
    return categories


def build_poi_type_nodes(
    G: nx.MultiDiGraph,
    pois_gdf: "gpd.GeoDataFrame",
    poi_types: List[str],
    nearest_node_col: str = "nearest_node_osmid",
    category_cols: Optional[List[str]] = None,
) -> Dict[str, Set[Any]]:
    target_types = [t.strip().lower() for t in poi_types]
    type_nodes: Dict[str, Set[Any]] = {t: set() for t in target_types}
    if nearest_node_col not in pois_gdf.columns:
        return type_nodes
    for _, row in pois_gdf.iterrows():
        node_id = resolve_graph_node_id(G, row.get(nearest_node_col))
        if node_id is None:
            continue
        for cat in poi_categories(row, category_cols or CATEGORY_COLS):
            cat = cat.strip().lower()
            if cat in type_nodes:
                type_nodes[cat].add(node_id)
    return type_nodes


# output / evaluation helpers
def save_route_gpkg(
    out_path: str,
    G: nx.MultiDiGraph,
    path: List[Any],
    nodes_gdf: "gpd.GeoDataFrame",
    *,
    states: Optional[List[Any]] = None,
    pois_gdf: Optional["gpd.GeoDataFrame"] = None,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    from shapely.geometry import LineString, Point

    ctx = context or {}
    crs = nodes_gdf.crs
    poi_types = ctx.get("poi_types", [])
    visit_time_s = ctx.get("visit_time_s", 0.0)
    charger_nodes = ctx.get("charger_nodes", set())

    route_lines = []
    for u, v in zip(path[:-1], path[1:]):
        edge_data = list(G.get_edge_data(u, v).values())[0] if G.has_edge(u, v) else {}
        geom = edge_data.get("geometry")
        if geom is None:
            u_d, v_d = G.nodes[u], G.nodes[v]
            uc = (float(u_d.get("x", u_d.get("lon", 0))), float(u_d.get("y", u_d.get("lat", 0))))
            vc = (float(v_d.get("x", v_d.get("lon", 0))), float(v_d.get("y", v_d.get("lat", 0))))
            geom = LineString([uc, vc])
        route_lines.append({"u": u, "v": v, "geometry": geom})
    gpd.GeoDataFrame(route_lines, geometry="geometry", crs=crs).to_file(out_path, layer="route", driver="GPKG")

    def node_point(nid: Any) -> Point:
        d = G.nodes[nid]
        return Point(float(d.get("x", d.get("lon", 0))), float(d.get("y", d.get("lat", 0))))

    gpd.GeoDataFrame(
        [{"node_id": path[0], "label": "start", "geometry": node_point(path[0])}],
        geometry="geometry", crs=crs,
    ).to_file(out_path, layer="route_start", driver="GPKG")
    gpd.GeoDataFrame(
        [{"node_id": path[-1], "label": "destination", "geometry": node_point(path[-1])}],
        geometry="geometry", crs=crs,
    ).to_file(out_path, layer="route_end", driver="GPKG")

    if states is not None and poi_types:
        def get_poi_idx(s: Any) -> int:
            if hasattr(s, "constraints") and isinstance(s.constraints, dict):
                return int(s.constraints.get("poi_idx", 0))
            return int(getattr(s, "poi_idx", 0))

        poi_stops = []
        prev_idx = 0
        for s in states:
            if get_poi_idx(s) > prev_idx:
                poi_stops.append(
                    {
                        "node_id": s.node,
                        "poi_type": poi_types[prev_idx],
                        "poi_idx": prev_idx,
                        "time_min": round((float(s.objective) - visit_time_s) / 60, 1),
                        "geometry": node_point(s.node),
                    }
                )
                prev_idx = get_poi_idx(s)
        if poi_stops:
            gpd.GeoDataFrame(poi_stops, geometry="geometry", crs=crs).to_file(
                out_path, layer="poi_stops", driver="GPKG"
            )

    if states is not None and charger_nodes:
        charger_stops = [{"node_id": n, "geometry": node_point(n)} for n in path if n in charger_nodes]
        if charger_stops:
            gpd.GeoDataFrame(charger_stops, geometry="geometry", crs=crs).to_file(
                out_path, layer="charger_stops", driver="GPKG"
            )

    if states is not None and pois_gdf is not None and poi_types and "type" in pois_gdf.columns:
        ctx_pois = pois_gdf[pois_gdf["type"].isin(poi_types)].copy()
        if not ctx_pois.empty:
            ctx_pois.to_file(out_path, layer="poi_context", driver="GPKG")


