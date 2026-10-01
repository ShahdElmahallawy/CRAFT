"""
result.py — Algorithm output writer.

Saves the minimal output.json produced by every generated algorithm:
  {query_id, route, run_time_ms}

route is null when no feasible path was found.

After execute, call enrich_output() to add poi_stops, objective, weight_key.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def save_output(
    query_id: str,
    route: Optional[List[Any]],
    run_time_ms: float,
    out_dir: Any,
    collected_pois: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Write output.json to out_dir.

    collected_pois (maximize_poi only): the (node, category) pairs the search
    actually collected along the winning path — see template.py::_derive_collected_pois.
    """
    output = {
        "query_id": query_id,
        "route": [str(n) for n in route] if route is not None else None,
        "run_time_ms": round(run_time_ms, 2),
    }
    if collected_pois is not None:
        output["collected_pois"] = collected_pois
    out_path = Path(out_dir) / "output.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")


_CAT_COLS = ["amenity", "shop", "leisure", "tourism", "highway", "power"]

_SKIP_TYPES = {
    "bus_stop", "bus_station", "tram_stop", "subway_entrance",
    "parking", "parking_space", "parking_entrance",
    "power", "tower", "pole", "generator",
    "highway", "street_lamp", "traffic_signals",
}


def _extract_poi_stops(path: List[str], G, pois_gdf) -> List[Dict]:
    """
    Extract visited POIs from a route (list of string node IDs) in visit order.
    Each POI node is included at most once (first occurrence).
    Transit/infrastructure types (bus_stop, parking, etc.) are excluded.
    Coordinates are returned in WGS84 (EPSG:4326).
    """
    if pois_gdf is None or "nearest_node_osmid" not in pois_gdf.columns:
        return []
    try:
        if pois_gdf.crs is not None and not pois_gdf.crs.equals("EPSG:4326"):
            pois_wgs = pois_gdf.to_crs("EPSG:4326")
        else:
            pois_wgs = pois_gdf
    except Exception:
        pois_wgs = pois_gdf

    poi_lookup: Dict[str, List] = {}
    for idx, row in pois_wgs.iterrows():
        nid = row.get("nearest_node_osmid")
        if nid is not None and str(nid) != "nan":
            poi_lookup.setdefault(str(nid), []).append(row)

    stops = []
    seen: set = set()
    for node in path:
        node_str = str(node)
        if node_str in poi_lookup and node_str not in seen:
            seen.add(node_str)
            row = poi_lookup[node_str][0]
            poi_type = None
            for col in _CAT_COLS:
                val = row.get(col)
                if val and str(val) != "nan":
                    poi_type = str(val)
                    break
            if poi_type in _SKIP_TYPES:
                continue
            name_raw = row.get("name")
            name = name_raw if name_raw and str(name_raw) != "nan" else None
            if name is None and poi_type is None:
                continue
            lat, lng = None, None
            geom = getattr(row, "geometry", None)
            if geom is not None and not geom.is_empty:
                pt = geom if geom.geom_type == "Point" else geom.centroid
                lat = round(float(pt.y), 7)
                lng = round(float(pt.x), 7)
            stops.append({
                "name": name,
                "type": poi_type or "unknown",
                "node_id": node_str,
                "lat": lat,
                "lng": lng,
            })
    return stops


def _compute_route_objective(path: List[str], G, weight_key: str) -> float:
    """Sum edge weights along the route. Uses min-key edge for multi-graphs."""
    total = 0.0
    for u_str, v_str in zip(path[:-1], path[1:]):
        for u in (u_str, _try_int(u_str)):
            for v in (v_str, _try_int(v_str)):
                data = G.get_edge_data(u, v)
                if data:
                    edge_data = data[min(data)]
                    total += _edge_cost(edge_data, weight_key)
                    goto_next = True
                    break
            else:
                continue
            break
    return round(total, 2)


def _try_int(s: str) -> Any:
    try:
        return int(s)
    except (ValueError, TypeError):
        return s


def _edge_cost(edge_data: Dict, weight_key: str) -> float:
    val = edge_data.get(weight_key)
    if val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            pass
    if weight_key == "travel_time":
        length = edge_data.get("length", 0.0)
        speed_kph = edge_data.get("speed_kph", 30.0) or 30.0
        speed_mps = max(float(speed_kph), 1e-6) * 1000.0 / 3600.0
        return float(length or 0.0) / speed_mps
    if weight_key == "length":
        return float(edge_data.get("length", 0.0) or 0.0)
    return 0.0

def enrich_output(out_dir: Any, G, pois_gdf, weight_key: str) -> None:
    """
    Read output.json from out_dir, add poi_stops / objective / weight_key, write back.
    No-op if output.json does not exist or route is None.
    """
    out_path = Path(out_dir) / "output.json"
    if not out_path.exists():
        return
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        return

    route = data.get("route")
    if not route:
        data.setdefault("poi_stops", [])
        data.setdefault("objective", None)
        data.setdefault("weight_key", weight_key)
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return

    data["poi_stops"] = _extract_poi_stops(route, G, pois_gdf)
    data["objective"] = _compute_route_objective(route, G, weight_key)
    data["weight_key"] = weight_key
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_output(path: Any) -> dict:
    """Load output.json from path."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_result(result_doc: dict, out_path: Any) -> None:
    """Write a generator result/summary JSON to out_path."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result_doc, indent=2, ensure_ascii=False), encoding="utf-8")


def save_route_visualization(
    out_path: Any,
    G,
    route: List[Any],
    context: dict,
    pois_gdf=None,
) -> None:
    """
    Save route + stops to a GeoPackage for visualization.

    Layers written:
      route          — LineString segments for the full path
      route_waypoints — start, destination, POI stops, charger stops as Points
      poi_candidates  — all candidate POIs (possible_targets) near the route (optional)

    POI stops are detected by checking which route nodes are in context['possible_targets'].
    Charger stops are detected by checking context['charger_nodes'].
    No dependency on State internals.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import LineString, Point
    except ImportError:
        print("[save_visualization] geopandas/shapely not available — skipping GPKG output.")
        return

    out_path = Path(out_path)
    crs = "EPSG:4326"

    def node_point(nid):
        d = G.nodes[nid]
        x = float(d.get("x", d.get("lon", 0)))
        y = float(d.get("y", d.get("lat", 0)))
        return Point(x, y)

    lines = []
    for u, v in zip(route[:-1], route[1:]):
        edge_data = (list(G.get_edge_data(u, v).values())[0]
                     if G.has_edge(u, v) else {})
        geom = edge_data.get("geometry")
        if geom is None:
            geom = LineString([
                (G.nodes[u].get("x", 0), G.nodes[u].get("y", 0)),
                (G.nodes[v].get("x", 0), G.nodes[v].get("y", 0)),
            ])
        lines.append({"u": u, "v": v, "geometry": geom})
    gpd.GeoDataFrame(lines, geometry="geometry", crs=crs).to_file(
        out_path, layer="route", driver="GPKG"
    )

    poi_types = context.get("poi_types", [])
    possible_targets = context.get("possible_targets", {})
    charger_nodes = context.get("charger_nodes", set())

    waypoints = []
    waypoints.append({
        "node_id": route[0], "label": "start", "stop_type": "origin",
        "poi_type": None, "geometry": node_point(route[0]),
    })

    poi_idx = 0
    charger_visited = set()
    for node in route[1:]:
        if poi_idx < len(poi_types) and node in possible_targets.get(poi_idx, set()):
            waypoints.append({
                "node_id": node,
                "label": f"stop_{poi_idx}_{poi_types[poi_idx]}",
                "stop_type": "poi",
                "poi_type": poi_types[poi_idx],
                "geometry": node_point(node),
            })
            poi_idx += 1
        elif node in charger_nodes and node not in charger_visited:
            waypoints.append({
                "node_id": node, "label": "charger",
                "stop_type": "charger", "poi_type": None,
                "geometry": node_point(node),
            })
            charger_visited.add(node)

    waypoints.append({
        "node_id": route[-1], "label": "destination", "stop_type": "destination",
        "poi_type": None, "geometry": node_point(route[-1]),
    })

    gpd.GeoDataFrame(waypoints, geometry="geometry", crs=crs).to_file(
        out_path, layer="route_waypoints", driver="GPKG"
    )

    if pois_gdf is not None and possible_targets and "nearest_node_osmid" in pois_gdf.columns:
        all_candidate_nodes = set()
        for nodes in possible_targets.values():
            all_candidate_nodes.update(str(n) for n in nodes)
        cand = pois_gdf[pois_gdf["nearest_node_osmid"].astype(str).isin(all_candidate_nodes)].copy()
        if not cand.empty and "geometry" in cand.columns:
            cand = cand[cand.geometry.notna()]
            if not cand.empty:
                cand.to_file(out_path, layer="poi_candidates", driver="GPKG")

    print(f"[visualization] Saved to {out_path}"
          f" — {len(waypoints)} waypoints, {len(lines)} route segments")
