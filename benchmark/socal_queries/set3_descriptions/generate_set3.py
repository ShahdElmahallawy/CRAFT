#!/usr/bin/env python3
"""
Generates Set 3: 200 ordered_poi queries, each filtering at least one stop by
a real POI attribute (cuisine, wheelchair, ...) or, for a few, a free-text
description (museum/attraction). A filter is only used in a city if the gpkg
actually has >= MIN_CANDIDATES matching POIs there.

Outputs: ordered+desc.json (query_id/nl_query, parser-facing), set3_meta.json
(ground truth: stop filters, src/dst, distance limits, difficulty).

Usage:
  python benchmark/socal_queries/set3_descriptions/generate_set3.py
"""

from __future__ import annotations

import json
import math
import random
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# set1-style connectivity gate: verify a src->dst path exists on the real road
# graph, same check generate_set1.py applies. Loaded lazily in main() so importing
# this module (e.g. for its filter vocab) doesn't force a 248k-node graph build.
from craft.generator.tools.common_handlers import (  # noqa: E402
    build_graph_from_gpkg,
    coerce_node_id,
    compute_shortest_path_length,
)
SET3_DIR = ROOT / "benchmark" / "socal_queries" / "set3_descriptions"
NODES_FILE = ROOT / "data" / "socal" / "catalog_nodes.json"
CITIES_FILE = ROOT / "data" / "socal" / "cities.json"
GPKG_FILE = ROOT / "data" / "socal" / "road_network.gpkg"

MI_TO_KM = 1.60934
TOTAL_QUERIES = 200
MIN_CANDIDATES = 2  # min real in-city POIs matching a filter to use it there
RNG = random.Random(42)

# Mirrors craft/generator/tools/common_handlers.py::CATEGORY_COLS
CATEGORY_COLS = [
    "amenity", "tourism", "leisure", "shop", "historic", "natural", "man_made",
    "aeroway", "railway", "public_transport", "highway", "healthcare",
    "emergency", "sport", "office", "craft", "building",
]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(min(1.0, a)))


_node_cache = {}


def get_city_nodes(city_name, cities, nodes):
    if city_name not in _node_cache:
        city = cities[city_name]
        _node_cache[city_name] = [
            n for n in nodes
            if haversine_km(city["lat"], city["lon"], n["lat"], n["lon"]) <= city["radius_km"]
        ]
    return _node_cache[city_name]


def pick_src_dst(city_name, cities, nodes, min_km=4.0, max_km=28.0, seed=0,
                 is_connected=None):
    """Return the first distance-band-valid (src, dst) pair. When is_connected is
    given (set1-style gate), the pair must also have a src->dst path on the road
    graph. is_connected is a pure graph lookup — it consumes no RNG, so passing it
    never perturbs the deterministic pick order; it only skips disconnected pairs."""
    city_nodes = get_city_nodes(city_name, cities, nodes)
    n = len(city_nodes)
    if n < 2:
        raise ValueError(f"{city_name} has only {n} catalog nodes")
    for i in range(n):
        src = city_nodes[(seed + i) % n]
        for step in range(1, min(50, n)):
            dst = city_nodes[(seed + i + step * 17) % n]
            dist = haversine_km(src["lat"], src["lon"], dst["lat"], dst["lon"])
            if min_km <= dist <= max_km:
                if is_connected is not None and not is_connected(src, dst):
                    continue
                return src, dst
    raise ValueError(f"No valid src/dst pair for {city_name} seed={seed}")


def stable_hash(text):
    """Deterministic string hash — Python's builtin hash() is salted per
    process (PYTHONHASHSEED) and is NOT safe for reproducible seeding."""
    return zlib.crc32(text.encode("utf-8"))


def coord(node):
    return [node["lat"], node["lon"]]


def coord_str(node):
    return f"({node['lat']}, {node['lon']})"


# Filter vocabulary — grounded in real gpkg column coverage. Structured filters
# are exact substring matches on real columns; description filters are
# keyword-in-description checks (a proxy for the embedding match
# craft/generator/tools/filtration.py performs).

def _article(word):
    return "an" if word[0].lower() in "aeiou" else "a"


RESTAURANT_CUISINES = [
    ("mexican", "Mexican"), ("italian", "Italian"), ("chinese", "Chinese"),
    ("japanese", "Japanese"), ("thai", "Thai"), ("vietnamese", "Vietnamese"),
    ("korean", "Korean"), ("mediterranean", "Mediterranean"),
    ("seafood", "seafood"), ("sushi", "sushi"), ("pizza", "pizza"),
    ("american", "American"),
]

STRUCTURED_FILTERS = []
for _val, _disp in RESTAURANT_CUISINES:
    STRUCTURED_FILTERS.append({
        "id": f"restaurant_cuisine_{_val}", "poi_type": "restaurant",
        "column": "cuisine", "value": _val,
        "nl_phrase": f"{_article(_disp)} {_disp} restaurant",
    })
STRUCTURED_FILTERS += [
    {"id": "restaurant_outdoor_seating", "poi_type": "restaurant",
     "column": "outdoor_seating", "value": "yes",
     "nl_phrase": "a restaurant with outdoor seating"},
    {"id": "restaurant_wheelchair", "poi_type": "restaurant",
     "column": "wheelchair", "value": "yes",
     "nl_phrase": "a wheelchair-accessible restaurant"},
    {"id": "restaurant_takeaway", "poi_type": "restaurant",
     "column": "takeaway", "value": "yes",
     "nl_phrase": "a restaurant that offers takeout"},
    {"id": "cafe_bubble_tea", "poi_type": "cafe",
     "column": "cuisine", "value": "bubble_tea",
     "nl_phrase": "a bubble tea cafe"},
    {"id": "cafe_wifi", "poi_type": "cafe",
     "column": "internet_access", "value": "wlan",
     "nl_phrase": "a cafe with Wi-Fi"},
    {"id": "cafe_takeaway", "poi_type": "cafe",
     "column": "takeaway", "value": "yes",
     "nl_phrase": "a cafe that offers takeout"},
    {"id": "cafe_outdoor_seating", "poi_type": "cafe",
     "column": "outdoor_seating", "value": "yes",
     "nl_phrase": "a cafe with outdoor seating"},
]

# description filters: keyword-grounded per-city — museum history/art, attraction theme-park cluster
DESCRIPTION_FILTERS = [
    {"id": "museum_history", "poi_type": "museum", "keywords": ["history"],
     "nl_phrase": "a history museum"},
    {"id": "museum_art", "poi_type": "museum", "keywords": ["art"],
     "nl_phrase": "an art museum"},
    {"id": "attraction_theme_park", "poi_type": "attraction",
     "keywords": ["theme", "amusement", "disney", "ride"],
     "nl_phrase": "a theme park attraction"},
]

PLAIN_TYPES = {
    "restaurant": "a restaurant",
    "cafe": "a cafe",
    "park": "a park",
}

ALL_POI_TYPES_USED = ["restaurant", "cafe", "park", "museum", "attraction"]

# NL skeleton phrasings — mirrors templates/ordered_poi.txt (ordered stops + per-leg
# distance limits), but text is generated fresh per query rather than templated

TWO_STOP_SKELETONS = [
    "I'm at {SRC} and need to get to {DST}. On the way, I want to stop at {STOP1} within {D1} miles of me, then {STOP2} about {D2} miles after that, in that order. What's the shortest-distance route?",
    "I'm at {SRC} and need to get to {DST}. I want to stop at {STOP1} first, no more than {D1} miles away, and then {STOP2} about {D2} miles after that, in that order. Please give me the shortest route.",
    "I need to get from {SRC} to {DST} with two stops: first {STOP1} within {D1} miles, then {STOP2} about {D2} miles later, in that order. What's the shortest way to get there?",
    "Can you plan a route from {SRC} to {DST}? I need to stop at {STOP1} first within {D1} miles, then {STOP2} within another {D2} miles, in that order. Please give me the shortest route.",
    "I'm planning a trip from {SRC} to {DST}. I need to visit {STOP1} first within {D1} miles, then {STOP2} about {D2} miles later. Both stops are required and have to be in order. What's the shortest-distance route?",
    "Starting from {SRC}, heading to {DST}. I need to stop at {STOP1} first within {D1} miles, then at {STOP2} about {D2} miles later, in that order. What's the shortest-distance route?",
]

THREE_STOP_SKELETONS = [
    "I need to get from {SRC} to {DST} with three stops in order: first {STOP1} within {D1} miles, then {STOP2} within {D2} miles after that, and then {STOP3} within {D3} miles after that. What's the shortest-distance route?",
    "I'm planning a three-stop trip from {SRC} to {DST}. I need to stop at {STOP1} first, then {STOP2}, then {STOP3} in that exact order. The leg limits are {D1}, {D2}, and {D3} miles respectively. Can you keep the route as short as possible?",
]

# difficulty -> per-leg mile budget (2:3:2 easy:medium:hard)
DIFFICULTY_MILES = {
    "easy": [7, 6, 5],
    "medium": [5, 5, 4],
    "hard": [4, 4, 3],
}
DIFFICULTY_CYCLE = ["easy", "easy", "medium", "medium", "medium", "hard", "hard"]


# feasibility: real gpkg-backed candidate counts per (city, filter)

def load_pois():
    gdf = gpd.read_file(GPKG_FILE, layer="pois")
    gdf["_cat"] = gdf[CATEGORY_COLS].bfill(axis=1).iloc[:, 0]
    return gdf


def build_feasibility(gdf, cities):
    """Return {city_name: {filter_id: candidate_count}} for all structured +
    description filters, and {city_name: {poi_type: plain_count}}."""
    lats = gdf["lat"].to_numpy()
    lons = gdf["lon"].to_numpy()
    cats = gdf["_cat"].to_numpy()

    struct_cols = {f["column"] for f in STRUCTURED_FILTERS}
    col_lower = {c: gdf[c].astype(str).str.lower() for c in struct_cols}
    desc_lower = gdf["description"].astype(str).str.lower()

    feas = {}
    plain = {}
    for c in cities:
        d = pd.Series(
            [haversine_km(c["lat"], c["lon"], la, lo) for la, lo in zip(lats, lons)]
        )
        within = (d.to_numpy() <= c["radius_km"])

        city_feas = {}
        for f in STRUCTURED_FILTERS:
            base = within & (cats == f["poi_type"])
            match = col_lower[f["column"]].str.contains(f["value"], na=False, regex=False).to_numpy()
            city_feas[f["id"]] = int((base & match).sum())
        for f in DESCRIPTION_FILTERS:
            base = within & (cats == f["poi_type"])
            kw_match = pd.Series(False, index=gdf.index)
            for kw in f["keywords"]:
                kw_match = kw_match | desc_lower.str.contains(kw, na=False, regex=False)
            city_feas[f["id"]] = int((base & kw_match.to_numpy()).sum())
        feas[c["name"]] = city_feas

        city_plain = {}
        for pt in PLAIN_TYPES:
            city_plain[pt] = int((within & (cats == pt)).sum())
        plain[c["name"]] = city_plain

    return feas, plain


# query assembly

def dist_str(v):
    return str(v)


def build_nl(skeleton, src, dst, stop_phrases, dists):
    text = skeleton.replace("{SRC}", coord_str(src)).replace("{DST}", coord_str(dst))
    for i, phrase in enumerate(stop_phrases):
        text = text.replace(f"{{STOP{i+1}}}", phrase)
    for i, d in enumerate(dists):
        text = text.replace(f"{{D{i+1}}}", dist_str(d))
    return text


def other_type(used_types):
    for t in ("restaurant", "cafe", "park"):
        if t not in used_types:
            return t
    return "park"


def assign_targets(cities):
    """Distribute 200 queries across 31 cities: 6 each + remainder to the
    richest cities (by combined restaurant+cafe count)."""
    base = TOTAL_QUERIES // len(cities)
    remainder = TOTAL_QUERIES - base * len(cities)
    order = sorted(cities, key=lambda c: -(c.get("_richness", 0)))
    targets = {c["name"]: base for c in cities}
    for c in order[:remainder]:
        targets[c["name"]] += 1
    return targets


def main():
    cities_list = json.loads(CITIES_FILE.read_text())
    cities = {c["name"]: c for c in cities_list}
    nodes = json.loads(NODES_FILE.read_text())

    print("Loading pois layer from gpkg ...")
    gdf = load_pois()
    print(f"  {len(gdf)} POIs loaded")

    # set1-style connectivity gate: build the routable graph once and reject any
    # src/dst pair with no path between them (same as generate_set1.py). Cached
    # per (node_id, node_id) so the repeated lookups in pick_src_dst stay cheap.
    print("Building road graph for connectivity gate ...")
    G = build_graph_from_gpkg(str(GPKG_FILE))
    _conn_cache: dict = {}

    def is_connected(src, dst) -> bool:
        key = (src["node_id"], dst["node_id"])
        cached = _conn_cache.get(key)
        if cached is not None:
            return cached
        s = coerce_node_id(G, str(src["node_id"]))
        d = coerce_node_id(G, str(dst["node_id"]))
        ok = (s is not None and d is not None and s != d
              and compute_shortest_path_length(G, s, d, "length") is not None)
        _conn_cache[key] = ok
        return ok

    feas, plain = build_feasibility(gdf, cities_list)
    for c in cities_list:
        c["_richness"] = plain[c["name"]]["restaurant"] + plain[c["name"]]["cafe"]

    targets = assign_targets(cities_list)
    print("Per-city query targets:", targets)

    records = []
    meta = []
    qi = 0

    for c in cities_list:
        name = c["name"]
        n_target = targets[name]
        city_feas = feas[name]

        feasible_structured = [
            f for f in STRUCTURED_FILTERS if city_feas[f["id"]] >= MIN_CANDIDATES
        ]
        feasible_desc = [
            f for f in DESCRIPTION_FILTERS if city_feas[f["id"]] >= MIN_CANDIDATES
        ]
        if len(feasible_structured) < 2:
            raise RuntimeError(f"{name}: not enough feasible structured filters ({len(feasible_structured)})")

        struct_cycle = list(feasible_structured)
        RNG.shuffle(struct_cycle)
        desc_cycle = list(feasible_desc)
        RNG.shuffle(desc_cycle)

        struct_i = 0
        desc_i = 0
        n_desc_used = 0
        max_desc_this_city = min(2, len(desc_cycle))

        for local_i in range(n_target):
            seed = 1000 * (stable_hash(name) % 97) + local_i * 13 + 7
            src, dst = pick_src_dst(name, cities, nodes, seed=seed,
                                    is_connected=is_connected)

            is_three_stop = (local_i % 7 == 6) and n_target >= 7  # ~1 in 7 queries
            difficulty = DIFFICULTY_CYCLE[local_i % len(DIFFICULTY_CYCLE)]
            miles = DIFFICULTY_MILES[difficulty]

            # Pick primary filtered stop (structured, round-robin)
            f1 = struct_cycle[struct_i % len(struct_cycle)]
            struct_i += 1

            use_desc_second = (
                desc_cycle and n_desc_used < max_desc_this_city and local_i in (2, 5)
            )

            stops = []  # list of dicts: poi_type, filter_kind, phrase, meta
            stops.append({
                "poi_type": f1["poi_type"], "filter_kind": "structured",
                "column": f1["column"], "value": f1["value"],
                "nl_phrase": f1["nl_phrase"],
                "candidate_count_in_city": city_feas[f1["id"]],
            })
            used_types = {f1["poi_type"]}

            if use_desc_second:
                f2 = desc_cycle[desc_i % len(desc_cycle)]
                desc_i += 1
                n_desc_used += 1
                stops.append({
                    "poi_type": f2["poi_type"], "filter_kind": "description",
                    "description": f2["nl_phrase"], "keywords": f2["keywords"],
                    "nl_phrase": f2["nl_phrase"],
                    "candidate_count_in_city": city_feas[f2["id"]],
                })
                used_types.add(f2["poi_type"])
            else:
                # second stop: another structured filter of a different type,
                # else fall back to a plain (unfiltered) companion type
                candidates = [f for f in feasible_structured if f["poi_type"] not in used_types]
                if candidates and local_i % 2 == 0:
                    f2 = candidates[struct_i % len(candidates)]
                    stops.append({
                        "poi_type": f2["poi_type"], "filter_kind": "structured",
                        "column": f2["column"], "value": f2["value"],
                        "nl_phrase": f2["nl_phrase"],
                        "candidate_count_in_city": city_feas[f2["id"]],
                    })
                    used_types.add(f2["poi_type"])
                else:
                    pt2 = other_type(used_types)
                    stops.append({
                        "poi_type": pt2, "filter_kind": "none",
                        "nl_phrase": PLAIN_TYPES[pt2],
                        "candidate_count_in_city": plain[name][pt2],
                    })
                    used_types.add(pt2)

            if is_three_stop:
                pt3 = other_type(used_types)
                stops.append({
                    "poi_type": pt3, "filter_kind": "none",
                    "nl_phrase": PLAIN_TYPES[pt3],
                    "candidate_count_in_city": plain[name][pt3],
                })

            # occasionally swap stop order for phrasing variety
            if local_i % 3 == 1 and len(stops) == 2:
                stops = [stops[1], stops[0]]

            n_stops = len(stops)
            dists = miles[:n_stops]
            phrases = [st["nl_phrase"] for st in stops]

            if n_stops == 3:
                skeleton = THREE_STOP_SKELETONS[local_i % len(THREE_STOP_SKELETONS)]
            else:
                skeleton = TWO_STOP_SKELETONS[local_i % len(TWO_STOP_SKELETONS)]

            nl_query = build_nl(skeleton, src, dst, phrases, dists)

            qi += 1
            query_id = f"set3_ordered+desc_{qi:03d}"
            records.append({"query_id": query_id, "nl_query": nl_query})

            poi_filters = []
            for st in stops:
                if st["filter_kind"] == "structured":
                    poi_filters.append({st["column"]: st["value"]})
                elif st["filter_kind"] == "description":
                    poi_filters.append({"description": st["description"]})
                else:
                    poi_filters.append(None)

            meta.append({
                "query_id": query_id,
                "city": name,
                "src": coord(src),
                "dst": coord(dst),
                "distance_limits_mi": dists,
                "difficulty": difficulty,
                "poi_types": [st["poi_type"] for st in stops],
                "stops": stops,
                "poi_filters": poi_filters,
            })

    assert qi == TOTAL_QUERIES, f"expected {TOTAL_QUERIES}, got {qi}"

    (SET3_DIR / "ordered+desc.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (SET3_DIR / "set3_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    filter_kind_counts = Counter()
    poi_type_counts = Counter()
    n_three_stop = 0
    for m in meta:
        n_three_stop += 1 if len(m["stops"]) == 3 else 0
        for st in m["stops"]:
            filter_kind_counts[st["filter_kind"]] += 1
            poi_type_counts[st["poi_type"]] += 1

    summary = {
        "total": len(records),
        "by_template": {"ordered_poi": len(records)},
        "three_stop_queries": n_three_stop,
        "two_stop_queries": len(records) - n_three_stop,
        "stop_filter_kind_counts": dict(filter_kind_counts),
        "stop_poi_type_counts": dict(poi_type_counts),
        "record_shape": ["query_id", "nl_query"],
        "output_file": "ordered+desc.json",
        "meta_file": "set3_meta.json",
    }
    (SET3_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nWrote {len(records)} records -> ordered+desc.json")
    print(f"Wrote {len(meta)} meta entries -> set3_meta.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
