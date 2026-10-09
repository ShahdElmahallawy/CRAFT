"""
Generates the poi+road_pref benchmark queries.

Same convention as generate_set1.py: numbered NL template variants in
benchmark/templates/poi+road_pref.txt, literal POI type names baked into each
variant's text (unordered, no "in that order" language), filling in just
{SRC_NODE}/{DST_NODE} per variation. Every stop is a plain required POI type —
no either/or groups — so this template isolates one axis: an always-on
"avoid highways" road-type preference (an edge-level constraint).

Only builds query text (query_id, nl_query) — doesn't touch the parser,
generator, or any system code.

Usage:
  conda run -n myenv python benchmark/socal_queries/poi_road_pref/generate_poi_road_pref.py

Output:
  benchmark/socal_queries/poi_road_pref/poi_road_pref.json
  benchmark/socal_queries/poi_road_pref/summary.json
"""

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from craft.generator.tools.common_handlers import (
    build_graph_from_gpkg,
    coerce_node_id,
    compute_shortest_path_length,
)

DATASET       = "socal"
TEMPLATE_PATH = ROOT / "benchmark" / "templates" / "poi+road_pref.txt"
QUERIES_DIR   = ROOT / "benchmark" / "socal_queries" / "poi_road_pref"

SEED           = 42
N_VARIATIONS   = 5   # 20 templates x 5 = 100
SRC_DST_MIN_KM = 4.0
SRC_DST_MAX_KM = 28.0
CITY_MIN_NODES = 20

POI_TYPES = ["pharmacy", "bank", "supermarket", "library"]


# Catalog + city loading

def load_catalogs(dataset: str):
    d = ROOT / "data" / dataset
    pois  = json.loads((d / "catalog_pois.json").read_text())
    nodes = json.loads((d / "catalog_nodes.json").read_text())
    return pois, nodes


def load_cities(dataset: str) -> List[Dict]:
    path = ROOT / "data" / dataset / "cities.json"
    return json.loads(path.read_text())


# Haversine

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(min(1.0, a)))


# City filtering

def filter_nodes_to_city(nodes: List[Dict], city: Dict) -> List[Dict]:
    clat, clon, r = city["lat"], city["lon"], city["radius_km"]
    return [n for n in nodes
            if n.get("lat") is not None and n.get("lon") is not None
            and haversine_km(clat, clon, n["lat"], n["lon"]) <= r]


def filter_pois_to_city(pois: Dict, city: Dict) -> Dict:
    clat, clon, r = city["lat"], city["lon"], city["radius_km"]
    return {
        cat: [p for p in items
              if p.get("lat") is not None and p.get("lon") is not None
              and haversine_km(clat, clon, p["lat"], p["lon"]) <= r]
        for cat, items in pois.items()
    }


def city_has_required_pois(city_pois: Dict, poi_types: List[str]) -> bool:
    return all(len(city_pois.get(t, [])) >= 1 for t in poi_types)


# Template parsing

def parse_template_file(path: Path) -> List[Tuple[str, str]]:
    templates = []
    stem = path.stem
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = re.match(r'^(\d+):\s+(.*)', stripped)
        if m:
            num, text = m.group(1), m.group(2).strip()
            if text:
                templates.append((f"{stem}_{num}", text))
    return templates


def fill_template(template: str, values: Dict[str, str]) -> str:
    result = template
    for key, val in values.items():
        result = result.replace("{" + key + "}", str(val))
    return result


def extract_poi_types(text: str) -> List[str]:
    """Return required POI types mentioned in template text, in first-appearance order."""
    lower = text.lower()
    hits: List[Tuple[int, str]] = []
    for t in POI_TYPES:
        for m in re.finditer(r'\b' + t + r's?\b', lower):
            hits.append((m.start(), t))
    hits.sort(key=lambda x: x[0])
    seen: set = set()
    result: List[str] = []
    for _, t in hits:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


# Smart src/dst

def smart_pick_pair(candidates_src, candidates_dst, rng,
                     min_km: float = SRC_DST_MIN_KM,
                     max_km: float = SRC_DST_MAX_KM,
                     max_tries: int = 80):
    valid_src = [c for c in candidates_src if c.get("lat") is not None]
    valid_dst = [c for c in candidates_dst if c.get("lat") is not None]
    if not valid_src or not valid_dst:
        return (candidates_src[int(rng.integers(len(candidates_src)))],
                candidates_dst[int(rng.integers(len(candidates_dst)))])

    for _ in range(max_tries):
        s = valid_src[int(rng.integers(len(valid_src)))]
        good = [d for d in valid_dst
                if d is not s
                and min_km <= haversine_km(s["lat"], s["lon"], d["lat"], d["lon"]) <= max_km]
        if good:
            return s, good[int(rng.integers(len(good)))]

    target_km = (min_km + max_km) / 2.0
    s = valid_src[int(rng.integers(len(valid_src)))]
    scored = sorted(
        [(abs(haversine_km(s["lat"], s["lon"], d["lat"], d["lon"]) - target_km), d)
         for d in valid_dst if d is not s]
    )
    dst = scored[0][1] if scored else valid_dst[int(rng.integers(len(valid_dst)))]
    return s, dst


# Per-template generation

def generate(templates, poi_catalog, node_catalog, G, rng, cities, n_variations):
    queries: List[Dict[str, Any]] = []

    total_slots = len(templates) * n_variations
    city_list = list(cities)
    rng.shuffle(city_list)
    city_seq = [city_list[i % len(city_list)] for i in range(total_slots)]
    slot_idx = 0
    global_counter = 0

    for t_idx, (tmpl_id, tmpl_text) in enumerate(templates):
        print(f"  [{t_idx + 1}/{len(templates)}] {tmpl_id}", flush=True)
        required_poi_types = extract_poi_types(tmpl_text)

        generated = 0
        attempts = 0
        max_attempts = n_variations * 30
        used_cities_this_tmpl: set = set()

        while generated < n_variations and attempts < max_attempts:
            attempts += 1
            try:
                candidate_city = city_seq[(slot_idx + attempts - 1) % len(city_seq)]
                if candidate_city["name"] in used_cities_this_tmpl and attempts < max_attempts // 2:
                    candidate_city = city_list[(slot_idx + generated + attempts) % len(city_list)]

                city = candidate_city
                city_nodes = filter_nodes_to_city(node_catalog, city)
                city_pois = filter_pois_to_city(poi_catalog, city)

                if len(city_nodes) < CITY_MIN_NODES:
                    continue
                if required_poi_types and not city_has_required_pois(city_pois, required_poi_types):
                    continue

                src_entry, dst_entry = smart_pick_pair(city_nodes, city_nodes, rng)

                pre_src = coerce_node_id(G, str(src_entry["node_id"]))
                pre_dst = coerce_node_id(G, str(dst_entry["node_id"]))
                if pre_src is None or pre_dst is None or pre_src == pre_dst:
                    continue
                if pre_src not in G or pre_dst not in G:
                    continue
                if compute_shortest_path_length(G, pre_src, pre_dst, "length") is None:
                    continue

                src_lat, src_lon = src_entry.get("lat"), src_entry.get("lon")
                dst_lat, dst_lon = dst_entry.get("lat"), dst_entry.get("lon")

                nl_query = fill_template(tmpl_text, {
                    "SRC_NODE": f"({src_lat}, {src_lon})",
                    "DST_NODE": f"({dst_lat}, {dst_lon})",
                })
                if re.search(r'\{[A-Z0-9_]+\}', nl_query):
                    continue

                global_counter += 1
                record = {
                    "query_id": f"pref{global_counter}",
                    "nl_query": nl_query,
                }
                queries.append(record)
                used_cities_this_tmpl.add(city["name"])
                generated += 1
                slot_idx += 1

            except Exception:
                continue

    return queries


# Main

def main():
    parser = argparse.ArgumentParser(description="Generate poi+road_pref SoCal benchmark queries.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-variations", type=int, default=N_VARIATIONS)
    args = parser.parse_args()

    QUERIES_DIR.mkdir(parents=True, exist_ok=True)

    gpkg = ROOT / "data" / DATASET / "road_network.gpkg"
    rng = np.random.default_rng(args.seed)

    print(f"Dataset    : {DATASET}")
    print(f"GPKG       : {gpkg}")
    print(f"Template   : {TEMPLATE_PATH}")
    print(f"Output     : {QUERIES_DIR}")
    print(f"Variations : {args.n_variations} per template")
    print()

    templates = parse_template_file(TEMPLATE_PATH)
    print(f"Templates  : {len(templates)}")
    print(f"Planned queries: {len(templates) * args.n_variations}")
    print()

    print("Loading graph and catalogs...")
    G = build_graph_from_gpkg(str(gpkg))
    poi_catalog, node_catalog = load_catalogs(DATASET)
    cities = load_cities(DATASET)
    print(f"  Road nodes   : {G.number_of_nodes()}")
    print(f"  Catalog nodes: {len(node_catalog)}")
    print(f"  Cities       : {len(cities)}")
    print()

    t0 = time.time()
    queries = generate(templates, poi_catalog, node_catalog, G, rng, cities,
                        n_variations=args.n_variations)
    elapsed = time.time() - t0
    print(f"  -> {len(queries)} queries in {elapsed:.1f}s")

    out_path = QUERIES_DIR / "poi_road_pref.json"
    out_path.write_text(json.dumps(queries, indent=2, ensure_ascii=False))

    summary = {
        "total_queries": len(queries),
        "templates": len(templates),
        "queries_per_template": args.n_variations,
        "dataset": DATASET,
        "set": "poi_road_pref",
        "note": (
            "Unordered plain POI stops (pharmacy/bank/supermarket/library, no "
            "either/or groups) + an always-on 'avoid highways' road-type "
            "preference clause, asking for the fastest route. Deliberately "
            "isolates the road-preference (edge-level constraint) axis from the "
            "either/or-group feature used in poi_or_pref.json, since the parser "
            "does not reliably support either/or groups (verified: it flattens "
            "them into required sequential stops) whereas road_preference is "
            "verified working end-to-end. Not wired into set1 or any system "
            "code yet."
        ),
    }
    (QUERIES_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\nTotal queries : {len(queries)}")
    print(f"Saved to {QUERIES_DIR}")


if __name__ == "__main__":
    main()
