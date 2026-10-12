"""
Converts baselines/LLMAP/dataset/HIPP.json (1000 records) into SoCal query
instances, adding src/dst coordinates and query_id/template/nl_query/city while
keeping every original HIPP field as-is.

Writes hipp_all.json (slim, all 1000), hipp_meta.json (full gold), and
hipp_ordered.json/hipp_unordered.json (split by whether synthetic_label.dependencies
is non-empty). shopping_mall maps to the shop=mall value specifically — the
generic shop column lacks the synthetic rating/opening_hours enrichment those
rows carry. src/dst sampling reuses set1's city + smart_pick_pair logic.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HIPP_PATH    = ROOT / "baselines" / "LLMAP" / "dataset" / "HIPP.json"
NODES_PATH   = ROOT / "data" / "socal" / "catalog_nodes.json"
CITIES_PATH  = ROOT / "data" / "socal" / "cities.json"
OUT_DIR      = ROOT / "benchmark" / "socal_queries" / "llmap_queries"

SEED = 42

SRC_DST_MIN_KM = 4.0
SRC_DST_MAX_KM = 28.0


HIPP_TO_CATALOG: Dict[str, str] = {
    "shopping_mall": "mall",
    "shopping mall": "mall",
    "supermarket":   "supermarket",
    "pharmacy":      "pharmacy",
    "bank":          "bank",
    "library":       "library",
}


def _map_type(t: str) -> str:
    t_norm = t.strip().lower().replace(" ", "_")
    direct = HIPP_TO_CATALOG.get(t_norm)
    if direct:
        return direct
    return HIPP_TO_CATALOG.get(t.strip().lower(), t.strip().lower())


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(min(1.0, a)))


def filter_nodes_to_city(nodes: List[Dict], city: Dict) -> List[Dict]:
    clat, clon, r = city["lat"], city["lon"], city["radius_km"]
    return [n for n in nodes
            if n.get("lat") is not None and n.get("lon") is not None
            and haversine_km(clat, clon, n["lat"], n["lon"]) <= r]


def smart_pick_pair(
    candidates: List[Dict],
    rng,
    min_km: float = SRC_DST_MIN_KM,
    max_km: float = SRC_DST_MAX_KM,
    max_tries: int = 80,
) -> Tuple[Dict, Dict]:
    valid = [c for c in candidates if c.get("lat") is not None]
    if not valid:
        raise ValueError("No valid candidates in city")

    for _ in range(max_tries):
        s = valid[int(rng.integers(len(valid)))]
        good = [d for d in valid
                if d is not s
                and min_km <= haversine_km(s["lat"], s["lon"], d["lat"], d["lon"]) <= max_km]
        if good:
            return s, good[int(rng.integers(len(good)))]

    target_km = (min_km + max_km) / 2.0
    s = valid[int(rng.integers(len(valid)))]
    scored = sorted(
        [(abs(haversine_km(s["lat"], s["lon"], d["lat"], d["lon"]) - target_km), d)
         for d in valid if d is not s]
    )
    return s, (scored[0][1] if scored else valid[(int(rng.integers(len(valid))) + 1) % len(valid)])


def convert_record(
    hipp_idx: int,
    record: Dict,
    cities: List[Dict],
    nodes: List[Dict],
    rng,
) -> Dict:
    """Return a benchmark query dict for any record (time_limit kept as field)."""
    sl = record["synthetic_label"]

    time_limit = str(sl.get("time_limit", "None"))
    raw_pois = sl.get("pois", [])
    raw_deps = sl.get("dependencies", [])
    quality_weight = sl.get("quality_weight")
    distance_weight = sl.get("distance_weight")

    poi_types_mapped = [_map_type(p) for p in raw_pois]

    deps_mapped = [[_map_type(a), _map_type(b)] for a, b in raw_deps]

    family = "ordered" if raw_deps else "unordered"

    record_rng = np.random.default_rng(SEED + hipp_idx)
    city = cities[int(record_rng.integers(len(cities)))]

    city_nodes = filter_nodes_to_city(nodes, city)
    if len(city_nodes) < 2:
        city = cities[(cities.index(city) + 1) % len(cities)]
        city_nodes = filter_nodes_to_city(nodes, city)

    try:
        src_node, dst_node = smart_pick_pair(city_nodes, record_rng)
    except ValueError:
        city = cities[0]
        city_nodes = filter_nodes_to_city(nodes, city)
        src_node, dst_node = smart_pick_pair(city_nodes, record_rng)

    src_lat = round(src_node["lat"], 6)
    src_lon = round(src_node["lon"], 6)
    dst_lat = round(dst_node["lat"], 6)
    dst_lon = round(dst_node["lon"], 6)

    human_instruction = record["human_instruction"]

    nl_query = (
        f"I'm at ({src_lat}, {src_lon}) and need to get to ({dst_lat}, {dst_lon}). "
        + human_instruction
    )

    return {
        "query_id":            f"hipp_{hipp_idx:04d}",
        "template":            f"hipp_{family}",
        "nl_query":            nl_query,
        "city":                city["name"],
        "hipp_index":          hipp_idx,
        "hipp_family":         family,
        "time_limit":          time_limit,
        "poi_types_mapped":    poi_types_mapped,
        "dependencies_mapped": deps_mapped,
        "quality_weight":      quality_weight,
        "distance_weight":     distance_weight,
        "human_instruction":   human_instruction,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="Process only first 10 records (smoke test)")
    args = ap.parse_args()

    print(f"Loading HIPP from {HIPP_PATH} ...")
    hipp = json.loads(HIPP_PATH.read_text())
    print(f"Loading cities ({CITIES_PATH}) and nodes ({NODES_PATH}) ...")
    cities = json.loads(CITIES_PATH.read_text())
    nodes  = json.loads(NODES_PATH.read_text())

    records = hipp[:10] if args.smoke else hipp
    label   = "SMOKE (first 10)" if args.smoke else "FULL"
    print(f"Processing {len(records)} records [{label}] ...")

    all_queries = []
    rng = np.random.default_rng(SEED)

    for idx, rec in enumerate(records):
        q = convert_record(idx, rec, cities, nodes, rng)
        all_queries.append(q)

    suffix = "_smoke" if args.smoke else ""
    out_all  = OUT_DIR / f"hipp_all{suffix}.json"
    out_meta = OUT_DIR / f"hipp_meta{suffix}.json"

    # hipp_all.json matches the other benchmark files: query_id + nl_query only.
    slim_queries = [{"query_id": q["query_id"], "nl_query": q["nl_query"]}
                    for q in all_queries]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_all.write_text(json.dumps(slim_queries, indent=2, ensure_ascii=False))
    # hipp_meta.json keeps the full records (poi/deps/time_limit/weights) for evaluation.
    out_meta.write_text(json.dumps(all_queries, indent=2, ensure_ascii=False))

    # Per-template files (query_id + nl_query only) so run_crafthipp.py can load
    # one file per template, matching the convention every other runner uses.
    out_paths_by_family = {}
    for family in ("ordered", "unordered"):
        family_queries = [
            {"query_id": q["query_id"], "nl_query": q["nl_query"]}
            for q in all_queries if q["hipp_family"] == family
        ]
        out_family = OUT_DIR / f"hipp_{family}{suffix}.json"
        out_family.write_text(json.dumps(family_queries, indent=2, ensure_ascii=False))
        out_paths_by_family[family] = (out_family, len(family_queries))

    ordered      = sum(1 for q in all_queries if q["hipp_family"] == "ordered")
    unordered    = sum(1 for q in all_queries if q["hipp_family"] == "unordered")
    with_timelim = sum(1 for q in all_queries if q["time_limit"] != "None")

    print(f"\n{'='*55}")
    print(f"Total records: {len(all_queries)}")
    print(f"  ordered:        {ordered}")
    print(f"  unordered:      {unordered}")
    print(f"  with time_limit: {with_timelim}")
    print(f"\nOutput → {out_all} (slim: query_id + nl_query)")
    print(f"       → {out_meta} (full records w/ poi/deps/time_limit/weights)")
    for family, (path, n) in out_paths_by_family.items():
        print(f"       → {path} ({n} queries, template=hipp_{family})")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
