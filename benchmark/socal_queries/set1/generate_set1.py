"""
Set 1 (city-focused) SoCal benchmark query generator.

Samples src/dst and POIs within a single California city per query,
using data/socal/ catalogs and cities.json.

Usage:
  python benchmark/socal_queries/set1/generate_set1.py
  python benchmark/socal_queries/set1/generate_set1.py --seed 99

Output (one file per template):
  benchmark/socal_queries/set1/ev_routing.json
  benchmark/socal_queries/set1/ordered_poi.json
  ... (7 files total)
  benchmark/socal_queries/set1/summary.json
"""

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
TEMPLATES_DIR = ROOT / "benchmark" / "templates"
QUERIES_DIR   = ROOT / "benchmark" / "socal_queries" / "set1"

KM_TO_MI    = 0.621371
ROAD_FACTOR = 1.3
SEED        = 42

# 7 variations per template → 1001 queries total (within [950, 1050] target)
N_VARIATIONS = 7

SRC_DST_MIN_KM = 4.0
SRC_DST_MAX_KM = 28.0

# Minimum catalog nodes required inside a city to attempt query generation
CITY_MIN_NODES = 20

# Variety grids (cycled deterministically across variations)

# Realistic EV max ranges in km — cycled per variation to guarantee spread
EV_MAX_RANGES_KM = [150, 200, 250, 300, 350, 400, 175, 225, 275, 325]

# Battery current charge percentages per difficulty
EV_PCT_BY_DIFF = {
    "easy":   [75, 80, 85, 90, 95, 100],
    "medium": [40, 50, 55, 60, 65, 70],
    "hard":   [15, 20, 25, 30, 35, 40],
}

# Time budget tiers in minutes (whole hours for LLMAP compatibility)
# City trips are short so we widen the range explicitly
TIME_BUDGET_TIERS = [60, 120, 180, 240, 300, 360]

# Minimum assumed travel time for budget calc (prevents collapse on short trips)
MIN_TRAVEL_MIN = 20.0

# Difficulty sequence for N_VARIATIONS=7: 2 easy, 3 medium, 2 hard
DIFFICULTY_CYCLE = ["easy", "medium", "hard", "medium", "easy", "hard", "medium"]

LLMAP_POI_TYPES = {"pharmacy", "bank", "supermarket", "library"}

LLMAP_DISPLAY_TO_CATEGORY: Dict[str, str] = {
    "pharmacy":    "pharmacy",
    "bank":        "bank",
    "supermarket": "supermarket",
    "library":     "library",
}


# Catalog + city loading

def load_catalogs(dataset: str):
    d        = ROOT / "data" / dataset
    charging = json.loads((d / "catalog_charging.json").read_text())
    pois     = json.loads((d / "catalog_pois.json").read_text())
    nodes    = json.loads((d / "catalog_nodes.json").read_text())
    counts   = json.loads((d / "catalog_count.json").read_text()) if (d / "catalog_count.json").exists() else {}
    return charging, pois, nodes, counts


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


def filter_charging_to_city(charging: List[Dict], city: Dict) -> List[Dict]:
    clat, clon, r = city["lat"], city["lon"], city["radius_km"]
    return [c for c in charging
            if c.get("lat") is not None and c.get("lon") is not None
            and haversine_km(clat, clon, c["lat"], c["lon"]) <= r]


def city_has_required_pois(city_pois: Dict, poi_types: List[str]) -> bool:
    """Return True if the city has ≥1 POI for each required type."""
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
            num  = m.group(1)
            text = m.group(2).strip()
            if text:
                templates.append((f"{stem}_{num}", text))
    return templates


def fill_template(template: str, values: Dict[str, str]) -> str:
    result = template
    for key, val in values.items():
        result = result.replace("{" + key + "}", str(val))
    return result


def placeholders_in(template: str) -> List[str]:
    return re.findall(r'\{([A-Z0-9_]+)\}', template)


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

    # Fallback: pick pair closest to target midpoint
    target_km = (min_km + max_km) / 2.0
    s = valid_src[int(rng.integers(len(valid_src)))]
    scored = sorted(
        [(abs(haversine_km(s["lat"], s["lon"], d["lat"], d["lon"]) - target_km), d)
         for d in valid_dst if d is not s]
    )
    dst = scored[0][1] if scored else valid_dst[int(rng.integers(len(valid_dst)))]
    return s, dst


# POI type helpers

def _extract_poi_types(text: str) -> List[str]:
    text_lower = text.lower()
    hits: List[Tuple[int, str]] = []
    for display, cat in sorted(LLMAP_DISPLAY_TO_CATEGORY.items(), key=lambda x: -len(x[0])):
        for m in re.finditer(r'\b' + re.escape(display) + r's?\b', text_lower):
            hits.append((m.start(), cat))
    hits.sort(key=lambda x: x[0])
    seen: set = set()
    result: List[str] = []
    for _, cat in hits:
        if cat not in seen:
            seen.add(cat)
            result.append(cat)
    return result


def _extract_cat_limits(text: str) -> Dict[str, int]:
    text_lower = text.lower()
    limits: Dict[str, int] = {}
    for display, cat in sorted(LLMAP_DISPLAY_TO_CATEGORY.items(), key=lambda x: -len(x[0])):
        pattern = r'(\d+)\s+' + re.escape(display) + r's?\b'
        m = re.search(pattern, text_lower)
        if m:
            limits[cat] = int(m.group(1))
    return limits or {"pharmacy": 1}


# Parameter samplers

def _mi(km: float) -> int:
    return round(km * KM_TO_MI)


def sample_difficulty(rng) -> str:
    r = float(rng.random())
    if r < 0.30:
        return "easy"
    elif r < 0.70:
        return "medium"
    return "hard"


def smart_battery(path_length_m: float, rng, difficulty: str = "medium",
                  force_charger: bool = False, variation_idx: int = 0):
    """
    Uses a pre-defined grid of realistic EV max ranges cycled by variation_idx
    so successive queries always span the full 150-400 km spectrum regardless
    of how short the city-scoped route is.
    """
    max_km = EV_MAX_RANGES_KM[variation_idx % len(EV_MAX_RANGES_KM)]

    if force_charger:
        road_km = path_length_m * ROAD_FACTOR / 1000
        # Current battery below what's needed to make it without a stop
        cur_km = max(5, min(round(road_km * float(rng.uniform(0.30, 0.65))), max_km - 1))
        pct    = round(cur_km / max_km * 100)
        return max_km, pct, cur_km

    pct_options = EV_PCT_BY_DIFF.get(difficulty, EV_PCT_BY_DIFF["medium"])
    pct = pct_options[variation_idx % len(pct_options)]
    cur_km = min(round(max_km * pct / 100), max_km)
    return max_km, pct, cur_km


def smart_distance_limits(src_lat: float, src_lon: float,
                           poi_types: List[str], poi_catalog: Dict,
                           rng, difficulty: str = "medium") -> List[int]:
    if difficulty == "easy":
        slack = float(rng.uniform(2.0, 3.0))
    elif difficulty == "medium":
        slack = float(rng.uniform(1.4, 2.0))
    else:
        slack = float(rng.uniform(1.2, 1.4))

    limits = []
    cur_lat, cur_lon = src_lat, src_lon

    for poi_type in poi_types:
        pois_of_type = poi_catalog.get(poi_type, [])
        if not pois_of_type:
            limits.append(10)
            continue

        min_dist_km  = None
        nearest_lat  = cur_lat
        nearest_lon  = cur_lon

        for poi in pois_of_type:
            if poi.get("lat") is None or poi.get("lon") is None:
                continue
            d = haversine_km(cur_lat, cur_lon, poi["lat"], poi["lon"])
            if min_dist_km is None or d < min_dist_km:
                min_dist_km  = d
                nearest_lat  = poi["lat"]
                nearest_lon  = poi["lon"]

        if min_dist_km is None:
            limits.append(10)
        else:
            limit_km = max(1.5, min_dist_km * ROAD_FACTOR * slack)
            limits.append(max(2, round(limit_km)))
            cur_lat, cur_lon = nearest_lat, nearest_lon

    return limits


def smart_time_budget(path_travel_s: float, cat_limits: Dict[str, int],
                      rng, difficulty: str = "medium",
                      variation_idx: int = 0) -> Tuple[int, int]:
    """
    Cycles through TIME_BUDGET_TIERS per variation_idx so city-scoped queries
    (which have short travel times) still produce budgets across 60-360 min.
    visit_min is varied by difficulty to match the tier plausibly.
    """
    total_pois = sum(cat_limits.values()) if cat_limits else 1

    if difficulty == "easy":
        visit_min = int(rng.choice([15, 20, 25]))
        slack     = float(rng.uniform(1.5, 2.0))
    elif difficulty == "medium":
        visit_min = int(rng.choice([20, 25, 30]))
        slack     = float(rng.uniform(1.2, 1.5))
    else:
        visit_min = int(rng.choice([25, 30, 45]))
        slack     = float(rng.uniform(1.05, 1.2))

    # Floor travel time so short city trips don't collapse budgets
    travel_min      = max(MIN_TRAVEL_MIN, path_travel_s / 60.0)
    total_visit_min = total_pois * visit_min
    raw_budget      = (travel_min + total_visit_min) * slack

    # Round to whole hour (LLMAP requirement), then ensure minimum tier spread
    computed = max(60, math.ceil(raw_budget / 60) * 60)

    # Pick from tier grid indexed by variation so successive queries differ
    tier = TIME_BUDGET_TIERS[variation_idx % len(TIME_BUDGET_TIERS)]
    # Use whichever is larger: what the route actually needs vs the tier
    budget_min = max(computed, tier)

    return budget_min, visit_min


# Template classification — exact-membership sets, not substring tests. A substring
# check on the stem name is fragile here: "unordered_poi" contains "ordered", and
# unordered trips still have distance limits despite not being "ordered" — sets
# avoid both false positives/negatives that a naive text match would hit.
_EV_STEMS          = frozenset({"ev_routing", "ev+ordered_poi", "ev+time"})
_ORDERED_STEMS     = frozenset({"ordered_poi", "ev+ordered_poi", "ordered+time_budget"})
_DIST_LIMIT_STEMS  = frozenset({"ordered_poi", "ev+ordered_poi", "ordered+time_budget",
                                 "unordered_poi"})
_COLLECTION_STEMS  = frozenset({"poi_collection"})
_TIME_BUDGET_STEMS = frozenset({"poi_collection", "ordered+time_budget"})

def _is_ev_template(stem: str) -> bool:
    return stem in _EV_STEMS

def _is_ordered_template(stem: str) -> bool:
    return stem in _ORDERED_STEMS

def _has_distance_limits_template(stem: str) -> bool:
    return stem in _DIST_LIMIT_STEMS

def _is_collection_template(stem: str) -> bool:
    return stem in _COLLECTION_STEMS

def _is_time_budget_template(stem: str) -> bool:
    return stem in _TIME_BUDGET_STEMS


# Value builder

def build_values(template, poi_catalog, node_catalog, rng,
                 src_entry, dst_entry,
                 path_m=None, path_s=None, difficulty=None,
                 force_charger: bool = False, variation_idx: int = 0):
    needed = set(placeholders_in(template))
    v      = {}

    if difficulty is None:
        difficulty = sample_difficulty(rng)

    src_lat = src_entry.get("lat")
    src_lon = src_entry.get("lon")
    dst_lat = dst_entry.get("lat")
    dst_lon = dst_entry.get("lon")

    if "SRC_NODE" in needed:
        v["SRC_NODE"] = f"({src_lat}, {src_lon})"
    if "DST_NODE" in needed:
        v["DST_NODE"] = f"({dst_lat}, {dst_lon})"

    if path_m is not None and path_m > 0:
        max_km, pct, cur_km = smart_battery(path_m, rng, difficulty,
                                            force_charger=force_charger,
                                            variation_idx=variation_idx)
    else:
        max_km = EV_MAX_RANGES_KM[variation_idx % len(EV_MAX_RANGES_KM)]
        pct    = EV_PCT_BY_DIFF.get(difficulty, EV_PCT_BY_DIFF["medium"])[variation_idx % 6]
        cur_km = int(max_km * pct / 100)

    cur_km = min(cur_km, max_km)
    pct    = min(pct, 100)
    max_mi = _mi(max_km)
    cur_mi = min(_mi(cur_km), max_mi)

    v["MAX_BATTERY_MI"]      = max_mi
    v["CURRENT_BATTERY_PCT"] = f"{pct}%"
    v["CURRENT_BATTERY_MI"]  = cur_mi

    max_battery_m = max_km * 1000
    cur_battery_m = cur_km * 1000

    poi_types = _extract_poi_types(template)

    if src_lat is not None and src_lon is not None and poi_types:
        dlims = smart_distance_limits(src_lat, src_lon, poi_types[:3], poi_catalog, rng, difficulty)
        while len(dlims) < 3:
            dlims.append(dlims[-1])
    else:
        dlims = [int(rng.choice([3, 4, 5, 6, 7, 8, 10, 12])) for _ in range(3)]

    v["DISTANCE_LIMIT_MI"]   = _mi(dlims[0])
    v["DISTANCE_LIMIT_1_MI"] = _mi(dlims[0])
    v["DISTANCE_LIMIT_2_MI"] = _mi(dlims[1])
    v["DISTANCE_LIMIT_3_MI"] = _mi(dlims[2])

    dlims_m = [d * 1000 for d in dlims]

    cat_limits = _extract_cat_limits(template)

    if path_s is not None and path_s > 0:
        budget_min, visit_min = smart_time_budget(path_s, cat_limits, rng, difficulty,
                                                   variation_idx=variation_idx)
    else:
        budget_min = TIME_BUDGET_TIERS[variation_idx % len(TIME_BUDGET_TIERS)]
        visit_min  = int(rng.choice([20, 25, 30, 45]))

    v["TIME_BUDGET_MIN"]   = budget_min
    v["TIME_BUDGET_HOURS"] = int(budget_min / 60)
    v["VISIT_MIN"]         = visit_min

    return (v, poi_types, dlims_m, cat_limits,
            budget_min, visit_min,
            max_battery_m, cur_battery_m, pct, difficulty)


# Per-file generator

FILE_CONFIG: Dict[str, int] = {
    "ev_routing":     N_VARIATIONS,
    "ordered_poi":    N_VARIATIONS,
    "poi_collection": N_VARIATIONS,
    "ev+ordered_poi":       N_VARIATIONS,
    "ev+time":              N_VARIATIONS,
    "ordered+time_budget":  N_VARIATIONS,
    "unordered_poi":        N_VARIATIONS,
}

# fail fast at import time if a new FILE_CONFIG stem isn't in any classification set above
assert set(FILE_CONFIG) >= (_EV_STEMS | _ORDERED_STEMS | _DIST_LIMIT_STEMS
                            | _COLLECTION_STEMS | _TIME_BUDGET_STEMS)


def generate_for_file(file_stem, templates, poi_catalog, node_catalog,
                      G, rng, cities, n_variations):
    """
    Generates n_variations queries per template text with guaranteed variety:
    - Cities assigned round-robin (shuffled once per file) → even spread
    - Difficulty follows DIFFICULTY_CYCLE → balanced easy/medium/hard
    - Battery max range cycles through EV_MAX_RANGES_KM → full 150-400 km spread
    - Time budget cycles through TIME_BUDGET_TIERS → 60-360 min spread
    """
    queries  = []
    is_ev    = _is_ev_template(file_stem)
    is_ord   = _is_ordered_template(file_stem)
    has_dist = _has_distance_limits_template(file_stem)
    is_coll  = _is_collection_template(file_stem)
    is_time  = _is_time_budget_template(file_stem)
    is_ev_routing = file_stem == "ev_routing"

    # Build a round-robin city sequence for the whole file so each template
    # text gets a different city for each of its n_variations slots.
    total_slots = len(templates) * n_variations
    city_list   = list(cities)
    rng.shuffle(city_list)
    # Tile the city list to cover all slots
    city_seq = [city_list[i % len(city_list)] for i in range(total_slots)]
    slot_idx = 0  # advances once per successfully generated query

    for t_idx, (tmpl_id, tmpl_text) in enumerate(templates):
        print(f"  [{t_idx+1}/{len(templates)}] {tmpl_id}", flush=True)

        required_poi_types = _extract_poi_types(tmpl_text)

        generated = 0
        attempts  = 0
        max_attempts = n_variations * 30
        # Track which cities we've already used for this template to avoid repeats
        used_cities_this_tmpl: set = set()

        while generated < n_variations and attempts < max_attempts:
            attempts += 1
            try:
                # Round-robin city assignment
                # Walk forward through city_seq until we find one that works
                candidate_city = city_seq[(slot_idx + attempts - 1) % len(city_seq)]

                # Prefer a city not yet used for this template
                if candidate_city["name"] in used_cities_this_tmpl and attempts < max_attempts // 2:
                    candidate_city = city_list[(slot_idx + generated + attempts) % len(city_list)]

                city       = candidate_city
                city_nodes = filter_nodes_to_city(node_catalog, city)
                city_pois  = filter_pois_to_city(poi_catalog, city)

                if len(city_nodes) < CITY_MIN_NODES:
                    continue
                if required_poi_types and not city_has_required_pois(city_pois, required_poi_types):
                    continue

                # Pick src/dst within the city
                src_entry, dst_entry = smart_pick_pair(
                    city_nodes, city_nodes, rng,
                    min_km=SRC_DST_MIN_KM, max_km=SRC_DST_MAX_KM,
                )

                pre_src = coerce_node_id(G, str(src_entry["node_id"]))
                pre_dst = coerce_node_id(G, str(dst_entry["node_id"]))
                if pre_src is None or pre_dst is None or pre_src == pre_dst:
                    continue

                path_m = compute_shortest_path_length(G, pre_src, pre_dst, "length")
                path_s = compute_shortest_path_length(G, pre_src, pre_dst, "travel_time")
                if path_m is None:
                    continue

                # Deterministic difficulty and variation index
                difficulty    = DIFFICULTY_CYCLE[generated % len(DIFFICULTY_CYCLE)]
                variation_idx = generated          # used for battery/time grids
                force_charger = (is_ev_routing and is_ev and (generated % 2 == 0))

                (v, poi_types, dlims_m, cat_limits,
                 budget_min, visit_min,
                 max_battery_m, cur_battery_m, pct,
                 difficulty) = build_values(
                    tmpl_text, city_pois, city_nodes, rng,
                    src_entry, dst_entry,
                    path_m=path_m, path_s=path_s, difficulty=difficulty,
                    force_charger=force_charger, variation_idx=variation_idx,
                )

                nl_query = fill_template(tmpl_text, {k: str(val) for k, val in v.items()})
                if re.search(r'\{[A-Z0-9_]+\}', nl_query):
                    continue

                record: Dict[str, Any] = {
                    "query_id": f"{tmpl_id}_v{generated+1:02d}",
                    "nl_query": nl_query,
                }

                queries.append(record)
                used_cities_this_tmpl.add(city["name"])
                generated += 1
                slot_idx  += 1

            except Exception:
                continue

    return queries


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Generate city-scoped SoCal benchmark queries."
    )
    parser.add_argument("--seed",      type=int, default=SEED)
    parser.add_argument("--templates", nargs="+", default=None,
                        help="Only regenerate specific template stems.")
    args = parser.parse_args()

    QUERIES_DIR.mkdir(parents=True, exist_ok=True)

    gpkg = ROOT / "data" / DATASET / "road_network.gpkg"
    rng  = np.random.default_rng(args.seed)

    print(f"Dataset    : {DATASET}")
    print(f"GPKG       : {gpkg}")
    print(f"Templates  : {TEMPLATES_DIR}")
    print(f"Output     : {QUERIES_DIR}")
    print(f"Variations : {N_VARIATIONS} per template")
    print()

    # Count planned queries
    planned = 0
    for stem, nv in FILE_CONFIG.items():
        path = TEMPLATES_DIR / f"{stem}.txt"
        if path.exists():
            lines = [l.strip() for l in path.read_text().splitlines()
                     if l.strip() and not l.strip().startswith("#")]
            n_tmpl = sum(1 for l in lines if re.match(r'^\d+:\s+', l))
            planned += n_tmpl * nv
    print(f"Planned queries: {planned}")
    print()

    print("Loading graph and catalogs...")
    G = build_graph_from_gpkg(str(gpkg))
    charging, poi_catalog, node_catalog, _ = load_catalogs(DATASET)
    cities = load_cities(DATASET)

    print(f"  Road nodes  : {G.number_of_nodes()}")
    print(f"  Catalog nodes: {len(node_catalog)}")
    print(f"  Cities       : {len(cities)}")
    print(f"  POI types    : {list(poi_catalog.keys())}")
    print()

    summary: Dict[str, Any] = {}
    total = 0
    city_counts: Dict[str, int] = {}
    diff_counts: Dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}

    for file_stem, n_var in FILE_CONFIG.items():
        if args.templates and file_stem not in args.templates:
            continue
        path = TEMPLATES_DIR / f"{file_stem}.txt"
        if not path.exists():
            print(f"  [skip] {file_stem}.txt not found")
            continue

        templates = parse_template_file(path)
        print(f"\n{file_stem}: {len(templates)} templates × {n_var} variations")

        t0 = time.time()
        queries = generate_for_file(
            file_stem, templates, poi_catalog, node_catalog,
            G, rng, cities, n_variations=n_var,
        )
        elapsed = time.time() - t0
        print(f"  → {len(queries)} queries in {elapsed:.1f}s")

        out_path = QUERIES_DIR / f"{file_stem}.json"
        out_path.write_text(json.dumps(queries, indent=2, ensure_ascii=False))

        summary[file_stem] = {"templates": len(templates), "queries": len(queries)}
        total += len(queries)

        for q in queries:
            d = q.get("difficulty", "medium")
            diff_counts[d] = diff_counts.get(d, 0) + 1

    # Merge with existing summary if partial run
    summary_path = QUERIES_DIR / "summary.json"
    if args.templates and summary_path.exists():
        existing = json.loads(summary_path.read_text())
        existing.update(summary)
        summary = existing
        # Rebuild totals from all files
        total = sum(v["queries"] for k, v in summary.items() if isinstance(v, dict) and "queries" in v)

    summary["total_queries"]       = total
    summary["queries_per_template"] = {k: v["queries"] for k, v in summary.items()
                                        if isinstance(v, dict) and "queries" in v}
    summary["queries_per_difficulty"] = diff_counts

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n{'='*50}")
    print(f"Total queries : {total}")
    print(f"\nPer template:")
    for stem, info in summary.items():
        if isinstance(info, dict) and "queries" in info:
            print(f"  {stem:<28} {info['queries']:>4}")
    print(f"\nPer difficulty:")
    for d, c in diff_counts.items():
        print(f"  {d:<8} {c:>4}")
    print(f"\nSaved to {QUERIES_DIR}")


if __name__ == "__main__":
    main()
