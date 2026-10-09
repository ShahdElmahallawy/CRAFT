"""
Bare LLM baseline — no route search, just asks the LLM to name stops and matches them to the dataset.

Given a query file ([{query_id, nl_query}, ...]), asks the LLM for stops as
structured JSON (name, address, type), then matches each to the dataset:
  1. Geocode the address via Google Geocoding API -> (lat, lng).
  2. Find the nearest same-type POI within threshold_m, preferring an exact
     name match as tie-breaker among candidates in radius.
  3. Drop the stop if geocoding fails or nothing matches within threshold.

Output (per query, saved to outputs/llm_baseline/<query_id>.json):
  {query_id, nl_query, llm_feasible, llm_infeasible_reason,
   llm_stops, matched_stops, dropped_stops, classification}

Usage:
  python baselines/llm/llm_baseline.py --queries benchmark/socal_queries/set1/ev_routing.json
  python baselines/llm/llm_baseline.py --queries benchmark/socal_queries/set1/ordered_poi.json --query-id ordered_poi_0_v01
  python baselines/llm/llm_baseline.py --queries benchmark/socal_queries/set1/ev_routing.json --no-geocode
"""

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.append(_p)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "outputs" / "baselines" / "llm"
GEOCODE_CACHE_PATH = Path(__file__).resolve().parent / "geocode_cache.json"

DEFAULT_GPKG = str(REPO_ROOT / "data" / "riverside" / "road_network.gpkg")
DEFAULT_GEOCODE_THRESHOLD_M = 200

# Normalise LLM-generated type strings to canonical dataset categories.
_NORMALIZE_TYPE: Dict[str, str] = {
    # charging_station aliases
    "ev charger": "charging_station",
    "ev charging": "charging_station",
    "ev station": "charging_station",
    "charger": "charging_station",
    "charging": "charging_station",
    "electric vehicle charging": "charging_station",
    # supermarket aliases
    "grocery": "supermarket",
    "grocery store": "supermarket",
    "market": "supermarket",
    "food store": "supermarket",
    # pharmacy aliases
    "drugstore": "pharmacy",
    "drug store": "pharmacy",
    "chemist": "pharmacy",
    # bank aliases
    "credit union": "bank",
    "financial institution": "bank",
}


def _normalize_type(raw: str) -> str:
    return _NORMALIZE_TYPE.get(raw.strip().lower(), raw.strip().lower())


# Env helpers


def _read_env(key: str) -> str:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return ""
    for line in env_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        k, v = text.split("=", 1)
        if k.strip() == key:
            return v.strip().strip("'").strip('"')
    return ""


# JSON parsing (handles markdown fences + trailing content from LLM)


def _extract_json(text: str) -> Dict[str, Any]:
    import json as _json

    try:
        return _json.loads(text)
    except _json.JSONDecodeError:
        pass
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        stripped = "\n".join(inner).strip()
        try:
            return _json.loads(stripped)
        except _json.JSONDecodeError:
            pass
    decoder = _json.JSONDecoder()
    for search_text in (text, stripped):
        start = search_text.find("{")
        if start == -1:
            continue
        try:
            obj, _ = decoder.raw_decode(search_text, start)
            return obj
        except _json.JSONDecodeError:
            continue
    return _json.loads(text)  # raises


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a travel/itinerary planner. The user will describe a trip and you will suggest \
the intermediate stops to visit along the way.

Do NOT include the starting point or the final destination in the stops list \
— only the places to visit in between.

Return ONLY a JSON object, no explanation, no markdown.

If you believe the trip is not possible at all, return exactly:
{"feasible": false}

Otherwise return:
{"feasible": true, "stops": [...]}

Each stop in the list must have exactly:
  "name"    : the place name (string)
  "address" : the full street address including city and state (string)
  "type"    : one of: pharmacy, library, bank, supermarket, charging_station

IMPORTANT: Don't just return a list of POIs  - Instead make sure to also consider the practical constraints of the trip, such as time limits, distance limits, specific order of POIs, and battery range if it's an EV trip.

Some rules to follow:
If the User had to stop at a charging station, the car's battery is fully recharged and can be considered full again for the next leg of the trip. 
Also include it as part of the stops list with type "charging_station" so it can be matched to the dataset and not accidentally skipped.\

If the user mentions a total time limit for the trip, the whole trip (driving \
time plus time spent at each stop) must fit within that limit.
If time spent at stop (POI) is not mentioned by the user, assume a default of 30 minutes. Otherwise use the user's specified time at stop.\

When deciding how many stops to include: if the user wants to visit as many \
places as possible, include as many valid stops as you can while following \
the rules above. If the user wants a quick or efficient route, prefer stops \
that do not add much extra driving. Respect any time or distance limits the user has given.\

So, to sum up: Always consider the practical constraints of the trip, and try to find a good balance between visiting interesting stops and keeping the trip feasible and efficient. The stops you suggest should make sense given the user's preferences and constraints.\

Example output:
{"feasible": true, "stops": [{"name": "CVS Pharmacy", "address": "123 W Foothill Blvd, Upland, CA 91786", \
"type": "pharmacy"}, {"name": "Upland Public Library", "address": "450 N Euclid Ave, Upland, CA 91786", \
"type": "library"}]}
"""


def _call_llm(
    nl_query: str,
    api_key: str,
    base_url: str,
    model: str,
    extra_payload: Optional[Dict] = None,
) -> Tuple[str, Dict]:
    """Returns (content_str, usage_dict). usage_dict may be empty if the API omits it."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": nl_query},
        ],
        "temperature": 0,
    }
    if extra_payload:
        payload.update(extra_payload)
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    max_attempts = max(1, int(os.environ.get("LLM_MAX_RETRIES", "10")))
    raw = None
    for attempt in range(1, max_attempts + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                raw = resp.read()
            break
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt == max_attempts:
                raise RuntimeError(
                    f"LLM request failed at {url} (HTTP {exc.code})"
                    + (f" after {attempt} attempts." if retryable else ".")
                ) from exc
            time.sleep(_retry_wait(exc, attempt))
    data = json.loads(raw)
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return content, usage


def _retry_wait(exc: "urllib.error.HTTPError", attempt: int) -> float:
    """Seconds to wait before retrying a 429/5xx, honoring server hints."""
    headers = getattr(exc, "headers", None)
    if headers is not None:
        ra = headers.get("Retry-After")
        if ra:
            try:
                return min(float(ra) + 0.5, 60.0)
            except ValueError:
                pass
    try:
        body = exc.read().decode("utf-8", "ignore")
    except Exception:
        body = ""
    m = re.search(r"try again in ([\d.]+)\s*(ms|s)", body)
    if m:
        val = float(m.group(1))
        secs = val / 1000.0 if m.group(2) == "ms" else val
        return min(secs + 0.5, 60.0)
    return min(2 ** attempt, 30.0)


# Haversine distance (metres)
def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# Geocode cache helpers
def _load_geocode_cache(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_geocode_cache(cache: Dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Google Geocoding API
# ---------------------------------------------------------------------------
_GEOCODE_BASE = (
    _read_env("GEOCODING_API_URL")
    or "https://maps.googleapis.com/maps/api/geocode/json"
)
_GEOCODE_COMPONENTS = "administrative_area:CA|country:US"
# SoCal POI extent (EPSG:4326): derived from data/socal/road_network.gpkg pois layer
_GEOCODE_BOUNDS = "33.396326,-118.954970|34.832025,-117.454112"


def _geocode(
    address: str, api_key: str, cache: Dict[str, Any]
) -> Optional[Tuple[float, float]]:
    """Geocode an address using Google. Returns (lat, lng) or None. Caches results."""
    cache_key = address.strip().lower()
    if cache_key in cache:
        cached = cache[cache_key]
        if cached is None:
            return None
        return (cached["lat"], cached["lng"])

    params = urllib.parse.urlencode(
        {
            "address": address,
            "components": _GEOCODE_COMPONENTS,
            "bounds": _GEOCODE_BOUNDS,
            "key": api_key,
        }
    )
    try:
        time.sleep(0.1)
        with urllib.request.urlopen(f"{_GEOCODE_BASE}?{params}", timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        print(f"    [geocode] error for '{address}': {exc}")
        cache[cache_key] = None
        return None

    results = data.get("results", [])
    if not results:
        cache[cache_key] = None
        return None

    loc = results[0]["geometry"]["location"]
    lat, lng = float(loc["lat"]), float(loc["lng"])
    cache[cache_key] = {"lat": lat, "lng": lng}
    return (lat, lng)


# ---------------------------------------------------------------------------
# POI matching helpers
# ---------------------------------------------------------------------------
_CATEGORY_COLS = ["amenity", "shop", "leisure", "tourism", "highway", "power"]

_TYPE_ALIASES: Dict[str, List[str]] = {
    "charging_station": ["charging_station"],
    "supermarket": ["supermarket"],
    "pharmacy": ["pharmacy"],
    "bank": ["bank"],
    "library": ["library"],
}


def _load_pois(gpkg_path: str):
    import geopandas as gpd

    return gpd.read_file(gpkg_path, layer="pois")


def _type_mask(pois_gdf, poi_type: str):
    """Vectorized type filter — returns a boolean array over pois_gdf.

    Uses exact equality, not substring match, because all five supported types
    are exact OSM tag values. substring matching would cause 'bank' to hit
    'food_bank', 'blood_bank', 'embankment', etc.
    """
    import numpy as np
    aliases = _TYPE_ALIASES.get(poi_type, [poi_type])
    mask = np.zeros(len(pois_gdf), dtype=bool)
    for col in _CATEGORY_COLS:
        if col not in pois_gdf.columns:
            continue
        col_str = pois_gdf[col].fillna("").astype(str).str.lower()
        for alias in aliases:
            mask |= (col_str == alias).values
    return mask


def _row_lat_lng(row, pois_gdf) -> Tuple[Optional[float], Optional[float]]:
    if "lat" in pois_gdf.columns and "lon" in pois_gdf.columns:
        try:
            return float(row["lat"]), float(row["lon"])
        except (TypeError, ValueError):
            pass
    geom = row.get("geometry")
    if geom is not None and not geom.is_empty:
        return float(geom.centroid.y), float(geom.centroid.x)
    return None, None


# Location-first matching pipeline
def match_stops(
    llm_stops: List[Dict],
    pois_gdf,
    google_api_key: str,
    threshold_m: float,
    geocode_cache: Dict[str, Any],
    geocode_enabled: bool = True,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Match each LLM stop to a dataset POI using a 4-tier location-first pipeline.

    Step 1 — Geocode the address → (lat, lng).
    Step 2 — Search same-type POIs within 1km of geocoded point, pick best by tier:
               Tier A: coordinate match   — within 50m   (same place, different data sources)
               Tier B: exact name match   — within 1km   (name matches exactly)
               Tier C: fuzzy name match   — within 1km   (one name is substring of the other)
               Tier D: proximity only     — within threshold_m  (nearest same-type, no name needed)
    Step 3 — Drop if geocode fails, address missing, or no same-type POI within threshold_m.

    Returns (matched_stops, dropped_stops).
    """
    import numpy as np

    COORD_MATCH_M = 50.0   # within 50m → treat as coordinate-level match
    NAME_SEARCH_M = 1000.0  # search radius for name-confirmed tiers

    matched: List[Dict] = []
    dropped: List[Dict] = []
    geocode_disabled = not geocode_enabled or not google_api_key

    # Pre-compute type masks once per unique type.
    _tmask_cache: Dict[str, Any] = {}
    for stop in llm_stops:
        pt = _normalize_type(stop.get("type", ""))
        if pt not in _tmask_cache:
            _tmask_cache[pt] = _type_mask(pois_gdf, pt)

    for stop in llm_stops:
        poi_type  = _normalize_type(stop.get("type", ""))
        stop_name = stop.get("name", "").strip()
        address   = stop.get("address", "")
        tmask     = _tmask_cache[poi_type]

        geocoded_lat: Optional[float] = None
        geocoded_lng: Optional[float] = None

        # Step 1: Geocode address
        if geocode_disabled:
            drop_reason = "geocoding disabled — cannot verify stop location"
        elif not address:
            drop_reason = "address field is empty — cannot geocode"
        else:
            result = _geocode(address, google_api_key, geocode_cache)
            if result is None:
                drop_reason = "geocoding API returned no results"
            else:
                geocoded_lat, geocoded_lng = result
                drop_reason = ""

        if geocoded_lat is None:
            dropped.append({
                "name": stop_name, "address": address, "type": poi_type,
                "drop_reason": drop_reason,
                "geocoded_lat": None, "geocoded_lng": None, "geocode_failed": True,
            })
            continue

        # Step 2: Compute distances to all same-type candidates
        if not tmask.any():
            dropped.append({
                "name": stop_name, "address": address, "type": poi_type,
                "drop_reason": f"poi type '{poi_type}' not found in dataset",
                "geocoded_lat": geocoded_lat, "geocoded_lng": geocoded_lng,
                "geocode_failed": False,
            })
            continue

        candidates = pois_gdf[tmask]

        lats  = candidates["lat"].values.astype(float)
        lngs  = candidates["lon"].values.astype(float)
        valid = ~(np.isnan(lats) | np.isnan(lngs))

        best_row     = None
        best_dist    = float("inf")
        match_method = None

        if valid.any():
            lats_v    = lats[valid]
            lngs_v    = lngs[valid]
            valid_idx = np.where(valid)[0]

            R    = 6_371_000.0
            phi1 = math.radians(geocoded_lat)
            phi2 = np.radians(lats_v)
            dphi = np.radians(lats_v - geocoded_lat)
            dlam = np.radians(lngs_v - geocoded_lng)
            a    = np.sin(dphi / 2) ** 2 + math.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
            dists = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

            name_lower = stop_name.lower()
            has_names  = "name" in candidates.columns and name_lower

            # Compute name_mask once — used by both Tier B and Tier C
            name_mask  = dists <= NAME_SEARCH_M
            name_idx   = valid_idx[name_mask]
            name_dists = dists[name_mask]

            # Tier A: coordinate match (within 50m, no name needed)
            coord_mask  = dists <= COORD_MATCH_M
            coord_idx   = valid_idx[coord_mask]
            coord_dists = dists[coord_mask]
            if coord_mask.any():
                pick      = int(np.argmin(coord_dists))
                best_row  = candidates.iloc[coord_idx[pick]]
                best_dist = float(coord_dists[pick])
                match_method = "coordinate_match"

            # Tier B: exact name within 1km
            if best_row is None and has_names and name_mask.any():
                ds_names  = candidates.iloc[name_idx]["name"].fillna("").astype(str).str.strip().str.lower()
                exact_hit = (ds_names == name_lower).values
                if exact_hit.any():
                    hit_dists = name_dists[exact_hit]
                    pick      = int(np.argmin(hit_dists))
                    best_row  = candidates.iloc[name_idx[exact_hit][pick]]
                    best_dist = float(hit_dists[pick])
                    match_method = "geocode_name"

            # Tier C: fuzzy name within 1km
            if best_row is None and has_names and name_mask.any():
                ds_names  = candidates.iloc[name_idx]["name"].fillna("").astype(str).str.strip().str.lower()
                fuzzy_hit = np.array([
                    bool(n) and (name_lower in n or n in name_lower)
                    for n in ds_names.tolist()
                ])
                if fuzzy_hit.any():
                    hit_dists = name_dists[fuzzy_hit]
                    pick      = int(np.argmin(hit_dists))
                    best_row  = candidates.iloc[name_idx[fuzzy_hit][pick]]
                    best_dist = float(hit_dists[pick])
                    match_method = "geocode_fuzzy"

            # Tier D: proximity only within threshold_m
            if best_row is None:
                prox_mask  = dists <= threshold_m
                prox_idx   = valid_idx[prox_mask]
                prox_dists = dists[prox_mask]
                if prox_mask.any():
                    pick      = int(np.argmin(prox_dists))
                    best_row  = candidates.iloc[prox_idx[pick]]
                    best_dist = float(prox_dists[pick])
                    match_method = "geocode_proximity"

        if best_row is not None:
            d_lat, d_lng = _row_lat_lng(best_row, pois_gdf)
            matched.append({
                "name":            stop_name,
                "address":         address,
                "type":            poi_type,
                "match_method":    match_method,
                "dataset_node":    str(best_row.get("nearest_node_osmid", "")),
                "dataset_name":    str(best_row.get("name", "")),
                "dataset_lat":     d_lat,
                "dataset_lng":     d_lng,
                "geocoded_lat":    geocoded_lat,
                "geocoded_lng":    geocoded_lng,
                "snap_distance_m": round(best_dist, 1),
                "geocode_failed":  False,
            })
        else:
            # Geocoded OK but no same-type POI within threshold_m.
            if tmask.any() and valid.any():
                nearest_dist = float(np.min(dists))
                drop_reason = (
                    f"geocoded to ({geocoded_lat:.5f}, {geocoded_lng:.5f}) but "
                    f"nearest '{poi_type}' is {nearest_dist:.0f}m away "
                    f"(threshold {threshold_m:.0f}m, name search {NAME_SEARCH_M:.0f}m)"
                )
            else:
                drop_reason = f"geocoded but no '{poi_type}' POIs found in dataset"
            dropped.append({
                "name": stop_name, "address": address, "type": poi_type,
                "drop_reason": drop_reason,
                "geocoded_lat": geocoded_lat, "geocoded_lng": geocoded_lng,
                "geocode_failed": False,
            })

    return matched, dropped


# Per-query pipeline


def run_query(
    query: Dict,
    api_key: str,
    base_url: str,
    model: str,
    pois_gdf,
    threshold_m: float,
    google_api_key: str,
    geocode_cache: Dict[str, Any],
    verbose: bool = True,
    extra_payload: Optional[Dict] = None,
) -> Dict:
    query_id = query["query_id"]
    nl_query = query["nl_query"]

    if verbose:
        print(f"\n[{query_id}] Calling LLM...")

    t0 = time.perf_counter()
    raw_text, usage = _call_llm(nl_query, api_key, base_url, model, extra_payload=extra_payload)
    latency_s = round(time.perf_counter() - t0, 3)

    llm_stops: List[Dict] = []
    llm_feasible: bool = True
    llm_infeasible_reason: Optional[str] = None

    try:
        parsed = _extract_json(raw_text)
        llm_feasible = bool(parsed.get("feasible", True))
        llm_stops = parsed.get("stops", []) if llm_feasible else []
        llm_infeasible_reason = None  # prompt no longer asks for a reason
    except Exception as e:
        print(f"[{query_id}] JSON parse failed: {e}")

    if verbose:
        if not llm_feasible:
            print(f"[{query_id}] LLM declared infeasible")
        else:
            names = ", ".join(
                f"{s.get('name', '?')} ({s.get('type', '?')})" for s in llm_stops
            )
            print(f"[{query_id}] LLM suggested {len(llm_stops)} stop(s): {names}")

    matched, dropped = match_stops(
        llm_stops=llm_stops,
        pois_gdf=pois_gdf,
        google_api_key=google_api_key,
        threshold_m=threshold_m,
        geocode_cache=geocode_cache,
        geocode_enabled=bool(google_api_key),
    )

    if verbose and llm_feasible:
        print(f"[{query_id}] Matched: {len(matched)}  Dropped: {len(dropped)}")
        for s in matched:
            dist_str = (
                f"{s['snap_distance_m']}m away"
                if s["snap_distance_m"] is not None
                else "name match"
            )
            print(
                f"  ✓ {s['name']} → node {s['dataset_node']}  ({dist_str})"
                f"  [{s['match_method']}]  [dataset: {s['dataset_name']}]"
            )
        for d in dropped:
            print(f"  ✗ {d['name']} — {d['drop_reason']}")

    return {
        "query_id": query_id,
        "nl_query": nl_query,
        "llm_feasible": llm_feasible,
        "llm_infeasible_reason": llm_infeasible_reason,
        "llm_stops": llm_stops,
        "matched_stops": matched,
        "dropped_stops": dropped,
        "classification": None,
        "model": model,
        "latency_s": latency_s,
        "tokens": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
        "geocode_bounds": _GEOCODE_BOUNDS,
        "geocode_threshold_m": threshold_m,
    }


#  ============================= Main =============================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bare LLM baseline — stop suggestion + 3-step dataset matching."
    )
    parser.add_argument(
        "--queries",
        required=True,
        help="Path to queries JSON file [{query_id, nl_query}, ...]",
    )
    parser.add_argument(
        "--gpkg", default=DEFAULT_GPKG, help="Path to road_network.gpkg"
    )
    parser.add_argument(
        "--geocode-threshold",
        type=float,
        default=DEFAULT_GEOCODE_THRESHOLD_M,
        help=f"Max distance in metres for geocode proximity match (default {DEFAULT_GEOCODE_THRESHOLD_M})",
    )
    parser.add_argument(
        "--out-dir", default=str(RESULTS_DIR), help="Directory to save result JSONs"
    )
    parser.add_argument(
        "--query-id", default=None, help="Run only this query_id from the file"
    )
    parser.add_argument(
        "--no-geocode", action="store_true", help="Disable geocoding (name match only)"
    )
    args = parser.parse_args()

    api_key = _read_env("GEMINI_API_KEY") or _read_env("OPENAI_API_KEY")
    base_url = (
        _read_env("GEMINI_BASE_URL")
        or _read_env("OPENAI_BASE_URL")
        or "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    model = _read_env("GEMINI_MODEL") or _read_env("OPENAI_MODEL") or "gemini-2.0-flash"
    google_api_key = (
        ""
        if args.no_geocode
        else (
            _read_env("GEOCODING_API_KEY")
            or _read_env("GOOGLE_MAPS_API_KEY")
            or _read_env("GOOGLE_API_KEY")
            or _read_env("key")
        )
    )

    if not api_key:
        sys.exit("No LLM API key found. Set GEMINI_API_KEY or OPENAI_API_KEY in .env")
    if not google_api_key and not args.no_geocode:
        print("Warning: GEOCODING_API_KEY not found in .env — geocoding disabled.")

    queries_path = Path(args.queries)
    print(f"Model     : {model}")
    print(f"GPKG      : {args.gpkg}")
    print(
        f"Geocode   : {'disabled' if args.no_geocode else f'{args.geocode_threshold}m threshold'}"
    )

    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    if args.query_id:
        queries = [q for q in queries if q["query_id"] == args.query_id]
        if not queries:
            sys.exit(f"query_id '{args.query_id}' not found in {args.queries}")

    print(f"Loading POIs from {args.gpkg}...")
    import geopandas as gpd

    pois_gdf = gpd.read_file(args.gpkg, layer="pois")
    print(f"  {len(pois_gdf)} POIs loaded.")

    geocode_cache = _load_geocode_cache(GEOCODE_CACHE_PATH)
    print(f"  Geocode cache: {len(geocode_cache)} entries loaded.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for query in queries:
        result = run_query(
            query=query,
            api_key=api_key,
            base_url=base_url,
            model=model,
            pois_gdf=pois_gdf,
            threshold_m=args.geocode_threshold,
            google_api_key=google_api_key,
            geocode_cache=geocode_cache,
        )
        all_results.append(result)

        out_path = out_dir / f"{result['query_id']}.json"
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  Saved → {out_path}")

        if google_api_key:
            _save_geocode_cache(geocode_cache, GEOCODE_CACHE_PATH)

    total_llm = sum(len(r["llm_stops"]) for r in all_results)
    total_matched = sum(len(r["matched_stops"]) for r in all_results)
    total_dropped = sum(len(r["dropped_stops"]) for r in all_results)
    total_infeasible = sum(1 for r in all_results if not r["llm_feasible"])
    print(
        f"\nDone. {len(all_results)} queries | "
        f"{total_llm} LLM stops | {total_matched} matched | {total_dropped} dropped | "
        f"{total_infeasible} declared infeasible"
    )


if __name__ == "__main__":
    main()
