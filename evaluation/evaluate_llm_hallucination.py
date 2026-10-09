"""
evaluate_llm_hallucination.py — Step 2: Hallucination detection for LLM baseline outputs.

Pipeline for each dropped stop (matched stops are already confirmed real):
  Step 1: Geocode the address         (Google Geocoding API)
  Step 2: Google Places Find Place    (name + address, 200 m location bias)
  Step 3: Type check                  (does Google's returned type match the claimed type?)

hallucination_type values:
  "address_not_found"  — geocoding failed (address doesn't exist)
  "place_not_found"    — address geocodes but Google can't find the named business there
  "type_mismatch"      — place found and nearby but category is wrong
  null                 — real place, correct type (not hallucinated, just not in our OSM dataset)

A query is hallucinated if any of its stops is hallucinated.

Run order:
  1. python scripts/run_llm_baseline_on_llmap.py    (Step 1: LLM call + name matching)
  2. python evaluation/evaluate_llm_hallucination.py (this script — Step 2)
  3. python evaluation/evaluate_llm_baseline.py      (Steps 3-5: constraints + comparison)

Usage:
  conda run -n myenv python evaluation/evaluate_llm_hallucination.py
  conda run -n myenv python evaluation/evaluate_llm_hallucination.py \\
      --templates ordered_poi --limit 5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_BASELINE_DIR = ROOT / "outputs" / "llm"
DEFAULT_QUERIES_DIR  = ROOT / "benchmark" / "socal_queries" / "set1"
OUT_DIR              = ROOT / "outputs" / "evaluation" / "llm_hallucination"

GEOCODE_URL   = "https://maps.googleapis.com/maps/api/geocode/json"
FINDPLACE_URL = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"

ALL_TEMPLATES = [
    "ev_routing",
    "ordered_poi",
    "poi_collection",
    "ev+ordered_poi",
    "ev+time",
    "ordered+time_budget",
    "unordered_poi",
]

ALL_MODELS = [
    "gemini-3_1-flash-lite",
    "gemini-3_5-flash",
    "gemini-3_1-pro-preview",
    "Qwen_Qwen3-32B-AWQ",
]

# Maps our POI categories to Google Places type strings for the type-match check
CATEGORY_TO_GOOGLE_TYPES: Dict[str, List[str]] = {
    "pharmacy":         ["pharmacy"],
    "bank":             ["bank", "finance"],
    "supermarket":      ["supermarket", "grocery_or_supermarket"],
    "library":          ["library"],
    "mall":             ["shopping_mall"],
    "shopping_mall":    ["shopping_mall"],
    "charging_station": ["electric_vehicle_charging_station"],
}

# If the Place result is further than this from the geocoded point, treat as not found
# (location bias is soft — API can return a far-away chain branch)
MAX_PLACE_DISTANCE_M = 500

HALL_KEYS = ["address_not_found", "place_not_found", "wrong_location", "type_mismatch"]


# Env

def _read_env(key: str) -> Optional[str]:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == key:
            return v.strip() or None
    return None


# Geometry

def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# Google API calls

def _geocode(address: str, api_key: str, cache: Dict) -> Optional[Tuple[float, float]]:
    if address in cache:
        v = cache[address]
        return tuple(v) if v else None
    params = {"address": address, "key": api_key}
    url = GEOCODE_URL + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read().decode())
    except Exception as exc:
        print(f"    [geocode error] {exc}")
        cache[address] = None
        return None
    results = data.get("results", [])
    if not results:
        cache[address] = None
        return None
    loc = results[0]["geometry"]["location"]
    result = [loc["lat"], loc["lng"]]
    cache[address] = result
    return tuple(result)


def _find_place(
    name: str,
    address: str,
    lat: Optional[float],
    lng: Optional[float],
    api_key: str,
    cache: Dict,
    radius_m: Optional[int] = 200,
) -> Optional[Dict]:
    """
    Google Places Find Place from Text.
    If radius_m and lat/lng are provided, applies a soft location bias of radius_m metres.
    If radius_m is None (or lat/lng are None), searches globally (no location bias).
    Returns the first candidate dict, or None if nothing found or API error.
    """
    cache_key = f"{name}||{lat:.5f},{lng:.5f}||r{radius_m}" if (lat is not None and lng is not None) \
                else f"{name}||global"
    if cache_key in cache:
        return cache[cache_key]

    params = {
        "input":     f"{name}, {address}",
        "inputtype": "textquery",
        "fields":    "name,geometry,types,formatted_address",
        "key":       api_key,
    }
    if radius_m is not None and lat is not None and lng is not None:
        params["locationbias"] = f"circle:{radius_m}@{lat},{lng}"
    url = FINDPLACE_URL + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            data = json.loads(r.read().decode())
    except Exception as exc:
        print(f"    [places error] {exc}")
        return None

    status = data.get("status", "")
    if status == "ZERO_RESULTS":
        cache[cache_key] = None
        return None
    if status != "OK":
        print(f"    [places status={status}]")
        return None

    candidates = data.get("candidates", [])
    result = candidates[0] if candidates else None
    cache[cache_key] = result
    return result


def _type_matches(google_types: List[str], claimed_type: str) -> bool:
    expected = CATEGORY_TO_GOOGLE_TYPES.get(claimed_type.lower().strip(), [])
    return bool(expected) and any(t in google_types for t in expected)


# Per-stop hallucination check

def check_stop(
    stop: Dict,
    geocode_key: str,
    places_key: str,
    geocode_cache: Dict,
    places_cache: Dict,
    places_radius_m: int = 200,
) -> Dict:
    """
    3-step hallucination pipeline for a single dropped stop.

    Step 1 — Geocode the address.
    Step 2 — Google Places Find Place: verify the named business exists nearby.
    Step 3 — Type check: does Google's returned category match the claimed type?
    """
    name         = stop.get("name", "")
    address      = stop.get("address", "")
    claimed_type = stop.get("type", "")

    out: Dict[str, Any] = {
        "name":              name,
        "address":           address,
        "claimed_type":      claimed_type,
        "geocoded_lat":      None,
        "geocoded_lng":      None,
        "places_found":      False,
        "places_name":       None,
        "places_types":      None,
        "places_lat":        None,
        "places_lng":        None,
        "places_distance_m": None,
        "type_match":        None,
        "hallucination_type": "address_not_found",
    }

    # Step 1: Geocode
    geo = _geocode(address, geocode_key, geocode_cache)
    if geo is None:
        return out  # address_not_found

    glat, glng = geo
    out["geocoded_lat"] = glat
    out["geocoded_lng"] = glng

    # Step 2: Google Places Find Place
    candidate = _find_place(name, address, glat, glng, places_key, places_cache, places_radius_m)

    if candidate is None:
        # Second pass: wider global search to distinguish fabricated names from wrong locations.
        wide = _find_place(name, address, glat, glng, places_key, places_cache, radius_m=None)
        out["hallucination_type"] = "wrong_location" if wide is not None else "place_not_found"
        return out

    p_name  = candidate.get("name", "")
    p_types = candidate.get("types", [])
    p_loc   = candidate.get("geometry", {}).get("location", {})
    p_lat   = p_loc.get("lat")
    p_lng   = p_loc.get("lng")

    out["places_found"] = True
    out["places_name"]  = p_name
    out["places_types"] = p_types

    if p_lat is not None and p_lng is not None:
        out["places_lat"] = p_lat
        out["places_lng"] = p_lng
        dist_m = _haversine_m(glat, glng, p_lat, p_lng)
        out["places_distance_m"] = round(dist_m, 1)
        # If the returned place is too far, Google found a distant branch — real business, wrong address
        if dist_m > MAX_PLACE_DISTANCE_M:
            out["hallucination_type"] = "wrong_location"
            return out

    # Step 3: Type check
    type_ok = _type_matches(p_types, claimed_type)
    out["type_match"] = type_ok

    if not type_ok:
        out["hallucination_type"] = "type_mismatch"
    else:
        out["hallucination_type"] = None  # real place, correct type

    return out


# Query metadata loader

def load_query_map(queries_dir: Path) -> Dict[str, Dict]:
    qmap: Dict[str, Dict] = {}
    for f in sorted(queries_dir.glob("*.json")):
        records = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(records, dict):
            records = [records]
        for q in records:
            if isinstance(q, dict) and "query_id" in q:
                qmap[q["query_id"]] = q
    return qmap


# Cache helpers

def _load_cache(path: Path) -> Dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cache(cache: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


# Main

def main() -> None:
    ap = argparse.ArgumentParser(description="LLM baseline hallucination detection.")
    ap.add_argument("--baseline-dir", default=str(DEFAULT_BASELINE_DIR),
                    help="Root dir containing {model}/{template}/*.json (default: outputs/llm)")
    ap.add_argument("--queries-dir",  default=str(DEFAULT_QUERIES_DIR),
                    help="Dir containing template JSON files (default: socal set1)")
    ap.add_argument("--models",    nargs="+", default=ALL_MODELS, metavar="MODEL")
    ap.add_argument("--templates", nargs="+", default=ALL_TEMPLATES, metavar="TEMPLATE")
    ap.add_argument("--limit",  type=int, default=None,
                    help="Max queries per template per model (smoke test)")
    ap.add_argument("--radius", type=int, default=200,
                    help="Google Places Find Place location bias radius in metres (default: 200)")
    args = ap.parse_args()

    baseline_dir = Path(args.baseline_dir)
    queries_dir  = Path(args.queries_dir)

    geocode_key = _read_env("GEOCODING_API_KEY")
    places_key  = _read_env("PLACES_API_KEY")
    if not geocode_key:
        sys.exit("GEOCODING_API_KEY not found in .env")
    if not places_key:
        sys.exit("PLACES_API_KEY not found in .env")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    eval_dir           = ROOT / "outputs" / "evaluation"
    geocode_cache_path = eval_dir / "geocode_cache.json"
    places_cache_path  = eval_dir / "places_cache.json"
    geocode_cache      = _load_cache(geocode_cache_path)
    places_cache       = _load_cache(places_cache_path)

    print(f"Run ID       : {run_id}")
    print(f"Baseline dir : {baseline_dir}")
    print(f"Models       : {args.models}")
    print(f"Templates    : {args.templates}")
    print(f"Places radius: {args.radius} m")
    print(f"Geocode cache: {len(geocode_cache)} entries")
    print(f"Places cache : {len(places_cache)} entries")

    print("\nLoading query metadata ...")
    query_map = load_query_map(queries_dir)
    print(f"  {len(query_map)} queries loaded")

    HALL_STAT_KEYS = ["total", "n_matched_stops", "n_dropped_stops"] + HALL_KEYS
    grand_total_records = 0

    for model in args.models:
        model_dir = baseline_dir / model
        if not model_dir.exists():
            print(f"\n[skip] {model} — no output dir")
            continue

        out_dir = OUT_DIR / model / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        zero_stat = lambda: {k: 0 for k in HALL_STAT_KEYS}
        tmpl_stats: Dict[str, Dict] = defaultdict(zero_stat)
        all_records: List[Dict] = []

        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        for template in args.templates:
            tmpl_dir = model_dir / template
            if not tmpl_dir.exists():
                print(f"\n[skip] {template} — not found for {model}")
                continue

            files = sorted(tmpl_dir.glob("*.json"))
            if args.limit:
                files = files[: args.limit]

            print(f"\n── {template}: {len(files)} queries ──")

            for fi, path in enumerate(files, 1):
                data     = json.loads(path.read_text(encoding="utf-8"))
                query_id = data["query_id"]
                matched  = data.get("matched_stops", [])
                dropped  = data.get("dropped_stops", [])

                matched_enriched = [
                    {**s, "hallucination_type": None,
                     "verification_method": s.get("match_method", "geocode_proximity")}
                    for s in matched
                ]

                dropped_enriched: List[Dict] = []
                for stop in dropped:
                    checked = check_stop(
                        stop, geocode_key, places_key,
                        geocode_cache, places_cache,
                        places_radius_m=args.radius,
                    )
                    dropped_enriched.append(checked)

                    h = checked["hallucination_type"]
                    s = tmpl_stats[template]
                    if   h == "address_not_found": s["address_not_found"] += 1
                    elif h == "place_not_found":   s["place_not_found"]   += 1
                    elif h == "wrong_location":    s["wrong_location"]    += 1
                    elif h == "type_mismatch":     s["type_mismatch"]     += 1

                s = tmpl_stats[template]
                s["total"]           += 1
                s["n_matched_stops"] += len(matched)
                s["n_dropped_stops"] += len(dropped)

                n_hall = sum(1 for d in dropped_enriched if d["hallucination_type"] is not None)
                print(f"  [{fi}/{len(files)}] {query_id}: "
                      f"{len(matched)} matched  {len(dropped)} dropped  "
                      f"{n_hall} hallucinated", flush=True)

                record: Dict[str, Any] = {
                    "query_id":         query_id,
                    "model":            model,
                    "template":         template,
                    "nl_query":         data.get("nl_query", ""),
                    "llm_feasible":     data.get("llm_feasible", True),
                    "llm_stops":        data.get("llm_stops", []),
                    "matched_stops":    matched_enriched,
                    "dropped_stops":    dropped_enriched,
                    "n_stops":          len(matched) + len(dropped),
                    "n_matched":        len(matched),
                    "n_dropped":        len(dropped),
                    "n_hallucinated":   n_hall,
                    "any_hallucinated": n_hall > 0,
                }
                all_records.append(record)
                (out_dir / f"{query_id}.json").write_text(
                    json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
                )

                if len(all_records) % 20 == 0:
                    _save_cache(geocode_cache, geocode_cache_path)
                    _save_cache(places_cache, places_cache_path)

        grand_total_records += len(all_records)

        # Per-model summary
        W = 10
        header = (
            f"{'Template':<30} {'Queries':>{W}} {'Matched':>{W}} {'Dropped':>{W}} "
            f"{'NoAddr':>{W}} {'NoPlace':>{W}} {'WrongLoc':>{W}} {'TypeMis':>{W}} {'RealNotDB':>{W}}"
        )
        sep = "=" * len(header)
        print(f"\n{sep}")
        print(header)
        print(sep.replace("=", "-"))
        for tmpl in args.templates:
            s = tmpl_stats.get(tmpl)
            if not s or s["total"] == 0:
                continue
            n_real = (s["n_dropped_stops"] - s["address_not_found"] - s["place_not_found"]
                      - s["wrong_location"] - s["type_mismatch"])
            print(
                f"{tmpl:<30} {s['total']:>{W}} {s['n_matched_stops']:>{W}} {s['n_dropped_stops']:>{W}} "
                f"{s['address_not_found']:>{W}} {s['place_not_found']:>{W}} {s['wrong_location']:>{W}} "
                f"{s['type_mismatch']:>{W}} {n_real:>{W}}"
            )
        totals = {k: sum(tmpl_stats[t].get(k, 0) for t in tmpl_stats) for k in HALL_STAT_KEYS}
        n_real_total = (totals["n_dropped_stops"] - totals["address_not_found"]
                        - totals["place_not_found"] - totals["wrong_location"] - totals["type_mismatch"])
        print(sep.replace("=", "-"))
        print(
            f"{'TOTAL':<30} {totals['total']:>{W}} {totals['n_matched_stops']:>{W}} {totals['n_dropped_stops']:>{W}} "
            f"{totals['address_not_found']:>{W}} {totals['place_not_found']:>{W}} {totals['wrong_location']:>{W}} "
            f"{totals['type_mismatch']:>{W}} {n_real_total:>{W}}"
        )
        print(sep)
        n_any_hall = sum(1 for r in all_records if r["any_hallucinated"])
        print(f"Queries with ≥1 hallucinated stop: {n_any_hall} / {len(all_records)}")
        print(f"Outputs: {out_dir}")

        summary = {
            "run_id": run_id, "model": model, "places_radius_m": args.radius,
            "n_queries": len(all_records), "n_any_hallucinated": n_any_hall,
            "by_template": {t: dict(s) for t, s in tmpl_stats.items()},
            "totals": totals,
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    _save_cache(geocode_cache, geocode_cache_path)
    _save_cache(places_cache, places_cache_path)
    print(f"\nDone. Total records: {grand_total_records}")
    print(f"Caches saved — geocode: {len(geocode_cache)}, places: {len(places_cache)}")


if __name__ == "__main__":
    main()
