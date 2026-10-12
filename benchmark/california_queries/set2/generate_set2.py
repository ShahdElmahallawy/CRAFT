"""
Generates Set 2: 200 ev_routing EV queries on the full California road network —
the long/moderate-distance counterpart to Set 1, same templates and schema,
just longer trips (moderate 15-75mi x100, long 75-300mi x100).

60% of queries (120) are built so the trip can't be completed on the starting
charge and must pass a charger; the rest get a comfortable margin. Samples
node pairs directly from the California gpkg (not the coarse 116-node catalog
grid). Deterministic, seed=42.

Usage:
  python benchmark/california_queries/set2/generate_set2.py
"""

import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

OUT_DIR       = ROOT / "benchmark" / "california_queries" / "set2"
TEMPLATES_DIR = ROOT / "benchmark" / "templates"
DATA_DIR      = ROOT / "data" / "california"
GPKG_PATH     = DATA_DIR / "california_road_network.gpkg"

SEED = 42

MAX_RANGE_BUCKETS_MI = [150, 200, 250, 300, 350]

# (group, lo_mi, hi_mi, count)
GROUPS = [
    ("moderate", 15,  75, 100),
    ("long",     75, 300, 100),
]

TARGET_NC_TRUE_FRAC = 0.6

MAX_ATTEMPTS = 500_000


def _load_base_module():
    """Reuse the dataset-agnostic helpers from the original hand-authored
    California set2 generator (read-only sibling import, same pattern as
    scripts/run_llmap.py loading LLMAP's HIPP.py)."""
    spec = importlib.util.spec_from_file_location(
        "generate_set2_california",
        ROOT / "benchmark" / "california_queries" / "set2_long_trips" / "generate_set2_california.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = _load_base_module()
haversine_km         = base.haversine_km
chargers_in_corridor = base.chargers_in_corridor
parse_templates      = base.parse_templates
fill                 = base.fill
KM_TO_MI             = base.KM_TO_MI
MI_TO_M              = base.MI_TO_M
ROAD_FACTOR          = base.ROAD_FACTOR


def load_nodes():
    gdf = gpd.read_file(GPKG_PATH, layer="nodes")
    return gdf["lat"].to_numpy(), gdf["lon"].to_numpy()


def compute_battery(trip_mi, max_range_mi, needs_charger, rng):
    """Picks a starting charge: needs_charger=True -> cur_mi in (trip*0.3, road_mi)
    (must stop to charge); False -> cur_mi >= trip*1.5 (comfortable margin).
    Returns (cur_mi_rounded, battery_pct)."""
    road_mi = trip_mi * ROAD_FACTOR
    false_thresh = trip_mi * 1.5

    if needs_charger:
        if max_range_mi <= road_mi:
            pct = rng.randint(50, 90)
            # Floor so the car can still reach a charger (cur > trip*0.3);
            # a small max_range on a long trip can otherwise fall under it.
            pct = max(pct, math.ceil(trip_mi * 0.31 / max_range_mi * 100))
            if pct > 95:
                raise ValueError("cannot satisfy charger-reachability floor at this range")
            cur_mi = max_range_mi * pct / 100
        else:
            cur_mi = rng.uniform(trip_mi * 0.35, road_mi * 0.95)
            pct = max(1, min(99, round(cur_mi / max_range_mi * 100)))
            cur_mi = max_range_mi * pct / 100
        assert cur_mi < road_mi,        f"NC=True but {cur_mi:.1f} >= {road_mi:.1f}"
        assert cur_mi > trip_mi * 0.29, f"NC=True but cur too low: {cur_mi:.1f}"
    else:
        if max_range_mi < false_thresh:
            raise ValueError("NC=False infeasible at this range/trip")
        target_mi = min(max(false_thresh, max_range_mi * 0.82), max_range_mi)
        pct = max(60, min(100, math.ceil(target_mi / max_range_mi * 100)))
        cur_mi = max_range_mi * pct / 100
        assert cur_mi >= false_thresh, f"NC=False but {cur_mi:.1f} < {false_thresh:.1f}"

    return round(cur_mi), pct


def pick_max_range(rng, trip_mi, needs_charger):
    ranges = MAX_RANGE_BUCKETS_MI[:]
    rng.shuffle(ranges)
    if needs_charger:
        return rng.choice(ranges)
    capable = [m for m in ranges if m >= trip_mi * 1.5]
    return rng.choice(capable) if capable else max(ranges)


def sample_group(rng, lats, lons, charging, lo, hi, n_true, n_false):
    """Rejection-sample distinct node pairs in [lo, hi) mi: n_true requiring
    a corridor charger, n_false without that requirement."""
    n_nodes = len(lats)
    used = set()
    out = []

    for want_true, target in ((True, n_true), (False, n_false)):
        made = 0
        attempts = 0
        while made < target and attempts < MAX_ATTEMPTS:
            attempts += 1
            i = rng.randrange(n_nodes)
            j = rng.randrange(n_nodes)
            if i == j:
                continue
            key = (min(i, j), max(i, j))
            if key in used:
                continue

            sla, slo = float(lats[i]), float(lons[i])
            dla, dlo = float(lats[j]), float(lons[j])
            trip_mi = haversine_km(sla, slo, dla, dlo) * KM_TO_MI
            if not (lo <= trip_mi < hi):
                continue

            # A no-stop trip needs range >= trip*1.5, so it is physically
            # impossible past ~233 mi given the 350 mi top of the range
            # catalog. Such long trips can only be needs_charger=True.
            if not want_true and trip_mi * 1.5 > max(MAX_RANGE_BUCKETS_MI):
                continue

            n_chg = len(chargers_in_corridor(charging, sla, slo, dla, dlo))
            if want_true and n_chg < 1:
                continue

            used.add(key)
            out.append({
                "src_latlon": (sla, slo),
                "dst_latlon": (dla, dlo),
                "trip_mi": trip_mi,
                "needs_charger": want_true,
                "chargers_in_corridor": n_chg,
            })
            made += 1

        if made < target:
            raise RuntimeError(
                f"band {lo}-{hi} mi: only sampled {made}/{target} with needs_charger={want_true}")

    return out


def main():
    rng = random.Random(SEED)

    print("Loading California nodes layer from gpkg...")
    lats, lons = load_nodes()
    print(f"California road-network nodes: {len(lats)}")

    charging = json.loads((DATA_DIR / "catalog_charging.json").read_text())
    tmpl = parse_templates(TEMPLATES_DIR / "ev_routing.txt")
    tmpl_indices = sorted(tmpl.keys())

    picks = []
    for group, lo, hi, count in GROUPS:
        n_true = round(count * TARGET_NC_TRUE_FRAC)
        n_false = count - n_true
        print(f"sampling {group:<9} {lo}-{hi} mi: {n_true} needing a charger + {n_false} without")
        for p in sample_group(rng, lats, lons, charging, lo, hi, n_true, n_false):
            p["group"] = group
            picks.append(p)

    # Interleave groups/flags so query_id order isn't correlated with either.
    rng.shuffle(picks)

    records = []
    meta = []
    for idx, p in enumerate(picks):
        trip_mi = p["trip_mi"]
        nc = p["needs_charger"]
        for _ in range(len(MAX_RANGE_BUCKETS_MI) * 4):
            max_mi = pick_max_range(rng, trip_mi, nc)
            try:
                cur_mi, pct = compute_battery(trip_mi, max_mi, nc, rng)
                break
            except ValueError:
                continue
        else:
            raise RuntimeError(f"no feasible battery config for trip {trip_mi:.1f} mi, nc={nc}")

        tmpl_text = tmpl[tmpl_indices[idx % len(tmpl_indices)]]
        nl = fill(
            tmpl_text,
            SRC_NODE=f"({p['src_latlon'][0]}, {p['src_latlon'][1]})",
            DST_NODE=f"({p['dst_latlon'][0]}, {p['dst_latlon'][1]})",
            CURRENT_BATTERY_PCT=f"{pct}%",
            CURRENT_BATTERY_MI=str(cur_mi),
            MAX_BATTERY_MI=str(max_mi),
        )
        assert "{" not in nl and "}" not in nl, f"unfilled placeholder: {nl}"

        qid = f"set2_ev_routing_{idx + 1:03d}"
        records.append({"query_id": qid, "nl_query": nl})
        meta.append({
            "query_id":             qid,
            "group":                p["group"],
            "src_latlon":           list(p["src_latlon"]),
            "dst_latlon":           list(p["dst_latlon"]),
            "trip_dist_mi":         round(trip_mi, 2),
            "current_battery_mi":   cur_mi,
            "max_battery_mi":       max_mi,
            "battery_pct":          pct,
            "needs_charger":        nc,
            "chargers_in_corridor": p["chargers_in_corridor"],
        })

    assert len(records) == 200, f"expected 200 records, got {len(records)}"
    ids = [r["query_id"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate query_id"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ev_routing.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False))
    (OUT_DIR / "set2_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))

    nc_true = sum(1 for m in meta if m["needs_charger"])
    group_counts = {g: sum(1 for m in meta if m["group"] == g) for g, *_ in GROUPS}
    dists = [m["trip_dist_mi"] for m in meta]
    summary = {
        "dataset":                 "california",
        "set":                     2,
        "template":                "ev_routing",
        "total":                   len(records),
        "group_counts":            group_counts,
        "group_ranges_mi":         {g: [lo, hi] for g, lo, hi, _ in GROUPS},
        "needs_charger_true":      nc_true,
        "needs_charger_true_frac": round(nc_true / len(records), 3),
        "trip_dist_min_mi":        min(dists),
        "trip_dist_max_mi":        max(dists),
        "seed":                    SEED,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nTotal: {len(records)}  {group_counts}")
    print(f"needs_charger=True: {nc_true}/{len(records)} ({summary['needs_charger_true_frac']:.1%})")
    print(f"Trip distance: {min(dists)}-{max(dists)} mi")
    print(f"\nWrote {OUT_DIR / 'ev_routing.json'}")
    print(f"Wrote {OUT_DIR / 'set2_meta.json'}")
    print(f"Wrote {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
