"""
build_schema.py — schema builder for craft_hipp.

Same as craft/build_schema.py with one addition: a numeric_filter section for
`rating` / `num_ratings`. Those are the synthetic quality columns the HIPP pilot
needs (seed 42), populated only for the five HIPP POI types — bank, library,
pharmacy, supermarket, shop=mall. Everything else (poi_attributes,
road_attributes, opening_hours as state_dependent) is the base logic unchanged.

Reads the GPKG pois/edges layers and writes data/socal/schema_hipp.json, a
separate file from data/socal/schema.json so craft/ keeps reading its own
schema untouched. craft_hipp/spec_builder/agent.py prefers schema_hipp.json and
falls back to the base one, so a dataset without ratings still runs.

Run:
  python craft_hipp/build_schema.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
GPKG_PATH = ROOT / "data" / "socal" / "road_network.gpkg"
SCHEMA_OUT = ROOT / "data" / "socal" / "schema_hipp.json"

SAMPLE_CAP = 25

STATIC_FILTER_COLS = [
    "name",
    "cuisine",
    "wheelchair",
    "outdoor_seating",
    "takeaway",
    "delivery",
    "drive_through",
    "internet_access",
    "smoking",
    "fee",
    "denomination",
    "religion",
    "attraction",
    "information",
    "museum",
    "artwork_type",
    "swimming_pool",
    "brand",
    "operator",
]

# HIPP-only addition: numeric attributes usable as a >= threshold filter.
# Populated ONLY for the 5 HIPP POI types (see enrich_synthetic_attrs.py) —
# non_null_pct below is computed over the WHOLE pois layer and will look low
# (~0.8%) even though coverage is 100% within bank/library/pharmacy/
# supermarket/shop=mall. The per-type coverage note documents this so the
# parser/generator LLMs don't misread the low global pct as "sparse data".
NUMERIC_FILTER_COLS = ["rating", "num_ratings"]

POI_TYPE_COLS = [
    "amenity", "tourism", "leisure", "shop", "historic", "natural",
    "man_made", "aeroway", "railway", "public_transport", "highway",
    "healthcare", "emergency", "sport", "office", "craft", "building",
]


def _find_project_root() -> Path:
    return ROOT


def _sample_values(series, cap: int = SAMPLE_CAP, split_multi: bool = False) -> Optional[List[str]]:
    """Return sorted distinct non-null values up to cap, or None if > cap.

    split_multi: OSM ';'-joined multi-values (e.g. "tertiary;residential")
    are split into their component tokens before dedup, since the compound
    strings themselves are not meaningful road classes.
    """
    raw = (str(v) for v in series.dropna().unique())
    if split_multi:
        tokens: set = set()
        for v in raw:
            tokens.update(v.split(";"))
        vals = sorted(t for t in tokens if t not in ("nan", "none", "None", ""))
    else:
        vals = sorted(v for v in raw if v not in ("nan", "none", "None", ""))
    if len(vals) > cap:
        return None
    return vals


def build_poi_attributes(pois_gdf) -> Dict[str, Any]:
    """Build the poi_attributes section from real GPKG columns."""
    attrs: Dict[str, Any] = {}

    for col in STATIC_FILTER_COLS:
        if col not in pois_gdf.columns:
            continue
        non_null = int(pois_gdf[col].notna().sum())
        total = len(pois_gdf)
        if non_null < 200:
            continue
        entry: Dict[str, Any] = {
            "kind": "static_filter",
            "match": "substring",
            "non_null": non_null,
            "non_null_pct": round(100.0 * non_null / total, 2),
        }
        svs = _sample_values(pois_gdf[col])
        if svs is None:
            entry["note"] = "high-cardinality"
        else:
            entry["sample_values"] = svs
        attrs[col] = entry

    if "description" in pois_gdf.columns:
        non_null = int(pois_gdf["description"].notna().sum())
        attrs["description"] = {
            "kind": "description",
            "non_null": non_null,
            "non_null_pct": round(100.0 * non_null / len(pois_gdf), 2),
        }

    if "opening_hours" in pois_gdf.columns:
        non_null = int(pois_gdf["opening_hours"].notna().sum())
        svs = _sample_values(pois_gdf["opening_hours"])
        entry = {
            "kind": "state_dependent",
            "non_null": non_null,
            "non_null_pct": round(100.0 * non_null / len(pois_gdf), 2),
            "format": "osm_opening_hours",
            "activation_field": "departure_time_sec",
            "parsing": "inline_in_generated_code",
            "note": (
                "Parsed inline in the generated expand() function. "
                "No pre-normalizer or lookup table. "
                "Supported formats: '24/7', 'Mo-Fr HH:MM-HH:MM', "
                "'Sa,Su HH:MM-HH:MM', ';'-separated rule chains. "
                "Anything else → treat as no hours data (POI always allowed). "
                "Check is time-of-day only (day-of-week ignored — documented approximation)."
            ),
        }
        if svs is None:
            entry["sample_note"] = "high-cardinality"
        else:
            entry["sample_values"] = svs
        attrs["opening_hours"] = entry

    total = len(pois_gdf)
    for col in NUMERIC_FILTER_COLS:
        if col not in pois_gdf.columns:
            continue
        series = pois_gdf[col].astype("float64", errors="ignore")
        non_null = int(pois_gdf[col].notna().sum())
        if non_null == 0:
            continue
        numeric = pois_gdf.loc[pois_gdf[col].notna(), col].astype(float)
        attrs[col] = {
            "kind": "numeric_filter",
            "match": ">=",
            "non_null": non_null,
            "non_null_pct": round(100.0 * non_null / total, 2),
            "min": round(float(numeric.min()), 2),
            "max": round(float(numeric.max()), 2),
            "note": (
                "SYNTHETIC (fabricated for evaluation, seed 42 — real OSM has "
                "almost no ratings). Populated ONLY for the 5 HIPP POI types "
                "(bank, library, pharmacy, supermarket, shop=mall) — 100% "
                "coverage within those types despite the low global non_null_pct "
                "here. Use as a >= threshold filter, e.g. rating >= 4.0."
            ),
        }

    return attrs


ROAD_ATTR_COLS = ["highway", "lanes", "maxspeed"]


def build_road_attributes(edges_gdf) -> Dict[str, Any]:
    """Build the road_attributes section from real GPKG edge columns.

    Mirrors build_poi_attributes but over the edges layer, so the parser and
    generator can both see real road-class values and turn a stated road-type
    preference (e.g. avoiding highways) into a structured constraint / a cost
    multiplier in expand(), instead of a hardcoded helper.
    """
    attrs: Dict[str, Any] = {}
    total = len(edges_gdf)
    for col in ROAD_ATTR_COLS:
        if col not in edges_gdf.columns:
            continue
        non_null = int(edges_gdf[col].notna().sum())
        if non_null < 200:
            continue
        entry: Dict[str, Any] = {
            "kind": "edge_attribute",
            "non_null": non_null,
            "non_null_pct": round(100.0 * non_null / total, 2),
        }
        svs = _sample_values(edges_gdf[col], split_multi=(col == "highway"))
        if svs is None:
            entry["note"] = "high-cardinality"
        else:
            entry["sample_values"] = svs
            if col == "highway":
                entry["note"] = (
                    "value may be a single class or ';'-joined multiple classes "
                    "(e.g. 'tertiary;residential') when OSM tagged conflicting "
                    "classes on the same way; check membership with 'in value.split(\";\")'."
                )
        attrs[col] = entry
    return attrs


def build_schema(gpkg_path: Path, out_path: Path) -> None:
    try:
        import geopandas as gpd
    except ImportError:
        sys.exit("geopandas not available — activate the conda env: conda run -n myenv python craft_hipp/build_schema.py")

    print(f"Reading {gpkg_path} ...")
    pois_gdf = gpd.read_file(str(gpkg_path), layer="pois")
    print(f"  {len(pois_gdf):,} POI rows, {len(pois_gdf.columns)} columns")
    edges_gdf = gpd.read_file(str(gpkg_path), layer="edges")
    print(f"  {len(edges_gdf):,} edge rows, {len(edges_gdf.columns)} columns")

    poi_attributes = build_poi_attributes(pois_gdf)
    road_attributes = build_road_attributes(edges_gdf)

    schema: Dict[str, Any] = {
        "dataset": "socal_hipp",
        "schema_version": "1",
        "source": str(gpkg_path.relative_to(ROOT)),
        "total_pois": len(pois_gdf),
        "poi_type_columns": POI_TYPE_COLS,
        "poi_attributes": poi_attributes,
        "road_attributes": road_attributes,
    }

    out_path.write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"  {len(poi_attributes)} filterable attributes")
    for col, entry in poi_attributes.items():
        kind = entry["kind"]
        nn = entry.get("non_null", "?")
        note = entry.get("note", entry.get("sample_note", ""))
        sv_count = len(entry.get("sample_values", [])) if "sample_values" in entry else "—"
        print(f"    {col:20s} {kind:16s}  non_null={nn:6}  vals={sv_count}  {note}")


if __name__ == "__main__":
    build_schema(GPKG_PATH, SCHEMA_OUT)
