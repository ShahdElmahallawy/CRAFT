"""
Scans a GPKG's pois layer and writes a schema.json describing which attribute
columns are usable as filters. Works on any OSM-style road network GPKG, not
just ours — point --gpkg at your own file (see the commented line below for
how this project's own SoCal dataset is wired in).

Each attribute column gets classified as:
  static_filter   - known before search, matched by substring (cuisine, wheelchair, ...)
  description     - free text, matched by embedding similarity
  state_dependent - depends on when the route arrives (opening_hours), parsed inline during search

Skips graph plumbing (lon/lat/nearest_node...), contact/wiki fields, near-empty columns,
and the POI type columns themselves (those go in poi_type_columns, not poi_attributes).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]

# --- Dataset location -------------------------------------------------------
# Enter the GPKG you want to build a schema for, or pass --gpkg on the CLI.
GPKG_PATH = None  # <-- enter your dataset's road_network.gpkg path here
# GPKG_PATH = ROOT / "data" / "socal" / "road_network.gpkg"  # uncomment to use this project's own SoCal dataset directly

DEFAULT_LAYER = "pois"
DEFAULT_MIN_NON_NULL = 200  

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

POI_TYPE_COLS = [
    "amenity", "tourism", "leisure", "shop", "historic", "natural",
    "man_made", "aeroway", "railway", "public_transport", "highway",
    "healthcare", "emergency", "sport", "office", "craft", "building",
]


def _find_project_root() -> Path:
    return ROOT


def _sample_values(series, cap: int = SAMPLE_CAP) -> Optional[List[str]]:
    """Return sorted distinct non-null values up to cap, or None if > cap."""
    vals = sorted(str(v) for v in series.dropna().unique() if str(v) not in ("nan", "none", "None", ""))
    if len(vals) > cap:
        return None  
    return vals


def build_poi_attributes(pois_gdf, min_non_null: int = DEFAULT_MIN_NON_NULL) -> Dict[str, Any]:
    """Build the poi_attributes section from real GPKG columns.

    min_non_null is an absolute row count, not a percentage, to keep this
    identical to the original SoCal build by default — raising it on a much
    smaller dataset would silently drop every column."""
    attrs: Dict[str, Any] = {}

    for col in STATIC_FILTER_COLS:
        if col not in pois_gdf.columns:
            continue
        non_null = int(pois_gdf[col].notna().sum())
        total = len(pois_gdf)
        if non_null < min_non_null:
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

    return attrs


def build_schema(
    gpkg_path: Path,
    out_path: Path,
    *,
    layer: str = DEFAULT_LAYER,
    dataset_name: Optional[str] = None,
    min_non_null: int = DEFAULT_MIN_NON_NULL,
) -> None:
    try:
        import geopandas as gpd
    except ImportError:
        sys.exit("geopandas not available — activate the conda env: conda run -n myenv python craft/build_schema.py")

    print(f"Reading {gpkg_path} (layer={layer}) ...")
    pois_gdf = gpd.read_file(str(gpkg_path), layer=layer)
    print(f"  {len(pois_gdf):,} POI rows, {len(pois_gdf.columns)} columns")

    poi_attributes = build_poi_attributes(pois_gdf, min_non_null=min_non_null)

    try:
        source = str(gpkg_path.relative_to(ROOT))
    except ValueError:
        source = str(gpkg_path)

    schema: Dict[str, Any] = {
        "dataset": dataset_name or gpkg_path.parent.name,
        "schema_version": "3",
        "source": source,
        "total_pois": len(pois_gdf),
        "poi_type_columns": POI_TYPE_COLS,
        "poi_attributes": poi_attributes,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpkg", type=Path, default=GPKG_PATH,
                        help="Path to road_network.gpkg (or uncomment GPKG_PATH above)")
    parser.add_argument("--layer", default=DEFAULT_LAYER,
                        help=f"GPKG layer containing POIs (default: {DEFAULT_LAYER})")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output schema.json path (default: <gpkg's folder>/schema.json)")
    parser.add_argument("--dataset-name", default=None,
                        help="Name stored in schema.json's 'dataset' field (default: gpkg's folder name)")
    parser.add_argument("--min-non-null", type=int, default=DEFAULT_MIN_NON_NULL,
                        help=f"Minimum non-null POIs to keep a column (default: {DEFAULT_MIN_NON_NULL}, "
                             "matches the original SoCal build — raise/lower for a smaller/larger dataset)")
    args = parser.parse_args()

    if args.gpkg is None:
        sys.exit("No GPKG given — pass --gpkg, or uncomment GPKG_PATH near the top of this file.")

    build_schema(
        args.gpkg,
        args.out or (args.gpkg.parent / "schema.json"),
        layer=args.layer,
        dataset_name=args.dataset_name,
        min_non_null=args.min_non_null,
    )
