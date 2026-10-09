"""
Parser agent — turns a natural-language routing query into structured JSON for the generator.

Two steps: the LLM extracts src/dst/objective/constraints from text, then plain
code resolves src/dst to OSM node IDs by coordinate snapping. The LLM never
touches node IDs; the resolver never touches language.

Output shape:
  {query_id, src: node_id|None, dst: node_id|None,
   objective: "minimize_travel_time|minimize_distance",
   constraints: {poi_types, ordered, time_budget_min, category_limits, ...}}

Usage:
  python craft/parser/agent.py --nl "Drive from X to Y, stop at a cafe..."
  python craft/parser/agent.py --input benchmark/socal_queries/set1/algo1_ev_routing.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
for _p in [str(ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from craft.generator.agent import LocalLLM
from craft.parser.tools import CatalogResolver, normalize_constraints_for_pipeline
from craft.generator.tools.common_handlers import (
    build_graph_from_gpkg,
    parse_lat_lon,
    nearest_node_to_coords,
)

PARSER_DIR = Path(__file__).resolve().parent
PROBLEM_PATH = PARSER_DIR / "prompt.md"
DEFAULT_GPKG = ROOT / "data" / "riverside" / "road_network.gpkg"
_SCHEMA_PATHS = [
    ROOT / "data" / "socal" / "schema.json",
]


def _load_dataset_schema() -> str:
    for schema_path in _SCHEMA_PATHS:
        if not schema_path.exists():
            continue
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        attrs = schema.get("poi_attributes", {})
        if attrs:
            return "Dataset POI attribute schema:\n" + json.dumps(attrs, indent=2)
    return ""


def build_instruction(problem_text: str, valid_categories: List[str]) -> str:
    schema_block = _load_dataset_schema()
    parts = [
        problem_text,
        f"Valid POI categories in this dataset:\n{', '.join(valid_categories)}",
    ]
    if schema_block:
        parts.append(schema_block)
    parts.append("Return only a valid JSON object. No explanation. No markdown.")
    return "\n\n".join(parts)


def extract_structured(
    query_id: str,
    nl_query: str,
    llm: LocalLLM,
    valid_categories: List[str],
    problem_text: str,
) -> Dict[str, Any]:
    """Call LLM to extract structured fields from the NL query."""
    instruction = build_instruction(problem_text, valid_categories)
    payload = {
        "query_id": query_id,
        "nl_query": nl_query,
    }
    result = llm.ask_json(instruction, payload)
    result["query_id"] = query_id  # ensure it's always set
    return result


def resolve_endpoint(
    name: str,
    G=None,
) -> Dict[str, Any]:
    """
    Resolve an endpoint to a node ID.

    Resolution order:
      1. Numeric node ID — use directly.
      2. "(lat, lon)" coordinate string — snap to nearest graph node.
      3. Any other format — fail (coordinate-only mode).

    Returns:
        {"node_id": str, "resolved_name": str, "score": float, "status": "resolved|ambiguous|failed"}
    """
    name_stripped = str(name).strip()

    # already a numeric node ID
    if name_stripped.lstrip("-").isdigit():
        return {"node_id": name_stripped, "resolved_name": name_stripped, "score": 100.0, "status": "resolved"}

    # coordinate string "(lat, lon)" -> nearest graph node
    coords = parse_lat_lon(name_stripped)
    if coords is not None:
        lat, lon = coords
        if G is not None:
            node = nearest_node_to_coords(G, lat, lon)
            if node is not None:
                return {"node_id": str(node), "resolved_name": f"({lat}, {lon})", "score": 100.0, "status": "resolved"}
        return {"node_id": None, "resolved_name": name_stripped, "score": 0.0, "status": "failed"}

    return {"node_id": None, "resolved_name": name, "score": 0.0, "status": "failed"}


# visit duration is fixed per POI category, matching LLMAP's hardcoded stay_time_mapping
# (minutes: shopping_mall 120, supermarket 30, pharmacy 15, bank 20, library 60) — not read
# from the query, so the two systems are compared fairly. Stored here in seconds.
LLMAP_VISIT_SEC = {
    "shopping_mall": 7200, "shopping mall": 7200,
    "supermarket": 1800, "pharmacy": 900, "bank": 1200, "library": 3600,
}


def parse(
    query_id: str,
    nl_query: str,
    *,
    llm: LocalLLM,
    resolver: CatalogResolver,
    problem_text: str,
    G=None,
) -> Dict[str, Any]:
    valid_categories = resolver.list_categories()

    # Step 1: LLM extraction.
    extracted = extract_structured(query_id, nl_query, llm, valid_categories, problem_text)

    src_raw = extracted.get("src", "")
    dst_raw = extracted.get("dst", "")

    # Step 2: Resolve src and dst.
    src_info = resolve_endpoint(src_raw, G=G)
    dst_info = resolve_endpoint(dst_raw, G=G)

    warnings = []
    if src_info["status"] != "resolved":
        warnings.append(f"src '{src_raw}' resolution status: {src_info['status']} (score={src_info['score']})")
    if dst_info["status"] != "resolved":
        warnings.append(f"dst '{dst_raw}' resolution status: {dst_info['status']} (score={dst_info['score']})")

    # Step 3: Assemble final structured input.
    objective = extracted.get("objective", "minimize_travel_time")
    result: Dict[str, Any] = {
        "query_id": query_id,
        "nl_query": nl_query,
        "src": src_info["node_id"],
        "dst": dst_info["node_id"],
        "objective": objective,
        "constraints": normalize_constraints_for_pipeline(extracted.get("constraints", {})),
        "_parse_meta": {
            "nl_query": nl_query,
            "src_raw": src_raw,
            "dst_raw": dst_raw,
            "src_resolution": src_info,
            "dst_resolution": dst_info,
            "warnings": warnings,
        },
    }

    # override any query-stated visit time with LLMAP's fixed per-category durations
    _c = result["constraints"]
    _types = _c.get("poi_types") or _c.get("stop_types") or []
    if _types:
        _c["visit_time"] = {str(t).lower(): LLMAP_VISIT_SEC.get(str(t).lower(), 900) for t in _types}

    # alternative: let the query's own stated visit time win (normalize_constraints_for_pipeline
    # already converted it to seconds under the same "visit_time" key) and only fall back to
    # LLMAP's fixed durations when the query never mentions one. Not fair-vs-LLMAP anymore, but
    # lets CRAFT actually honor a stated time like "each stop takes 25 minutes". To use this,
    # comment out the block above and uncomment the one below.
    # if _types and "visit_time" not in _c:
    #     _c["visit_time"] = {str(t).lower(): LLMAP_VISIT_SEC.get(str(t).lower(), 900) for t in _types}

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse NL routing query to structured JSON.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--nl", help="Natural language query string")
    group.add_argument("--input", help="Path to JSON file with {query_id, nl_query} or a benchmark query object")
    parser.add_argument("--query-id", default="query", help="Query ID (used with --nl)")
    parser.add_argument("--catalog", default=None, help="Path to catalog_pois.json (optional — gpkg_path alone already covers its categories)")
    parser.add_argument("--gpkg", default=str(DEFAULT_GPKG), help="Path to road_network.gpkg (primary source for the category list)")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.0-flash")))
    parser.add_argument("--output", default=None, help="Write result JSON to this path")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.input:
        raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
        if isinstance(raw, list):
            raw = raw[0]
        query_id = raw.get("query_id", raw.get("id", "query"))
        nl_query = raw.get("nl_query", raw.get("nl", ""))
    else:
        query_id = args.query_id
        nl_query = args.nl

    gpkg_path = args.gpkg if Path(args.gpkg).exists() else None
    resolver = CatalogResolver(args.catalog, gpkg_path=gpkg_path)
    print(f"Catalog loaded: {len(resolver.list_categories())} categories")
    G = build_graph_from_gpkg(gpkg_path) if gpkg_path else None
    if G is not None:
        print(f"Graph loaded: {G.number_of_nodes()} nodes")

    llm = LocalLLM(model=args.model, debug=args.debug)
    problem_text = PROBLEM_PATH.read_text(encoding="utf-8")

    print(f"Parsing: [{query_id}] {nl_query[:80]}...")
    result = parse(query_id, nl_query, llm=llm, resolver=resolver, problem_text=problem_text, G=G)

    meta = result.pop("_parse_meta")
    print(json.dumps(result, indent=2))
    if meta["warnings"]:
        print("\nWarnings:")
        for w in meta["warnings"]:
            print(f"  - {w}")
    if args.debug:
        print("\nResolution details:")
        print(json.dumps(meta, indent=2))

    if args.output:
        out = dict(result)
        out["_parse_meta"] = meta
        Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
