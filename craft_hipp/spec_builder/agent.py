"""
Spec Builder (paper Section 5) — turns a natural-language routing query into
the structured query the ORCStrator consumes.

Two steps: the LLM extracts src/dst/objective/constraints from text, then plain
code resolves src/dst to OSM node IDs by coordinate snapping. The LLM never
touches node IDs; the resolver never touches language.

Output shape:
  {query_id, src: node_id|None, dst: node_id|None,
   objective: "minimize_travel_time|minimize_distance",
   constraints: {poi_types, ordered, time_budget_min, category_limits, ...}}

Usage:
  python craft/spec_builder/agent.py --nl "Drive from X to Y, stop at a cafe..."
  python craft/spec_builder/agent.py --input benchmark/socal_queries/set1/algo1_ev_routing.json
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

from craft_hipp.orcstrator.agent import LocalLLM
from craft_hipp.spec_builder.tools import CatalogResolver, normalize_constraints_for_pipeline
from craft_hipp.orcstrator.tools.common_handlers import (
    build_graph_from_gpkg,
    parse_lat_lon,
    nearest_node_to_coords,
)

PARSER_DIR = Path(__file__).resolve().parent
PROBLEM_PATH = PARSER_DIR / "prompt.md"
DEFAULT_GPKG = ROOT / "data" / "riverside" / "road_network.gpkg"
_SCHEMA_PATHS = [
    # HIPP needs its own schema: it adds the numeric rating / num_ratings
    # attributes on top of the base one (data/socal/build_schema_hipp.py).
    # Falls back to the base schema so this still runs on a dataset without it.
    ROOT / "data" / "socal" / "schema_hipp.json",
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


_RATING_SCHEMA_NOTE = (
    "If the query emphasizes wanting well-rated / high-quality / \"best\" places "
    "over route efficiency (e.g. \"focus on places with higher ratings\", "
    "\"prioritize high-rated places\"), emit a numeric rating filter aligned to "
    "poi_types: \"rating_min\": [<threshold>, ...] — one entry per poi_types "
    "element, same order. Use threshold 4.0 unless the query implies otherwise. "
    "This is a hard filter (candidates below the threshold are excluded), not a "
    "soft preference. Only emit it when quality is clearly emphasized over "
    "efficiency in the query language — omit it entirely for efficiency-focused "
    "or balanced queries."
)

_DEADLINE_NOTE = (
    "If the query states an absolute return/arrival time (e.g. \"be home by "
    "17:00\", \"return by 8pm\"), emit both fields below. Departure is always "
    "assumed to be 10:00 AM (36000 seconds after midnight) — do not change this. "
    "Convert the stated deadline clock time to seconds since midnight "
    "(e.g. \"17:00\" -> 61200):\n"
    '  "departure_time_sec": {"value": 36000, "unit": "sec"},\n'
    '  "arrival_deadline_sec": {"value": <deadline_seconds>, "unit": "sec"}\n'
    "Omit both fields entirely if no deadline is stated."
)


def _load_rating_schema() -> str:
    """The rating attribute on its own, plus how to turn it into a filter.

    Separate from _load_dataset_schema because rating is a numeric_filter rather
    than a value to match, and the model needs to be told that explicitly.
    """
    for schema_path in _SCHEMA_PATHS:
        if not schema_path.exists():
            continue
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rating_entry = schema.get("poi_attributes", {}).get("rating")
        if rating_entry:
            return (
                "Dataset numeric POI attribute schema (rating):\n"
                + json.dumps({"rating": rating_entry}, indent=2)
                + "\n\n" + _RATING_SCHEMA_NOTE
            )
    return ""


def build_instruction(problem_text: str, valid_categories: List[str]) -> str:
    schema_block = _load_dataset_schema()
    rating_block = _load_rating_schema()
    parts = [
        problem_text,
        f"Valid POI categories in this dataset:\n{', '.join(valid_categories)}",
    ]
    if schema_block:
        parts.append(schema_block)
    if rating_block:
        parts.append(rating_block)
    parts.append(_DEADLINE_NOTE)
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


# The dwell table lives in one place — orcstrator/tools/common_handlers.py, the
# same module the solver reads it from — so generation and the spec can never
# disagree about how long a stop takes. ("mall" is HIPP's own name for
# shopping_mall, and is in that table.)
from craft_hipp.orcstrator.tools.common_handlers import (
    LLMAP_VISIT_SEC,
    LLMAP_VISIT_DEFAULT_SEC,
)

# HIPP gold injection
#
# HIPP queries carry their structure as ground truth — poi types, dependency
# ordering, clock deadline, quality/distance weights — in
# benchmark/socal_queries/llmap_queries/hipp_meta.json. When a caller passes
# `hipp_gold` (that record, joined by query_id), parse() overwrites those
# structural fields after the LLM call, so a CRAFT-vs-LLMAP routing comparison
# is not contaminated by spec-builder error on fields we already know. The LLM's
# own extraction is kept under _parse_meta (raw_llm_objective /
# raw_llm_constraints) so spec fidelity can still be measured on its own.

_MALL_ALIASES = {"shopping_mall", "shopping mall"}

# Default "high-rated" floor. The synthetic ratings are ~U(3.0, 5.0), so 4.0
# keeps roughly the top half. A constant so a different rule (a per-query top
# third, say) can replace it without touching call sites.
HIPP_RATING_THRESHOLD = 4.0

# HIPP's assumed departure, in seconds since midnight. Matches LLMAP's own
# hardcoded `departure_time = 10` in baselines/LLMAP/main/utils.py.
HIPP_DEPARTURE_SEC = 10 * 3600


def _remap_mall_alias(cat: str) -> str:
    c = str(cat).strip().lower()
    return "mall" if c in _MALL_ALIASES else c


def _hipp_deadline_to_sec(time_limit: Any) -> Optional[int]:
    """HIPP's 'HH:00' clock deadline -> seconds since midnight.

    None for missing / 'None' / unparseable, which is how a query with no
    deadline is expressed.
    """
    if not time_limit or str(time_limit) == "None":
        return None
    try:
        hour = int(str(time_limit).split(":")[0])
    except (ValueError, IndexError):
        return None
    return hour * 3600


def _apply_hipp_gold(result: Dict[str, Any], hipp_gold: Dict[str, Any]) -> None:
    constraints = dict(result.get("constraints") or {})

    poi_types = [_remap_mall_alias(t) for t in (hipp_gold.get("poi_types_mapped") or [])]
    if poi_types:
        constraints["poi_types"] = poi_types
        # poi_filters is index-aligned with poi_types, so it has to be re-cut to
        # the gold length or every filter after a changed index applies to the
        # wrong stop.
        if isinstance(constraints.get("poi_filters"), list):
            aligned = list(constraints["poi_filters"])[: len(poi_types)]
            aligned += [None] * (len(poi_types) - len(aligned))
            constraints["poi_filters"] = aligned

    deps = hipp_gold.get("dependencies_mapped") or []
    constraints["ordered"] = bool(deps)
    if deps:
        constraints["dependencies"] = deps
    else:
        constraints.pop("dependencies", None)

    deadline_sec = _hipp_deadline_to_sec(hipp_gold.get("time_limit"))
    if deadline_sec is not None:
        constraints["departure_time_sec"] = HIPP_DEPARTURE_SEC
        constraints["arrival_deadline_sec"] = deadline_sec
    else:
        constraints.pop("departure_time_sec", None)
        constraints.pop("arrival_deadline_sec", None)

    # HIPP states a quality/distance weight pair rather than a rating threshold.
    # Quality-dominant (> 0.5) is what "wants well-rated places" means here.
    quality_weight = hipp_gold.get("quality_weight")
    if quality_weight is not None and float(quality_weight) > 0.5 and poi_types:
        constraints["rating_min"] = [HIPP_RATING_THRESHOLD] * len(poi_types)
    else:
        constraints.pop("rating_min", None)

    result["constraints"] = constraints
    result["objective"] = "minimize_distance"


def parse(
    query_id: str,
    nl_query: str,
    *,
    llm: LocalLLM,
    resolver: CatalogResolver,
    problem_text: str,
    G=None,
    hipp_gold: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    valid_categories = resolver.list_categories()

    # Step 1: LLM extraction.
    extracted = extract_structured(query_id, nl_query, llm, valid_categories, problem_text)

    src_raw = extracted.get("src", "")
    dst_raw = extracted.get("dst", "")

    # Keep the LLM's own answer before gold overwrites any of it, so spec
    # fidelity stays measurable even on the gold-injected runs.
    raw_objective = extracted.get("objective", "minimize_travel_time")
    raw_constraints = dict(extracted.get("constraints", {}) or {})
    if isinstance(raw_constraints.get("poi_types"), list):
        raw_constraints["poi_types"] = [_remap_mall_alias(t) for t in raw_constraints["poi_types"]]

    # Step 2: Resolve src and dst.
    src_info = resolve_endpoint(src_raw, G=G)
    dst_info = resolve_endpoint(dst_raw, G=G)

    warnings = []
    if src_info["status"] != "resolved":
        warnings.append(f"src '{src_raw}' resolution status: {src_info['status']} (score={src_info['score']})")
    if dst_info["status"] != "resolved":
        warnings.append(f"dst '{dst_raw}' resolution status: {dst_info['status']} (score={dst_info['score']})")

    # Step 3: Assemble final structured input.
    result: Dict[str, Any] = {
        "query_id": query_id,
        "nl_query": nl_query,
        "src": src_info["node_id"],
        "dst": dst_info["node_id"],
        "objective": raw_objective,
        "constraints": normalize_constraints_for_pipeline(raw_constraints),
        "_parse_meta": {
            "nl_query": nl_query,
            "src_raw": src_raw,
            "dst_raw": dst_raw,
            "src_resolution": src_info,
            "dst_resolution": dst_info,
            "warnings": warnings,
            "raw_llm_objective": raw_objective,
            "raw_llm_constraints": raw_constraints,
        },
    }

    if hipp_gold is not None:
        _apply_hipp_gold(result, hipp_gold)

    # Fill visit_time from LLMAP's own per-category durations. HIPP is LLMAP's
    # benchmark and its queries never state a dwell, so this is what makes the
    # clock-deadline comparison apples-to-apples. Written into the spec rather
    # than assumed by the solver, which reads it from here — so a query that
    # DOES state its own dwell keeps it (see visit_seconds_for_poi_type).
    _c = result["constraints"]
    _types = _c.get("poi_types") or _c.get("stop_types") or []
    if _types and "visit_time" not in _c:
        _c["visit_time"] = {
            str(t).lower(): LLMAP_VISIT_SEC.get(str(t).lower(), LLMAP_VISIT_DEFAULT_SEC)
            for t in _types
        }

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
