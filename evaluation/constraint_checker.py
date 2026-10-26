"""
Single source of truth for cross-system evaluation.

All three systems (LLM baseline, LLMAP baseline, Craft) are scored through the
same logic here, so columns, constraint sets, and objective values are
directly comparable.

Design
------
1. Per-template constraint matrix (TEMPLATE_CHECKS) — every template has a fixed
   set of constraints. The same set is reported for every system; a check that
   doesn't apply is reported as None, so all systems share identical columns
   (including ev_battery_feasible).

2. Fixed per-template objective weight (TEMPLATE_WEIGHT_KEY), derived from the
   NL objective wording and uniform within each template family: length for
   "shortest-distance" families, travel_time for "fastest" families. Every
   system's objective_value uses this same weight for a given query, so the
   numbers are in the same unit and comparable.

3. One comparable objective_value for all systems: leg-by-leg shortest-path cost
   through [src -> intended POI stops -> dst]. Craft additionally reports
   dijkstra_objective_value, the cost of its own realized route (same weight).

4. n_stops counts only the intended POI stops the query asked for, never the
   incidental POI nodes a route happens to pass.

Public API
----------
  templates, template_checks, template_weight_key, check_fields, core_fields
  normalize_type(raw)
  weight_key(template) / objective_type(template) / required_stops(nl, template)
  NodeSnapper.from_gpkg(gpkg, G)         — nearest-node lookup shared by baselines
  evaluate_baseline(nl, template, matched_stops, G, snapper, all_stops=None) -> dict
  evaluate_system(nl, template, route, G, pois_gdf, snapper)                 -> dict

HIPP pilot (benchmark/socal_queries/llmap_queries/) — gold-driven, not NL-regex-driven,
since hipp_meta.json is reliable ground truth (see load_hipp_gold):
  HIPP_TEMPLATES, load_hipp_gold(), build_hipp_rating_cache(G, pois_gdf)
  evaluate_hipp_system(gold, route, G, pois_gdf, snapper, caches=None, rating_cache=None)   -> dict
  evaluate_hipp_baseline(gold, matched_stops, G, snapper, rating_cache=None)                -> dict

set3_descriptions (benchmark/socal_queries/set3_descriptions/) — also gold-driven,
for the same reason plus one more: the filter a stop must satisfy exists only in
set3_meta.json, there is nothing in the NL text to regex it out of. See
load_set3_gold:
  SET3_TEMPLATE, SET3_POI_TYPES, load_set3_gold()
  build_set3_filter_cache(G, pois_gdf, gold_records)
  build_set3_query_embeddings(gold_records, api_key=None)
  evaluate_set3_system(gold, parser_output, route, G, pois_gdf, snapper, ...)   -> dict
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from craft.generator.tools.common_handlers import (
    build_poi_type_nodes,
    resolve_graph_node_id,
)

MILES_TO_M = 1609.344

# POI types that are charging/fuel infrastructure, never counted as user "stops"
# nor included in the objective stop chain.
EV_INFRA: Set[str] = {"charging_station", "fuel", "car_sharing"}

# EV refuelling policy (set 2026-08-10) — applies to EVERY system
# THE RULE, identical for every system: the battery refills ONLY at charging
# stations that system itself COMMITTED to visiting. What differs between
# systems is not the rule but where the commitment is legible in their output:
#
#   CRAFT    — commits via its route. Its LLM-generated search code refuses any
#              edge it lacks the charge for (`if new_battery < 0: continue`) and
#              sets `new_battery = <full range>` on reaching a charging_station.
#              Verified across every routed EV record of all 3 models: ~100%
#              reference chargers and recharge at them. So a charging station on
#              a CRAFT route is a deliberate, searched-for charging stop, NOT an
#              incidental fly-by. Its declared set = chargers ∩ route.
#   LLM      — commits via stops it types "charging_station" (11–30 of 400).
#   LLMAP    — never commits to one (0 of 400): it has no EV support at all
#              (hardcoded 5 POI types, no battery concept). That is a genuine
#              capability gap, not an artefact of this rule.
#
# HISTORY — read before "simplifying" this. Between the 2026-08-10 battery fix
# and this note, the checker briefly treated CRAFT's declared set as EMPTY, on
# the theory that a bare node route cannot express "I stop and charge here".
# That was WRONG and cost CRAFT 269 rows (algo1 solved-correctly fell 185→86,
# putting it below the far weaker LLM baseline — the anomaly that exposed it).
# CRAFT's commitment lives in its generated code's semantics, not in a stop
# list. Do not reinstate the empty-set reading.
#
# "any_charger_on_route" remains selectable ONLY to reproduce pre-fix numbers;
# for CRAFT it is equivalent to the declared reading above, since its declared
# set is derived from the route.
EV_REFUEL_POLICY = "declared_stops_only"


# Type normalisation

_NORMALIZE_TYPE: Dict[str, str] = {
    "coffee shop": "cafe", "coffee": "cafe",
    "gas station": "fuel", "gas": "fuel", "petrol": "fuel", "petrol station": "fuel",
    "ev charger": "charging_station", "ev charging": "charging_station",
    "charger": "charging_station", "charging": "charging_station",
    "pub": "bar", "motel": "hotel", "inn": "hotel",
    "diner": "restaurant", "eatery": "restaurant",
    "burger": "fast_food", "playground": "park",
    "grocery": "supermarket", "grocery store": "supermarket",
    "drugstore": "pharmacy", "clinic": "hospital",
}


def normalize_type(raw: Optional[str]) -> str:
    s = (raw or "").strip().lower()
    return _NORMALIZE_TYPE.get(s, s)


# Template → constraint matrix + objective weight

TEMPLATES = [
    "ev_routing",
    "ordered_poi",
    "poi_collection",
    "ev+ordered_poi",
    "ev+time",
    "ordered+time_budget",
    "unordered_poi",
]

# Hard constraints active per template. Reported as bool when applicable,
# None when not. Every system reports the SAME set for a given template.
# Each comment states the pass condition in the same shorthand: cs=1 (constraints_satisfied)
# when every listed check holds; cs=None if the route doesn't exist, so none can be evaluated.
TEMPLATE_CHECKS: Dict[str, Set[str]] = {
    # cs=1 when the EV never runs the battery negative anywhere along the route
    "ev_routing":     {"ev_battery_feasible"},
    # cs=1 when the POI types/order match the query AND every leg is within its distance limit
    "ordered_poi":    {"poi_type_match", "poi_order", "distance_limits_met"},
    # cs=1 when each category's visited count is <= its cap (>=0, up to the requested count) AND total trip time is <= the time budget
    "poi_collection": {"category_limits_met", "time_budget_met"},
    # cs=1 when POI types/order match AND battery never goes negative AND every leg is within its distance limit
    "ev+ordered_poi":       {"poi_type_match", "poi_order", "ev_battery_feasible", "distance_limits_met"},
    # cs=1 when the EV never runs the battery negative anywhere along the route
    "ev+time":              {"ev_battery_feasible"},
    # cs=1 when POI types/order match AND every leg is within its distance limit AND total trip time is <= the time budget
    "ordered+time_budget":  {"poi_type_match", "poi_order", "distance_limits_met", "time_budget_met"},
    # cs=1 when the required POI types are all visited (any order) AND every leg is within its distance limit
    "unordered_poi":        {"poi_type_match", "distance_limits_met"},  # unordered → no poi_order
    # cs=1 when the required POI types are all visited AND the route uses no more highway than the preference-weighted shortest path through the same stops
    "poi_road_pref":        {"poi_type_match", "road_preference_met"},  # unordered, no distance/time/battery
}

# HIPP pilot (benchmark/socal_queries/llmap_queries/) — kept out of the base dict
# literal above so the existing entries are untouched. Objective is always
# distance (CRAFT's own algorithm minimizes distance for these queries; see
# evaluate_hipp_system/evaluate_hipp_baseline below). NOT added to TEMPLATES
# (the default template list other scripts iterate) to avoid changing any
# existing script's default behavior — HIPP runners pass HIPP_TEMPLATES explicitly.
HIPP_TEMPLATES = ["hipp_ordered", "hipp_unordered"]
# cs=1 when POI types/order match AND every stop meets the rating floor AND every stop is reached before its opening-hours/clock deadline
TEMPLATE_CHECKS["hipp_ordered"]   = {"poi_type_match", "poi_order", "rating_ok", "clock_deadline_met"}
# cs=1 when the required POI types are all visited (any order) AND every stop meets the rating floor AND every stop is reached before its deadline
TEMPLATE_CHECKS["hipp_unordered"] = {"poi_type_match", "rating_ok", "clock_deadline_met"}

# Fixed objective weight per template (uniform within each family; verified
# against the benchmark NL wording).
TEMPLATE_WEIGHT_KEY: Dict[str, str] = {
    "ev_routing":     "length",
    "ordered_poi":    "length",
    "poi_collection": "length",
    "ev+ordered_poi":       "length",
    "ev+time":              "travel_time",
    "ordered+time_budget":  "length",
    "unordered_poi":        "travel_time",
    "poi_road_pref":        "travel_time",
}
TEMPLATE_WEIGHT_KEY["hipp_ordered"]   = "length"
TEMPLATE_WEIGHT_KEY["hipp_unordered"] = "length"

# Templates whose POI stops are order-free (set match, no sequence check).
_UNORDERED_TEMPLATES = {"unordered_poi", "poi_road_pref"}
_ORDERED_TEMPLATES   = {"ordered_poi", "ev+ordered_poi", "ordered+time_budget"}
_CATEGORY_TEMPLATES  = {"poi_collection"}

# Column schema shared by ALL systems (system-specific extras are appended by
# each exporter after these).
CHECK_FIELDS = [
    "poi_type_match",
    "poi_order",
    "category_limits_met",
    "ev_battery_feasible",
    "time_budget_met",
    "distance_limits_met",
    "road_preference_met",
]
# HIPP pilot additions.
CHECK_FIELDS.append("rating_ok")
CHECK_FIELDS.append("clock_deadline_met")
# set3_descriptions addition. Must be appended HERE, before CORE_FIELDS is built
# below — CORE_FIELDS snapshots CHECK_FIELDS with a splat, so a later append
# would land in CHECK_FIELDS (and thus in _satisfaction) but never reach the
# exported column list.
CHECK_FIELDS.append("poi_filters_respected")

# Road-preference REPORTING metrics — deliberately NOT in CHECK_FIELDS: they
# are magnitudes ("how much highway"), not pass/fail constraints, so they must
# not feed _satisfaction / _all_satisfied. All None outside poi_road_pref.
ROAD_PREF_METRIC_FIELDS = [
    "route_segments",
    "hwy_segments",
    "hwy_segment_share",
    "hwy_distance_share",
    "hwy_distance_share_neutral",
    "hwy_fully_avoided",
    # Absolute distances. A share alone hides trip length (40% of 3 mi vs 40% of
    # 60 mi) and cannot be summed across queries; these can.
    "hwy_miles",
    "nonhwy_miles",
    "total_miles",
    # AVOID-SP reference (added 2026-08-25): the preference-weighted SP through
    # the SAME stops — the rule the generated code is asked to implement. A
    # stronger bar than the neutral SP (see _road_pref_verdict); NOT a proven
    # highway minimum (see AVOIDSP_TOL_MI).
    "hwy_miles_avoidsp",
    "hwy_distance_share_avoidsp",
    "hwy_miles_excess_vs_avoidsp",
    "matches_avoidsp",
]

# CRAFT-only mirror of the above, measured on the route CRAFT's generated code
# actually produced. Prefer these whenever CRAFT is reported on its own — the
# reconstructed set exists only for cross-system comparability with LLM/LLMAP.
ROAD_PREF_METRIC_FIELDS_REALIZED = [
    "route_segments_realized",
    "hwy_segments_realized",
    "hwy_segment_share_realized",
    "hwy_distance_share_realized",
    "hwy_distance_share_neutral_realized",
    "hwy_fully_avoided_realized",
    "hwy_miles_realized",
    "nonhwy_miles_realized",
    "total_miles_realized",
    "hwy_miles_avoidsp_realized",
    "hwy_distance_share_avoidsp_realized",
    "hwy_miles_excess_vs_avoidsp_realized",
    "matches_avoidsp_realized",
]

CORE_FIELDS = [
    "required_stops", "n_stops", "stops_count_ok",
    *CHECK_FIELDS,
    "constraint_satisfaction",
    "constraints_satisfied",
    "objective_type", "objective_value",
    *ROAD_PREF_METRIC_FIELDS,
]

CORE_FIELD_DESCRIPTIONS: List[Tuple[str, str]] = [
    ("required_stops",   "Number of POI stops the query requires (from NL text); None for pure-EV templates"),
    ("n_stops",          "Number of *intended* POI stops the system visited (required types / collection visits) — not incidental POIs on the path"),
    ("stops_count_ok",   "True if n_stops == required_stops; None if required_stops is unknown"),
    ("poi_type_match",   "True if visited stop types match the required types (ordered: sequence; unordered: set); None if not applicable"),
    ("poi_order",        "True if required types visited in the required order; None for unordered / no-order templates"),
    ("category_limits_met", "True if every category count is within its limit and the collection target is met; None if not a collection template"),
    ("ev_battery_feasible", "True if battery never goes negative along the route (refilling at chargers); None if no EV constraint"),
    ("time_budget_met",  "True if SP travel time + visit time ≤ budget; None if no time-budget constraint"),
    ("distance_limits_met", "True if every per-leg SP distance ≤ its limit; None if no distance constraint"),
    ("constraint_satisfaction", "Fraction of applicable (non-None) constraint checks that are True (0–1); None if no checks apply"),
    ("constraints_satisfied", "Boolean: True iff EVERY applicable hard constraint is True; False if any is violated; None if no constraint was evaluable (e.g. no route). Use 'feasible AND constraints_satisfied' counted over all N as the comparable solved-correctly rate"),
    ("objective_type",   "'distance' (weight=length) or 'time' (weight=travel_time); fixed per template"),
    ("objective_value",  "Leg-by-leg SP cost through [src → intended stops → dst] with the fixed per-template weight (meters or seconds); same computation for every system; None if any leg is unreachable"),
    ("route_segments",   "poi_road_pref only: number of road segments (graph edges) in the scored route. Same route road_preference_met is decided on — the preference-weighted SP through the system's own stops"),
    ("hwy_segments",     "poi_road_pref only: how many of those segments run on a highway (motorway/trunk/primary, incl. _link)"),
    ("hwy_segment_share", "poi_road_pref only: hwy_segments / route_segments. Counts hops, so short ramps weigh as much as long stretches — read alongside hwy_distance_share"),
    ("hwy_distance_share", "poi_road_pref only: share of the route's distance on highways. This is the quantity road_preference_met compares against the neutral baseline"),
    ("hwy_distance_share_neutral", "poi_road_pref only: same share for the NEUTRAL (no-preference) SP through the identical stops — the counterfactual. The gap vs hwy_distance_share is how much highway the preference actually removed"),
    ("hwy_fully_avoided", "poi_road_pref only: True if the route uses zero highway segments. Stricter than road_preference_met, which only requires beating the neutral baseline"),
]
# HIPP pilot additions.
CORE_FIELD_DESCRIPTIONS.append(
    ("rating_ok", "True if every claimed stop's real (synthetic) gpkg rating >= HIPP_RATING_THRESHOLD; "
                  "None if the query is not quality-emphasized (gold quality_weight <= 0.5) or not a HIPP template"))
CORE_FIELD_DESCRIPTIONS.append(
    ("clock_deadline_met", "True if 10:00 AM departure + SP travel through claimed stops + per-category "
                           "dwell time <= the query's clock deadline; None if no deadline stated"))
# set3_descriptions addition.
CORE_FIELD_DESCRIPTIONS.append(
    ("poi_filters_respected", "True iff EVERY filtered stop the query asks for was both visited and satisfies "
                              "its gold filter (structured: substring match on the real gpkg column; description: "
                              "cosine >= 0.65 against the same poi_embeddings cache the solver used). A filtered "
                              "stop the route never visited counts as False. None if the query has no filtered "
                              "stops or is not a set3 template. This is a HARD constraint — it enters "
                              "constraints_satisfied"))


def weight_key(template: str) -> str:
    return TEMPLATE_WEIGHT_KEY.get(template, "length")


def objective_type(template: str) -> str:
    return "time" if weight_key(template) == "travel_time" else "distance"


# NL query parsing → canonical parameters

def parse_coords(nl: str) -> List[Tuple[float, float]]:
    """All (lat, lon) pairs in the NL text. First = src, last = dst."""
    return [(float(lat), float(lon))
            for lat, lon in re.findall(r'\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)', nl)]


def parse_time_budget_s(nl: str) -> Optional[float]:
    m = re.search(r'(\d+(?:\.\d+)?)\s*hour', nl, re.I)
    if m:
        return float(m.group(1)) * 3600
    m = re.search(r'(?:in|within)\s+(\d+)\s*minute', nl, re.I)
    if m:
        return float(m.group(1)) * 60
    return None


def parse_visit_time_s(nl: str) -> float:
    for pat in [
        r'(\d+)\s*minutes?\s*(?:per|at each)\s*stop',
        r'assume\s+(\d+)\s*minute',
        r'spend(?:ing)?\s+(?:about\s+)?(\d+)\s*minute',
        r'about\s+(\d+)\s*minute',
    ]:
        m = re.search(pat, nl, re.I)
        if m:
            return float(m.group(1)) * 60
    return 15 * 60


# Fixed per-category POI visit/dwell durations, in seconds. Mirrors LLMAP's
# stay_time_mapping and the craft parser, so time_budget_met charges each stop
# the same dwell time LLMAP assumes instead of one uniform per-stop value.
# Any category not listed here defaults to LLMAP_VISIT_DEFAULT_SEC.
LLMAP_VISIT_SEC: Dict[str, int] = {
    "shopping_mall": 7200,
    "shopping mall": 7200,
    "supermarket":   1800,
    "pharmacy":       900,
    "bank":          1200,
    "library":       3600,
}
# HIPP pilot: "mall" is the actual category value (shop=mall), distinct from
# the "shopping_mall"/"shopping mall" display-name keys above — same dwell time.
LLMAP_VISIT_SEC["mall"] = 7200
LLMAP_VISIT_DEFAULT_SEC = 900


def _category_visit_total_s(
    stop_nodes: List[Any],
    poi_types: Optional[List[str]],
    type_nodes: Optional[Dict[str, Set[Any]]],
) -> float:
    """Total visit seconds for a set of stop nodes, charging LLMAP's fixed
    per-category dwell time to each stop.

    A stop node's category is the first ``poi_type`` whose
    ``type_nodes[poi_type]`` set contains it; a node matching none (or when
    categories/caches are unavailable) is charged LLMAP_VISIT_DEFAULT_SEC.
    """
    types = poi_types or []
    tn = type_nodes or {}
    total = 0.0
    for node in stop_nodes:
        cat: Optional[str] = None
        for t in types:
            if node in (tn.get(t) or set()):
                cat = t
                break
        total += LLMAP_VISIT_SEC.get(cat, LLMAP_VISIT_DEFAULT_SEC) if cat is not None \
            else LLMAP_VISIT_DEFAULT_SEC
    return total


# A run of comma-separated numbers sharing ONE trailing unit, optionally with
# "and" before the last: "9, 2, and 4 miles" / "4, 1, 1 miles". Group 1 is the
# leading list (may be absent), group 2 the final number that carries the unit.
#
# Without the group-1 alternative this returned ONLY the last value, because the
# older pattern required "mile" to sit directly after each number — so "The leg
# limits are 9, 2, and 4 miles respectively" parsed as [4], which the caller then
# broadcast to EVERY leg. That mis-scored the 7 algo2 variant-27 queries (stricter
# on legs 1 and 4, looser on leg 2). Every other phrasing in the benchmark repeats
# the unit per number ("3 miles, leg 2 within 2 miles"), which matched correctly
# before and still does: `\d+\s*,` cannot match "3 miles," (a letter follows the
# digits), so those never form a false run.
_LIMIT_RUN = re.compile(
    r'((?:\d+(?:\.\d+)?\s*,\s*)+(?:and\s+)?)?(\d+(?:\.\d+)?)\s*mile', re.I)
_NUM = re.compile(r'\d+(?:\.\d+)?')


def parse_distance_limits_m(nl: str) -> List[float]:
    """Per-leg distance limits, in meters, in order of appearance. EV ranges are
    matched separately and do not appear here because parse_battery_m strips the
    range phrase first.

    Handles both benchmark phrasings:
      "a pharmacy within 7 miles ... a library about 2 miles"  -> [7, 2]
      "The leg limits are 9, 2, and 4 miles respectively"      -> [9, 2, 4]
    """
    out: List[float] = []
    for m in _LIMIT_RUN.finditer(_strip_range_phrase(nl)):
        if m.group(1):
            out.extend(float(v) * MILES_TO_M for v in _NUM.findall(m.group(1)))
        out.append(float(m.group(2)) * MILES_TO_M)
    return out


# EV battery is stated many ways in the benchmark NL. We parse a max (full)
# range and a current charge (either a percentage of full, or remaining miles).
_PCT = re.compile(r'(\d+(?:\.\d+)?)\s*%')

# Max / full range patterns, most specific first.
_MAX_PATTERNS = [
    r'maximum range of\s*(\d+(?:\.\d+)?)\s*miles?',
    r'max(?:imum)?\s+range\s+(?:is|of)\s*(\d+(?:\.\d+)?)\s*miles?',
    r'full charge gives(?:\s+me)?\s*(\d+(?:\.\d+)?)\s*miles?',
    r'full range is\s*(\d+(?:\.\d+)?)\s*miles?',
    r'(\d+(?:\.\d+)?)\s*-?\s*mile full range',
    r'out of a maximum of\s*(\d+(?:\.\d+)?)\s*miles?',
    r'out of a\s*(\d+(?:\.\d+)?)\s*-?\s*mile',
    r'maximum of\s*(\d+(?:\.\d+)?)\s*miles?',
    r'range\s+(?:is|of)\s*(\d+(?:\.\d+)?)\s*miles?',
    r'out of\s*(\d+(?:\.\d+)?)\s*miles?',
    r'out of\s*(\d+(?:\.\d+)?)\b',
]

# Current remaining range (in miles) patterns.
_CUR_MILES_PATTERNS = [
    r'(\d+(?:\.\d+)?)\s*miles?\s+of\s+range\s+left',
    r'(\d+(?:\.\d+)?)\s*miles?\s+remaining',
    r'(\d+(?:\.\d+)?)\s*miles?\s+left',
    r'current range is\s*(\d+(?:\.\d+)?)\s*miles?',
    r'currently have\s*(\d+(?:\.\d+)?)\s*miles?(?:\s+of\s+range)?',
    r'have\s*(\d+(?:\.\d+)?)\s*miles?\s+of\s+range',
    r'(\d+(?:\.\d+)?)\s*miles?\s+of\s+range',
]


def _first_group(patterns: List[str], nl: str) -> Optional[float]:
    for pat in patterns:
        m = re.search(pat, nl, re.I)
        if m:
            return float(m.group(1))
    return None


def _strip_range_phrase(nl: str) -> str:
    """Remove EV battery clauses so per-leg 'N miles' distance limits aren't
    contaminated by battery range values."""
    out = nl
    for pat in _CUR_MILES_PATTERNS + _MAX_PATTERNS:
        out = re.sub(pat, " ", out, flags=re.I)
    return out


def parse_battery_m(nl: str) -> Tuple[Optional[float], Optional[float]]:
    """(current_battery_m, max_battery_m). Handles the benchmark's EV phrasings:
       'battery is at 75% ... full charge gives me 93 miles', 'at 4% charge,
       maximum range of 93 miles', 'I have 3 miles of range left out of 93',
       'X miles remaining out of a 93-mile full range', 'full range is 93 miles',
       'my current range is 3 miles out of a maximum of 93 miles', etc."""
    max_mi = _first_group(_MAX_PATTERNS, nl)
    max_m = max_mi * MILES_TO_M if max_mi is not None else None

    cur_m: Optional[float] = None
    m_pct = _PCT.search(nl)
    if m_pct is not None and max_m is not None:
        cur_m = float(m_pct.group(1)) / 100.0 * max_m
    else:
        cur_mi = _first_group(_CUR_MILES_PATTERNS, nl)
        if cur_mi is not None:
            cur_m = cur_mi * MILES_TO_M

    return cur_m, max_m


# Road-type preference: parsed from NL the same way as every other constraint
# here (re-derived from text, not trusted from any system's own parser output,
# so every system is scored on the same basis).
_AVOID_HIGHWAY_PATTERNS = [
    r"avoid highways?",
    r"avoid (?:the )?freeways?",
    r"keep me off (?:the )?highways?",
    r"stay off (?:the )?freeways?",
    r"stay off (?:the )?highways?",
    r"don'?t want to take (?:any )?highways?",
    r"rather not (?:drive|take) (?:on )?highways?",
    r"dislike highways?",
]
_PREFER_HIGHWAY_PATTERNS = [
    r"don'?t mind (?:taking |the )?highways?",
    r"highways? (?:is|are) fine",
    r"fine with highways?",
    r"no problem (?:with|taking) highways?",
    r"okay with me",
    r"like highways?",
]


def parse_road_preference(nl: str) -> Optional[str]:
    """'avoid_highways' | 'prefer_highways' | None, re-derived from NL text."""
    text = nl.lower()
    for pat in _AVOID_HIGHWAY_PATTERNS:
        if re.search(pat, text):
            return "avoid_highways"
    for pat in _PREFER_HIGHWAY_PATTERNS:
        if re.search(pat, text):
            return "prefer_highways"
    return None


# Major-road classes, matching the generator's road_attribute_schema note
# (craft/generator/agent.py) — kept in sync manually since there is no
# single shared source of truth for this set yet.
_HIGHWAY_TYPES = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"}


def highway_usage(G: nx.MultiDiGraph, route: List[Any], weight: str = "length") -> Dict[str, Any]:
    """Segment- and distance-level highway usage along a node route.

    Two complementary views of the same route, both reported because they
    answer different questions:
      * segments  — how many road segments are on a motorway/trunk/primary
                    (+ _link). Counts hops, so many short highway ramps read
                    as heavy usage.
      * distance  — what share of the route's `weight` runs on those roads.
                    The measure `road_preference_met` is decided on.
    All-None when the route is empty/unresolvable, so callers can pass the
    dict straight through to a report row.
    """
    total = 0.0
    hwy = 0.0
    n_seg = 0
    n_hwy_seg = 0
    for u_raw, v_raw in zip(route[:-1], route[1:]):
        u, v = coerce(G, u_raw), coerce(G, v_raw)
        if u is None or v is None:
            continue
        data = G.get_edge_data(u, v)
        if not data:
            continue
        d = data[min(data)]
        cost = float(d.get(weight) or 0.0)
        total += cost
        n_seg += 1
        types = str(d.get("highway") or "").split(";")
        if any(t in _HIGHWAY_TYPES for t in types):
            hwy += cost
            n_hwy_seg += 1
    if n_seg == 0:
        return {"route_segments": None, "hwy_segments": None,
                "hwy_segment_share": None, "hwy_distance_share": None,
                "hwy_miles": None, "nonhwy_miles": None, "total_miles": None}
    return {
        "route_segments":     n_seg,
        "hwy_segments":       n_hwy_seg,
        "hwy_segment_share":  n_hwy_seg / n_seg,
        "hwy_distance_share": (hwy / total) if total > 0 else None,
        # Absolute distances alongside the share: a 40% share means something very
        # different on a 3-mile trip than on a 60-mile one, and the share alone
        # cannot be summed across queries. Only meaningful when `weight` is length.
        "hwy_miles":          hwy / MILES_TO_M,
        "nonhwy_miles":       (total - hwy) / MILES_TO_M,
        "total_miles":        total / MILES_TO_M,
    }


def _highway_fraction(G: nx.MultiDiGraph, route: List[Any], weight: str = "length") -> Optional[float]:
    """Fraction of `weight` (default: distance) on major roads along a realized route."""
    return highway_usage(G, route, weight)["hwy_distance_share"]


def sp_chain_path(G: nx.MultiDiGraph, chain: Sequence[Any], weight: str) -> Optional[List[Any]]:
    """Concatenated shortest-path node list through a chain of waypoints
    (ignoring any preference), used as the neutral baseline for
    road_preference_met. None if any leg has no path."""
    full: List[Any] = []
    for a, b in zip(chain, chain[1:]):
        if a is None or b is None:
            return None
        try:
            leg = nx.shortest_path(G, a, b, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.exception.NetworkXError):
            return None
        full = full[:-1] + leg if full else leg
    return full


# Tolerance, in MILES, for "did this route use no more highway than a correct
# preference-aware router would". Absorbs float noise and a single short ramp;
# not a grading curve.
#
# ⚠️ WHAT THE avoidsp REFERENCE IS — AND IS NOT. It is the preference-WEIGHTED
# shortest path through the same stops (highway edges costed x3), i.e. exactly
# the rule CRAFT's generated code is asked to implement. It therefore answers
# "did this code route as well as a correct implementation of the same rule?".
# It is NOT a proven minimum of highway distance: it minimises WEIGHTED distance,
# so a route with slightly more highway but much shorter overall can win. A true
# highway-distance minimiser would be a different (and useless) objective — it
# would accept an unbounded detour to shed the last 100 m of ramp.
# CONSEQUENCE: `hwy_miles_excess_vs_avoidsp` CAN BE NEGATIVE — the realized route
# legitimately beating the reference on highway miles while costing more overall.
# Measured on real routes: e.g. poi_road_pref pref100, realized 0.116 hwy-mi vs
# reference 0.143. Do not "fix" that as a bug, and do not read the reference as
# a floor.
AVOIDSP_TOL_MI = 0.1


def _road_pref_verdict(realized_frac: Optional[float], neutral_frac: Optional[float],
                       pref: str) -> Optional[bool]:
    """Shared avoid/prefer comparison: realized highway-fraction vs. the
    neutral (no-preference) SP baseline through the same stops. Not an
    arbitrary absolute threshold — a route-selection comparison.

    ⚠️ THIS IS A WEAK BAR, kept for continuity of the published column. It asks
    only "did the route use less highway than doing nothing", so ANY avoidance
    passes — which is why it reads 78-96% while `hwy_fully_avoided` reads 24-38%
    on the same routes. The stronger question — "did it use as little highway as
    was ACHIEVABLE given the stops it chose?" — is answered by
    `matches_avoidsp*` / `hwy_miles_excess_vs_avoidsp*`, which compare against
    the preference-weighted SP instead. Prefer those for a capability claim."""
    if realized_frac is None or neutral_frac is None:
        return None
    if pref == "avoid_highways":
        if neutral_frac <= 0.01:
            return realized_frac <= 0.01
        return realized_frac < neutral_frac
    if pref == "prefer_highways":
        # Weak preference: shouldn't go out of its way to dodge highways the
        # neutral/fastest path would have used anyway.
        return realized_frac >= neutral_frac - 0.05
    return None


def road_preference_met(
    G: nx.MultiDiGraph, route: List[Any], neutral_chain: Sequence[Any],
    nl_query: str, weight: str = "length",
) -> Optional[bool]:
    """Whether a stated road-type preference actually shaped the realized
    route, compared against a neutral SP baseline through the same stops
    (fair basis — not an arbitrary absolute threshold). None if no
    preference is stated or either route can't be computed."""
    pref = parse_road_preference(nl_query)
    if pref is None or not route:
        return None
    realized_frac = _highway_fraction(G, route, weight)
    neutral_route = sp_chain_path(G, neutral_chain, weight)
    neutral_frac = _highway_fraction(G, neutral_route, weight) if neutral_route else None
    return _road_pref_verdict(realized_frac, neutral_frac, pref)


def _pref_multiplier(highway_val: Any, pref: str) -> float:
    if not highway_val:
        return 1.0
    types = str(highway_val).split(";")
    if not any(t in _HIGHWAY_TYPES for t in types):
        return 1.0
    if pref == "avoid_highways":
        return 3.0
    if pref == "prefer_highways":
        return 0.7
    return 1.0


def _pref_weight_fn(weight_key: str, pref: str):
    """nx weight callable: (u, v, d) -> float, min over parallel edges,
    same highway multiplier convention as craft's generated expand()."""
    def w(u: Any, v: Any, d: Dict[Any, Dict]) -> float:
        best = None
        for attr in d.values():
            cost = float(attr.get(weight_key) or 0.0) * _pref_multiplier(attr.get("highway"), pref)
            if best is None or cost < best:
                best = cost
        return best if best is not None else 0.0
    return w


def sp_chain_path_weighted(G: nx.MultiDiGraph, chain: Sequence[Any],
                          weight_key: str, pref: str) -> Optional[List[Any]]:
    """Like sp_chain_path but using a preference-weighted cost function —
    the "what would a preference-aware shortest path look like" reconstruction
    used to score baselines (LLM/LLMAP) that have no real route of their own."""
    wf = _pref_weight_fn(weight_key, pref)
    full: List[Any] = []
    for a, b in zip(chain, chain[1:]):
        if a is None or b is None:
            return None
        try:
            leg = nx.shortest_path(G, a, b, weight=wf)
        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.exception.NetworkXError):
            return None
        full = full[:-1] + leg if full else leg
    return full


def road_preference_met_baseline(
    G: nx.MultiDiGraph, src: Any, dst: Any, stop_nodes: Sequence[Any],
    nl_query: str, weight: str = "length",
) -> Optional[bool]:
    """Baseline (LLM/LLMAP) variant: neither system computes a real road
    route (LLM: stops only; LLMAP: haversine, not real edges), so there is
    no realized route to score. Instead, reconstruct a route via a
    preference-weighted SP through the baseline's OWN chosen stops (same
    multiplier craft's generated code uses) and compare it to the neutral
    SP through the same stops. This tests "given identical stops, does a
    preference-aware SP respect the preference" — NOT the baseline's own
    capability, since neither system reasons about road type at all. See
    CLAUDE.md's poi_road_pref section for the fairness caveat this implies."""
    pref = parse_road_preference(nl_query)
    if pref is None:
        return None
    chain = [src] + list(stop_nodes) + [dst]
    if any(n is None for n in chain):
        return None
    neutral_route = sp_chain_path(G, chain, weight)
    pref_route = sp_chain_path_weighted(G, chain, weight, pref)
    neutral_frac = _highway_fraction(G, neutral_route, weight) if neutral_route else None
    pref_frac = _highway_fraction(G, pref_route, weight) if pref_route else None
    return _road_pref_verdict(pref_frac, neutral_frac, pref)


_EMPTY_ROAD_PREF_METRICS: Dict[str, Any] = {k: None for k in ROAD_PREF_METRIC_FIELDS}
_EMPTY_ROAD_PREF_METRICS_REALIZED: Dict[str, Any] = {
    k: None for k in ROAD_PREF_METRIC_FIELDS_REALIZED}


def road_pref_metrics_realized(
    G: nx.MultiDiGraph, route: List[Any], neutral_chain: Sequence[Any],
    nl_query: str, weight: str = "length",
) -> Dict[str, Any]:
    """CRAFT-only: the same highway magnitudes measured on the route CRAFT's
    generated code ACTUALLY produced, rather than a reconstruction. This is
    the real capability signal — use it whenever CRAFT is reported on its own
    (the reconstructed variant exists only to make LLM/LLMAP, which never
    compute a route, comparable). `..._neutral` stays the no-preference SP
    through the same intended stops, so the counterfactual is unchanged."""
    pref = parse_road_preference(nl_query)
    if pref is None or not route:
        return dict(_EMPTY_ROAD_PREF_METRICS_REALIZED)
    stats = highway_usage(G, route, weight)
    out = dict(_EMPTY_ROAD_PREF_METRICS_REALIZED)
    out.update({
        "route_segments_realized":     stats["route_segments"],
        "hwy_segments_realized":       stats["hwy_segments"],
        "hwy_segment_share_realized":  stats["hwy_segment_share"],
        "hwy_distance_share_realized": stats["hwy_distance_share"],
        "hwy_miles_realized":          stats["hwy_miles"],
        "nonhwy_miles_realized":       stats["nonhwy_miles"],
        "total_miles_realized":        stats["total_miles"],
        "hwy_fully_avoided_realized": (
            (stats["hwy_segments"] == 0) if stats["hwy_segments"] is not None else None),
    })
    if not any(n is None for n in neutral_chain):
        neutral_route = sp_chain_path(G, neutral_chain, weight)
        out["hwy_distance_share_neutral_realized"] = (
            _highway_fraction(G, neutral_route, weight) if neutral_route else None)
        # best achievable given the SAME stops: a preference-weighted SP
        best = sp_chain_path_weighted(G, neutral_chain, weight, pref)
        if best:
            bstats = highway_usage(G, best, weight)
            out["hwy_miles_avoidsp_realized"] = bstats["hwy_miles"]
            out["hwy_distance_share_avoidsp_realized"] = bstats["hwy_distance_share"]
            if stats["hwy_miles"] is not None and bstats["hwy_miles"] is not None:
                excess = stats["hwy_miles"] - bstats["hwy_miles"]
                out["hwy_miles_excess_vs_avoidsp_realized"] = excess
                out["matches_avoidsp_realized"] = excess <= AVOIDSP_TOL_MI
    return out


def road_pref_metrics(
    G: nx.MultiDiGraph, src: Any, dst: Any, stop_nodes: Sequence[Any],
    nl_query: str, weight: str = "length",
) -> Dict[str, Any]:
    """Highway-usage magnitudes for a road-preference query, on the SAME
    preference-weighted reconstructed route `road_preference_met` is decided
    on — so the metrics and the verdict can never disagree, and the numbers
    are comparable across all three systems. `..._neutral` is the
    no-preference counterfactual through the identical stops; the gap between
    them is how much highway the preference actually removed."""
    pref = parse_road_preference(nl_query)
    if pref is None:
        return dict(_EMPTY_ROAD_PREF_METRICS)
    chain = [src] + list(stop_nodes) + [dst]
    if any(n is None for n in chain):
        return dict(_EMPTY_ROAD_PREF_METRICS)
    pref_route = sp_chain_path_weighted(G, chain, weight, pref)
    if not pref_route:
        return dict(_EMPTY_ROAD_PREF_METRICS)
    stats = highway_usage(G, pref_route, weight)
    neutral_route = sp_chain_path(G, chain, weight)
    out = dict(_EMPTY_ROAD_PREF_METRICS)
    out.update({k: stats.get(k) for k in
                ("route_segments", "hwy_segments", "hwy_segment_share", "hwy_distance_share",
                 "hwy_miles", "nonhwy_miles", "total_miles")})
    out["hwy_distance_share_neutral"] = (
        _highway_fraction(G, neutral_route, weight) if neutral_route else None)
    out["hwy_fully_avoided"] = (
        (stats["hwy_segments"] == 0) if stats["hwy_segments"] is not None else None)
    # For the RECONSTRUCTED path the scored route IS the preference-weighted SP,
    # so it is its own best-achievable reference and the excess is 0 by
    # construction. Recorded explicitly rather than left blank so the column means
    # the same thing in both variants — and so nobody reads a 100% match rate here
    # as a capability result. The informative version is the *_realized one.
    out["hwy_miles_avoidsp"] = stats["hwy_miles"]
    out["hwy_distance_share_avoidsp"] = stats["hwy_distance_share"]
    out["hwy_miles_excess_vs_avoidsp"] = 0.0 if stats["hwy_miles"] is not None else None
    out["matches_avoidsp"] = True if stats["hwy_miles"] is not None else None
    return out


# POI display term → dataset category (the collection/ordered types the
# benchmark uses). Mirrors the generator's LLMAP_DISPLAY_TO_CATEGORY.
_DISPLAY_TO_CATEGORY: Dict[str, str] = {
    "pharmacy":    "pharmacy",
    "bank":        "bank",
    "supermarket": "supermarket",
    "library":     "library",
}


def _plural_pattern(display: str) -> str:
    """Regex fragment matching singular or plural, incl. y→ies (pharmacy→pharmacies)."""
    if display.endswith("y"):
        return re.escape(display[:-1]) + r"(?:y|ies)"
    return re.escape(display) + r"(?:s|es)?"


def required_poi_types(nl: str) -> List[str]:
    """Ordered (by first appearance) list of required POI categories in the NL."""
    text = nl.lower()
    hits: List[Tuple[int, str]] = []
    for display, cat in sorted(_DISPLAY_TO_CATEGORY.items(), key=lambda x: -len(x[0])):
        for m in re.finditer(r'\b' + _plural_pattern(display) + r'\b', text):
            hits.append((m.start(), cat))
    hits.sort(key=lambda x: x[0])
    seen: Set[str] = set()
    out: List[str] = []
    for _, cat in hits:
        if cat not in seen:
            seen.add(cat)
            out.append(cat)
    return out


def required_cat_limits(nl: str) -> Dict[str, int]:
    """Per-category upper limits. Handles 'up to 2 pharmacies', 'pharmacies (max 2)',
    'libraries, up to 2' / 'supermarket stops as possible, up to 3' (the limit
    trails the noun rather than leading it — poi_collection template variants
    24/27/29), and a bare 'a library' (→ 1). Returns {} if nothing parses.

    The post-noun form needs its own window scan (same 35-char/no-period-or-
    semicolon rule as the leading-number form) since the number trails the
    category instead of preceding it. Without it, e.g. "up to 3" on variant 29
    gets missed entirely and silently falls back to the bare-mention default
    of 1, which then reports a real 3-collection route as an over-cap violation.
    """
    text = nl.lower()
    limits: Dict[str, int] = {}
    for display, cat in sorted(_DISPLAY_TO_CATEGORY.items(), key=lambda x: -len(x[0])):
        plural = _plural_pattern(display)
        m = re.search(r'(\d+)\s+' + plural + r'\b', text)                 # "2 pharmacies"
        if m is None:
            m = re.search(plural + r'\b\s*\(?\s*max\s+(\d+)', text)       # "pharmacies (max 2)"
        if m is None:
            m = re.search(plural + r'\b[^.;]{0,35}?\bup to\s+(\d+)', text)  # "libraries, up to 2"
        if m is not None:
            limits[cat] = int(m.group(1))
        elif re.search(r'\b' + plural + r'\b', text):                    # "a library" → 1
            limits[cat] = 1
    return limits


def required_stops(nl: str, template: str) -> Optional[int]:
    if template in _ORDERED_TEMPLATES or template in _UNORDERED_TEMPLATES:
        types = required_poi_types(nl)
        return len(types) if types else None
    if template in _CATEGORY_TEMPLATES:
        limits = required_cat_limits(nl)
        return sum(int(v) for v in limits.values()) if limits else None
    return None


# Node resolution

def coerce(G: nx.MultiDiGraph, raw: Any) -> Optional[Any]:
    """Return the node id as it exists in G (int or str), or None."""
    if raw is None:
        return None
    if raw in G:
        return raw
    for cast in (int, str):
        try:
            v = cast(raw)
        except (ValueError, TypeError):
            continue
        if v in G:
            return v
    return resolve_graph_node_id(G, raw)


class NodeSnapper:
    """Nearest-graph-node lookup from (lat, lon), shared by the baselines."""

    def __init__(self, G: nx.MultiDiGraph, tree, osmids: List[Any]):
        self.G = G
        self.tree = tree
        self.osmids = osmids

    @classmethod
    def from_gpkg(cls, gpkg_path: str, G: nx.MultiDiGraph) -> "NodeSnapper":
        import geopandas as gpd
        from scipy.spatial import KDTree
        import numpy as np

        nodes_gdf = gpd.read_file(gpkg_path, layer="nodes")
        lats = nodes_gdf["lat"].values
        lons = nodes_gdf["lon"].values
        osmids: List[Any] = []
        for v in nodes_gdf["osmid_original"].values:
            s = str(v).split(";")[0].strip()
            try:
                osmids.append(int(s))
            except ValueError:
                osmids.append(s)
        tree = KDTree(np.column_stack([lats, lons]))
        return cls(G, tree, osmids)

    def snap(self, lat: float, lon: float) -> Optional[Any]:
        _, idx = self.tree.query([lat, lon])
        return coerce(self.G, self.osmids[idx])


# Shortest-path helpers

def sp_chain(G: nx.MultiDiGraph, chain: Sequence[Any], weight: str
             ) -> Tuple[Optional[float], Optional[List[float]]]:
    """Sum of SP costs along the node chain. Returns (total, per_leg) or (None, None)."""
    total = 0.0
    legs: List[float] = []
    for a, b in zip(chain, chain[1:]):
        if a is None or b is None:
            return None, None
        try:
            cost = nx.shortest_path_length(G, a, b, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound, nx.exception.NetworkXError):
            return None, None
        total += cost
        legs.append(cost)
    return round(total, 2), legs


def _road_network_checks(G, src, dst, stop_nodes, nl_query, active,
                         poi_types=None, type_nodes=None):
    """SP-through-stops distance/time checks — the SAME basis for every system
    (baselines AND craft). Each leg is the shortest path between consecutive
    chosen stops, not a realized traversal, so distance_limits_met /
    time_budget_met are apples-to-apples across systems."""
    out: Dict[str, Optional[bool]] = {}
    need_dist = "distance_limits_met" in active
    need_time = "time_budget_met" in active
    if not (need_dist or need_time) or G is None or src is None or dst is None:
        return out
    stop_nodes = list(stop_nodes)
    if any(n is None for n in stop_nodes):
        return out
    chain = [src] + stop_nodes + [dst]
    if need_dist:
        limits_m = parse_distance_limits_m(nl_query)
        _, legs = sp_chain(G, chain, "length")
        if limits_m and legs is not None:
            # Single parsed limit means "each leg within X" — broadcast to all legs.
            if len(limits_m) == 1 and len(legs) > 1:
                limits_m = limits_m * len(legs)
            n = min(len(limits_m), len(legs))
            out["distance_limits_met"] = all(legs[i] <= limits_m[i] for i in range(n))
        elif limits_m and legs is None:
            out["distance_limits_met"] = False
    if need_time:
        budget_s = parse_time_budget_s(nl_query)
        _, legs_t = sp_chain(G, chain, "travel_time")
        if budget_s is not None and legs_t is not None:
            # Per-category LLMAP dwell times when categories/caches are supplied,
            # else fall back to the old uniform per-stop visit time.
            if poi_types is not None and type_nodes is not None:
                visit_total = _category_visit_total_s(stop_nodes, poi_types, type_nodes)
            else:
                visit_total = len(stop_nodes) * parse_visit_time_s(nl_query)
            total = sum(legs_t) + visit_total
            out["time_budget_met"] = total <= budget_s
        elif budget_s is not None and legs_t is None:
            out["time_budget_met"] = False
    return out


def compute_system_objective(route: List[Any], G: nx.MultiDiGraph, weight: str
                             ) -> Optional[float]:
    """Cost of a realized route (sum of actual edge weights along it)."""
    total = 0.0
    for u_raw, v_raw in zip(route[:-1], route[1:]):
        u, v = coerce(G, u_raw), coerce(G, v_raw)
        data = G.get_edge_data(u, v) if u is not None and v is not None else None
        if not data:
            return None
        d = data[min(data)]
        cost = d.get(weight)
        if cost is None:
            if weight == "travel_time":
                length = float(d.get("length", 0.0) or 0.0)
                speed = float(d.get("speed_kph", 30.0) or 30.0)
                cost = length / (speed * 1000.0 / 3600.0) if speed else 0.0
            else:
                cost = 0.0
        total += float(cost)
    return round(total, 2)


def _edge_cost(G: nx.MultiDiGraph, u: Any, v: Any, weight: str) -> float:
    data = G.get_edge_data(u, v)
    if not data:
        return 0.0
    d = data[min(data)]
    val = d.get(weight)
    if val is None and weight == "travel_time":
        length = float(d.get("length", 0.0) or 0.0)
        speed = float(d.get("speed_kph", 30.0) or 30.0)
        val = length / (speed * 1000.0 / 3600.0) if speed else 0.0
    return float(val or 0.0)


def build_poi_caches(G: nx.MultiDiGraph, pois_gdf, poi_types: Optional[List[str]] = None
                     ) -> Tuple[Dict[str, Set[Any]], Set[Any]]:
    """Precompute (type_nodes, charger_nodes) ONCE for a whole export. Scanning the
    POI table per query is prohibitively slow, so callers build this once and pass
    it to evaluate_system."""
    if poi_types is None:
        poi_types = list(_DISPLAY_TO_CATEGORY.values())
    type_nodes = build_poi_type_nodes(G, pois_gdf, poi_types, "nearest_node_osmid")

    charger: Set[Any] = set()
    mask = None
    if "amenity" in pois_gdf.columns:
        mask = pois_gdf["amenity"].eq("charging_station")
    if "power" in pois_gdf.columns:
        pm = pois_gdf["power"].eq("charging_station")
        mask = pm if mask is None else (mask | pm)
    if mask is not None and "nearest_node_osmid" in pois_gdf.columns:
        for val in pois_gdf[mask]["nearest_node_osmid"].dropna():
            nid = resolve_graph_node_id(G, val)
            if nid is not None:
                charger.add(nid)
    return type_nodes, charger


def score_collection(
    route: List[Any],
    category_limits: Dict[str, int],
    type_nodes: Dict[str, Set[Any]],
    collected_pois: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Score a maximize_poi / poi_collection route's category collection.
    SINGLE SOURCE OF TRUTH for both the live scorer (evaluation.metrics.
    check_constraints, via import) and the offline/export scorer
    (_system_checks below). These were two independent reimplementations that
    quietly drifted apart — confirmed disagreeing on 17/200 real records on
    2026-10-01 (constraints_satisfied 185/200 vs the live metrics' 191/200 on
    gpt-4.1; 132/200 vs 153/200 on gpt-oss-20b). This function exists so they
    cannot diverge again.

    Category caps are UPPER LIMITS, not requirements: a route collecting zero
    POIs, as long as it is under budget, is valid — category_caps_respected
    does NOT require >=1 collected.

    category_membership_valid fails only on a genuine violation: a claimed
    collection matching NONE of the requested categories (a bug — either in
    the generated search's own objective bookkeeping, or a node-id mismatch
    between the route and the search's own possible_targets). A node matching
    MULTIPLE requested categories is genuine, harmless ambiguity — it IS a
    real collected POI, just not attributable to one category, so it counts
    toward num_collection_visits but not any specific category's count, and
    does NOT fail this check. The two are distinguished by collected_pois'
    own `ambiguous_candidates` list: empty => no match (invalid); non-empty
    => real ambiguity between candidates. (craft/generator/tools/
    template.py::_derive_collected_pois already encodes this distinction —
    the bug was only that no consumer checked the list length.)

    When collected_pois is given (craft's own recorded collection
    decisions) it is authoritative. Otherwise falls back to geometric
    route/POI-location overlap, capping each category's count at its limit
    during the scan so an incidental pass-by of an already-full category
    can't manufacture a false over-cap. This fallback can't distinguish "the
    search deliberately stopped here" from "the route happens to drive past
    another same-category POI it never intended to collect" — it is a guess,
    not ground truth, and never fails category_membership_valid (only a
    recorded collected_pois entry can).
    """
    counts: Dict[str, int] = {c: 0 for c in category_limits}
    visited: Set[Any] = set()
    n_ambiguous = 0
    n_invalid = 0

    if collected_pois is not None:
        for entry in collected_pois:
            node = entry.get("node_id")
            if node in visited:
                continue
            visited.add(node)
            cat = entry.get("category")
            if cat is not None and cat in counts:
                counts[cat] += 1
            elif entry.get("ambiguous_candidates"):
                n_ambiguous += 1
            else:
                n_invalid += 1
    else:
        for node in route:
            if node in visited:
                continue
            for cat, lim in category_limits.items():
                if counts[cat] < int(lim) and node in type_nodes.get(cat, set()):
                    counts[cat] += 1
                    visited.add(node)
                    break

    return {
        "counts": counts,
        "visited": visited,
        "category_caps_respected": all(counts[c] <= int(category_limits[c]) for c in category_limits),
        "category_membership_valid": n_invalid == 0,
        "num_collection_visits": sum(counts.values()) + n_ambiguous,
    }


def _system_checks(
    route: List[Any], params: Dict[str, Any], G: nx.MultiDiGraph,
    type_nodes: Dict[str, Set[Any]], charger_nodes: Set[Any],
    template: str = "", collected_pois: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Cache-based reimplementation of evaluation.metrics.check_constraints
    (identical logic, no per-query POI-table scan). Returns the same raw keys.
    collected_pois (maximize_poi only): the search's own recorded collection
    decisions — see score_collection() above for why this makes the
    collection checks authoritative instead of a geometric guess."""
    out: Dict[str, Any] = {}
    route = [coerce(G, n) for n in route]
    route = [n for n in route if n is not None]
    route_set = set(route)

    # EV battery
    # Feasible iff the battery is non-negative at EVERY point along the route,
    # not merely at the end. Checking only the final value (the behaviour up to
    # 2026-08-10) passes routes that run the battery flat and *then* reach a
    # charger, which resets it to max — physically the car is stranded before it
    # ever gets there. Found via the optimal oracle (evaluation/optimal/): three
    # gpt-4.1 ev_routing routes scored feasible while dipping to −348 m,
    # −494 m and −121 m, and were *shorter than the proven optimum* precisely
    # because they skipped the charging detour that makes the trip legal.
    # This now matches _ev_feasible (the LLM/LLMAP baseline path), which has
    # always latched feasible=False on a negative — so all three systems are
    # scored identically. Mirrored in evaluation/metrics.py.
    #
    # Refuelling follows EV_REFUEL_POLICY (see the constant's comment). Under
    # the active "declared_stops_only" policy the refill set is
    # params["declared_charger_nodes"] — the chargers this system committed to.
    # For CRAFT that is the chargers ON ITS ROUTE (set in evaluate_system): its
    # generated code refuses edges it lacks charge for and recharges on reaching
    # a charging_station, so landing on one is a deliberate choice. The
    # baselines' equivalent (_ev_feasible) uses the stops they typed
    # "charging_station".
    if "current_battery" in params or "max_battery" in params:
        max_b = float(params.get("max_battery", float("inf")))
        cur_b = float(params.get("current_battery", max_b))
        if EV_REFUEL_POLICY == "declared_stops_only":
            refill_at = set(params.get("declared_charger_nodes") or ())
        else:
            refill_at = charger_nodes
        battery = cur_b
        nonneg = True
        for u, v in zip(route, route[1:]):
            battery -= _edge_cost(G, u, v, "length")
            if battery < 0.0:
                nonneg = False
            if v in refill_at:
                battery = max_b
        out["ev_battery_nonnegative"] = nonneg

    # Ordered POI + per-leg distance
    poi_types: List[str] = params.get("poi_types", []) or []
    poi_idx = 0
    if poi_types:
        poi_stop_positions: List[int] = []
        if template in _UNORDERED_TEMPLATES:
            # Any-order scan: at each route position, check if node matches any
            # remaining unvisited type. Necessary for unordered_poi where the
            # Dijkstra may visit POIs in a different order than poi_types lists them.
            remaining = list(range(len(poi_types)))
            for pos, node in enumerate(route):
                for ri in range(len(remaining)):
                    if node in type_nodes.get(poi_types[remaining[ri]], set()):
                        poi_stop_positions.append(pos)
                        remaining.pop(ri)
                        poi_idx += 1
                        break
        else:
            for pos, node in enumerate(route):
                if poi_idx < len(poi_types) and node in type_nodes.get(poi_types[poi_idx], set()):
                    poi_stop_positions.append(pos)
                    poi_idx += 1
        out["ordered_all_pois_visited"] = poi_idx == len(poi_types)
        found = sum(1 for t in poi_types if type_nodes.get(t, set()) & route_set)
        out["ordered_type_coverage"] = round(found / len(poi_types), 4)

        limits_m = [float(x) for x in (params.get("distance_limits") or [])]
        if limits_m:
            breakpoints = [0] + poi_stop_positions + [len(route) - 1]
            n_legs = len(breakpoints) - 1
            # Broadcast single limit to all legs ("each leg within X miles").
            if len(limits_m) == 1 and n_legs > 1:
                limits_m = limits_m * n_legs
            all_ok = True
            for i in range(n_legs):
                leg = sum(_edge_cost(G, route[j], route[j + 1], "length")
                          for j in range(breakpoints[i], breakpoints[i + 1]))
                if i < len(limits_m) and leg > limits_m[i]:
                    all_ok = False
                    break
            out["ordered_distance_limits_met"] = all_ok

    # Collection. Category caps are UPPER LIMITS, not requirements — see
    # score_collection() above, the single source of truth shared with
    # evaluation.metrics.check_constraints. collection_within_limits folds
    # BOTH signals (never exceed a cap, never claim a non-matching node) into
    # the one category_limits_met column this template's schema exposes.
    cat_limits: Dict[str, int] = params.get("category_limits", {}) or {}
    collection_counts: Dict[str, int] = {}
    if cat_limits:
        sc = score_collection(route, cat_limits, type_nodes, collected_pois=collected_pois)
        collection_counts = sc["counts"]
        total_req = sum(int(v) for v in cat_limits.values())
        total_col = sum(collection_counts.values())
        out["collection_within_limits"] = sc["category_caps_respected"] and sc["category_membership_valid"]
        out["collection_coverage"] = round(total_col / total_req, 4) if total_req else 1.0

    # Time budget
    if "time_budget_s" in params:
        budget_s = float(params["time_budget_s"])
        travel_s = sum(_edge_cost(G, u, v, "travel_time") for u, v in zip(route, route[1:]))
        if collection_counts:
            # Authoritative per-category counts from score_collection above —
            # charge each category's own LLMAP dwell time directly, instead of
            # _category_visit_total_s's geometric re-derivation, which could
            # assign a node to a different category than what was actually
            # counted (or than what collected_pois itself recorded).
            visit_total_s = sum(
                collection_counts[c] * LLMAP_VISIT_SEC.get(c, LLMAP_VISIT_DEFAULT_SEC)
                for c in collection_counts
            )
        elif poi_types:
            visit_stop_nodes = [route[p] for p in poi_stop_positions]
            visit_total_s = _category_visit_total_s(visit_stop_nodes, poi_types, type_nodes)
        else:
            visit_total_s = 0.0
        out["time_budget_met"] = (travel_s + visit_total_s) <= budget_s

    return out


def _satisfaction(checks: Dict[str, Optional[bool]]) -> Optional[float]:
    """Mean over the applicable (non-None) constraint checks. stops_count_ok is
    intentionally excluded — it is reported separately, not as a hard constraint."""
    vals = [float(v) for k, v in checks.items()
            if k in CHECK_FIELDS and v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def _all_satisfied(checks: Dict[str, Optional[bool]]) -> Optional[bool]:
    """Boolean: True iff EVERY applicable (non-None) hard constraint is True.
    None when no constraint could be evaluated (e.g. no route returned) — an
    empty set must NOT read as vacuously True. stops_count_ok is excluded, same
    as _satisfaction. Count `== True` over a fixed N for a fair cross-system rate
    (a None/empty result is simply not a success, no 0-substitution needed)."""
    vals = [v for k, v in checks.items()
            if k in CHECK_FIELDS and v is not None]
    if not vals:
        return None
    return all(bool(v) for v in vals)


# EV battery feasibility (works for matched and geocode-snapped stops)

def _stop_type(s: Dict) -> str:
    return normalize_type(s.get("type") or s.get("poi_type") or "")


def _ev_feasible(stops: List[Dict], src: Any, dst: Any, cur_m: float, max_m: float,
                 G: nx.MultiDiGraph, snapper: Optional[NodeSnapper]) -> Optional[bool]:
    """Battery never negative along [src → stops → dst] on the road network,
    refilling to max at charging-station stops. None if a stop can't be resolved."""
    stop_nodes: List[Any] = []
    types: List[str] = []
    for s in stops:
        node = coerce(G, s.get("dataset_node"))
        if node is None and snapper is not None:
            glat, glng = s.get("geocoded_lat"), s.get("geocoded_lng")
            if glat is not None and glng is not None and not s.get("geocode_failed"):
                node = snapper.snap(float(glat), float(glng))
        if node is None:
            return None
        stop_nodes.append(node)
        types.append(_stop_type(s))

    chain = [src] + stop_nodes + [dst]
    total, legs = sp_chain(G, chain, "length")
    if legs is None:
        return False
    battery = cur_m
    feasible = True
    for i, leg_m in enumerate(legs):
        battery -= leg_m
        if battery < 0:
            feasible = False
        if i < len(types) and types[i] == "charging_station":
            battery = max_m
    return feasible


def _stop_identity(s: Dict) -> Tuple[str, str, str]:
    return (str(s.get("name", "")).strip(), str(s.get("address", "")),
            str(s.get("type", s.get("poi_type", s.get("claimed_type", "")))))


def _reconstruct_stop_order(
    llm_stops: List[Dict], matched_stops: List[Dict], dropped_stops: List[Dict],
) -> List[Tuple[str, Dict]]:
    """matched_stops/dropped_stops are each an order-preserving SUBSEQUENCE of
    llm_stops (match_stops() partitions llm_stops into one or the other while
    iterating it in order — see baselines/llm/llm_baseline.py::match_stops).
    Recover the true interleaved visiting order via a two-pointer merge on
    (name, address, type) identity. A stop that can't be matched to either
    pointer (shouldn't happen) is skipped rather than risking a wrong order."""
    mi = di = 0
    out: List[Tuple[str, Dict]] = []
    for ls in llm_stops:
        k = _stop_identity(ls)
        if mi < len(matched_stops) and _stop_identity(matched_stops[mi]) == k:
            out.append(("matched", matched_stops[mi]))
            mi += 1
        elif di < len(dropped_stops) and _stop_identity(dropped_stops[di]) == k:
            out.append(("dropped", dropped_stops[di]))
            di += 1
    return out


def objective_stops_incl_unmatched_real(
    llm_stops: List[Dict], matched_stops: List[Dict], dropped_stops: List[Dict],
    hallucination_dropped: Optional[List[Dict]], G: nx.MultiDiGraph,
    snapper: Optional[NodeSnapper],
) -> List[Dict]:
    """Objective stop list for the LLM baseline, WIDENED to also include a
    dropped stop when the hallucination pass (evaluate_llm_hallucination.py,
    Google Places) confirms it is a REAL place of the correct type — just one
    our OSM extract doesn't contain. Such a stop is snapped to its nearest
    road-network node (Places-verified coords preferred over the raw geocode)
    and inserted at its true position in the sequence, so the objective isn't
    silently cheaper just because our dataset has a coverage gap. A dropped
    stop that is NOT verified real (actually hallucinated, or the
    hallucination pass hasn't been run on it) is still excluded — never
    fabricate a waypoint for a place we can't confirm exists.
    EV infrastructure stops (chargers) are excluded, matching obj_stops'
    existing convention — the objective is the cost of the required POIs only.
    """
    if snapper is None:
        return [s for s in matched_stops if _stop_type(s) not in EV_INFRA]

    hlut: Dict[Tuple[str, str, str], Dict] = {}
    for h in (hallucination_dropped or []):
        hlut[_stop_identity(h)] = h

    out: List[Dict] = []
    for kind, s in _reconstruct_stop_order(llm_stops, matched_stops, dropped_stops):
        if _stop_type(s) in EV_INFRA:
            continue
        if kind == "matched":
            out.append(s)
            continue
        h = hlut.get(_stop_identity(s))
        if h is None or h.get("hallucination_type") is not None:
            continue  # unverified, or actually hallucinated — don't fabricate a stop
        lat = h.get("places_lat") if h.get("places_lat") is not None else s.get("geocoded_lat")
        lng = h.get("places_lng") if h.get("places_lng") is not None else s.get("geocoded_lng")
        if lat is None or lng is None:
            continue
        node = snapper.snap(float(lat), float(lng))
        if node is None:
            continue
        out.append({**s, "dataset_node": str(node), "match_method": "verified_real_unmatched_snap"})
    return out


# Baseline (LLM / LLMAP) record evaluation

def evaluate_baseline(
    nl_query: str,
    template: str,
    matched_stops: List[Dict],
    G: Optional[nx.MultiDiGraph],
    snapper: Optional[NodeSnapper],
    all_stops: Optional[List[Dict]] = None,
    objective_stops_incl_unmatched_real: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Score one baseline route (a list of matched stops) against the template's
    constraint set, and compute the comparable objective. Returns the CORE_FIELDS
    dict (identical schema for every system).

    matched_stops : stops resolved to dataset_node (used for checks + objective).
    all_stops     : full proposed stop list (used for EV battery, may include
                    geocode-only stops). Defaults to matched_stops.
    objective_stops_incl_unmatched_real : optional widened stop list from
                    objective_stops_incl_unmatched_real() (LLM baseline only) —
                    when given, also reports objective_value_incl_unmatched_real,
                    the SP cost through matched stops PLUS any dropped stop the
                    hallucination pass verified as real. None (the default)
                    leaves that column None — this never affects
                    objective_value itself.
    """
    active = TEMPLATE_CHECKS.get(template, set())
    wkey = weight_key(template)
    checks: Dict[str, Optional[bool]] = {f: None for f in CHECK_FIELDS}

    req = required_stops(nl_query, template)
    n_stops = len(matched_stops)
    stops_count_ok = (n_stops == req) if req is not None else None

    stop_types = [_stop_type(s) for s in matched_stops]
    # Objective / distance chain excludes EV infrastructure stops.
    obj_stops = [s for s in matched_stops if _stop_type(s) not in EV_INFRA]

    src = dst = None
    coords = parse_coords(nl_query)
    if G is not None and snapper is not None and len(coords) >= 2:
        src = snapper.snap(*coords[0])
        dst = snapper.snap(*coords[-1])

    # POI type / order
    if "poi_type_match" in active:
        required = required_poi_types(nl_query)
        non_infra_types = [t for t in stop_types if t not in EV_INFRA]
        if required:
            if template in _UNORDERED_TEMPLATES:
                checks["poi_type_match"] = (sorted(non_infra_types) == sorted(required))
            else:
                seq_ok = (len(non_infra_types) == len(required)
                          and all(a == b for a, b in zip(non_infra_types, required)))
                checks["poi_type_match"] = seq_ok
                if "poi_order" in active:
                    checks["poi_order"] = seq_ok

    # Category limits
    if "category_limits_met" in active:
        limits = required_cat_limits(nl_query)
        if limits:
            counts: Dict[str, int] = {}
            for t in stop_types:
                counts[t] = counts.get(t, 0) + 1
            within = all(counts.get(c, 0) <= lim for c, lim in limits.items())
            checks["category_limits_met"] = within and len(stop_types) > 0

    # Road-network legs (distance / time) — SP through chosen stops
    if G is not None and src is not None and dst is not None:
        obj_nodes = [coerce(G, s.get("dataset_node")) for s in obj_stops]
        # Build a category -> {nodes} mapping from the matched stops so the
        # time_budget_met check can charge per-category LLMAP dwell times.
        obj_type_nodes: Dict[str, Set[Any]] = {}
        obj_cats: List[str] = []
        for s, nd in zip(obj_stops, obj_nodes):
            t = _stop_type(s)
            if t not in obj_type_nodes:
                obj_type_nodes[t] = set()
                obj_cats.append(t)
            if nd is not None:
                obj_type_nodes[t].add(nd)
        checks.update(_road_network_checks(G, src, dst, obj_nodes, nl_query, active,
                                           poi_types=obj_cats, type_nodes=obj_type_nodes))
        if "road_preference_met" in active and all(n is not None for n in obj_nodes):
            # weight="length", NOT wkey. Highway usage is a DISTANCE question, and
            # poi_road_pref's template weight is travel_time — so passing wkey here
            # summed SECONDS and the mileage columns divided them by metres-per-mile,
            # reporting a 5.1-mile route as "0.36 miles" (14x low). The CRAFT path
            # below already hard-codes "length"; this is the same basis, so the two
            # are comparable. The share columns were only mildly affected (a
            # time-share rather than a distance-share) which is why this survived
            # until absolute miles were added and made it obvious.
            checks["road_preference_met"] = road_preference_met_baseline(
                G, src, dst, obj_nodes, nl_query, weight="length")
            checks.update(road_pref_metrics(G, src, dst, obj_nodes, nl_query, weight="length"))

    # EV battery
    if "ev_battery_feasible" in active and G is not None and src is not None and dst is not None:
        cur_m, max_m = parse_battery_m(nl_query)
        if cur_m is not None and max_m is not None:
            # Keep all proposed stops — charger stops refill battery mid-route.
            ev_stops = all_stops if all_stops is not None else matched_stops
            checks["ev_battery_feasible"] = _ev_feasible(
                ev_stops, src, dst, cur_m, max_m, G, snapper)

    # Objective
    obj_value: Optional[float] = None
    if G is not None and src is not None and dst is not None:
        obj_nodes = [coerce(G, s.get("dataset_node")) for s in obj_stops]
        if all(n is not None for n in obj_nodes):
            obj_value, _ = sp_chain(G, [src] + obj_nodes + [dst], wkey)

    obj_value_incl_unmatched_real: Optional[float] = None
    n_unmatched_real_included = 0
    if objective_stops_incl_unmatched_real is not None and G is not None \
            and src is not None and dst is not None:
        wide_stops = [s for s in objective_stops_incl_unmatched_real if _stop_type(s) not in EV_INFRA]
        n_unmatched_real_included = sum(
            1 for s in wide_stops if s.get("match_method") == "verified_real_unmatched_snap")
        wide_nodes = [coerce(G, s.get("dataset_node")) for s in wide_stops]
        if all(n is not None for n in wide_nodes):
            obj_value_incl_unmatched_real, _ = sp_chain(G, [src] + wide_nodes + [dst], wkey)

    row: Dict[str, Any] = {
        "required_stops": req,
        "n_stops": n_stops,
        "stops_count_ok": stops_count_ok,
        # Seeded so every row carries the road-pref metric keys (None outside
        # poi_road_pref); `**checks` overrides them where they were computed.
        **_EMPTY_ROAD_PREF_METRICS,
        **checks,
        "constraint_satisfaction": _satisfaction(checks),
        "constraints_satisfied": _all_satisfied(checks),
        "objective_type": objective_type(template),
        "objective_value": obj_value,
        "objective_value_incl_unmatched_real": obj_value_incl_unmatched_real,
        "n_unmatched_real_included": n_unmatched_real_included,
    }
    return row


# System (Craft) record evaluation

def _intended_stop_nodes(
    route: List[Any], template: str, poi_types: List[str],
    cat_limits: Dict[str, int], G: nx.MultiDiGraph, type_nodes: Dict[str, Set[Any]],
) -> List[Any]:
    """Ordered list of route nodes that are the *intended* POI stops."""
    route = [coerce(G, n) for n in route]
    route = [n for n in route if n is not None]

    if poi_types:
        if template in _UNORDERED_TEMPLATES:
            picked: List[Tuple[int, Any]] = []
            used_types: Set[str] = set()
            for pos, node in enumerate(route):
                for t in poi_types:
                    if t in used_types:
                        continue
                    if node in type_nodes.get(t, set()):
                        picked.append((pos, node))
                        used_types.add(t)
                        break
            return [n for _, n in sorted(picked)]
        # ordered: match required types in sequence
        idx = 0
        out: List[Any] = []
        for node in route:
            if idx < len(poi_types) and node in type_nodes.get(poi_types[idx], set()):
                out.append(node)
                idx += 1
        return out

    if cat_limits:
        counts = {c: 0 for c in cat_limits}
        visited: Set[Any] = set()
        out = []
        for node in route:
            if node in visited:
                continue
            for c, lim in cat_limits.items():
                if counts[c] < int(lim) and node in type_nodes.get(c, set()):
                    counts[c] += 1
                    visited.add(node)
                    out.append(node)
                    break
        return out

    return []


def evaluate_system(
    nl_query: str,
    template: str,
    route: List[Any],
    G: Optional[nx.MultiDiGraph],
    pois_gdf,
    snapper: Optional[NodeSnapper],
    caches: Optional[Tuple[Dict[str, Set[Any]], Set[Any]]] = None,
    collected_pois: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Score one Craft route (full ordered node list) against the template's
    constraint set. Returns CORE_FIELDS plus dijkstra_objective_value.

    Constraint logic mirrors evaluation.metrics.check_constraints exactly but
    uses precomputed POI-type / charger node sets (caches) so a whole export
    doesn't rescan the 600k-row POI table per query. Pass caches from
    build_poi_caches(G, pois_gdf); if None it is built once here from pois_gdf.

    collected_pois (maximize_poi / poi_collection only, optional): the
    record's own result.collected_pois — see score_collection() for why this
    makes the collection checks authoritative instead of a geometric guess.
    """
    active = TEMPLATE_CHECKS.get(template, set())
    wkey = weight_key(template)
    checks: Dict[str, Optional[bool]] = {f: None for f in CHECK_FIELDS}

    poi_types = required_poi_types(nl_query) if (
        {"poi_type_match", "poi_order"} & active) else []
    cat_limits = required_cat_limits(nl_query) if "category_limits_met" in active else {}

    req = required_stops(nl_query, template)

    type_nodes: Dict[str, Set[Any]] = {}
    charger_nodes: Set[Any] = set()
    if G is not None and route:
        if caches is None and pois_gdf is not None:
            caches = build_poi_caches(G, pois_gdf)
        if caches is not None:
            type_nodes, charger_nodes = caches

    # Constraint checks (cache-based mirror of metrics.check_constraints)
    if G is not None and route:
        params: Dict[str, Any] = {}
        if poi_types:
            params["poi_types"] = poi_types
        if cat_limits:
            params["category_limits"] = cat_limits
        if "distance_limits_met" in active:
            params["distance_limits"] = parse_distance_limits_m(nl_query)
        if "time_budget_met" in active:
            b = parse_time_budget_s(nl_query)
            if b is not None:
                params["time_budget_s"] = b
                params["visit_time_s"] = parse_visit_time_s(nl_query)
        if "ev_battery_feasible" in active:
            cur_m, max_m = parse_battery_m(nl_query)
            if cur_m is not None and max_m is not None:
                params["current_battery"] = cur_m
                params["max_battery"] = max_m
                # Chargers CRAFT COMMITTED to = the charging stations on the
                # route it returned. Its generated search code recharges at
                # every charging_station it steps onto and refuses edges it
                # lacks the charge for, so reaching one is a deliberate choice
                # by the search, not an incidental pass-by. See the
                # EV_REFUEL_POLICY note — treating this set as empty was tried
                # and was wrong.
                params["declared_charger_nodes"] = {
                    n for n in (coerce(G, x) for x in route) if n in charger_nodes
                }

        try:
            raw = _system_checks(route, params, G, type_nodes, charger_nodes, template=template,
                                 collected_pois=collected_pois)
        except Exception:
            raw = {}

        def _b(v):
            if v is None:
                return None
            if isinstance(v, bool):
                return v
            try:
                return float(v) >= 1.0
            except (TypeError, ValueError):
                return None

        if "poi_type_match" in active:
            # Ordered templates: sequence check (match baseline behaviour where
            # poi_type_match == poi_order). Unordered: set check (all types anywhere).
            if template in _ORDERED_TEMPLATES:
                checks["poi_type_match"] = _b(raw.get("ordered_all_pois_visited"))
            else:
                checks["poi_type_match"] = _b(raw.get("ordered_type_coverage"))
        if "poi_order" in active:
            checks["poi_order"] = _b(raw.get("ordered_all_pois_visited"))
        if "category_limits_met" in active:
            # within-caps + ≥1 collected, matching the baseline definition
            checks["category_limits_met"] = _b(raw.get("collection_within_limits"))
        if "ev_battery_feasible" in active:
            checks["ev_battery_feasible"] = _b(raw.get("ev_battery_nonnegative"))
        # Realized-route values kept as *_realized extras; the SHARED comparison
        # columns (distance_limits_met / time_budget_met) are set below via SP
        # through the intended stops, matching the baselines exactly.
        if "time_budget_met" in active:
            checks["time_budget_met_realized"] = _b(raw.get("time_budget_met"))
        if "distance_limits_met" in active:
            checks["distance_limits_met_realized"] = _b(raw.get("ordered_distance_limits_met"))

    # Intended stops + objective
    n_stops = 0
    obj_value: Optional[float] = None
    dijkstra_obj: Optional[float] = None
    if G is not None and route:
        intended = _intended_stop_nodes(route, template, poi_types, cat_limits, G, type_nodes)
        n_stops = len(intended)

        src = dst = None
        coords = parse_coords(nl_query)
        if snapper is not None and len(coords) >= 2:
            src = snapper.snap(*coords[0])
            dst = snapper.snap(*coords[-1])
        if src is None:
            src = coerce(G, route[0])
        if dst is None:
            dst = coerce(G, route[-1])

        if src is not None and dst is not None:
            obj_value, _ = sp_chain(G, [src] + intended + [dst], wkey)
            # Shared distance/time checks on the SAME SP-through-stops basis as baselines.
            sp_cats = poi_types or list(cat_limits)
            checks.update(_road_network_checks(G, src, dst, intended, nl_query, active,
                                               poi_types=sp_cats, type_nodes=type_nodes))
            if "road_preference_met" in active:
                # Shared column: SAME reconstructed-SP method as baselines, through
                # CRAFT's own chosen stops — apples-to-apples across all 3 systems.
                checks["road_preference_met"] = road_preference_met_baseline(
                    G, src, dst, intended, nl_query, weight="length",
                )
                checks.update(road_pref_metrics(G, src, dst, intended, nl_query, weight="length"))
                # Craft-only extras: scores CRAFT's actual realized route (the real
                # capability signal — whether its own generated code respects the
                # preference), same convention as distance/time_budget_met_realized.
                checks["road_preference_met_realized"] = road_preference_met(
                    G, route, [src] + intended + [dst], nl_query, weight="length",
                )
                checks.update(road_pref_metrics_realized(
                    G, route, [src] + intended + [dst], nl_query, weight="length"))
        dijkstra_obj = compute_system_objective(route, G, wkey)

    stops_count_ok = (n_stops == req) if req is not None else None

    row: Dict[str, Any] = {
        "required_stops": req,
        "n_stops": n_stops,
        "stops_count_ok": stops_count_ok,
        # Seeded so every row carries the road-pref metric keys (None outside
        # poi_road_pref); `**checks` overrides them where they were computed.
        **_EMPTY_ROAD_PREF_METRICS,
        **_EMPTY_ROAD_PREF_METRICS_REALIZED,
        **checks,
        "constraint_satisfaction": _satisfaction(checks),
        "constraints_satisfied": _all_satisfied(checks),
        "objective_type": objective_type(template),
        "objective_value": obj_value,
        "dijkstra_objective_value": dijkstra_obj,
    }
    return row


# ---------------------------------------------------------------------------
# HIPP pilot — gold-driven evaluation (benchmark/socal_queries/llmap_queries/hipp_meta.json)
#
# Unlike every template above, HIPP's "required" spec (poi_types, dependencies,
# clock deadline, quality/distance weights) is NOT re-derived from NL text —
# hipp_meta.json is reliable synthetic ground truth (it IS the generation
# input), and the display-name regex above (_DISPLAY_TO_CATEGORY) doesn't even
# know about "mall". Using gold keeps CRAFT, LLMAP, and the LLM baseline
# scored against the exact same required-stop/order/deadline spec.
# ---------------------------------------------------------------------------

HIPP_META_PATH = ROOT / "benchmark" / "socal_queries" / "llmap_queries" / "hipp_meta.json"

# Must match craft_hipp/parser/agent.py::HIPP_RATING_THRESHOLD /
# HIPP_DEPARTURE_SEC exactly — these three places define the same rule
# (generation-time filter, generation-time deadline, and this scoring).
HIPP_RATING_THRESHOLD = 4.0
HIPP_DEPARTURE_SEC = 10 * 3600

_HIPP_MALL_ALIASES = {"shop", "shopping_mall", "shopping mall"}


def load_hipp_gold(meta_path: Path = HIPP_META_PATH) -> Dict[str, Dict[str, Any]]:
    """query_id -> hipp_meta.json record. Empty dict if the file doesn't exist."""
    if not meta_path.exists():
        return {}
    import json
    return {r["query_id"]: r for r in json.loads(meta_path.read_text(encoding="utf-8"))}


def _normalize_hipp_type(t: Any) -> str:
    """'shopping mall' / 'shopping_mall' / the old buggy 'shop' -> 'mall';
    everything else passes through lowercased."""
    tn = str(t or "").strip().lower()
    return "mall" if tn in _HIPP_MALL_ALIASES else tn


def build_hipp_rating_cache(G: nx.MultiDiGraph, pois_gdf, node_col: str = "nearest_node_osmid",
                           rating_col: str = "rating") -> Dict[Any, float]:
    """node_id -> max synthetic `rating` among POI rows resolving to that node.
    Build once per export (like build_poi_caches) and pass to
    evaluate_hipp_system/evaluate_hipp_baseline. Empty if the gpkg has no
    rating column (non-HIPP dataset)."""
    cache: Dict[Any, float] = {}
    if pois_gdf is None or rating_col not in pois_gdf.columns or node_col not in pois_gdf.columns:
        return cache
    import pandas as pd
    numeric = pd.to_numeric(pois_gdf[rating_col], errors="coerce")
    for node_raw, r in zip(pois_gdf[node_col], numeric):
        if r != r:  # NaN
            continue
        nid = resolve_graph_node_id(G, node_raw)
        if nid is None:
            continue
        prev = cache.get(nid)
        if prev is None or r > prev:
            cache[nid] = float(r)
    return cache


def _hipp_type_nodes_for_gold(
    required: List[str], gold: Dict[str, Any],
    base_type_nodes: Dict[str, Set[Any]], rating_cache: Optional[Dict[Any, float]],
) -> Dict[str, Set[Any]]:
    """Candidate node set per category, matching what the generator's
    self.possible_targets actually used — plain category membership, UNLESS
    the query is quality-emphasized (gold quality_weight>0.5), in which case
    craft_hipp's generator rating-filters candidates (rating>=HIPP_RATING_THRESHOLD,
    with craft_hipp/generator/tools/filtration.py's own broaden-on-empty
    fallback). Reconstructing route "claims" against the wrong (unfiltered)
    node set lets a physically-nearby but non-qualifying node get credited
    instead of the one the search actually used — this must match generation
    exactly or rating_ok gets scored against the wrong node.

    Uses the already-built rating_cache (node -> rating) for an O(1) set
    intersection per category — NOT a fresh pois_gdf scan per query. An
    earlier version called filtration.build_possible_targets directly here,
    which rebuilds its own category->node map from scratch (a full 627k-row
    pois_gdf scan) on EVERY quality-emphasized query; across ~100 such
    queries in a 200-query export that took over 20 minutes and counting.
    """
    qw = gold.get("quality_weight")
    if qw is None or float(qw) <= 0.5 or not rating_cache:
        return base_type_nodes
    rated_nodes = {n for n, r in rating_cache.items() if r >= HIPP_RATING_THRESHOLD}
    out: Dict[str, Set[Any]] = {}
    for cat in required:
        nodes = base_type_nodes.get(cat, set())
        narrowed = nodes & rated_nodes
        out[cat] = narrowed if narrowed else nodes  # broaden-on-empty, matches filtration.py
    return out


def _typed_seq_from_route(
    route: List[Any], required_types: List[str], type_nodes: Dict[str, Set[Any]],
) -> List[Tuple[Any, str]]:
    """(node, category) pairs in route order. A node that belongs to more than
    one required category's node set yields one entry per matching category
    (in required_types order), NOT just the first — this mirrors the
    generated expand()'s own `for i, poi_type in enumerate(self.poi_types):
    if neighbor in self.possible_targets.get(i, set()): ...` loop, which has
    no `break` and can mark multiple categories satisfied by the same
    neighbor in one step (its dependency gate even uses the just-updated
    flag from earlier in that same loop, so co-located stops can satisfy a
    dependency at the same arrival). Reconstructing with a first-match
    `break` here would silently undercount real successes."""
    out: List[Tuple[Any, str]] = []
    for node in route:
        for t in required_types:
            if node in type_nodes.get(t, set()):
                out.append((node, t))
    return out


def _hipp_claim_from_typed_sequence(
    typed_seq: List[Tuple[Any, str]], required_types: List[str], dependencies: List[List[str]],
) -> Dict[str, Any]:
    """Greedy left-to-right claim of each required category to the first
    (node, category) pair where the category is not yet claimed AND every
    dependency predecessor of that category (per `dependencies`, [before,
    after] pairs) is already claimed. Mirrors the craft_hipp generator's own
    dependency-gated visited-tracking (see craft_hipp/generator/prompt.md,
    "Ordering" section), so evaluation checks the SAME interpretation of
    "visited in the required order" as generation enforces — not just whether
    a correctly-ordered occurrence exists anywhere in the sequence.

    Returns {category: node_id}, insertion-ordered by claim position (i.e. by
    visitation order) — used directly as the objective/deadline stop chain.
    """
    claimed: Dict[str, Any] = {}
    for node, cat in typed_seq:
        if cat not in required_types or cat in claimed:
            continue
        blocked = any(dep[1] == cat and dep[0] not in claimed for dep in dependencies)
        if blocked:
            continue
        claimed[cat] = node
    return claimed


def _hipp_rating_ok(
    stop_nodes: List[Any], gold: Dict[str, Any], rating_cache: Dict[Any, float],
) -> Optional[bool]:
    """True iff every claimed stop's real rating >= HIPP_RATING_THRESHOLD.
    None if the query isn't quality-emphasized (gold quality_weight <= 0.5) —
    matches the same gate used to decide whether CRAFT applies the filter at
    generation time, so the check is only active when the query actually asks
    for it."""
    qw = gold.get("quality_weight")
    if qw is None:
        return None
    try:
        if float(qw) <= 0.5:
            return None
    except (TypeError, ValueError):
        return None
    if not stop_nodes:
        return False
    return all((rating_cache.get(n) or 0.0) >= HIPP_RATING_THRESHOLD for n in stop_nodes)


def _hipp_clock_deadline_met(
    G: nx.MultiDiGraph, src: Any, dst: Any, claimed: Dict[str, Any], gold: Dict[str, Any],
) -> Optional[bool]:
    """Absolute arrival check: HIPP_DEPARTURE_SEC (10:00 AM) + SP travel time
    through [src, claimed stops in visitation order, dst] + per-category dwell
    time (LLMAP_VISIT_SEC) <= the query's clock deadline. None if the query
    has no deadline. Same computation for every system — matches LLMAP's own
    10 AM departure convention (baselines/LLMAP/main/utils.py) and the
    craft_hipp generator's elapsed_s model exactly."""
    time_limit = gold.get("time_limit")
    if not time_limit or str(time_limit) == "None":
        return None
    try:
        deadline_sec = int(str(time_limit).split(":")[0]) * 3600
    except (ValueError, IndexError):
        return None
    if G is None or src is None or dst is None:
        return False
    chain = [src] + list(claimed.values()) + [dst]
    _, legs = sp_chain(G, chain, "travel_time")
    if legs is None:
        return False
    dwell = sum(LLMAP_VISIT_SEC.get(c, LLMAP_VISIT_DEFAULT_SEC) for c in claimed)
    elapsed = sum(legs) + dwell
    return (HIPP_DEPARTURE_SEC + elapsed) <= deadline_sec


def _hipp_src_dst(gold: Dict[str, Any], G: Optional[nx.MultiDiGraph],
                  snapper: Optional[NodeSnapper]) -> Tuple[Optional[Any], Optional[Any]]:
    coords = parse_coords(gold.get("nl_query", ""))
    if G is not None and snapper is not None and len(coords) >= 2:
        return snapper.snap(*coords[0]), snapper.snap(*coords[-1])
    return None, None


def evaluate_hipp_system(
    gold: Dict[str, Any],
    route: List[Any],
    G: Optional[nx.MultiDiGraph],
    pois_gdf,
    snapper: Optional[NodeSnapper],
    caches: Optional[Tuple[Dict[str, Set[Any]], Set[Any]]] = None,
    rating_cache: Optional[Dict[Any, float]] = None,
) -> Dict[str, Any]:
    """Score one craft_hipp route (full realized node list) against its HIPP
    gold record. Returns CORE_FIELDS (same schema as evaluate_system)."""
    template = gold.get("template") or "hipp_unordered"
    active = TEMPLATE_CHECKS.get(template, set())
    wkey = weight_key(template)
    checks: Dict[str, Optional[bool]] = {f: None for f in CHECK_FIELDS}

    required = [_normalize_hipp_type(t) for t in (gold.get("poi_types_mapped") or [])]
    dependencies = [[_normalize_hipp_type(a), _normalize_hipp_type(b)]
                    for a, b in (gold.get("dependencies_mapped") or [])]
    req = len(required) if required else None

    route_c = [coerce(G, n) for n in (route or [])] if G is not None else list(route or [])
    route_c = [n for n in route_c if n is not None]

    type_nodes: Dict[str, Set[Any]] = {}
    if G is not None and route_c and required:
        if caches is None and pois_gdf is not None:
            caches = build_poi_caches(G, pois_gdf, poi_types=required)
        if caches is not None:
            type_nodes, _ = caches
        type_nodes = _hipp_type_nodes_for_gold(required, gold, type_nodes, rating_cache)

    claimed: Dict[str, Any] = {}
    if required and type_nodes:
        claimed = _hipp_claim_from_typed_sequence(
            _typed_seq_from_route(route_c, required, type_nodes), required, dependencies)

    n_stops = len(claimed)
    stops_count_ok = (n_stops == req) if req is not None else None

    if required:
        type_ok = (set(claimed) == set(required))
        if "poi_type_match" in active:
            checks["poi_type_match"] = type_ok
        if "poi_order" in active:
            checks["poi_order"] = type_ok

    src, dst = _hipp_src_dst(gold, G, snapper)
    if src is None and route_c:
        src = route_c[0]
    if dst is None and route_c:
        dst = route_c[-1]

    ordered_stop_nodes = list(claimed.values())
    obj_value: Optional[float] = None
    if G is not None and src is not None and dst is not None:
        obj_value, _ = sp_chain(G, [src] + ordered_stop_nodes + [dst], wkey)

    if "rating_ok" in active:
        checks["rating_ok"] = _hipp_rating_ok(ordered_stop_nodes, gold, rating_cache or {})
    if "clock_deadline_met" in active:
        checks["clock_deadline_met"] = _hipp_clock_deadline_met(G, src, dst, claimed, gold)

    return {
        "required_stops": req,
        "n_stops": n_stops,
        "stops_count_ok": stops_count_ok,
        **checks,
        "constraint_satisfaction": _satisfaction(checks),
        "constraints_satisfied": _all_satisfied(checks),
        "objective_type": objective_type(template),
        "objective_value": obj_value,
    }


def evaluate_hipp_baseline(
    gold: Dict[str, Any],
    matched_stops: List[Dict],
    G: Optional[nx.MultiDiGraph],
    snapper: Optional[NodeSnapper],
    rating_cache: Optional[Dict[Any, float]] = None,
) -> Dict[str, Any]:
    """Score one baseline (LLM / LLMAP) HIPP result — a list of matched stop
    dicts (each with 'dataset_node' and 'poi_type'/'type'), in the order the
    system itself proposed them — against its HIPP gold record. Returns
    CORE_FIELDS (same schema as evaluate_baseline)."""
    template = gold.get("template") or "hipp_unordered"
    active = TEMPLATE_CHECKS.get(template, set())
    wkey = weight_key(template)
    checks: Dict[str, Optional[bool]] = {f: None for f in CHECK_FIELDS}

    required = [_normalize_hipp_type(t) for t in (gold.get("poi_types_mapped") or [])]
    dependencies = [[_normalize_hipp_type(a), _normalize_hipp_type(b)]
                    for a, b in (gold.get("dependencies_mapped") or [])]
    req = len(required) if required else None

    typed_seq: List[Tuple[Any, str]] = []
    for s in matched_stops:
        node = s.get("dataset_node")
        node = coerce(G, node) if G is not None else node
        cat = _normalize_hipp_type(s.get("poi_type") or s.get("type"))
        if node is not None:
            typed_seq.append((node, cat))

    claimed = _hipp_claim_from_typed_sequence(typed_seq, required, dependencies) if required else {}
    n_stops = len(claimed)
    stops_count_ok = (n_stops == req) if req is not None else None

    if required:
        type_ok = (set(claimed) == set(required))
        if "poi_type_match" in active:
            checks["poi_type_match"] = type_ok
        if "poi_order" in active:
            checks["poi_order"] = type_ok

    src, dst = _hipp_src_dst(gold, G, snapper)

    ordered_stop_nodes = list(claimed.values())
    obj_value: Optional[float] = None
    if G is not None and src is not None and dst is not None:
        obj_value, _ = sp_chain(G, [src] + ordered_stop_nodes + [dst], wkey)

    if "rating_ok" in active:
        checks["rating_ok"] = _hipp_rating_ok(ordered_stop_nodes, gold, rating_cache or {})
    if "clock_deadline_met" in active:
        checks["clock_deadline_met"] = _hipp_clock_deadline_met(G, src, dst, claimed, gold)

    return {
        "required_stops": req,
        "n_stops": n_stops,
        "stops_count_ok": stops_count_ok,
        **checks,
        "constraint_satisfaction": _satisfaction(checks),
        "constraints_satisfied": _all_satisfied(checks),
        "objective_type": objective_type(template),
        "objective_value": obj_value,
    }


# ---------------------------------------------------------------------------
# set3_descriptions pilot — gold-driven filtered-POI evaluation
# (benchmark/socal_queries/set3_descriptions/)
#
# Gold-driven for the same reason HIPP is, plus a stronger one. The required
# poi_types could *almost* be regexed out of the NL text, but the FILTER a stop
# must satisfy ("a restaurant that offers takeout", "an aviation museum") has no
# machine-checkable form anywhere in the query string — the column/value pair and
# the description phrase exist only in set3_meta.json, which IS the generation
# input. required_poi_types() would also mis-split set3's phrasing ("a restaurant
# that offers takeout within 7 miles of me, then a cafe with Wi-Fi"), so both the
# types and the limits are taken from gold too, keeping one consistent basis.
# ---------------------------------------------------------------------------

SET3_META_PATH        = ROOT / "benchmark" / "socal_queries" / "set3_descriptions" / "set3_meta.json"
SET3_QUERY_EMBED_PATH = ROOT / "data" / "socal" / "set3_query_embeddings.json"
SET3_POI_EMBED_PATH   = ROOT / "data" / "socal" / "poi_embeddings.json"

SET3_TEMPLATE = "ordered+desc"

# The 5 categories set3 actually uses (restaurant/cafe/park are the base three;
# museum/attraction carry the description filters). build_poi_caches() defaults
# to the 4 LLMAP display types, so callers MUST pass this list explicitly or
# every set3 stop silently fails to match — same trap as HIPP_POI_TYPES.
SET3_POI_TYPES = ["restaurant", "cafe", "park", "museum", "attraction"]

# Both MUST match craft/generator/tools/filtration.py exactly (its sim_threshold
# default and the model the runtime poi_embeddings cache was built with). If
# filtration's threshold moves, this moves with it or the check stops describing
# what the search actually did.
# 2026-08-17: was 0.65 while filtration used 0.75 — the "MUST match" above was not
# actually true, and neither value was right. Measured against LLM-labelled ground
# truth (results/craft_description_filter.xlsx), the best-separating cutoff for the
# set3 description phrases is ~0.55; the maximum similarity any POI attains is 0.748,
# so 0.75 admitted nothing at all. This is a POINT judgement ("does the POI the route
# visited match the description?"), so it uses the absolute floor only — the relative
# band in filtration is for SET selection and has no meaning for a single POI.
# 2026-08-22: tracked filtration's floor 0.55 -> 0.50 when the relative band and the
# contrastive rule were removed there.
# ⚠️ ASYMMETRY, deliberate and worth understanding before reading set3 scores: the
# production FILTER is now 0.50 + an LLM verification pass, while this SCORER stays
# 0.50 cosine only. The scorer is therefore strictly the more permissive of the two —
# it accepts every POI the filter would have accepted, plus those the LLM pruned. So
# `poi_filters_respected` cannot fail merely because the LLM disagreed, and it does
# NOT independently re-verify the description; it checks the floor only. Making the
# scorer call the LLM too would tighten it, but changes every published set3 number,
# so it is left as an explicit decision rather than a silent one.
SET3_SIM_THRESHOLD = 0.50
SET3_EMBED_MODEL   = "gemini-embedding-001"

TEMPLATE_CHECKS[SET3_TEMPLATE] = {
    "poi_type_match", "poi_order", "distance_limits_met", "poi_filters_respected",
}
# Every set3 query asks for "the shortest-distance route".
TEMPLATE_WEIGHT_KEY[SET3_TEMPLATE] = "length"


def load_set3_gold(meta_path: Path = SET3_META_PATH) -> Dict[str, Dict[str, Any]]:
    """query_id -> set3_meta.json record. Empty dict if the file doesn't exist."""
    if not meta_path.exists():
        return {}
    import json
    return {r["query_id"]: r for r in json.loads(meta_path.read_text(encoding="utf-8"))}


def _set3_gold_filter_pairs(gold_records: Sequence[Dict[str, Any]]) -> Set[Tuple[str, str]]:
    """Distinct (column, value) structured filter pairs across all gold records —
    ~20 of them, vs 414 stops, so the node sets are built once per pair."""
    pairs: Set[Tuple[str, str]] = set()
    for rec in gold_records:
        for stop in rec.get("stops") or []:
            if stop.get("filter_kind") == "structured" and stop.get("column"):
                pairs.add((str(stop["column"]), str(stop.get("value"))))
    return pairs


def build_set3_filter_cache(
    G: nx.MultiDiGraph, pois_gdf, gold_records: Sequence[Dict[str, Any]],
    node_col: str = "nearest_node_osmid",
    poi_embed_path: Path = SET3_POI_EMBED_PATH,
) -> Dict[str, Any]:
    """Build ONCE per export (like build_hipp_rating_cache). Returns:

      {"structured": {(column, value): set(node_id)},
       "node_rows":  {node_id: [pois_gdf index, ...]},   # embedded rows only
       "poi_embeddings": {df_index: vector}}

    `structured` uses the EXACT rule filtration.py:111-117 applies at run time
    (case-insensitive substring, regex off) so a stop is scored by the same
    predicate that selected it.

    `node_rows` is restricted to rows present in the POI embedding cache (~1.5k
    of 627k), which is all the description check can ever look at.

    Do NOT reach for filtration.build_possible_targets per query instead: it
    rebuilds its own category->node map from the full POI table on every call.
    The HIPP version of this mistake cost >20 minutes across ~100 queries
    (see _hipp_type_nodes_for_gold).
    """
    from craft.generator.tools.poi_embeddings import load_embedding_cache

    structured: Dict[Tuple[str, str], Set[Any]] = {}
    for col, val in _set3_gold_filter_pairs(gold_records):
        nodes: Set[Any] = set()
        if col in pois_gdf.columns and node_col in pois_gdf.columns:
            mask = (
                pois_gdf[col]
                .astype(str)
                .str.lower()
                .str.contains(str(val).lower(), na=False, regex=False)
            )
            for raw in pois_gdf[mask][node_col].dropna():
                nid = resolve_graph_node_id(G, raw)
                if nid is not None:
                    nodes.add(nid)
        structured[(col, val)] = nodes

    poi_embeddings = load_embedding_cache(str(poi_embed_path))
    node_rows: Dict[Any, List[int]] = {}
    if poi_embeddings and node_col in pois_gdf.columns:
        embedded_idx = set(poi_embeddings)
        subset = pois_gdf.loc[pois_gdf.index.isin(embedded_idx)]
        for df_idx, raw in zip(subset.index, subset[node_col]):
            if raw is None or raw != raw:  # NaN
                continue
            nid = resolve_graph_node_id(G, raw)
            if nid is not None:
                node_rows.setdefault(nid, []).append(int(df_idx))

    return {"structured": structured, "node_rows": node_rows,
            "poi_embeddings": poi_embeddings}


def _set3_gold_query_texts(gold_records: Sequence[Dict[str, Any]]) -> List[str]:
    """Distinct "{category}: {phrase}" strings for the description stops — the
    same string shape filtration.py:146 embeds ("museum: an aviation museum"),
    so the query-side vector is directly comparable to what the search used.
    Only 13 distinct pairs exist across the 29 description stops."""
    texts: Set[str] = set()
    for rec in gold_records:
        for stop in rec.get("stops") or []:
            if stop.get("filter_kind") == "description":
                phrase = stop.get("description") or stop.get("nl_phrase")
                if phrase:
                    texts.add(f"{stop.get('poi_type')}: {phrase}")
    return sorted(texts)


def build_set3_query_embeddings(
    gold_records: Sequence[Dict[str, Any]],
    api_key: Optional[str] = None,
    cache_path: Path = SET3_QUERY_EMBED_PATH,
    model: str = SET3_EMBED_MODEL,
    base_url: Optional[str] = None,
) -> Dict[str, List[float]]:
    """text -> embedding vector for the gold description phrases, persisted to
    `cache_path`. Only the phrases missing from the cache are embedded, so the
    API key is needed on the FIRST export only and re-exports are offline and
    deterministic. Missing phrases (no key, or an API failure) simply stay
    absent — the description check then reports None rather than a false False."""
    import json

    cache: Dict[str, List[float]] = {}
    if Path(cache_path).exists():
        cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))

    wanted = _set3_gold_query_texts(gold_records)
    missing = [t for t in wanted if t not in cache]
    if missing and api_key:
        from craft.generator.tools.poi_embeddings import embed_text
        for text in missing:
            try:
                cache[text] = embed_text(text, api_key, model=model, base_url=base_url)
            except Exception as exc:
                print(f"[set3] embedding '{text}' failed ({exc}) — description check will be None")
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(cache), encoding="utf-8")
    elif missing:
        print(f"[set3] {len(missing)} gold phrases not cached and no API key — "
              f"description filters will score None")
    return cache


def _set3_parser_filters(parser_output: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The index-aligned poi_filters the SOLVER was given, normalised to a list
    of dicts (filtration.py accepts several shapes — see _filters_for_stop)."""
    if not isinstance(parser_output, dict):
        return []
    raw = parser_output.get("poi_filters")
    out: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for v in raw:
            if isinstance(v, dict):
                out.append(dict(v))
            elif isinstance(v, str):
                out.append({"description": v})
            else:
                out.append({})
    return out


def _set3_narrow_by_filter(
    nodes: Set[Any], flt: Dict[str, Any], category: str,
    filter_cache: Dict[str, Any], query_embeddings: Dict[str, List[float]],
) -> Set[Any]:
    """Replay filtration.py's cascade for ONE stop, including its broaden-on-empty
    fallback (filtration.py:124-131 / 158-161): a filter that narrows to nothing
    is dropped and the broader set kept, so the reconstruction lands on the same
    candidate set the search actually had."""
    from craft.generator.tools.poi_embeddings import cosine_sim

    structured = filter_cache.get("structured") or {}
    node_rows = filter_cache.get("node_rows") or {}
    poi_emb = filter_cache.get("poi_embeddings") or {}

    desc_filter: Optional[str] = None
    for col, val in (flt or {}).items():
        if col == "description":
            desc_filter = val
            continue
        key = (str(col), str(val))
        col_nodes = structured.get(key)
        if col_nodes is None:
            # Not a gold pair — compute nothing rather than scan the POI table
            # per query; leaving the set broader can only make the claim more
            # permissive, never credit a stop the route did not visit.
            continue
        narrowed = nodes & col_nodes
        if narrowed:
            nodes = narrowed

    if desc_filter:
        qvec = query_embeddings.get(f"{category}: {desc_filter}")
        if qvec and poi_emb:
            narrowed = set()
            for n in nodes:
                for df_idx in node_rows.get(n, ()):
                    vec = poi_emb.get(df_idx)
                    if vec and cosine_sim(qvec, vec) >= SET3_SIM_THRESHOLD:
                        narrowed.add(n)
                        break
            if narrowed:
                nodes = narrowed
    return nodes


def _set3_claim(
    route: List[Any], gold_types: List[str], candidate_sets: List[Set[Any]],
) -> List[Optional[Any]]:
    """Ordered greedy claim: stop i is the first route node in candidate_sets[i]
    at or after stop i-1's claim. This mirrors the generated expand()'s own
    ordered-index advance (it bumps the stop index the first time a neighbour
    lands in possible_targets[idx]), so the reconstructed stop list is the one
    the search committed to rather than an arbitrary same-type POI the path
    happened to pass. Returns one entry per required stop; None where the route
    never satisfied that stop."""
    claims: List[Optional[Any]] = [None] * len(gold_types)
    pos = 0
    for i in range(len(gold_types)):
        cands = candidate_sets[i] if i < len(candidate_sets) else set()
        found = None
        while pos < len(route):
            if route[pos] in cands:
                found = route[pos]
                pos += 1
                break
            pos += 1
        claims[i] = found
        if found is None:
            break
    return claims


# Frozen, LLM-labelled reference for set3's description phrases, built by
# evaluation/description_gt/build_description_labels.py (two independent passes,
# disagreements kept as `uncertain` and EXCLUDED rather than resolved).
SET3_DESC_LABELS_PATH = ROOT / "outputs" / "evaluation" / "description_gt" / "description_labels.json"
_SET3_DESC_LABELS: Optional[Dict[str, Any]] = None
_SET3_DESC_LABELS_WARNED = False


def load_set3_description_labels(path: Path = SET3_DESC_LABELS_PATH) -> Dict[str, Any]:
    """{row_id_str: {phrase: {label, uncertain, ...}}}. Cached; {} if absent."""
    global _SET3_DESC_LABELS
    if _SET3_DESC_LABELS is None:
        import json as _json
        try:
            _SET3_DESC_LABELS = _json.loads(path.read_text()).get("rows", {})
        except FileNotFoundError:
            _SET3_DESC_LABELS = {}
        except Exception as e:
            # Do NOT swallow a malformed/unreadable label file into a silent {},
            # which would make every description stop score None and look like
            # "not applicable" rather than "the reference is broken".
            raise RuntimeError(f"could not read set3 description labels at {path}: {e}") from e
    return _SET3_DESC_LABELS


def _set3_stop_filter_ok(
    node: Optional[Any], stop: Dict[str, Any],
    filter_cache: Dict[str, Any], query_embeddings: Dict[str, List[float]],
) -> Optional[bool]:
    """Does the claimed node satisfy this stop's GOLD filter?
      structured  -> membership in the (column, value) node set
      description -> the FROZEN LLM label for that POI row (see below)
      none        -> None (nothing to check)
    None is also returned when the evidence is missing (gold pair absent from the
    cache, phrase never embedded/labelled, node has no scoreable row) so an
    unmeasurable stop is never scored as a violation. An unvisited (None) node
    with a real filter IS False — the query asked for that stop.

    WHY THE DESCRIPTION BRANCH USES FROZEN LABELS, NOT A COSINE OR A LIVE LLM
    (changed 2026-08-23). Production now filters with cosine>=0.50 PLUS an LLM
    verification pass. Re-checking the same cosine here graded a POI against a
    bar it had already cleared by construction, so `Desc%` could not fail and
    read 100% on all 3 models — an artefact, not a result. Calling the LLM here
    instead would be worse: it grades the system with the system's own component
    (the agreement is guaranteed, not evidence) and it is non-deterministic —
    the same input measured F1 0.95 and 0.97 on two runs, so scores would not
    reproduce. The frozen two-pass labels are the fixed, auditable reference:
    independent of the filter, deterministic, free, and complete for set3 (they
    cover exactly its 3 description phrases).
    Node-level semantics are unchanged: a node qualifies if ANY of its POI rows
    qualifies — the same "max over the node's rows" rule the cosine used."""
    kind = stop.get("filter_kind")
    if kind == "structured":
        key = (str(stop.get("column")), str(stop.get("value")))
        nodes = (filter_cache.get("structured") or {}).get(key)
        if nodes is None:
            return None
        if node is None:
            return False
        return node in nodes

    if kind == "description":
        phrase = stop.get("description") or stop.get("nl_phrase")
        if not phrase:
            return None
        if node is None:
            return False
        rows = (filter_cache.get("node_rows") or {}).get(node, ())
        labels = load_set3_description_labels()
        if labels:
            verdicts = []
            for i in rows:
                rec = labels.get(str(i), {}).get(phrase)
                if isinstance(rec, dict) and not rec.get("uncertain"):
                    verdicts.append(bool(rec.get("label")))
            if verdicts:
                return any(verdicts)
            # node has no definitely-labelled row for this phrase -> unmeasurable
            return None
        # No label file (e.g. a different dataset): fall back to the old cosine
        # rule so the scorer still runs, but say so — the two are NOT equivalent.
        global _SET3_DESC_LABELS_WARNED
        if not _SET3_DESC_LABELS_WARNED:
            print(f"[constraint_checker] {SET3_DESC_LABELS_PATH.name} not found — description "
                  f"filters fall back to cosine>={SET3_SIM_THRESHOLD}, which is self-confirming "
                  f"against a 0.50+LLM filter. Build the labels for a real measurement.")
            _SET3_DESC_LABELS_WARNED = True
        from craft.generator.tools.poi_embeddings import cosine_sim
        qvec = query_embeddings.get(f"{stop.get('poi_type')}: {phrase}")
        if not qvec:
            return None
        poi_emb = filter_cache.get("poi_embeddings") or {}
        sims = [cosine_sim(qvec, poi_emb[i]) for i in rows if i in poi_emb]
        if not sims:
            return None
        return max(sims) >= SET3_SIM_THRESHOLD

    return None


def _set3_distance_limits_met(
    legs: Optional[List[float]], limits_mi: List[float],
) -> Optional[bool]:
    """SP-through-chosen-stops per-leg distance check, gold limits instead of
    NL-parsed ones. Same basis and same min(limits, legs) semantics as
    _road_network_checks — set3 gold always carries exactly one limit per stop,
    so the final stop->dst leg is unconstrained, matching what the parser hands
    the solver (leg_distances has one entry per stop).

    Takes the already-computed legs rather than a chain: set3's objective weight
    is `length` too, so the objective's sp_chain is the very same call and doing
    it twice would double the Dijkstra cost of the whole export."""
    if not limits_mi:
        return None
    if legs is None:
        return False
    n = min(len(limits_mi), len(legs))
    return all(legs[i] <= float(limits_mi[i]) * MILES_TO_M for i in range(n))


def evaluate_set3_system(
    gold: Dict[str, Any],
    parser_output: Optional[Dict[str, Any]],
    route: List[Any],
    G: Optional[nx.MultiDiGraph],
    pois_gdf,
    snapper: Optional[NodeSnapper],
    caches: Optional[Tuple[Dict[str, Set[Any]], Set[Any]]] = None,
    filter_cache: Optional[Dict[str, Any]] = None,
    query_embeddings: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, Any]:
    """Score one set3 (ordered+desc) craft route against its set3_meta.json gold
    record. Returns CORE_FIELDS plus dijkstra_objective_value and the set3-only
    diagnostics.

    Claiming uses the gold category for the base node set, narrowed by the
    PARSER's filter for that stop when the parser agreed with gold on the
    category — i.e. the candidate set the search really had, broaden-on-empty
    included. Where the parser dropped or mistyped a filter the base set is used,
    which is the honest treatment: the solver genuinely searched wider there, and
    the gold check below then usually fails, which is the signal we want.

    Scoring is always against GOLD, never the parser — a dropped filter must not
    become a free pass.
    """
    template = SET3_TEMPLATE
    active = TEMPLATE_CHECKS.get(template, set())
    wkey = weight_key(template)
    checks: Dict[str, Optional[bool]] = {f: None for f in CHECK_FIELDS}

    filter_cache = filter_cache or {}
    query_embeddings = query_embeddings or {}

    gold_types: List[str] = [normalize_type(t) for t in (gold.get("poi_types") or [])]
    stops_spec: List[Dict[str, Any]] = list(gold.get("stops") or [])
    req = len(gold_types) if gold_types else None

    route_c = [coerce(G, n) for n in (route or [])] if G is not None else list(route or [])
    route_c = [n for n in route_c if n is not None]

    type_nodes: Dict[str, Set[Any]] = {}
    if G is not None and route_c and gold_types:
        if caches is None and pois_gdf is not None:
            caches = build_poi_caches(G, pois_gdf, poi_types=SET3_POI_TYPES)
        if caches is not None:
            type_nodes, _ = caches

    parser_types = [normalize_type(t) for t in ((parser_output or {}).get("poi_types") or [])]
    parser_filters = _set3_parser_filters(parser_output)

    candidate_sets: List[Set[Any]] = []
    for i, cat in enumerate(gold_types):
        nodes: Set[Any] = set(type_nodes.get(cat, set()))
        same_cat = i < len(parser_types) and parser_types[i] == cat
        flt = parser_filters[i] if (same_cat and i < len(parser_filters)) else {}
        if flt:
            nodes = _set3_narrow_by_filter(nodes, flt, cat, filter_cache, query_embeddings)
        candidate_sets.append(nodes)

    claims = _set3_claim(route_c, gold_types, candidate_sets) if (route_c and gold_types) \
        else [None] * len(gold_types)

    # Second, deliberately INDEPENDENT reconstruction: claim on the plain gold
    # category set, with no filter narrowing at all. The primary claim above is
    # solver-faithful but partly self-confirming — a node picked *because* it is
    # in the filtered candidate set will of course pass the filter, so the
    # primary check can only fail via a parser error or filtration's
    # broaden-on-empty. This variant asks the stricter question ("the first
    # POI of the right category the route reaches — does it qualify?") and is
    # exported as a diagnostic only. It reads LOW by construction, because a
    # route passes many incidental same-category POIs before its real stop; the
    # truth for any single query sits between the two columns.
    base_claims = _set3_claim(route_c, gold_types,
                              [set(type_nodes.get(c, set())) for c in gold_types]) \
        if (route_c and gold_types) else [None] * len(gold_types)
    claimed_nodes = [n for n in claims if n is not None]
    n_stops = len(claimed_nodes)
    stops_count_ok = (n_stops == req) if req is not None else None

    if gold_types and route_c:
        # Ordered template: _set3_claim already enforces route order, so "all
        # stops claimed" IS the ordered-visit check (same identity HIPP uses).
        all_claimed = all(n is not None for n in claims)
        if "poi_type_match" in active:
            checks["poi_type_match"] = all_claimed
        if "poi_order" in active:
            checks["poi_order"] = all_claimed

    # Filter compliance
    per_stop: List[Optional[bool]] = [
        _set3_stop_filter_ok(claims[i] if i < len(claims) else None, stop,
                             filter_cache, query_embeddings)
        for i, stop in enumerate(stops_spec)
    ]
    kinds = [s.get("filter_kind") for s in stops_spec]
    scored = [v for v in per_stop if v is not None]
    if route_c and scored:
        checks["poi_filters_respected"] = all(scored)

    def _rate(kind: str) -> Optional[bool]:
        """Diagnostic: filter compliance over the CLAIMED stops of one kind only.
        Excludes stops the route never reached, so it answers 'when the search
        committed to a stop, did it respect the filter' — the standalone
        capability number, as opposed to the hard constraint above which counts
        an unvisited required stop as a failure."""
        vals = [v for i, (v, k) in enumerate(zip(per_stop, kinds))
                if k == kind and v is not None
                and i < len(claims) and claims[i] is not None]
        return all(vals) if vals else None

    base_per_stop: List[Optional[bool]] = [
        _set3_stop_filter_ok(base_claims[i] if i < len(base_claims) else None, stop,
                             filter_cache, query_embeddings)
        for i, stop in enumerate(stops_spec)
    ]
    base_scored = [v for i, v in enumerate(base_per_stop)
                   if v is not None and i < len(base_claims) and base_claims[i] is not None]
    filters_respected_basetype = all(base_scored) if (route_c and base_scored) else None

    # src/dst, objective, distance limits (all gold-driven)
    src = dst = None
    coords_src, coords_dst = gold.get("src"), gold.get("dst")
    if snapper is not None and coords_src and coords_dst:
        src = snapper.snap(float(coords_src[0]), float(coords_src[1]))
        dst = snapper.snap(float(coords_dst[0]), float(coords_dst[1]))
    if src is None and route_c:
        src = route_c[0]
    if dst is None and route_c:
        dst = route_c[-1]

    obj_value: Optional[float] = None
    dijkstra_obj: Optional[float] = None
    if G is not None and route_c and src is not None and dst is not None:
        # wkey is "length" for set3, so this single chain serves BOTH the
        # comparable objective and the per-leg distance check below.
        obj_value, legs = sp_chain(G, [src] + claimed_nodes + [dst], wkey)
        dijkstra_obj = compute_system_objective(route_c, G, wkey)
        if "distance_limits_met" in active:
            all_claimed = all(n is not None for n in claims)
            checks["distance_limits_met"] = _set3_distance_limits_met(
                legs if all_claimed else None, gold.get("distance_limits_mi") or [])

    # constraints_satisfied WITHOUT the filter check, so the pre-set3-scoring
    # headline (feasible AND the type/order/distance checks) stays recoverable
    # from the same export.
    checks_nofilter = {k: v for k, v in checks.items() if k != "poi_filters_respected"}

    return {
        "required_stops": req,
        "n_stops": n_stops,
        "stops_count_ok": stops_count_ok,
        **checks,
        "constraint_satisfaction": _satisfaction(checks),
        "constraints_satisfied": _all_satisfied(checks),
        "objective_type": objective_type(template),
        "objective_value": obj_value,
        "dijkstra_objective_value": dijkstra_obj,
        "constraints_satisfied_nofilter": _all_satisfied(checks_nofilter),
        "structured_filter_respected": _rate("structured"),
        "desc_filter_respected": _rate("description"),
        "filters_respected_basetype": filters_respected_basetype,
        "n_filtered_stops": sum(1 for k in kinds if k in ("structured", "description")),
        "n_filter_stops_scored": len(scored) if route_c else 0,
    }
