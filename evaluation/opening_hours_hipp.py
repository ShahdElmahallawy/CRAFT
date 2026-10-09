"""
opening_hours_hipp.py — shared OSM opening_hours parsing for the HIPP pilot.

Single source of truth so every consumer treats a POI's opening_hours the
same way. Used by:
  - scripts/run_llmap.py       (feeds real per-POI hours into LLMAP's MSGS
                                 day-of-week availability check instead of a
                                 hardcoded wide-open stub)
  - craft_hipp/generator       (arrival-time feasibility check in expand())
  - evaluation/constraint_checker.py (opening_hours_met check, scored the
                                 same way for every system)

Supports exactly the formats data/socal/enrich_synthetic_attrs.py emits:
  '24/7'
  'Mo-Fr HH:MM-HH:MM'
  'Sa,Su HH:MM-HH:MM'
  ';'-separated chains of the above
Anything else (or missing/None) -> treated as always-open, matching the
documented approximation in data/socal/schema_hipp.json's opening_hours
entry ("Anything else -> treat as no hours data (POI always allowed)").

Day-of-week handling mirrors LLMAP's own approximation: LLMAP's
check_path_availability (baselines/LLMAP/main/utils.py) fixes
`current_day = "Monday"` — a single representative day, not a real
multi-day schedule. We keep that same fixed-day semantics here so both
systems are checking identical logic, not just similarly-shaped data.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

DAY_ABBR: List[str] = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
DAY_NAMES: List[str] = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
]

# Fixed reference day used everywhere a single day must be assumed (matches
# LLMAP's own hardcoded `current_day = "Monday"`).
DEFAULT_DAY = "Mo"

_RULE_RE = re.compile(
    r"^([A-Za-z,\-]+)\s+(\d{2}):(\d{2})-(\d{2}):(\d{2})$"
)

# A rule's day span, e.g. "Mo-Fr" or "Sa,Su" or "Mo".
Rule = Tuple[Set[str], int, int]  # (days, open_sec, close_sec)


def _expand_days(days_part: str) -> Set[str]:
    days: Set[str] = set()
    for token in days_part.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            if a in DAY_ABBR and b in DAY_ABBR:
                ai, bi = DAY_ABBR.index(a), DAY_ABBR.index(b)
                if ai <= bi:
                    days.update(DAY_ABBR[ai : bi + 1])
                else:  # wrap, e.g. "Fr-Mo"
                    days.update(DAY_ABBR[ai:] + DAY_ABBR[: bi + 1])
        elif token in DAY_ABBR:
            days.add(token)
    return days


def parse_opening_hours(osm_str: Optional[str]) -> Optional[List[Rule]]:
    """Parse an OSM opening_hours string into a list of (days, open_sec, close_sec)
    rules. Returns None if the string is missing/unparseable — callers must
    treat None as "always open" (the documented fallback)."""
    if not osm_str or not isinstance(osm_str, str):
        return None
    osm_str = osm_str.strip()
    if osm_str == "24/7":
        return [(set(DAY_ABBR), 0, 24 * 3600)]

    rules: List[Rule] = []
    for chunk in osm_str.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = _RULE_RE.match(chunk)
        if not m:
            continue
        days_part, h1, m1, h2, m2 = m.groups()
        days = _expand_days(days_part)
        if not days:
            continue
        open_sec = int(h1) * 3600 + int(m1) * 60
        close_sec = int(h2) * 3600 + int(m2) * 60
        rules.append((days, open_sec, close_sec))
    return rules or None


def is_open_at(rules: Optional[List[Rule]], clock_sec: float, day_abbr: str = DEFAULT_DAY) -> bool:
    """True iff open at `clock_sec` (seconds since midnight) on `day_abbr`.
    `rules=None` (unparseable/missing) means always open."""
    if rules is None:
        return True
    for days, open_sec, close_sec in rules:
        if day_abbr in days and open_sec <= clock_sec <= close_sec:
            return True
    return False


def _fmt_ampm(sec: int) -> str:
    h = (sec // 3600) % 24
    m = (sec % 3600) // 60
    period = "AM" if h < 12 else "PM"
    h12 = h % 12
    if h12 == 0:
        h12 = 12
    return f"{h12}:{m:02d} {period}"


def to_llmap_day_string(osm_str: Optional[str], day_name: str = "Monday") -> str:
    """Convert an OSM opening_hours string into LLMAP's own expected format
    (baselines/LLMAP/main/utils.py::parse_time / check_path_availability):
    "<Day>: H:MM AM/PM - H:MM AM/PM", "<Day>: Closed", or "<Day>: Open 24 hours".
    Only the DEFAULT_DAY / Monday entry is ever read by LLMAP (current_day is
    hardcoded), so this is the only day worth converting."""
    day_abbr = DAY_ABBR[DAY_NAMES.index(day_name)]
    rules = parse_opening_hours(osm_str)
    if rules is None:
        return f"{day_name}: Open 24 hours"

    matching = [(o, c) for days, o, c in rules if day_abbr in days]
    if not matching:
        return f"{day_name}: Closed"
    if len(matching) == 1 and matching[0] == (0, 24 * 3600):
        return f"{day_name}: Open 24 hours"

    parts = [f"{_fmt_ampm(o)} – {_fmt_ampm(c)}" for o, c in matching]
    return f"{day_name}: " + ", ".join(parts)
