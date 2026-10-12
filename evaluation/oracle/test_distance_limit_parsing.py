"""
evaluation/oracle/test_distance_limit_parsing.py — exhaustive coverage test for
constraint_checker.parse_distance_limits_m.

Renders every variant of every benchmark template that carries a per-leg
distance limit, substituting a distinct sentinel value per placeholder, and
asserts the parser recovers exactly those values in exactly that order. This
proves coverage across phrasings rather than spot-checking the one that broke.

Motivating bug (fixed 2026-08-10): algo2 variant 27 writes the limits as a
comma-separated run sharing one trailing unit — "The leg limits are 9, 2, and 4
miles respectively" — and the old pattern, which required "mile" directly after
each number, returned only [4]. The caller then broadcast that single value to
every leg, mis-scoring 7 queries.

The EV templates are included on purpose: they state a battery range in miles
too, and parse_battery_m must strip it first, so these variants check that leg
limits and battery ranges do not contaminate each other.

Run:
  python evaluation/oracle/test_distance_limit_parsing.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.constraint_checker import MILES_TO_M, parse_distance_limits_m

TEMPLATES_DIR = ROOT / "benchmark" / "templates"
TEMPLATE_FILES = [
    "ordered_poi.txt",
    "ev+ordered_poi.txt",
    "unordered_poi.txt",
    "ordered+time_budget.txt",
    "poi+road_pref.txt",
    "ev_routing.txt",      # no leg limits — must yield []
    "poi_collection.txt",  # no leg limits — must yield []
]

# Distinct, unambiguous sentinels. Deliberately not equal to any battery/time
# value used below, so a mis-parse cannot coincidentally look correct.
DIST_VALUES: Dict[str, str] = {
    "DISTANCE_LIMIT_1_MI": "11",
    "DISTANCE_LIMIT_2_MI": "22",
    "DISTANCE_LIMIT_3_MI": "33",
    "DISTANCE_LIMIT_MI":   "44",
}
# Everything else a variant might contain. The battery values are in MILES and
# must be stripped by _strip_range_phrase, never counted as leg limits.
OTHER_VALUES: Dict[str, str] = {
    "SRC_NODE": "(34.05, -118.25)",
    "DST_NODE": "(33.99, -118.44)",
    "CURRENT_BATTERY_PCT": "55%",
    "CURRENT_BATTERY_MI": "77",
    "MAX_BATTERY_MI": "99",
    "TIME_BUDGET_HOURS": "3",
    "TIME_BUDGET_MIN": "180",
    "VISIT_MIN": "15",
}

PH = re.compile(r"\{([A-Z0-9_]+)\}")


def load_variants(path: Path) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^(\d+):\s*(.+)$", line.strip())
        if m:
            out[int(m.group(1))] = m.group(2).strip()
    return out


def main() -> None:
    total = failures = 0
    unknown_placeholders = set()

    for fname in TEMPLATE_FILES:
        path = TEMPLATES_DIR / fname
        if not path.exists():
            print(f"  (missing template {fname}, skipped)")
            continue
        variants = load_variants(path)
        file_fail = 0

        for vid, text in sorted(variants.items()):
            names = PH.findall(text)
            for n in names:
                if n not in DIST_VALUES and n not in OTHER_VALUES:
                    unknown_placeholders.add(n)

            # expected = the distance placeholders, in order of appearance
            expected = [float(DIST_VALUES[n]) for n in names if n in DIST_VALUES]

            rendered = text
            for n in names:
                val = DIST_VALUES.get(n) or OTHER_VALUES.get(n)
                if val is not None:
                    rendered = rendered.replace("{" + n + "}", val, 1)

            got = [round(v / MILES_TO_M, 6) for v in parse_distance_limits_m(rendered)]
            total += 1
            if got != expected:
                failures += 1
                file_fail += 1
                print(f"  FAIL {fname}#{vid}: expected {expected} got {got}")
                print(f"       {rendered[:150]}")

        status = "OK" if file_fail == 0 else f"{file_fail} FAILED"
        print(f"{fname}: {len(variants)} variants — {status}")

    if unknown_placeholders:
        print(f"\n!! placeholders with no test value (would render literally): "
              f"{sorted(unknown_placeholders)}")
        failures += 1

    print(f"\n{total - failures}/{total} variants parsed correctly")
    if failures:
        sys.exit(1)
    print("PASS — every variant of every distance-limit template parses exactly.")


if __name__ == "__main__":
    main()
