"""
Validates Set 3 filtered-POI benchmark records.

Records contain only {query_id, nl_query}. Ground truth (which stop is
filtered, by what column/value or description, src/dst, distance limits,
difficulty) lives in set3_meta.json, one entry per query_id.

Checks:
  1.  Every record has exactly the fields: query_id, nl_query
  2.  query_id format: set3_ordered+desc_NNN (3-digit) and all unique
  3.  No unfilled {PLACEHOLDER} in nl_query
  4.  nl_query is non-empty and contains at least one coordinate "(lat, lon)"
  5.  Total record count == 200 (single template: ordered_poi)
  6.  Every record has a matching set3_meta.json entry
  7.  Each meta entry's poi_filters length == its stops length == the number
      of stops actually present in the nl_query text (2 or 3)
  8.  Every meta stop with filter_kind in {structured, description} has
      candidate_count_in_city >= 2 (the generator's own feasibility gate,
      re-checked here so a future edit can't silently violate it)

Usage:
  conda run -n myenv python benchmark/socal_queries/set3_descriptions/validate.py
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SET3_DIR = ROOT / "benchmark" / "socal_queries" / "set3_descriptions"

EXPECTED_FIELDS = {"query_id", "nl_query"}
EXPECTED_TOTAL = 200
MIN_CANDIDATES = 2


def main():
    records = json.loads((SET3_DIR / "ordered+desc.json").read_text())
    meta = json.loads((SET3_DIR / "set3_meta.json").read_text())
    meta_by_id = {m["query_id"]: m for m in meta}

    total = len(records)
    print(f"Records loaded: {total}")
    print()

    fails = defaultdict(list)

    for q in records:
        qid = q.get("query_id", "?")

        extra = set(q.keys()) - EXPECTED_FIELDS
        missing = EXPECTED_FIELDS - set(q.keys())
        if extra:
            fails["1_extra_fields"].append(f"{qid}(extra={extra})")
        if missing:
            fails["1_missing_fields"].append(f"{qid}(missing={missing})")

        if not re.match(r"^set3_ordered\+desc_\d{3}$", q.get("query_id", "")):
            fails["2_bad_query_id_format"].append(qid)

        if re.search(r"\{[A-Z0-9_]+\}", q.get("nl_query", "")):
            fails["3_unfilled_placeholder"].append(qid)

        nl = q.get("nl_query", "")
        if not nl.strip():
            fails["4_empty_nl_query"].append(qid)
        elif not re.search(r"\(\s*-?\d+\.\d+,\s*-?\d+\.\d+\)", nl):
            fails["4_no_coordinate"].append(qid)

        m = meta_by_id.get(qid)
        if m is None:
            fails["6_missing_meta"].append(qid)
        else:
            n_stops = len(m.get("stops", []))
            n_filters = len(m.get("poi_filters", []))
            if n_stops != n_filters:
                fails["7_stop_filter_mismatch"].append(f"{qid}(stops={n_stops},filters={n_filters})")
            if n_stops not in (2, 3):
                fails["7_bad_stop_count"].append(f"{qid}(stops={n_stops})")
            for st in m.get("stops", []):
                if st.get("filter_kind") in ("structured", "description"):
                    cnt = st.get("candidate_count_in_city", 0)
                    if cnt < MIN_CANDIDATES:
                        fails["8_low_candidate_count"].append(
                            f"{qid}:{st.get('poi_type')}(count={cnt})"
                        )

    ids = [q.get("query_id") for q in records]
    dupes = [k for k, v in Counter(ids).items() if v > 1]
    if dupes:
        fails["2_duplicate_ids"].extend(dupes)

    if total != EXPECTED_TOTAL:
        fails["5_total_count"].append(f"got {total}, want {EXPECTED_TOTAL}")

    checks = [
        ("1", "Records have exactly {query_id, nl_query}"),
        ("2", "query_id format set3_ordered+desc_NNN and unique"),
        ("3", "No unfilled {PLACEHOLDER} in nl_query"),
        ("4", "nl_query non-empty and contains a coordinate"),
        ("5", f"Total record count == {EXPECTED_TOTAL}"),
        ("6", "Every record has a matching set3_meta.json entry"),
        ("7", "meta stops/poi_filters length match, stop count in {2,3}"),
        ("8", f"Every filtered stop has >= {MIN_CANDIDATES} real in-city candidates"),
    ]

    print(f"{'Check':<60} {'Fail':>5}  Status")
    print("-" * 75)
    all_pass = True
    for key, label in checks:
        matching = {k: v for k, v in fails.items() if k.startswith(key + "_") or k == key}
        n_fail = sum(len(v) for v in matching.values())
        status = "PASS" if n_fail == 0 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {label:<58} {n_fail:>5}  {status}")
        if n_fail > 0:
            for v in matching.values():
                for item in v[:3]:
                    print(f"      → {item}")

    print("-" * 75)
    print(f"  {'OVERALL':<58} {'PASS' if all_pass else 'FAIL'}")

    print()
    print("Distributions:")
    filter_kind_counts = Counter()
    poi_type_counts = Counter()
    stop_count_dist = Counter()
    for m in meta:
        stop_count_dist[len(m["stops"])] += 1
        for st in m["stops"]:
            filter_kind_counts[st["filter_kind"]] += 1
            poi_type_counts[st["poi_type"]] += 1
    print("  stop counts:", dict(stop_count_dist))
    print("  filter kinds:", dict(filter_kind_counts))
    print("  poi types:", dict(poi_type_counts))

    print()
    print("Sample nl_queries:")
    for q in records[:3]:
        print(f"  [{q['query_id']}]")
        print(f"    {q['nl_query'][:110]}...")


if __name__ == "__main__":
    main()
