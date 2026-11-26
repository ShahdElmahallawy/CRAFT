#!/usr/bin/env bash
# Set 2 — long-distance EV routing on the California network (200 queries).
#
# Same template as set1's ev_routing, but city-to-city instead of intra-city:
# 100 trips at 15-75 mi and 100 at 75-300 mi, 120 of which cannot be completed
# without a charging stop. Set 1's EV trips are short enough that the battery
# rarely binds, so this is the set where the EV constraint actually does work.
#
#   raw records -> outputs/craft_set2_ca/{model}/ev_routing/
#   table       -> results/craft_set2.xlsx
#
# Not part of run_all.sh: it needs the California graph, which is a separate
# ~2.7GB download and is 5x the size of SoCal (1.19M nodes, ~100s to load per
# query, ~7.3GB resident). The guard below fails early with that explanation
# rather than letting every query die in its subprocess.
#
#   ./scripts/run_set2.sh
#   ./scripts/run_set2.sh --limit 2        # smoke test
#   DRY_RUN=1 ./scripts/run_set2.sh        # print commands only
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/_common.sh

QUERIES="benchmark/california_queries/set2"
GPKG="data/california/california_road_network.gpkg"
CATALOG="data/california/catalog_pois.json"
OUT_DIR="outputs/craft_set2_ca"
TMPL="ev_routing"

if [[ ! -f "$GPKG" ]]; then
  echo "error: $GPKG not found." >&2
  echo "Set 2 runs on the California network, which is an optional download" >&2
  echo "separate from data/socal/. Fetch it before running this set." >&2
  exit 1
fi

# One worker, not WORKERS. The California graph is ~7.3GB resident per query
# subprocess, so even two concurrent loads risk the OOM kill that took out this
# box before (CLAUDE.md, HARD-WON LESSONS). Override only with headroom.
SET2_WORKERS="${SET2_WORKERS:-1}"

for model in "${MODELS[@]}"; do
  banner "CRAFT  $model  (set2, California)"
  run python scripts/run_craft.py --model "$model" $(model_flags "$model") \
      --queries-dir "$QUERIES" --templates "$TMPL" \
      --gpkg "$GPKG" --catalog-pois "$CATALOG" --out-dir "$OUT_DIR" \
      --workers "$SET2_WORKERS" "$@"
done

# Same evaluator as set1 — only the records, queries and graph differ, so they
# are passed as flags. --templates keeps it to ev_routing: the set2 directory
# holds no other template.
banner "Evaluate -> results/craft_set2.xlsx"
run python evaluation/evaluate_craft.py --models "${MODELS[@]}" \
    --templates "$TMPL" --baseline-dir "$OUT_DIR" --queries-dir "$QUERIES" \
    --gpkg "$GPKG" --out results/craft_set2.xlsx

echo
echo "done. table in results/craft_set2.xlsx"
