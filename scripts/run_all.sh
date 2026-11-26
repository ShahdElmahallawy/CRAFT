#!/usr/bin/env bash
# Every query set, in order. Hours, not minutes — each query loads the road
# graph in its own subprocess, and there are thousands of them.
#
#   ./scripts/run_all.sh                # the lot
#   DRY_RUN=1 ./scripts/run_all.sh      # print the commands, run nothing
#   ./scripts/run_all.sh --limit 2      # smoke test end to end
#
# Set 2 (California long trips) is NOT included: it needs data/california/
# (~2.7GB), which is an optional download. It has its own pipeline —
#   ./scripts/run_set2.sh
# which fails early with an explanation if that graph is missing.
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/_common.sh

banner "1/4  Set 1 — core benchmark (craft, llm, llmap)"
./scripts/run_set1.sh "$@"

banner "2/4  poi_road_pref — avoid highways (craft, llm, llmap)"
./scripts/run_poi_road_pref.sh "$@"

banner "3/4  Set 3 — filtered POIs (craft only)"
./scripts/run_set3.sh "$@"

# Last because it is the longest by far: 1000 queries x 3 systems x 3 models.
banner "4/4  HIPP — LLMAP's own benchmark (craft_hipp, llm, llmap)"
./scripts/run_hipp.sh "$@"

echo
echo "all done. tables in results/"
ls -1 results/*.xlsx 2>/dev/null || true
