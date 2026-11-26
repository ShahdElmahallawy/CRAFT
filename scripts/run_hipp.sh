#!/usr/bin/env bash
# HIPP — LLMAP's own benchmark (1000 queries), run on all three systems.
#
# This is the comparison on the baseline's home turf: the queries come from
# LLMAP's HIPP.json, and its three constraint types (dependency ordering, a
# rating floor, an arrival deadline with opening hours) are the ones LLMAP was
# built for. Two families, split by whether the query states dependencies:
# hipp_ordered (439) and hipp_unordered (561).
#
#   raw records -> outputs/crafthipp/{model}/{family}/   (craft_hipp)
#                  outputs/llm/{model}/{family}/         (LLM baseline)
#                  outputs/llmap/{model}/{family}/        (LLMAP)
#   tables      -> results/{craft,llm,llmap}_hipp.xlsx
#
# CRAFT runs through craft_hipp/, not craft/. It is the same solver and the same
# template; what differs is that the spec is overwritten with HIPP's ground
# truth for the fields HIPP provides (poi types, ordering, deadline, rating
# floor), so the comparison measures routing rather than spec extraction. See
# craft_hipp/spec_builder/agent.py::_apply_hipp_gold.
#
#   ./scripts/run_hipp.sh
#   ./scripts/run_hipp.sh --limit 5         # smoke test
#   DRY_RUN=1 ./scripts/run_hipp.sh         # print commands only
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/_common.sh

QUERIES="benchmark/socal_queries/llmap_queries"
FAMILIES=(hipp_ordered hipp_unordered)

# Ground truth and the per-family query files are written by generate_hipp.py.
# It is deterministic (seed 42) and reproduces hipp_all.json byte for byte, so
# running it is safe — but do it explicitly rather than silently from here.
for f in hipp_meta.json hipp_ordered.json hipp_unordered.json; do
  if [[ ! -f "$QUERIES/$f" ]]; then
    echo "error: $QUERIES/$f missing." >&2
    echo "Build the HIPP query set and its ground truth first:" >&2
    echo "  python $QUERIES/generate_hipp.py" >&2
    exit 1
  fi
done

# craft_hipp reads data/socal/schema_hipp.json for the numeric rating attribute.
if [[ ! -f data/socal/schema_hipp.json ]]; then
  echo "note: data/socal/schema_hipp.json missing — building it." >&2
  run python craft_hipp/build_schema.py
fi

for model in "${MODELS[@]}"; do
  banner "CRAFT (craft_hipp)  $model"
  run python scripts/run_crafthipp.py --model "$model" $(model_flags "$model") \
      --queries-dir "$QUERIES" --templates "${FAMILIES[@]}" \
      --workers "$WORKERS" "$@"
done

for model in "${MODELS[@]}"; do
  banner "LLM baseline  $model  (HIPP)"
  run python scripts/run_llm.py --model "$model" $(model_flags "$model") \
      --queries-dir "$QUERIES" --templates "${FAMILIES[@]}" \
      --out-dir outputs/llm "$@"
done

for model in "${MODELS[@]}"; do
  banner "LLMAP  $model  (HIPP)"
  run python scripts/run_llmap.py --model "$model" $(model_flags "$model") \
      --queries-dir "$QUERIES" --templates "${FAMILIES[@]}" \
      --out-dir outputs/llmap "$@"
done

banner "Evaluate -> results/{craft,llm,llmap}_hipp.xlsx"
run python evaluation/evaluate_crafthipp.py  --models "${MODELS[@]}" --out results/craft_hipp.xlsx
run python evaluation/evaluate_llm_hipp.py   --models "${MODELS[@]}" --out results/llm_hipp.xlsx
run python evaluation/evaluate_llmap_hipp.py --models "${MODELS[@]}" --out results/llmap_hipp.xlsx

echo
echo "done. tables in results/"
ls -1 results/*_hipp.xlsx 2>/dev/null || true
