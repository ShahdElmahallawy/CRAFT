#!/usr/bin/env bash
# Set 3 — filtered-POI ordered routes (attribute + free-text description).
#
# CRAFT only: it is the only system with POI attribute/description filtering.
# LLMAP is hardcoded to five POI types, and the LLM baseline has no dataset
# access, so neither can be meaningfully scored on this.
#
#   raw records -> outputs/craft/{model}/ordered+desc/
#   table       -> results/craft_set3.xlsx
#
# The description filter needs the embedding cache and a local LLM endpoint
# (Sec 7.3: embeddings drop clear non-matches, the LLM accepts). Without the
# endpoint the check degrades to stage 1 and prints a warning.
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/_common.sh

QUERIES="benchmark/socal_queries/set3_descriptions"
TMPL="ordered+desc"

for model in "${MODELS[@]}"; do
  banner "CRAFT  $model  (set3)"
  run python scripts/run_craft.py --model "$model" $(model_flags "$model") \
      --queries-dir "$QUERIES" --templates "$TMPL" --workers "$WORKERS" "$@"
done

banner "Evaluate -> results/craft_set3.xlsx"
run python evaluation/evaluate_craft_set3.py --models "${MODELS[@]}" \
    --templates "$TMPL" --out results/craft_set3.xlsx

echo
echo "done. table in results/craft_set3.xlsx"
