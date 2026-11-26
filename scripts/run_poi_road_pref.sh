#!/usr/bin/env bash
# Avoid-highways template — 200 queries, all three systems.
#
#   raw records -> outputs/{craft,llm,llmap}/{model}/poi_road_pref/
#   tables      -> results/{craft,llm,llmap}_poi_road_pref.xlsx
#
# Two-step export: the evaluator writes a plain sheet, then
# build_poi_road_pref_tables.py drops empty columns and builds the Paper Table.
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/_common.sh

QUERIES="benchmark/socal_queries/poi_road_pref"
TMPL="poi_road_pref"
TMP="outputs/_poi_road_pref_tmp"
mkdir -p "$TMP"

for model in "${MODELS[@]}"; do
  banner "CRAFT  $model  ($TMPL)"
  run python scripts/run_craft.py --model "$model" $(model_flags "$model") \
      --queries-dir "$QUERIES" --templates "$TMPL" --workers "$WORKERS" "$@"

  banner "LLM    $model  ($TMPL)"
  run python scripts/run_llm.py --model "$model" $(model_flags "$model") \
      --queries-dir "$QUERIES" --templates "$TMPL" "$@"

  banner "LLMAP  $model  ($TMPL)"
  run python scripts/run_llmap.py --model "$model" $(model_flags "$model") \
      --queries-dir "$QUERIES" --templates "$TMPL" "$@"
done

banner "Evaluate -> results/"
for sys in craft llm llmap; do
  run python "evaluation/evaluate_${sys}.py" --models "${MODELS[@]}" \
      --templates "$TMPL" --queries-dir "$QUERIES" --out "$TMP/${sys}.xlsx"
  run python evaluation/build_poi_road_pref_tables.py --system "$sys" \
      --in "$TMP/${sys}.xlsx" --out "results/${sys}_poi_road_pref.xlsx"
done

echo
echo "done. tables in results/{craft,llm,llmap}_poi_road_pref.xlsx"
