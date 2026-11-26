#!/usr/bin/env bash
# Set 1 — the core benchmark: 5 templates x 200 queries, all three systems.
#
#   raw records -> outputs/{craft,llm,llmap}/{model}/{template}/{query_id}.json
#   tables      -> results/{craft,llm,llmap}_set1.xlsx
#
#   ./scripts/run_set1.sh                 # everything
#   ./scripts/run_set1.sh --limit 3       # smoke test, 3 queries per template
set -euo pipefail
cd "$(dirname "$0")/.."

source scripts/_common.sh

QUERIES="benchmark/socal_queries/set1"
TEMPLATES="ev_routing ordered_poi poi_collection ev+ordered_poi unordered_poi"

for model in "${MODELS[@]}"; do
  banner "CRAFT  $model  (set1)"
  run python scripts/run_craft.py --model "$model" $(model_flags "$model") \
      --queries-dir "$QUERIES" --templates $TEMPLATES \
      --workers "$WORKERS" "$@"

  banner "LLM    $model  (set1)"
  run python scripts/run_llm.py --model "$model" $(model_flags "$model") \
      --queries-dir "$QUERIES" --templates $TEMPLATES "$@"

  banner "LLMAP  $model  (set1)"
  run python scripts/run_llmap.py --model "$model" $(model_flags "$model") \
      --queries-dir "$QUERIES" --templates $TEMPLATES "$@"
done

banner "Hallucination check (LLM only — a separate pass that does NOT re-run itself)"
run python evaluation/evaluate_llm_hallucination.py \
    --models "${MODELS[@]}" --templates $TEMPLATES

banner "Evaluate -> results/"
run python evaluation/evaluate_craft.py --models "${MODELS[@]}" \
    --templates $TEMPLATES --queries-dir "$QUERIES" --out results/craft_set1.xlsx
run python evaluation/evaluate_llm.py   --models "${MODELS[@]}" \
    --templates $TEMPLATES --queries-dir "$QUERIES" --out results/llm_set1.xlsx
run python evaluation/evaluate_llmap.py --models "${MODELS[@]}" \
    --templates $TEMPLATES --queries-dir "$QUERIES" --out results/llmap_set1.xlsx

echo
echo "done. tables in results/{craft,llm,llmap}_set1.xlsx"
