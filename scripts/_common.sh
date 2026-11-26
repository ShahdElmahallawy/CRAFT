#!/usr/bin/env bash
# Shared settings for the run_*.sh pipelines. Sourced, not executed.

# The three active models. Keep in step with evaluation/*.py MODELS — an
# evaluator only exports models it is told about, so a mismatch silently
# produces an empty column rather than an error.
MODELS=(gemini-3_1-flash-lite gpt-4_1 openai_gpt-oss-20b)

# Each query execution loads the road graph in its own subprocess (~460MB for
# SoCal), so workers are bounded by RAM, not CPU. Raise only if you have room.
WORKERS="${WORKERS:-4}"

# Run ONE graph-resident chain at a time. Several concurrent graph loads is how
# this box got OOM-killed before (see CLAUDE.md, HARD-WON LESSONS).
DRY_RUN="${DRY_RUN:-0}"

banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }

run() {
  echo "+ $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

# gpt-4.1 lives on the OpenAI endpoint and gpt-oss-20b on the local one;
# only Gemini is the default route. run_llm.py in particular 404s without
# this (run_craft/run_llmap auto-route by model name, but passing it is
# harmless and keeps all three calls identical).
model_flags() {
  case "$1" in
    gpt-4_1|gpt-4.1)       echo "--openai" ;;
    openai_gpt-oss-20b|*gpt-oss*) echo "--local" ;;
    *)                     echo "" ;;
  esac
}
