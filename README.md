# CRAFT

A schema-driven approach to constrained routing. Instead of hand-coding a new
algorithm per problem type (EV routing, ordered stops, POI collection, ...),
CRAFT uses one general Dijkstra template and has an LLM fill in the
problem-specific logic — `State.start`, `expand`, `dominates`, `is_goal` — from
a natural-language query. A separate parser agent turns the NL query into
structured constraints first; the generator agent then writes the search code.

## Structure

| Folder | What's in it |
|---|---|
| `craft/` | The core system: parser, generator, the Dijkstra template/core |
| `baselines/` | An LLM-only baseline, and LLMAP (vendored, MIT licensed) |
| `benchmark/` | Query sets, NL templates, ground truth |
| `evaluation/` | The scoring pipeline shared across all three systems |
| `examples/` | A runnable single-query demo — **start here** |
| `data/` | Datasets (gitignored — see Setup) |

## Setup

1. `pip install -r requirements.txt` (Python 3.12)
2. Create a `.env` at this folder's root with at least one LLM provider:
   ```
   GEMINI_API_KEY=...
   GEMINI_BASE_URL=...
   ```
3. Get the dataset: `data/socal/road_network.gpkg` + `schema.json` aren't
   checked into git (too large). Download from **[Drive link — ask for
   access]** and place both in `data/socal/`.

Using your own road network instead of SoCal? See the "Running your own
query" section below.

## Running a query

The fastest way to see the system work is `examples/run_example.py` — it runs
one query through parse → generate → execute and saves everything it
produced:
```
cd CRAFT
python examples/run_example.py
```
That uses a bundled example query. To run your own:
```
python examples/run_example.py --query "I'm at (33.83, -117.95) and need to get to (33.85, -117.86). Stop at a pharmacy, then a library, in that order."
```
or `--query-file your_query.json` with `{"query_id": ..., "nl_query": ...}`.
No special syntax needed — just describe the trip; see `examples/README.md`
for what kinds of constraints the parser understands (EV battery, ordered
stops, POI attribute filters, time budgets, road preference, ...) and
`benchmark/templates/*.txt` for real phrasing examples.

To run against your own dataset instead of SoCal: build its schema first —
`python craft/build_schema.py --gpkg <your.gpkg>` — then pass
`--gpkg <your.gpkg>` to `run_example.py`.

## How it works

1. **Parse** (`craft/parser/`) — NL query → structured JSON (src/dst, objective,
   constraints). Grounded against the dataset's real POI categories and
   attribute schema so the LLM can't invent a category or column that doesn't
   exist.
2. **Generate** (`craft/generator/`) — the LLM fills four method bodies into a
   fixed Dijkstra template (`craft/generator/tools/template.py`); everything
   else (the search core, the State interface) is fixed.
3. **Execute** — the filled-in template runs the actual search and writes the
   route (or `null` if infeasible).

`evaluation/constraint_checker.py` scores all three systems (this one, the LLM
baseline, LLMAP) through identical logic, so results are directly comparable.

## Baselines

- `baselines/llm/` — a bare LLM baseline (no search, just asks the model to
  name stops and matches them to the dataset).
- `baselines/LLMAP/` — vendored from Yuan et al., EMNLP Findings 2025
  (arXiv:2509.12273), MIT License (see `baselines/LLMAP/LICENSE`). Not
  modified — treat as read-only.

## License

This project's own license is not yet finalized — add one before any public
release. `baselines/LLMAP/` keeps its original MIT license regardless; see
`baselines/LLMAP/LICENSE` for the terms governing that subfolder.
