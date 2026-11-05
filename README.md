# CRAFT

## Structure

| Folder | What's in it |
|---|---|
| `craft/` | The core system: parser, generator, the Dijkstra template/core |
| `baselines/` | An LLM-only baseline, and LLMAP (vendored, MIT licensed) |
| `benchmark/` | Query sets, NL templates, ground truth |
| `evaluation/` | The scoring pipeline shared across all three systems |
| `examples/` | A runnable single-query demo — **start here** |
| `results/` | Published result tables (xlsx) for all three systems, already scored |
| `data/` | Datasets (gitignored — see "Downloading the datasets") |

## Setup

1. `pip install -r requirements.txt` (Python 3.12)
2. Get an LLM. By default the demo runs fully locally with no API key:
   [install Ollama](https://ollama.com), `ollama pull qwen2.5:7b`, then
   point a `.env` at this folder's root to it:
   ```
   OPENAI_BASE_URL=http://localhost:11434/v1
   OPENAI_API_KEY=ollama
   ```
   Prefer a cloud model? Set `GEMINI_API_KEY`/`GEMINI_BASE_URL` (or
   `OPENAI_API_KEY`/`OPENAI_BASE_URL` for a real OpenAI key) instead — see
   `examples/README.md` for details.
3. Get the dataset — see "Downloading the datasets" below.

Using your own road network instead of SoCal? See the "Running your own
query" section below.

## Downloading the datasets

`data/` isn't checked into git (too large) — download and place the pieces
you need from **https://drive.google.com/drive/u/0/folders/1i7Ku5QhfrSrxAGIoQWsQ9LkATmeaz2vd**.

**SoCal — required for everything in this repo.** Download the bundle and
place it at `data/socal/`:
```
data/socal/road_network.gpkg      # the road graph + POIs (~460MB)
data/socal/schema.json            # POI attribute schema
data/socal/catalog_pois.json      # POI catalog (benchmark query generation)
data/socal/catalog_nodes.json     # candidate node catalog (benchmark query generation)
data/socal/catalog_charging.json  # charging-station catalog (EV query generation)
data/socal/catalog_count.json     # per-category POI counts
data/socal/cities.json            # the 31 benchmark cities (name/lat/lon/radius)
data/socal/set3_query_embeddings.json  # query-side embeddings for set3's description filter
```
This is everything needed to run the demo, regenerate any benchmark query set,
and evaluate all three systems — the ground truth and oracle-optimal results
needed for scoring already ship with the code (`benchmark/ground_truth/`,
`evaluation/optimal/results/`, `outputs/evaluation/description_gt/`), no
separate download needed for those.

**Optional, large — skip unless you specifically need these:**
- `data/socal/poi_embeddings.json` (~3.5GB) — only used as a live fallback for
  set3's description filter when the frozen ground-truth labels are missing
  (which they aren't; they're already included). Only needed if you're
  rebuilding those labels from scratch via `evaluation/description_gt/`.
- `data/california/` (~2.7GB, dominated by `california_road_network.gpkg`) —
  only needed to regenerate or re-run Set 2's long-trip EV queries
  (`benchmark/california_queries/set2/`). Set 2's query text and ground truth
  are already included; you only need this to execute CRAFT/the baselines
  against those queries yourself.

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
