# Try it out

This runs ONE query through the full pipeline — parse → generate → execute — and
shows you what comes out. No benchmark, no batch runs, just one query so you can
see how the system works before using it at scale.

## 1. Setup

**Python env.** Activate whatever environment has the packages in
`requirements.txt` installed (`pip install -r requirements.txt`).

**LLM.** By default this demo runs fully locally against a small Qwen model, no
API key or internet access needed once the model is pulled.

Install [Ollama](https://ollama.com), then:
```
ollama pull qwen2.5:7b
ollama serve   # usually already running after install — skip if so
```
Create a `.env` file at the `CRAFT/` root (same folder this `examples/` dir
lives in) pointing at it:
```
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
```
(`OPENAI_API_KEY` can be any non-empty string — Ollama doesn't check it.) A
7B model is slow on CPU-only machines; if you don't have a GPU, either expect
a wait and pass `--timeout-generate 300`, or pull a smaller tag
(`qwen2.5:3b` or `qwen2.5:1.5b`) and pass it via `--model`.

Prefer a cloud model instead? Skip Ollama and set the matching keys instead —
`GEMINI_API_KEY`/`GEMINI_BASE_URL` for `--model gemini-3.1-flash-lite`, or
`OPENAI_API_KEY`/`OPENAI_BASE_URL` for `--model gpt-4.1`. Both the parser and
the generator call the same model.

**Dataset.** This demo only needs a road network GPKG with `nodes`, `edges`,
and `pois` layers, plus its schema:
```
CRAFT/data/socal/road_network.gpkg
CRAFT/data/socal/schema.json
```
Download both from **https://drive.google.com/drive/u/0/folders/1i7Ku5QhfrSrxAGIoQWsQ9LkATmeaz2vd** and place them in
`data/socal/` — see the top-level `README.md`'s "Downloading the datasets"
section for the full dataset (needed if you want to regenerate benchmark
query sets or run the full evaluation, not just this single-query demo).

Using your own dataset instead? Point `--gpkg` at it and build its
`schema.json` first with `python craft/build_schema.py --gpkg <your.gpkg>`.

## 2. Run it

```
cd CRAFT
python examples/run_example.py
```

This uses the bundled `example_query.json` and your local Qwen model by default.
To try your own query:
```
python examples/run_example.py --query "I'm at (33.83, -117.95) and need to get to (33.85, -117.86). Stop at a pharmacy, then a library, in that order."
```
or point it at your own query file:
```
python examples/run_example.py --query-file my_query.json --query-id my_query
```
(`my_query.json` should look like `example_query.json`: `{"query_id": ..., "nl_query": ...}`.)

Other useful flags: `--model gpt-4.1` or `--model gemini-3.1-flash-lite` to use
a cloud model instead, `--timeout-generate 300` (a local model on CPU needs
more time), `--timeout-execute 300` (large or hard searches).

## 3. What you'll see

Console output walks through the three pipeline steps (parse → generate →
execute) and ends with a summary: whether a route was found, how many stops it
hit, and the route's total cost. Everything it produced along the way gets
copied to `examples/output/<query_id>/`:

| File | What it is |
|---|---|
| `input.json` | The structured query the parser produced from your NL text |
| `generated.py` | The actual Dijkstra solver the LLM wrote for this query |
| `raw_response.json` | The LLM's raw response before validation |
| `llm_meta.json` | Model/latency/token-usage info for the generation call |
| `output.json` | The route, its cost, which POIs it stopped at |
| `output.gpkg` | The route + stops, viewable in QGIS or similar |

## 4. Writing your own query
A query is just natural language — no special syntax.

