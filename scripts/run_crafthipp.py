"""
run_crafthipp.py — Run the craft_hipp pipeline on the HIPP benchmark.

A fork of scripts/run_craft.py, kept separate for the same reason craft_hipp/
is separate from craft/: the headline runner should not carry HIPP branches.
Everything structural is identical; the HIPP-only part is that each query's
ground-truth record is looked up in hipp_meta.json and passed to the spec
builder, which overwrites the fields we have gold for (poi types, dependency
ordering, clock deadline, rating floor) — see
craft_hipp/spec_builder/agent.py::_apply_hipp_gold.

Pipeline per query:
  1. Spec builder : NL query → structured JSON, gold-overridden for HIPP fields
  2. ORCStrator   : input.json → LLM fills the solver template → generated.py
  3. Execute      : generated.py → output.json
  4. Score        : compute_metrics on the result

Output layout:
  outputs/crafthipp/{model_slug}/{template}/{query_id}.json          ← eval records
  outputs/crafthipp/{model_slug}/hipp/{query_id}_output/output.json  ← raw copies

Usage:
  python scripts/run_crafthipp.py
  python scripts/run_crafthipp.py --templates hipp_unordered --limit 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
for _p in [str(ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import geopandas as gpd
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from craft_hipp.orcstrator.agent import LocalLLM, read_env
from craft_hipp.spec_builder.tools import CatalogResolver
from craft_hipp.orcstrator.tools.common_handlers import build_graph_from_gpkg
from craft_hipp.orcstrator.tools.result import enrich_output, load_output
from craft_hipp.pipeline import run_parser, run_generator, run_execute, ORCSTRATOR_OUTPUTS
from evaluation.metrics import compute_metrics, compute_metrics_error

HIPP_QUERIES_DIR = ROOT / "benchmark" / "socal_queries" / "llmap_queries"
# Ground truth for the HIPP fields, written by that directory's generate_hipp.py.
HIPP_META_PATH   = HIPP_QUERIES_DIR / "hipp_meta.json"
CATALOG_POIS_PATH = ROOT / "data" / "socal" / "catalog_pois.json"
DEFAULT_GPKG      = str(ROOT / "data" / "socal" / "road_network.gpkg")
STAGING_DIR       = ROOT / "outputs" / "eval_staging_crafthipp"


def read_output(query_id: str):
    out_path = ORCSTRATOR_OUTPUTS / f"{query_id}_output" / "output.json"
    if not out_path.exists():
        return None
    try:
        return load_output(str(out_path))
    except Exception:
        return None


def read_input_params(query_id: str):
    inp_path = ORCSTRATOR_OUTPUTS / f"{query_id}_output" / "input.json"
    if not inp_path.exists():
        return {}
    try:
        import json as _json
        raw = _json.loads(inp_path.read_text(encoding="utf-8"))
        return raw[0] if isinstance(raw, list) else raw
    except Exception:
        return {}

SPEC_BUILDER_PROMPT = ROOT / "craft_hipp" / "spec_builder" / "prompt.md"

# The two HIPP families. Both come from the same hipp_all.json; the split is by
# whether the query states dependency ordering.
ALL_TEMPLATES = [
    "hipp_ordered",
    "hipp_unordered",
]


def _sanitize_model(model: str) -> str:
    return re.sub(r"[/\\. ]+", "_", model)


def load_queries(template: str, queries_dir: Path) -> List[Dict[str, Any]]:
    path = queries_dir / f"{template}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Not found: {path}\n"
            f"Run {queries_dir}/generate_hipp.py first — it writes hipp_all.json, "
            f"hipp_meta.json and the two per-family files this runner reads."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_hipp_gold(path: Path) -> Dict[str, Dict[str, Any]]:
    """query_id → its hipp_meta.json record.

    Hard failure if absent: without it every query would be solved from the
    LLM's own reading of the text, which is a different experiment from the one
    this runner exists to measure, and nothing downstream would say so.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"HIPP ground truth not found: {path}\n"
            f"Run benchmark/socal_queries/llmap_queries/generate_hipp.py to build it."
        )
    records = json.loads(path.read_text(encoding="utf-8"))
    return {r["query_id"]: r for r in records}


# ── Per-query runner ──────────────────────────────────────────────────────────

def run_one_query(
    query: Dict[str, Any],
    *,
    model: str,
    gpkg: str,
    G: Any,
    pois_gdf: Any,
    staging_dir: Path,
    timeout_generate_s: float,
    timeout_execute_s: float,
    python_exe: str,
    parser_llm: "LocalLLM",
    resolver: "CatalogResolver",
    parser_prompt: str,
    hipp_gold: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    query_id = query["query_id"]
    template = query.get("template", "unknown")
    # Model-namespaced scratch key so parallel runs on the same query_id never
    # share generator/staging dirs.
    scratch_id = f"{_sanitize_model(model)}__{query_id}"

    record: Dict[str, Any] = {
        "query_id": query_id,
        "template": template,
        "nl_query": query.get("nl_query", ""),
        "parser_output":    None,
        "generated_bodies": None,
        "generate": {},
        "execute":  {},
        "llm_meta": None,
        "result":   None,
        "metrics":  None,
    }

    # Step 1: parse
    try:
        staged_path = run_parser(
            query, work_dir=staging_dir, gpkg=gpkg,
            llm=parser_llm, resolver=resolver, problem_text=parser_prompt,
            G=G, stage_name=scratch_id, hipp_gold=hipp_gold,
        )
    except Exception as exc:
        record["generate"] = {"ok": False, "error": f"parse: {exc}", "elapsed_s": 0.0}
        record["metrics"]  = compute_metrics_error()
        return record

    # Capture parser_output inline as soon as parsing succeeds, so it is saved
    # in the eval record even if generate/execute later fail (and never depends
    # on the staging dir surviving).
    try:
        _pin = read_input_params(scratch_id)
        record["parser_output"] = {k: v for k, v in _pin.items()
                                   if k not in ("nl_query", "gpkg", "query_id")}
    except Exception:
        pass

    # Step 2: generate
    gen_info = run_generator(scratch_id, str(staged_path), model, timeout_generate_s, python_exe)
    record["generate"] = gen_info

    llm_meta_path = ORCSTRATOR_OUTPUTS / f"{scratch_id}_output" / "llm_meta.json"
    if llm_meta_path.exists():
        try:
            record["llm_meta"] = json.loads(llm_meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    raw_resp_path = ORCSTRATOR_OUTPUTS / f"{scratch_id}_output" / "raw_response.json"
    if raw_resp_path.exists():
        try:
            record["generated_bodies"] = json.loads(raw_resp_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    if not gen_info["ok"]:
        record["metrics"] = compute_metrics_error()
        return record

    # Step 3: execute
    exe_info = run_execute(scratch_id, timeout_execute_s, python_exe)
    record["execute"] = exe_info

    input_params = read_input_params(scratch_id)
    record["parser_output"] = {k: v for k, v in input_params.items()
                               if k not in ("nl_query", "gpkg", "query_id")}
    if exe_info["ok"]:
        weight_key = input_params.get("weight_key", "travel_time")
        enrich_output(ORCSTRATOR_OUTPUTS / f"{scratch_id}_output", G, pois_gdf, weight_key)

    output = read_output(scratch_id)
    record["result"] = output

    if not exe_info["ok"] or output is None:
        record["metrics"] = compute_metrics_error()
        return record

    record["metrics"] = compute_metrics(output, input_params, G, pois_gdf)
    return record


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Run the craft_hipp pipeline on the HIPP benchmark.")
    ap.add_argument("--templates", nargs="+", default=ALL_TEMPLATES, metavar="TEMPLATE")
    ap.add_argument("--queries-dir", default=str(HIPP_QUERIES_DIR),
                    help=f"Directory containing template JSON files (default: {HIPP_QUERIES_DIR})")
    ap.add_argument("--limit", type=int, default=None, help="Max queries per template (smoke test)")
    ap.add_argument("--gpkg", default=DEFAULT_GPKG)
    ap.add_argument("--catalog-pois", default=str(CATALOG_POIS_PATH),
                    help=f"POI catalog JSON for the parser's category grounding "
                         f"(default: {CATALOG_POIS_PATH}). Point this at the catalog "
                         f"matching --gpkg when running a non-SoCal dataset.")
    ap.add_argument("--timeout-generate", type=float, default=90.0)
    ap.add_argument("--timeout-execute",  type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=4, help="Parallel query workers")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--model", default=None, help="Override LLM model")
    ap.add_argument("--local", action="store_true",
                    help="Use OPENAI_BASE_URL + OPENAI_API_KEY (for local QWEN)")
    ap.add_argument("--no-thinking", action="store_true",
                    help="Disable thinking mode (Qwen3)")
    ap.add_argument("--out-dir", default=None,
                    help="Root output dir (default: outputs/crafthipp). Records go to {out-dir}/{model_slug}/...")
    args = ap.parse_args()

    if args.local:
        os.environ.setdefault("OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
        model = args.model or os.getenv("OPENAI_MODEL") or "Qwen/Qwen3-32B-AWQ"
    else:
        model = (
            args.model
            or os.getenv("GEMINI_MODEL") or read_env("GEMINI_MODEL")
            or os.getenv("OPENAI_MODEL") or read_env("OPENAI_MODEL")
            or "gemini-2.0-flash"
        )

    if args.no_thinking:
        os.environ["QWEN_NO_THINKING"] = "1"

    model_slug  = _sanitize_model(model)
    queries_dir = Path(args.queries_dir).resolve()
    args.gpkg   = str(Path(args.gpkg).resolve())
    staging_dir = STAGING_DIR
    staging_dir.mkdir(parents=True, exist_ok=True)

    craft_out_root = (Path(args.out_dir) if args.out_dir else ROOT / "outputs" / "crafthipp") / model_slug

    print(f"Model      : {model}")
    print(f"Queries dir: {queries_dir}")
    print(f"Templates  : {args.templates}")
    print(f"Limit      : {args.limit or 'all'}")
    print(f"Workers    : {args.workers}")
    print(f"GPKG       : {args.gpkg}")
    print(f"Output   : {craft_out_root}")

    print("\nInitialising parser ...")
    parser_llm    = LocalLLM(model=model, timeout_s=args.timeout_generate)
    # gpkg_path is REQUIRED for category grounding: catalog_pois.json alone
    # carries only 26 categories and is missing e.g. cafe and attraction, so the
    # parser silently degrades them ("a cafe" -> restaurant + cuisine:cafe).
    # Adding the gpkg brings the full POI-column category list (~1438). Same fix
    # already present in run_crafthipp.py and craft/spec_builder/agent.py's CLI path.
    resolver      = CatalogResolver(str(Path(args.catalog_pois).resolve()), gpkg_path=args.gpkg)
    parser_prompt = SPEC_BUILDER_PROMPT.read_text(encoding="utf-8")

    hipp_gold_by_id = load_hipp_gold(HIPP_META_PATH)
    print(f"HIPP gold  : {len(hipp_gold_by_id)} records from {HIPP_META_PATH.name}")

    print("Loading graph and POIs ...")
    G        = build_graph_from_gpkg(args.gpkg)
    pois_gdf = gpd.read_file(args.gpkg, layer="pois")
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(pois_gdf)} POIs")

    all_records: List[Dict[str, Any]] = []

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    print_lock   = threading.Lock()
    records_lock = threading.Lock()

    def _run_and_save(i: int, n_total: int, query: Dict[str, Any], template: str) -> None:
        query_id   = query["query_id"]
        # Output locations
        tmpl_dir   = craft_out_root / template
        tmpl_dir.mkdir(parents=True, exist_ok=True)
        record_path = tmpl_dir / f"{query_id}.json"

        if record_path.exists():
            with print_lock:
                print(f"  [{i}/{n_total}] {query_id} ... skipped (exists)", flush=True)
            return

        t0     = time.perf_counter()
        record = run_one_query(
            query,
            model=model, gpkg=args.gpkg, G=G, pois_gdf=pois_gdf,
            staging_dir=staging_dir,
            timeout_generate_s=args.timeout_generate,
            timeout_execute_s=args.timeout_execute,
            python_exe=args.python,
            parser_llm=parser_llm, resolver=resolver,
            parser_prompt=parser_prompt,
            hipp_gold=hipp_gold_by_id.get(query_id),
        )
        elapsed = time.perf_counter() - t0
        m       = record.get("metrics") or {}

        with print_lock:
            print(
                f"  [{i}/{n_total}] {query_id} ... "
                f"ran={m.get('ran','?')}  feasible={m.get('feasible','?')}  "
                f"CS={m.get('constraint_satisfaction', 0.0):.2f}  ({elapsed:.1f}s)",
                flush=True,
            )
            if args.verbose:
                for k in ("generate", "execute"):
                    err = record.get(k, {}).get("error", "")
                    if err:
                        print(f"       {k} error: {err[:300]}")

        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))

        # Copy raw output.json to per-model location for later comparison
        src = ORCSTRATOR_OUTPUTS / f"{_sanitize_model(model)}__{query_id}_output" / "output.json"
        if src.exists():
            raw_dir = craft_out_root / "hipp" / f"{query_id}_output"
            raw_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, raw_dir / "output.json")

        with records_lock:
            all_records.append(record)

    for template in args.templates:
        try:
            raw_queries = load_queries(template, queries_dir)
        except FileNotFoundError as exc:
            print(f"[skip] {exc}")
            continue

        queries = raw_queries[:args.limit] if args.limit else raw_queries
        print(f"\n── {template}: {len(queries)} queries ──")

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [
                pool.submit(
                    _run_and_save, i, len(queries),
                    {**raw_query, "template": template, "gpkg": args.gpkg},
                    template,
                )
                for i, raw_query in enumerate(queries, 1)
            ]
            for fut in as_completed(futs):
                if fut.exception():
                    with print_lock:
                        print(f"  [worker error] {fut.exception()}", flush=True)

    # Summary
    from collections import defaultdict
    tmpl_stats: dict = defaultdict(lambda: {"total": 0, "ran": 0, "feasible": 0})
    for r in all_records:
        tmpl = r.get("template", "unknown")
        m    = r.get("metrics") or {}
        tmpl_stats[tmpl]["total"]    += 1
        tmpl_stats[tmpl]["ran"]      += int(bool(m.get("ran")))
        tmpl_stats[tmpl]["feasible"] += int(bool(m.get("feasible")))

    n_ran      = sum(1 for r in all_records if (r.get("metrics") or {}).get("ran"))
    n_feasible = sum(1 for r in all_records if (r.get("metrics") or {}).get("feasible"))

    print(f"\n{'='*60}")
    print(f"Model : {model}")
    print(f"{'Template':<30} {'Total':>5} {'Ran':>5} {'Feasible':>8}")
    print(f"{'-'*60}")
    for tmpl in args.templates:
        s = tmpl_stats.get(tmpl)
        if not s:
            continue
        print(f"{tmpl:<30} {s['total']:>5} {s['ran']:>5} {s['feasible']:>8}")
    print(f"{'-'*60}")
    print(f"{'TOTAL':<30} {len(all_records):>5} {n_ran:>5} {n_feasible:>8}")
    print(f"{'='*60}")
    print(f"\nCraft outputs : {craft_out_root}")


if __name__ == "__main__":
    main()
