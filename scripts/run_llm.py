"""
run_llm.py — Run the bare LLM baseline on benchmark queries.

Auto-discovers all template JSON files in the queries directory,
calls the LLM for each query, matches stops to the dataset, and saves one JSON
per query to: <out-dir>/<model>/<template>/<query_id>.json

Usage:
  # Default (SoCal Set 1, model from .env):
  conda run -n myenv python scripts/run_llm.py

  # Override model:
  conda run -n myenv python scripts/run_llm.py --model gemini-2.5-pro

  # Local endpoint (Qwen):
  conda run -n myenv python scripts/run_llm.py --local --no-thinking

  # Different queries or graph:
  conda run -n myenv python scripts/run_llm.py \\
    --queries-dir benchmark/socal_queries/set2_long_trips \\
    --gpkg data/socal/road_network.gpkg \\
    --out-dir outputs/llm_set2

  # Smoke test:
  conda run -n myenv python scripts/run_llm.py --limit 5 --no-geocode
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import threading
import concurrent.futures
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.llm.llm_baseline import (
    _read_env,
    _load_geocode_cache,
    _save_geocode_cache,
    _load_pois,
    run_query,
    GEOCODE_CACHE_PATH,
)

DEFAULT_QUERIES_DIR = ROOT / "benchmark" / "socal_queries" / "set1"
DEFAULT_GPKG        = str(ROOT / "data" / "socal" / "road_network.gpkg")
DEFAULT_OUT_DIR     = ROOT / "outputs" / "llm"


def _sanitize_model_name(model: str) -> str:
    return re.sub(r"[/\\. ]+", "_", model)


def _discover_templates(queries_dir: Path) -> List[str]:
    return sorted(
        p.stem
        for p in queries_dir.glob("*.json")
        if p.stem not in ("summary", "validate")
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM baseline on benchmark queries.")
    ap.add_argument("--model", default=None,
                    help="LLM model name (overrides OPENAI_MODEL / GEMINI_MODEL)")
    ap.add_argument("--base-url", default=None,
                    help="API base URL (overrides OPENAI_BASE_URL / GEMINI_BASE_URL)")
    ap.add_argument("--local", action="store_true",
                    help="Shorthand: sets --base-url to http://127.0.0.1:8000/v1")
    ap.add_argument("--openai", action="store_true",
                    help="Use OpenAI: OPENAI_BASE_URL + OPENAI_API_KEY from .env (for gpt-4.1)")
    ap.add_argument("--no-thinking", action="store_true",
                    help="Inject enable_thinking=false (Qwen3 thinking mode)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="Override temperature (gpt-5.x only accepts the default, 1)")
    ap.add_argument("--templates", nargs="+", default=None, metavar="TEMPLATE",
                    help="Subset of templates to run (default: all discovered)")
    ap.add_argument("--queries-dir", default=str(DEFAULT_QUERIES_DIR),
                    help=f"Directory containing template JSON files (default: {DEFAULT_QUERIES_DIR})")
    ap.add_argument("--gpkg", default=DEFAULT_GPKG,
                    help=f"Path to road_network.gpkg (default: {DEFAULT_GPKG})")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                    help=f"Root output directory (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max queries per template (smoke test)")
    ap.add_argument("--geocode-threshold", type=float, default=200.0)
    ap.add_argument("--no-geocode", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel worker threads (default: 1). Use 2 for remote APIs.")
    args = ap.parse_args()

    # Resolve base URL — Gemini takes priority over OpenAI (Qwen uses --local which overrides)
    if args.local:
        base_url = "http://127.0.0.1:8000/v1"
    elif args.openai:
        base_url = _read_env("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    elif args.base_url:
        base_url = args.base_url
    else:
        base_url = (
            _read_env("GEMINI_BASE_URL")
            or _read_env("OPENAI_BASE_URL")
            or "https://generativelanguage.googleapis.com/v1beta/openai/"
        )

    if args.openai:
        api_key = _read_env("OPENAI_API_KEY")
    else:
        api_key = _read_env("GEMINI_API_KEY") or _read_env("OPENAI_API_KEY") or ""
    model = (
        args.model
        or _read_env("OPENAI_MODEL")
        or _read_env("GEMINI_MODEL")
        or "gemini-2.0-flash"
    )

    extra_payload = {"enable_thinking": False} if args.no_thinking else {}
    if args.temperature is not None:
        extra_payload["temperature"] = args.temperature
    extra_payload = extra_payload or None
    google_api_key = (
        ""
        if args.no_geocode
        else (
            _read_env("GEOCODING_API_KEY")
            or _read_env("GOOGLE_MAPS_API_KEY")
            or _read_env("GOOGLE_API_KEY")
            or _read_env("key")
        )
    )

    if not api_key:
        sys.exit("No LLM API key found. Set OPENAI_API_KEY or GEMINI_API_KEY in .env")
    if not google_api_key and not args.no_geocode:
        print("Warning: no geocoding API key found — geocoding disabled.")

    queries_dir = Path(args.queries_dir)
    all_templates = _discover_templates(queries_dir)
    templates = args.templates if args.templates else all_templates

    missing = [t for t in templates if not (queries_dir / f"{t}.json").exists()]
    if missing:
        sys.exit(f"Template files not found in {queries_dir}: {missing}")

    model_slug   = _sanitize_model_name(model)
    out_root     = Path(args.out_dir)
    out_root_mdl = out_root / model_slug

    print(f"Model     : {model}")
    print(f"Endpoint  : {base_url}")
    print(f"Thinking  : {'disabled' if args.no_thinking else 'default'}")
    print(f"Out dir   : {out_root_mdl}")
    print(f"Templates : {templates}")
    print(f"Limit     : {args.limit or 'all'}")
    print(f"Geocode   : {'disabled' if args.no_geocode else f'{args.geocode_threshold}m threshold'}")
    print(f"Queries   : {queries_dir}")

    print(f"\nLoading POIs from {args.gpkg} ...")
    pois_gdf = _load_pois(args.gpkg)
    print(f"  {len(pois_gdf)} POIs loaded.")

    geocode_cache = _load_geocode_cache(GEOCODE_CACHE_PATH)
    print(f"  Geocode cache: {len(geocode_cache)} entries.")

    tmpl_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "feasible": 0, "matched": 0, "dropped": 0, "errors": 0}
    )
    total_queries    = 0
    model_all_failed = True
    _lock            = threading.Lock()

    def _process_one(template: str, i: int, total: int, query: Dict[str, Any],
                     out_dir: Path) -> None:
        nonlocal model_all_failed, total_queries
        qid      = query["query_id"]
        out_path = out_dir / f"{qid}.json"
        if out_path.exists():
            print(f"  [{i}/{total}] {qid} ... skipped (exists)")
            with _lock:
                tmpl_stats[template]["total"] += 1
                total_queries += 1
            return

        print(f"  [{i}/{total}] {qid} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        error_flag = False
        try:
            result = run_query(
                query=query,
                api_key=api_key,
                base_url=base_url,
                model=model,
                pois_gdf=pois_gdf,
                threshold_m=args.geocode_threshold,
                google_api_key=google_api_key,
                geocode_cache=geocode_cache,
                verbose=False,
                extra_payload=extra_payload,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"ERROR: {exc} ({elapsed:.1f}s)")
            result = {
                "query_id": qid,
                "nl_query": query.get("nl_query", ""),
                "llm_feasible": False,
                "llm_infeasible_reason": str(exc),
                "llm_stops": [],
                "matched_stops": [],
                "dropped_stops": [],
                "classification": None,
                "model": model,
                "latency_s": round(time.perf_counter() - t0, 3),
                "tokens": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                "geocode_bounds": None,
                "geocode_threshold_m": args.geocode_threshold,
                "error": str(exc),
            }
            error_flag = True

        elapsed   = time.perf_counter() - t0
        n_stops   = len(result.get("llm_stops", []))
        n_matched = len(result.get("matched_stops", []))
        n_dropped = len(result.get("dropped_stops", []))
        feasible  = result.get("llm_feasible", False)
        print(
            f"feasible={feasible}  stops={n_stops}  matched={n_matched}"
            f"  dropped={n_dropped}  ({elapsed:.1f}s)"
        )

        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        with _lock:
            if not error_flag:
                model_all_failed = False
            s = tmpl_stats[template]
            s["total"]    += 1
            s["feasible"] += int(feasible)
            s["matched"]  += n_matched
            s["dropped"]  += n_dropped
            if error_flag:
                s["errors"] += 1
            total_queries += 1
            if google_api_key and total_queries % 10 == 0:
                _save_geocode_cache(geocode_cache, GEOCODE_CACHE_PATH)

    for template in templates:
        query_file = queries_dir / f"{template}.json"
        queries: List[Dict[str, Any]] = json.loads(query_file.read_text(encoding="utf-8"))
        if args.limit:
            queries = queries[: args.limit]

        out_dir = out_root_mdl / template
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n── {template}: {len(queries)} queries (workers={args.workers}) ──")

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [
                ex.submit(_process_one, template, i, len(queries), q, out_dir)
                for i, q in enumerate(queries, 1)
            ]
            for f in concurrent.futures.as_completed(futs):
                f.result()  # re-raise any unexpected exception

        if google_api_key:
            _save_geocode_cache(geocode_cache, GEOCODE_CACHE_PATH)

    if model_all_failed and total_queries > 0:
        print(f"\n[WARNING] Model '{model}' failed on every query — likely invalid model name.")

    print(f"\n{'='*75}")
    print(f"Model: {model}")
    print(f"{'Template':<32} {'Total':>5} {'Feasible':>8} {'Matched':>7} {'Dropped':>7} {'Errors':>6}")
    print(f"{'-'*75}")
    for tmpl in templates:
        s = tmpl_stats.get(tmpl)
        if not s:
            continue
        print(
            f"{tmpl:<32} {s['total']:>5} {s['feasible']:>8} {s['matched']:>7}"
            f" {s['dropped']:>7} {s['errors']:>6}"
        )
    all_s = {
        k: sum(tmpl_stats[t].get(k, 0) for t in tmpl_stats)
        for k in ("total", "feasible", "matched", "dropped", "errors")
    }
    print(f"{'-'*75}")
    print(
        f"{'TOTAL':<32} {all_s['total']:>5} {all_s['feasible']:>8} {all_s['matched']:>7}"
        f" {all_s['dropped']:>7} {all_s['errors']:>6}"
    )
    print(f"{'='*75}")
    print(f"\nOutputs: {out_root_mdl}")


if __name__ == "__main__":
    main()
