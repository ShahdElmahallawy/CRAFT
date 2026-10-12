"""
evaluate_craft_set3.py — Export set3_descriptions (ordered+desc) craft results to Excel.

Sibling of evaluate_craft.py, kept as a SEPARATE script (not a flag on the
set1 evaluator) for the same reason evaluate_crafthipp.py is: set3 must never
perturb the locked set1 evaluation path.

Reads outputs/craft/{model}/ordered+desc/{query_id}.json and scores each record
against its benchmark/socal_queries/set3_descriptions/set3_meta.json gold record
via evaluation.constraint_checker.evaluate_set3_system — gold-driven, not
NL-regex-driven (the filter a stop must satisfy exists only in the gold sidecar;
see the set3 section of constraint_checker.py).

Two solved-correctly definitions are exported side by side:
  solved_correctly           = feasible AND constraints_satisfied
                               (INCLUDES poi_filters_respected — the headline)
  solved_correctly_nofilter  = feasible AND constraints_satisfied_nofilter
                               (type/order/distance only — reproduces the
                               pre-filter-scoring numbers for continuity)

Usage:
  python evaluation/evaluate_craft_set3.py
  python evaluation/evaluate_craft_set3.py --models gemini-3_1-flash-lite
  python evaluation/evaluate_craft_set3.py --no-sp
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluation.constraint_checker as cc
from evaluation.evaluate_llm import _write_table

CRAFT_DIR    = ROOT / "outputs" / "craft"
DEFAULT_GPKG = ROOT / "data" / "socal" / "road_network.gpkg"
DEFAULT_OUT  = ROOT / "results" / "craft_set3.xlsx"

MODELS = [
    "gemini-3_1-flash-lite",
    "gpt-4_1",
    "openai_gpt-oss-20b",
]
TEMPLATES = [cc.SET3_TEMPLATE]


def _read_env(key: str) -> str:
    """Minimal .env reader (same convention as craft/generator/agent.py::read_env).
    Only needed the FIRST time this script runs, to embed the 13 distinct gold
    description phrases; afterwards the vectors come from the on-disk cache."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return ""
    for line in env_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        k, v = text.split("=", 1)
        if k.strip() == key:
            return v.strip().strip("'").strip('"')
    return ""


def _exc_line(err) -> Optional[str]:
    if not err:
        return None
    lines = [ln.strip() for ln in str(err).splitlines() if ln.strip()]
    return lines[-1][:300] if lines else None


def _craft_error(gen: dict, exe: dict):
    if gen and gen.get("ok") is False:
        return "generate", _exc_line(gen.get("error"))
    if exe and exe.get("ok") is False:
        return "execute", _exc_line(exe.get("error"))
    return "none", None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",          default=str(DEFAULT_OUT))
    ap.add_argument("--models",       nargs="+", default=MODELS)
    ap.add_argument("--templates",    nargs="+", default=TEMPLATES)
    ap.add_argument("--gpkg",         default=str(DEFAULT_GPKG))
    ap.add_argument("--baseline-dir", default=str(CRAFT_DIR))
    ap.add_argument("--no-sp",        action="store_true", help="Skip SP/graph computation")
    args = ap.parse_args()

    baseline_dir = Path(args.baseline_dir)

    print("Loading set3 gold (set3_meta.json) ...")
    gold_by_qid = cc.load_set3_gold()
    print(f"  {len(gold_by_qid)} gold records")
    gold_records = list(gold_by_qid.values())

    print("Building set3 query-phrase embeddings ...")
    query_embeddings = cc.build_set3_query_embeddings(
        gold_records, api_key=_read_env("GEMINI_API_KEY") or None)
    print(f"  {len(query_embeddings)} phrase vectors available")

    G = None
    pois_gdf = None
    snapper = None
    caches = None
    filter_cache: Dict = {}
    if not args.no_sp:
        print("Loading graph + POIs for constraint/objective computation ...")
        import geopandas as gpd
        from craft.generator.tools.common_handlers import build_graph_from_gpkg
        G = build_graph_from_gpkg(args.gpkg)
        pois_gdf = gpd.read_file(args.gpkg, layer="pois")
        snapper = cc.NodeSnapper.from_gpkg(args.gpkg, G)
        print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(pois_gdf)} POIs")
        print("Precomputing POI-type caches + set3 filter cache (one pass each) ...")
        # MUST pass the set3 categories explicitly — build_poi_caches defaults to
        # the 4 LLMAP display types and every set3 stop would silently miss.
        caches = cc.build_poi_caches(G, pois_gdf, poi_types=cc.SET3_POI_TYPES)
        filter_cache = cc.build_set3_filter_cache(G, pois_gdf, gold_records)
        print(f"  cached {sum(len(v) for v in caches[0].values())} type-nodes, "
              f"{len(filter_cache['structured'])} structured filter sets, "
              f"{len(filter_cache['node_rows'])} embedded POI nodes")

    rows = []
    for model in args.models:
        model_dir = baseline_dir / model
        if not model_dir.exists():
            print(f"[skip] {model} — no craft output dir")
            continue

        n_loaded = 0
        print(f"Loading {model} ...")
        for tmpl in args.templates:
            tmpl_dir = model_dir / tmpl
            if not tmpl_dir.exists():
                continue
            for fpath in sorted(tmpl_dir.glob("*.json")):
                try:
                    d = json.loads(fpath.read_text())
                except Exception as exc:
                    print(f"  [warn] {fpath.name}: {exc}")
                    continue

                qid = d.get("query_id", fpath.stem)
                gold = gold_by_qid.get(qid)
                if gold is None:
                    print(f"  [warn] {qid}: no set3_meta gold record — skipping")
                    continue

                metrics  = d.get("metrics") or {}
                result   = d.get("result") or {}
                llm_meta = d.get("llm_meta") or {}
                gen_info = d.get("generate") or {}
                exe_info = d.get("execute") or {}

                ran_ok   = bool(metrics.get("ran"))
                feasible = bool(metrics.get("feasible"))
                route    = result.get("route") if (isinstance(result, dict) and ran_ok) else None

                core = cc.evaluate_set3_system(
                    gold, d.get("parser_output"), route or [], G, pois_gdf, snapper,
                    caches=caches, filter_cache=filter_cache,
                    query_embeddings=query_embeddings,
                )
                if not ran_ok:
                    core = {k: None for k in core}

                tok = llm_meta.get("tokens") or {}
                error_stage, error = _craft_error(gen_info, exe_info)
                row = {
                    "query_id":   qid,
                    "template":   tmpl,
                    "model":      model,
                    "city":       gold.get("city"),
                    "difficulty": gold.get("difficulty"),
                    "feasible":   feasible,
                    "ran":        ran_ok,
                    "solved_correctly":
                        bool(feasible and core.get("constraints_satisfied") is True),
                    "solved_correctly_nofilter":
                        bool(feasible and core.get("constraints_satisfied_nofilter") is True),
                    "generate_ok": bool(gen_info.get("ok")) if gen_info else False,
                    "execute_ok":  bool(exe_info.get("ok")) if exe_info else False,
                    "error_stage": error_stage,
                    "error":       error,
                    "search_time_ms": metrics.get("run_time_ms"),
                    # .get, not [k]: CORE_FIELDS is the SHARED column list and gains
                    # fields for templates set3 does not have — the 6 highway-usage
                    # metrics added 2026-08-17 are emitted only by the poi_road_pref
                    # path, so strict indexing made this exporter crash outright
                    # (KeyError: 'route_segments') and set3 could not be re-exported
                    # at all. Absent field = not applicable here = None, which is the
                    # same convention the rest of this dict already uses.
                    **{k: core.get(k) for k in cc.CORE_FIELDS},
                    "constraints_satisfied_nofilter": core.get("constraints_satisfied_nofilter"),
                    "structured_filter_respected":    core.get("structured_filter_respected"),
                    "desc_filter_respected":          core.get("desc_filter_respected"),
                    "filters_respected_basetype":     core.get("filters_respected_basetype"),
                    "n_filtered_stops":               core.get("n_filtered_stops"),
                    "n_filter_stops_scored":          core.get("n_filter_stops_scored"),
                    "dijkstra_objective_value":       core.get("dijkstra_objective_value"),
                    "latency_s":         gen_info.get("elapsed_s"),
                    "prompt_tokens":     tok.get("prompt_tokens"),
                    "completion_tokens": tok.get("completion_tokens"),
                }
                rows.append(row)
                n_loaded += 1
        print(f"  {n_loaded} records loaded")

    if not rows:
        sys.exit("No records found")

    out = Path(args.out)
    if out.suffix.lower() != ".xlsx":
        out = out.with_suffix(".xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    from openpyxl import Workbook
    wb = Workbook()
    ws_data = wb.active
    ws_data.title = "Results"
    _write_table(ws_data, fieldnames,
                 [[r[f] for f in fieldnames] for r in rows],
                 title="Craft set3 Results (filtered-POI ordered routes)")

    ws_legend = wb.create_sheet("Column Descriptions")
    legend = [
        ("query_id", "set3 query id (set3_ordered+desc_NNN)"),
        ("template", "ordered+desc — ordered POI stops with attribute/description filters"),
        ("model",    "LLM used for code generation"),
        ("city",     "SoCal city the query was generated in (gold)"),
        ("difficulty", "easy / medium / hard (gold) — drives how tight the per-leg limits are"),
        ("feasible", "True if the Dijkstra search returned a valid route"),
        ("ran",      "True if generated.py executed without crashing"),
        ("solved_correctly", "HEADLINE: feasible AND constraints_satisfied — INCLUDES filter compliance, "
                             "i.e. the route visited the right POI types in order, within the per-leg "
                             "distance limits, AND every filtered stop actually satisfies its filter"),
        ("solved_correctly_nofilter", "feasible AND constraints_satisfied_nofilter — the same check with "
                                      "poi_filters_respected excluded. Reproduces the pre-filter-scoring "
                                      "numbers so the two are directly comparable"),
        ("generate_ok", "True if the LLM code-generation step succeeded"),
        ("execute_ok",  "True if generated.py ran without errors"),
        ("error_stage", "Stage that threw an exception: 'generate', 'execute', or 'none'. error_stage='none' "
                        "with feasible=False means the search ran cleanly but found no route"),
        ("error",       "Exception message (last traceback line); None if no exception occurred"),
        ("search_time_ms", "Dijkstra search time in milliseconds"),
        *cc.CORE_FIELD_DESCRIPTIONS,
        ("constraints_satisfied_nofilter", "constraints_satisfied computed without poi_filters_respected"),
        ("structured_filter_respected", "DIAGNOSTIC (not a hard constraint): over the structured-filter stops "
                                        "the route actually VISITED, did every one satisfy its gold column/value? "
                                        "Excludes unvisited stops, so this is the 'when it picked a stop, was the "
                                        "filter respected' capability rate; None if no such stop was visited"),
        ("desc_filter_respected", "Same diagnostic for description-filter stops, scored by cosine >= 0.65 against "
                                  "the same poi_embeddings cache and threshold filtration.py used at run time"),
        ("filters_respected_basetype", "DIAGNOSTIC, strict lower bound. Same gold filter check but with the stop "
                                       "reconstructed WITHOUT filter narrowing — the first POI of the right category "
                                       "the route reaches. The main columns reconstruct the stop from the filtered "
                                       "candidate set the solver actually searched, which is faithful but partly "
                                       "self-confirming (a node chosen because it is in the filtered set passes the "
                                       "filter by construction); this column removes that circularity at the cost of "
                                       "crediting incidental same-category POIs the route merely passes. Read the "
                                       "two as an upper and lower bound"),
        ("n_filtered_stops", "How many of this query's stops carry a filter (gold); the rest are plain companions"),
        ("n_filter_stops_scored", "How many filtered stops were actually evaluable (gold pair cached, phrase "
                                  "embedded, POI row has an embedding) — the denominator behind poi_filters_respected"),
        ("dijkstra_objective_value", "Cost of Craft's realized Dijkstra route with weight=length (meters)"),
        ("latency_s", "Wall-clock time for the LLM code-generation subprocess (seconds)"),
        ("prompt_tokens", "Prompt tokens used for code generation"),
        ("completion_tokens", "Completion tokens generated"),
    ]
    _write_table(ws_legend, ["Column", "Description"], legend, title="Column Descriptions")

    wb.save(out)
    print(f"Wrote {len(rows)} rows → {out}")

    # Console summary
    by_model: Dict[str, List] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    print(f"\n{'Model':<24} {'N':>4} {'Ran':>5} {'Feas':>5} {'Solved':>7} {'NoFilt':>7} "
          f"{'Struct%':>8} {'Desc%':>7}")
    print("-" * 74)
    for model in dict.fromkeys(r["model"] for r in rows):
        rs = by_model[model]
        n = len(rs)
        n_ran    = sum(1 for r in rs if r["ran"])
        n_feas   = sum(1 for r in rs if r["feasible"])
        n_solved = sum(1 for r in rs if r["solved_correctly"])
        n_nofilt = sum(1 for r in rs if r["solved_correctly_nofilter"])
        st = [r["structured_filter_respected"] for r in rs
              if r["structured_filter_respected"] is not None]
        ds = [r["desc_filter_respected"] for r in rs
              if r["desc_filter_respected"] is not None]
        st_pct = f"{100*sum(1 for v in st if v)/len(st):.1f}%" if st else "n/a"
        ds_pct = f"{100*sum(1 for v in ds if v)/len(ds):.1f}%" if ds else "n/a"
        print(f"{model:<24} {n:>4} {n_ran:>5} {n_feas:>5} {n_solved:>7} {n_nofilt:>7} "
              f"{st_pct:>8} {ds_pct:>7}")
    print("\nStruct%/Desc% are over queries where such a stop was VISITED "
          "(capability rate), not over all N.")


if __name__ == "__main__":
    main()
