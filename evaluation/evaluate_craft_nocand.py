"""
evaluate_craft_nocand.py — Export the craft "nocand" ablation to Excel.

The nocand ablation (`craft_nocand/`, run via `scripts/run_craft_nofilter.py --variant nocand`)
removes the CANDIDATE-SET component from craft: no `possible_targets`, no `type_nodes`.
The generated solver still receives `pois_gdf` + the POI schema + `poi_types`, so it *can*
derive POI-category membership itself — whether it does is exactly what this ablation tests.

This exporter puts both arms in one workbook so the ablation is self-contained:
  arm="baseline"  → outputs/craft/{model}/{template}      (candidate set present)
  arm="nocand"    → outputs/craft_nocand/{model}/{template} (candidate set removed)

Both arms are scored through the same shared evaluation.constraint_checker used by
results/craft_set1.xlsx, so the baseline rows here reproduce that file's algo2 numbers
and the comparison is apples-to-apples.

Headline metric follows the project convention (see CLAUDE.md "EVALUATION & SCORING"):
  solved_correctly = feasible AND constraints_satisfied, counted over FIXED N (=200/model/arm).
Do NOT compare arms on the raw `feasible` column or the `constraint_satisfaction` mean.

Usage:
  conda run -n myenv python evaluation/evaluate_craft_nocand.py
  conda run -n myenv python evaluation/evaluate_craft_nocand.py --templates ordered_poi
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluation.constraint_checker as cc
from evaluation.evaluate_llm import _write_table

BASELINE_DIR = ROOT / "outputs" / "craft"
NOCAND_DIR   = ROOT / "outputs" / "craft_nocand"
QUERIES_DIR  = ROOT / "benchmark" / "socal_queries" / "set1"
DEFAULT_GPKG = ROOT / "data" / "socal" / "road_network.gpkg"
DEFAULT_OUT  = ROOT / "results" / "craft_nocand_algo2.xlsx"

MODELS = ["gemini-3_1-flash-lite", "gpt-4_1", "openai_gpt-oss-20b"]
TEMPLATES = ["ordered_poi"]

ARMS = [("baseline", BASELINE_DIR), ("nocand", NOCAND_DIR)]


def _exc_line(err) -> Optional[str]:
    if not err:
        return None
    lines = [ln.strip() for ln in str(err).splitlines() if ln.strip()]
    return lines[-1][:300] if lines else None


def _error_bucket(gen: dict, exe: dict) -> tuple:
    """(error_stage, error, bucket) — bucket groups the nocand failure modes."""
    if gen and gen.get("ok") is False:
        msg = _exc_line(gen.get("error"))
        return "generate", msg, "generate_fail"
    if exe and exe.get("ok") is False:
        msg = _exc_line(exe.get("error")) or ""
        raw = str(exe.get("error") or "")
        if "timeout" in raw.lower():
            bucket = "execute_timeout"
        elif "has no attribute" in raw:
            bucket = "execute_attributeerror"
        elif "KeyError" in raw:
            bucket = "execute_keyerror"
        else:
            bucket = "execute_other"
        return "execute", msg, bucket
    return "none", None, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--templates", nargs="+", default=TEMPLATES)
    ap.add_argument("--gpkg", default=str(DEFAULT_GPKG))
    ap.add_argument("--queries-dir", default=str(QUERIES_DIR))
    args = ap.parse_args()

    queries_dir = Path(args.queries_dir)

    print("Loading graph + POIs for constraint/objective computation ...")
    import geopandas as gpd
    from craft.generator.tools.common_handlers import build_graph_from_gpkg
    G = build_graph_from_gpkg(args.gpkg)
    pois_gdf = gpd.read_file(args.gpkg, layer="pois")
    snapper = cc.NodeSnapper.from_gpkg(args.gpkg, G)
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(pois_gdf)} POIs")
    caches = cc.build_poi_caches(G, pois_gdf)

    nl_index: Dict[str, str] = {}
    valid_ids: set = set()
    for tmpl in args.templates:
        q_file = queries_dir / f"{tmpl}.json"
        if q_file.exists():
            for q in json.loads(q_file.read_text()):
                nl_index[q["query_id"]] = q["nl_query"]
                valid_ids.add(q["query_id"])
    print(f"  {len(valid_ids)} benchmark query ids in scope")

    rows: List[dict] = []
    for arm, arm_dir in ARMS:
        for model in args.models:
            model_dir = Path(arm_dir) / model
            if not model_dir.exists():
                print(f"[skip] {arm}/{model} — no output dir")
                continue
            n_loaded = 0
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
                    if valid_ids and qid not in valid_ids:
                        continue
                    nl_query = d.get("nl_query") or nl_index.get(qid, "")
                    metrics  = d.get("metrics") or {}
                    result   = d.get("result") or {}
                    llm_meta = d.get("llm_meta") or {}
                    gen_info = d.get("generate") or {}
                    exe_info = d.get("execute") or {}

                    ran_ok   = bool(metrics.get("ran"))
                    feasible = bool(metrics.get("feasible"))
                    route    = result.get("route") if (isinstance(result, dict) and ran_ok) else None

                    core = cc.evaluate_system(nl_query, tmpl, route or [], G, pois_gdf,
                                              snapper, caches=caches)
                    if not ran_ok:
                        core = {k: None for k in core}

                    tok = llm_meta.get("tokens") or {}
                    error_stage, error, bucket = _error_bucket(gen_info, exe_info)
                    solved_correctly = bool(feasible and core.get("constraints_satisfied") is True)

                    rows.append({
                        "arm":          arm,
                        "query_id":     qid,
                        "template":     tmpl,
                        "model":        model,
                        "feasible":     feasible,
                        "ran":          ran_ok,
                        "solved_correctly": solved_correctly,
                        "generate_ok":  bool(gen_info.get("ok")) if gen_info else False,
                        "execute_ok":   bool(exe_info.get("ok")) if exe_info else False,
                        "error_stage":  error_stage,
                        "error_bucket": bucket,
                        "error":        error,
                        "search_time_ms": metrics.get("run_time_ms"),
                        "states_expanded": metrics.get("states_expanded"),
                        "states_generated": metrics.get("states_generated"),
                        "states_pruned":   metrics.get("states_pruned"),
                        "peak_heap_size":  metrics.get("peak_heap_size"),
                        **{k: core[k] for k in cc.CORE_FIELDS},
                        "dijkstra_objective_value": core["dijkstra_objective_value"],
                        "distance_limits_met_realized": core.get("distance_limits_met_realized"),
                        "time_budget_met_realized":     core.get("time_budget_met_realized"),
                        "latency_s":         gen_info.get("elapsed_s"),
                        "prompt_tokens":     tok.get("prompt_tokens"),
                        "completion_tokens": tok.get("completion_tokens"),
                    })
                    n_loaded += 1
            print(f"  {arm}/{model}: {n_loaded} records")

    if not rows:
        sys.exit("No records found")

    out = Path(args.out)
    if out.suffix.lower() != ".xlsx":
        out = out.with_suffix(".xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    ws = wb.active
    ws.title = "Results"
    _write_table(ws, fieldnames, [[r[f] for f in fieldnames] for r in rows],
                 title="CRAFT nocand ablation — candidate set present (baseline) vs removed (nocand)")

    # Header row index inside the Results sheet (title + blank + header)
    HEADER_ROW = 3
    DATA_FIRST = HEADER_ROW + 1
    DATA_LAST  = HEADER_ROW + len(rows)

    def colref(name: str) -> str:
        """Resolve a column letter by header NAME (never hard-code letters —
        adding a column silently invalidates letter-based formulas)."""
        idx = fieldnames.index(name) + 1
        L = get_column_letter(idx)
        return f"Results!${L}${DATA_FIRST}:${L}${DATA_LAST}"

    C_ARM   = colref("arm")
    C_MODEL = colref("model")
    C_SOLVED = colref("solved_correctly")
    C_FEAS  = colref("feasible")
    C_RAN   = colref("ran")
    C_BUCKET = colref("error_bucket")

    # Paper Table
    ws2 = wb.create_sheet("Paper Tables")
    ws2.append(["Table 1 — CRAFT nocand ablation on ordered_poi (N=200 per model per arm)"])
    ws2.cell(ws2.max_row, 1).font = Font(bold=True, size=13)
    ws2.append([])
    ws2.append(["Removing the candidate set (possible_targets / type_nodes) forces the generated "
                "solver to derive POI-category membership from pois_gdf + schema itself."])
    ws2.append(["solved_correctly = feasible AND constraints_satisfied, over FIXED N. "
                "Check must equal N (200) for each row."])
    ws2.append([])

    headers = ["Model", "Arm", "Solved correctly", "Solved %", "Feasible", "Ran",
               "Execute timeout", "Execute AttributeError", "Execute KeyError",
               "Execute other", "Generate fail", "N", "Check"]
    ws2.append(headers)
    hrow = ws2.max_row
    from openpyxl.styles import PatternFill, Alignment
    for col in range(1, len(headers) + 1):
        c = ws2.cell(hrow, col)
        c.fill = PatternFill("solid", fgColor="1F4E79")
        c.font = Font(bold=True, color="FFFFFF")
        c.alignment = Alignment(horizontal="center", wrap_text=True)

    for model in args.models:
        for arm, _ in ARMS:
            cond = f'{C_MODEL},"{model}",{C_ARM},"{arm}"'
            r = ws2.max_row + 1
            ws2.append([
                model,
                arm,
                f'=COUNTIFS({cond},{C_SOLVED},TRUE)',
                None,  # filled below as a formula referencing this row
                f'=COUNTIFS({cond},{C_FEAS},TRUE)',
                f'=COUNTIFS({cond},{C_RAN},TRUE)',
                f'=COUNTIFS({cond},{C_BUCKET},"execute_timeout")',
                f'=COUNTIFS({cond},{C_BUCKET},"execute_attributeerror")',
                f'=COUNTIFS({cond},{C_BUCKET},"execute_keyerror")',
                f'=COUNTIFS({cond},{C_BUCKET},"execute_other")',
                f'=COUNTIFS({cond},{C_BUCKET},"generate_fail")',
                f'=COUNTIFS({cond})',
                # Check: ok + all failure buckets must sum to N
                f'=COUNTIFS({cond},{C_BUCKET},"ok")+G{r}+H{r}+I{r}+J{r}+K{r}',
            ])
            ws2.cell(r, 4).value = f'=IF(L{r}=0,"",C{r}/L{r})'
            ws2.cell(r, 4).number_format = "0.0%"

    for col in range(1, len(headers) + 1):
        ws2.column_dimensions[get_column_letter(col)].width = 20 if col <= 2 else 15

    # Column descriptions
    ws3 = wb.create_sheet("Column Descriptions")
    legend = [
        ("arm", "baseline = normal craft (candidate set present, outputs/craft). "
                "nocand = candidate-set component removed (outputs/craft_nocand): no possible_targets, "
                "no type_nodes; the solver still gets pois_gdf + POI schema + poi_types and must derive "
                "category membership itself."),
        ("solved_correctly", "feasible AND constraints_satisfied — THE headline metric, counted over "
                             "fixed N=200. Do not compare arms on raw `feasible` or on the "
                             "constraint_satisfaction mean (the latter drops None/empty rows from its "
                             "denominator and inflates the failing arm)."),
        ("feasible", "True if the Dijkstra search returned a valid route"),
        ("ran", "True if generated.py executed without crashing"),
        ("error_stage", "'generate', 'execute', or 'none'. 'none' with feasible=False means the search "
                        "ran cleanly but found no route (a genuine no-route, not an error)."),
        ("error_bucket", "Grouped failure mode: ok / execute_timeout / execute_attributeerror / "
                         "execute_keyerror / execute_other / generate_fail. In the nocand arm, "
                         "execute_attributeerror is the signature failure — the model reaches for a "
                         "candidate-set attribute that no longer exists (e.g. self.poi_nodes_by_type)."),
        ("error", "Last traceback line from the failing stage"),
        ("constraints_satisfied", "True iff every applicable hard constraint is True; None when nothing "
                                  "was evaluable (no route) — never vacuously True."),
        ("objective_value", "Leg-by-leg SP cost through the intended POI stops (comparable across arms)"),
        ("dijkstra_objective_value", "Cost of craft's realized Dijkstra route, same weight"),
        ("distance_limits_met / time_budget_met", "SP-through-chosen-stops for all systems (2026-06-17 "
                                                  "fairness convention); *_realized are craft's own route."),
        ("states_expanded", "Search-effort counter; populated only for runs made after 2026-08-05."),
    ]
    _write_table(ws3, ["Column", "Description"], [list(x) for x in legend])

    wb.calculation.fullCalcOnLoad = True
    wb.save(out)
    print(f"\nWrote {len(rows)} rows → {out}")

    # console summary
    print("\nSummary (solved_correctly / N):")
    for model in args.models:
        line = f"  {model:24s}"
        for arm, _ in ARMS:
            sub = [r for r in rows if r["model"] == model and r["arm"] == arm]
            if not sub:
                line += f"  {arm}: n/a"
                continue
            s = sum(1 for r in sub if r["solved_correctly"])
            line += f"  {arm}: {s}/{len(sub)} ({s/len(sub):.1%})"
        print(line)


if __name__ == "__main__":
    main()
