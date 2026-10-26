"""
evaluation/optimal/export_optimality_table.py — builds results/optimality_gap.xlsx,
the workbook behind the paper's optimality-gap table.

Table shape (rows = templates, columns = the LLM backing CRAFT):
    Opt  % of ORACLE-SOLVABLE queries CRAFT solved OPTIMALLY
    Med  median absolute percentage error over CRAFT's CORRECT routes
    Max  maximum absolute percentage error over CRAFT's CORRECT routes

DEFINITIONS — each of these changes the number, so they are pinned here and
repeated on the Column Descriptions sheet.

* Denominator for Opt is ORACLE-SOLVABLE queries, not "queries CRAFT answered".
  A fixed denominator is the project's standing rule (CLAUDE.md, EVALUATION &
  SCORING): dividing by what a system happened to answer lets a model score well
  by refusing the hard ones. Queries the oracle proved infeasible, never
  attempted, or left unproven are excluded from the denominator entirely — they
  have no optimum to be measured against.

* CORRECT = feasible AND constraints_satisfied. Med/Max are computed over
  correct routes only, because an invalid route has no meaningful "gap": it is
  typically SHORTER (it skips the detour that makes it legal), so including it
  would flatter the system rather than penalise it.

* OBJECTIVE BASIS = `dijkstra_objective_value`, CRAFT's REALIZED route cost.
  NOT the shared `objective_value` column. Measured reason: on ev_routing
  `objective_value` collapses to the unconstrained SP(src,dst) — that template
  has no POI stops, and chargers are excluded from the objective chain by design
  (EV_INFRA) — so it ignores the charging detour and "beats" the proven optimum
  on 226 of 521 correct rows, which is impossible. `dijkstra_objective_value`
  beats it on 0 rows for every template, so it is the only basis valid across
  the whole table. The shared column is still exported per query for reference.

* POI COLLECTION is scored on COUNT, not distance: gold records its objective as
  "maximize_poi" and the scorer does not check distance-optimality there. So its
  APE is (optimal_count - collected) / optimal_count, and "optimal" means the
  collected count equals the proven maximum.

* A row is OPTIMAL when it is within ATOL of the optimum (0.5 m / 0.5 s, or an
  exact count match). The tolerance only absorbs float noise: on algo1 every
  optimal row matches to 0.00 m and the next-largest gap is 5.42 m, so nothing
  sits inside the tolerance band by accident.

Usage:
  python evaluation/optimal/export_optimality_table.py \
      --craft-xlsx <regenerated craft set1 xlsx>
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluate_llm import _write_table

ORACLE_DIR = ROOT / "evaluation" / "optimal" / "results"
PARSER_EVAL = ROOT / "results" / "craft_parser_eval.xlsx"
DEFAULT_OUT = ROOT / "results" / "optimality_gap.xlsx"
ATOL = 0.5  # metres / seconds; exact-count templates compare exactly

# template -> (display name, oracle file, optimum field, kind, paper row?)
TEMPLATES = [
    ("ev_routing",     "EV Routing",      "ev_routing_oracle.json",        "optimal_distance_m", "distance", True),
    ("ordered_poi",    "Ordered POI",     "ordered_poi_oracle.json", "optimal_distance_m", "distance", True),
    ("unordered_poi",        "Unordered POI",   "poi_dist_time_oracle.json",     "optimal_time_s",     "time",     True),
    ("poi_collection", "POI Collection",  "poi_collection_oracle.json", "count",           "count",    True),
    ("ev+ordered_poi",       "EV + Ordered POI","ev_ordered_poi_v2_oracle.json",    "optimal_distance_m", "distance", True),
]
MODELS = [
    ("gemini-3_1-flash-lite", "Gemini Flash Lite"),
    ("gpt-4_1",               "GPT-4.1"),
    ("openai_gpt-oss-20b",    "gpt-oss-20b"),
]


def load_optima(fname: str, key: str) -> Dict[str, float]:
    recs = json.loads((ORACLE_DIR / fname).read_text())
    return {r["query_id"]: r.get(key) for r in recs
            if r.get("status") in ("optimal", "solved") and r.get(key) is not None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--craft-xlsx", required=True)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    craft = pd.read_excel(args.craft_xlsx, sheet_name="Results", header=2)

    # Per-query parse verdict vs gold, from the existing parser evaluation.
    # Needed because a large APE can come from CRAFT SEARCHING BADLY or from it
    # PARSING the query wrongly and so solving a different problem — and the two
    # mean opposite things. Measured example: ev_routing_5_v07 shows a
    # 90.6% gap purely because CRAFT read only the current charge (4 mi) and
    # missed the 109 mi max range. By contrast unordered_poi_r079's 147% gap has
    # a perfectly correct parse and is a genuine search failure.
    parse_ok: Dict[tuple, bool] = {}
    if PARSER_EVAL.exists():
        pe = pd.read_excel(PARSER_EVAL, sheet_name="per_query", header=2)
        for _, r in pe.iterrows():
            parse_ok[(str(r["model"]), str(r["query_id"]))] = bool(r.get("all_correct") == True)
    else:
        print(f"WARNING: {PARSER_EVAL} missing — parse-matched columns will be blank")

    detail: List[list] = []
    per_query: List[list] = []
    cells: Dict[tuple, Dict[str, Any]] = {}

    for tmpl, disp, fname, key, kind, in_paper in TEMPLATES:
        optima = load_optima(fname, key)
        for model, mdisp in MODELS:
            d = craft[(craft.template == tmpl) & (craft.model == model)]
            n_solvable = len(optima)
            apes: List[float] = []
            apes_pm: List[float] = []          # parse-matched subset
            n_correct = n_optimal = 0
            n_correct_pm = n_optimal_pm = 0
            n_solvable_pm = sum(1 for q in optima if parse_ok.get((model, q), False))

            for _, r in d.iterrows():
                opt = optima.get(r.query_id)
                if opt is None:
                    continue  # not oracle-solvable -> outside the denominator
                correct = (r.get("feasible") == True) and (r.get("constraints_satisfied") == True)
                got = r.get("n_stops") if kind == "count" else r.get("dijkstra_objective_value")
                shared = r.get("objective_value")
                ape = None
                is_opt = False
                if correct and got is not None and not pd.isna(got) and float(opt) > 0:
                    got = float(got)
                    ape = abs(got - float(opt)) / float(opt) * 100.0
                    is_opt = (got == float(opt)) if kind == "count" else (got <= float(opt) + ATOL)
                    n_correct += 1
                    apes.append(ape)
                    n_optimal += int(is_opt)
                    if parse_ok.get((model, r.query_id), False):
                        n_correct_pm += 1
                        apes_pm.append(ape)
                        n_optimal_pm += int(is_opt)
                per_query.append([
                    tmpl, disp, model, mdisp, r.query_id, kind, opt,
                    (None if got is None or pd.isna(got) else float(got)),
                    (None if shared is None or pd.isna(shared) else float(shared)),
                    bool(correct), (None if ape is None else round(ape, 6)),
                    (None if ape is None else bool(is_opt)),
                    parse_ok.get((model, r.query_id), None),
                ])

            opt_pct = (n_optimal / n_solvable * 100.0) if n_solvable else None
            med = statistics.median(apes) if apes else None
            mx = max(apes) if apes else None
            mean = (sum(apes) / len(apes)) if apes else None
            opt_pct_pm = (n_optimal_pm / n_solvable_pm * 100.0) if n_solvable_pm else None
            cells[(tmpl, model)] = {
                "opt": opt_pct, "med": med, "max": mx, "mean": mean,
                "opt_pm": opt_pct_pm,
                "mean_pm": (sum(apes_pm) / len(apes_pm)) if apes_pm else None,
                "med_pm": statistics.median(apes_pm) if apes_pm else None,
                "max_pm": max(apes_pm) if apes_pm else None,
                "n_solvable_pm": n_solvable_pm, "n_correct_pm": n_correct_pm,
            }
            detail.append([
                disp, tmpl, mdisp, model, kind, n_solvable, n_correct, n_optimal,
                None if opt_pct is None else round(opt_pct, 1),
                None if med is None else round(med, 2),
                None if mx is None else round(mx, 2),
                None if not apes else round(sum(apes) / len(apes), 3),
                n_solvable - n_correct,
                n_solvable_pm, n_correct_pm, n_optimal_pm,
                None if opt_pct_pm is None else round(opt_pct_pm, 1),
                None if not apes_pm else round(statistics.median(apes_pm), 2),
                None if not apes_pm else round(max(apes_pm), 2),
            ])

    # ── Paper Table: exactly the LaTeX layout ──────────────────────────────
    header = ["Template"]
    for _, mdisp in MODELS:
        header += [f"{mdisp} Opt", f"{mdisp} Mean", f"{mdisp} Med", f"{mdisp} Max"]

    def _row(tmpl, disp):
        row = [disp]
        for model, _ in MODELS:
            c = cells[(tmpl, model)]
            row += [None if c["opt"] is None else round(c["opt"], 1),
                    None if c["mean"] is None else round(c["mean"], 2),
                    None if c["med"] is None else round(c["med"], 2),
                    None if c["max"] is None else round(c["max"], 2)]
        return row

    paper_rows = [_row(tmpl, disp) for tmpl, disp, _, _, _, in_paper in TEMPLATES if in_paper]
    # supplementary rows, kept out of the paper table but not lost
    for tmpl, disp, _, _, _, in_paper in TEMPLATES:
        if not in_paper:
            paper_rows.append(_row(tmpl, f"{disp} (supplementary, not in paper table)"))

    import openpyxl
    wb = openpyxl.Workbook(); wb.remove(wb.active)
    _write_table(wb.create_sheet("Paper Table"), header, paper_rows,
                 title="Optimality gap against the exact oracle (CRAFT). "
                       "Opt = % of oracle-solvable queries solved optimally; "
                       "Mean/Med/Max = mean/median/max APE over correct routes "
                       "(Med is usually 0 since most correct routes are exactly optimal — "
                       "Mean is the number that reflects the error tail). All values in %.")
    pm_rows = []
    for tmpl, disp, _, _, _, in_paper in TEMPLATES:
        row = [disp if in_paper else f"{disp} (supplementary)"]
        for model, _ in MODELS:
            c = cells[(tmpl, model)]
            row += [None if c["opt_pm"] is None else round(c["opt_pm"], 1),
                    None if c["mean_pm"] is None else round(c["mean_pm"], 2),
                    None if c["med_pm"] is None else round(c["med_pm"], 2),
                    None if c["max_pm"] is None else round(c["max_pm"], 2)]
        pm_rows.append(row)
    _write_table(wb.create_sheet("Paper Table (parse-matched)"), header, pm_rows,
                 title="Same metrics restricted to queries CRAFT PARSED CORRECTLY "
                       "(craft_parser_eval all_correct). Isolates SEARCH quality: a large gap "
                       "here cannot be blamed on misreading the query. Denominator is "
                       "oracle-solvable AND parse-correct, so Opt is not comparable to the "
                       "main sheet's Opt.")
    _write_table(wb.create_sheet("Detail"),
                 ["Template", "template_id", "Model", "model_id", "objective_kind",
                  "n_oracle_solvable", "n_correct", "n_optimal", "Opt_pct",
                  "Median_APE_pct", "Max_APE_pct", "Mean_APE_pct", "n_not_correct",
                  "n_solvable_parsematched", "n_correct_parsematched",
                  "n_optimal_parsematched", "Opt_pct_parsematched",
                  "Median_APE_pct_parsematched", "Max_APE_pct_parsematched"],
                 detail, title="Per template x model detail (denominators made explicit)")
    _write_table(wb.create_sheet("Per-Query"),
                 ["template_id", "Template", "model_id", "Model", "query_id",
                  "objective_kind", "oracle_optimum", "craft_realized",
                  "craft_shared_objective_value", "is_correct", "APE_pct", "is_optimal",
                  "parse_matches_gold"],
                 per_query, title="Every oracle-solvable query x model")

    legend = [
        ["Opt", "n_optimal / n_oracle_solvable x100. FIXED denominator: every query the oracle "
                "solved, whether or not CRAFT answered it. Queries proved infeasible, not attempted, "
                "or left unproven are excluded — they have no optimum."],
        ["Med / Max", "Median / maximum absolute percentage error over CORRECT routes only "
                      "(feasible AND constraints_satisfied). Invalid routes are excluded: they are "
                      "usually SHORTER because they skip the detour that makes them legal, so "
                      "including them would flatter the system."],
        ["objective basis", "dijkstra_objective_value = CRAFT's REALIZED route cost. The shared "
                            "objective_value column is NOT used: on algo1 it collapses to the "
                            "unconstrained SP(src,dst) and 'beats' the proven optimum on 226/521 "
                            "correct rows. Realized beats it on 0 rows for every template."],
        ["POI Collection", "Scored on COUNT (gold objective = maximize_poi; the scorer does not "
                           "check distance there). APE = (optimum - collected)/optimum; optimal "
                           "means the count equals the proven maximum."],
        ["tolerance", f"A row counts as optimal within {ATOL} m / {ATOL} s (exact match for counts). "
                      "This only absorbs float noise: optimal rows match to 0.00 and the next "
                      "smallest gap observed is 5.42 m."],
        ["n_not_correct", "Oracle-solvable queries where CRAFT produced no route or an invalid one. "
                          "These count against Opt (they are in the denominator) but are excluded "
                          "from Med/Max."],
        ["*_parsematched / Paper Table (parse-matched)",
         "Same metrics over queries CRAFT parsed correctly (craft_parser_eval per_query "
         "all_correct). A large APE has two very different causes: bad SEARCH, or a wrong PARSE "
         "that made CRAFT solve a different problem. Measured: ev_routing_5_v07's 90.6% gap "
         "is purely a parse failure (CRAFT read the 4 mi current charge and missed the 109 mi max "
         "range), whereas unordered_poi_r079's 147% gap has a fully correct parse and is genuine. "
         "The parse-matched denominator is smaller, so its Opt is NOT comparable to the main Opt."],
    ]
    _write_table(wb.create_sheet("Column Descriptions"), ["Column", "Description"], legend,
                 title="Column Descriptions")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f"wrote {out}\n")

    # console: the LaTeX body, ready to paste
    print("LaTeX table body:")
    for tmpl, disp, _, _, _, in_paper in TEMPLATES:
        if not in_paper:
            continue
        vals = []
        for model, _ in MODELS:
            c = cells[(tmpl, model)]
            vals += [f"{c['opt']:.1f}" if c["opt"] is not None else "--",
                     f"{c['med']:.2f}" if c["med"] is not None else "--",
                     f"{c['max']:.2f}" if c["max"] is not None else "--"]
        print(f"{disp:<15}& " + " & ".join(vals) + r" \\")
    print()
    print(f"{'template':<16}{'model':<20}{'solvable':>9}{'correct':>8}{'optimal':>8}{'Opt%':>7}{'Med%':>8}{'Max%':>9}")
    for row in detail:
        print(f"{row[0]:<16}{row[2]:<20}{row[5]:>9}{row[6]:>8}{row[7]:>8}"
              f"{str(row[8]):>7}{str(row[9]):>8}{str(row[10]):>9}")


if __name__ == "__main__":
    main()
