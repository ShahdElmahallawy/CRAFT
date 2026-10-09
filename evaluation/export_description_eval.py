#!/usr/bin/env python3
"""
export_description_eval.py — results/desc.xlsx

Compares the candidate set CRAFT's description filter builds (possible_targets)
against the LLM-labelled ground truth, for set3's 3 description phrases, under the
OLD rule (absolute 0.75) and the NEW rule (floor 0.55 + within 0.20 of best).

METRICS — all by comparing NODE IDs against the ground-truth node IDs:
  pct_correct  of the node ids the filter returned, the % that are genuinely correct
  pct_found    of the ground-truth node ids, the % the filter actually returned
  exact_match  the two id sets are identical
  Jaccard      |intersection| / |union| — single size-independent summary
(An earlier version reported MAPE on set SIZE; it was dropped because it compares
counts only — a set can be exactly the right size and hold entirely the wrong ids,
which "a theme park attraction" demonstrated at 9.9% MAPE with 0.19 precision.)

Node level is the primary view: possible_targets is a NODE set, so that is the
apples-to-apples comparison with what the search consumes. Row level is included
because the ground truth is labelled per POI row, and the gap between the two views
is the node-level leakage (several POIs share one nearest node).

Usage:
  conda run -n myenv python evaluation/export_description_eval.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "craft" / "generator" / "tools"))

GT_DIR = ROOT / "outputs" / "evaluation" / "description_gt"
OUT = ROOT / "results" / "desc.xlsx"

OLD_THRESHOLD = 0.75
NEW_FLOOR, NEW_MARGIN = 0.55, 0.20
# 2026-08-22: the rule the project settled on — a plain absolute floor, no
# relative band, no contrastive subtraction. Measured (see
# results/craft_description_filter_llm_rerank.xlsx): at 0.50 the embedding
# keeps recall = 1.00 on all 3 phrases, and a single local-LLM verification
# pass then lifts precision from 0.20-0.50 to 0.90-1.00. Because that second
# stage can only PRUNE, stage-1 recall caps the cascade — which is exactly why
# the lenient floor beats the cleverer rules that discard true matches early.
PLAIN_FLOOR = 0.50
VERDICTS = GT_DIR / "llm_rerank_verdicts.json"
# Stage-2 timing is REUSED from the rerank run, never re-measured here: those are
# real local-LLM calls and re-timing them would cost minutes per export for a
# number that is already recorded. Stage-1 timing IS measured live below, because
# it is milliseconds.
RERANK_SUMMARY = ROOT / "results" / "craft_description_filter_llm_rerank_summary.json"


def cos(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d else 0.0


def cat_mask(p, cat: str):
    from common_handlers import CATEGORY_COLS
    m = None
    for c in CATEGORY_COLS:
        n = (p[c].astype(str).str.lower()
             .str.replace("|", ";", regex=False).str.replace(",", ";", regex=False))
        h = n.str.contains(r"(?:^|;)\s*" + re.escape(cat) + r"\s*(?:;|$)",
                           regex=True, na=False)
        m = h if m is None else (m | h)
    return m


QUERY_EMBED_S = 0.0   # set by measure_query_embed_s() at run time


def measure_query_embed_s(probe: str = "an art museum") -> float:
    """Wall time of ONE query-embedding API call — the per-query stage-1 cost that
    the cached vectors hide. Returns 0.0 (and says so) if the endpoint or key is
    unavailable, so the export still works offline; the timing sheet then reports
    scoring cost only, which is the honest thing to show rather than a guess."""
    try:
        from poi_embeddings import embed_text
        key = base = None
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
            if line.startswith("GEMINI_BASE_URL="):
                base = line.split("=", 1)[1].strip().strip("'\"")
        if not key:
            raise RuntimeError("no GEMINI_API_KEY in .env")
        t0 = time.perf_counter()
        embed_text(probe, key, base_url=base)
        dt = time.perf_counter() - t0
        print(f"[timing] query-embed API call: {dt:.3f}s")
        return dt
    except Exception as e:
        print(f"[timing] query-embed NOT measured ({e}) — reporting scoring cost only")
        return 0.0


def metrics(sys_set: Set, gt_set: Set) -> Dict[str, Any]:
    tp = len(sys_set & gt_set); fp = len(sys_set - gt_set); fn = len(gt_set - sys_set)
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    union = len(sys_set | gt_set)
    return {
        "n_sys": len(sys_set), "n_gt": len(gt_set),
        "correct_ids": tp, "wrong_ids": fp, "missed_ids": fn,
        "pct_correct": round(100 * P, 1), "pct_found": round(100 * R, 1),
        "exact_match": sys_set == gt_set,
        "f1": round(F, 4), "jaccard": round(tp / union, 4) if union else None,
    }


def main() -> None:
    import pyogrio
    from common_handlers import CATEGORY_COLS

    p = pyogrio.read_dataframe(
        ROOT / "data" / "socal" / "road_network.gpkg", layer="pois",
        columns=CATEGORY_COLS + ["description", "nearest_node_osmid"],
        read_geometry=False)
    bad = {"nan", "none", "", "–", "-", "n/a"}
    hasd = p["description"].notna() & ~p["description"].astype(str).str.strip().str.lower().isin(bad)
    node_of = p["nearest_node_osmid"]

    emb = {int(k): v for k, v in json.loads(
        (GT_DIR / "poi_embeddings_replay.json").read_text()).items()}
    bare = json.loads((GT_DIR / "bare_query_embeddings.json").read_text())
    lab = json.loads((GT_DIR / "description_labels.json").read_text())
    rows_lab, meta = lab["rows"], lab["phrases"]

    verdicts: Dict[str, Dict[str, bool]] = {}
    if VERDICTS.exists():
        verdicts = json.loads(VERDICTS.read_text())
    else:
        print(f"[warn] {VERDICTS} missing — the '+ LLM rerank' rule will score as "
              f"empty. Run evaluation/description_gt/evaluate_llm_rerank.py --arms plain")

    # stage-2 wall time, per phrase, from the plain arm of the rerank run
    llm_timing: Dict[str, Dict[str, Any]] = {}
    if RERANK_SUMMARY.exists():
        for r in json.loads(RERANK_SUMMARY.read_text()).get("rows", []):
            if r.get("arm") == "plain":
                llm_timing[r["phrase"]] = r
    else:
        print(f"[warn] {RERANK_SUMMARY} missing — stage-2 timing will be blank")

    timing_rows: List[list] = []

    # One real embedding call, timed. This is the only stage-1 cost that is NOT
    # already covered by the cached vectors, and it is per-QUERY (not per
    # candidate), so a single measurement applies to every phrase.
    global QUERY_EMBED_S
    QUERY_EMBED_S = measure_query_embed_s()

    gold = json.loads((ROOT / "benchmark" / "ground_truth"
                       / "set3_ordered+desc_gold.json").read_text())
    stops_by_phrase: Dict[str, List[str]] = {}
    for r in gold:
        for s in (r["_provenance"].get("stops") or []):
            if s.get("filter_kind") == "description":
                ph = s.get("nl_phrase") or s.get("description")
                stops_by_phrase.setdefault(ph, []).append(r["query_id"])

    summary: List[list] = []
    stop_rows: List[list] = []

    for ph, info in meta.items():
        cat = info["category"]
        cmask = cat_mask(p, cat)
        # Scoring population = rows with a DEFINITE label for this phrase. Rows the
        # two labelling passes disagreed on (`uncertain`) are excluded, per the
        # ground truth's own convention — counting them would score the filter
        # against a label the ground truth itself declines to assert. This also
        # keeps the population identical to evaluate_llm_rerank.py's, so the
        # cached LLM verdicts cover every candidate.
        def _definite(i: int) -> bool:
            rec = rows_lab.get(str(i), {}).get(ph)
            return isinstance(rec, dict) and not rec.get("uncertain")

        cand = [int(i) for i in p.index[cmask & hasd]
                if int(i) in emb and _definite(int(i))]
        q = bare[ph]
        # Stage-1 cost, measured. Two parts, reported separately because they scale
        # differently: the query embedding is ONE network call regardless of dataset
        # size, while scoring grows with the candidate count.
        t0 = time.perf_counter()
        sims = {i: cos(q, emb[i]) for i in cand}
        score_s = time.perf_counter() - t0
        best = max(sims.values()) if sims else 0.0

        T_rows = {i for i in cand
                  if rows_lab.get(str(i), {}).get(ph, {}).get("label") is True}
        base_nodes = set(node_of[cmask].dropna())
        T_nodes = {node_of[i] for i in T_rows} - {None}

        plain_rows = {i for i, s in sims.items() if s >= PLAIN_FLOOR}
        # stage 2: keep only the plain-floor candidates the LLM also judged true.
        # Rows with no cached verdict were never asked, so they are DROPPED rather
        # than assumed true — and counted, so a stale/partial cache is visible
        # instead of silently deflating recall.
        ph_verdicts = verdicts.get(ph, {})
        unjudged = {i for i in plain_rows if str(i) not in ph_verdicts}
        plain_llm_rows = {i for i in plain_rows if ph_verdicts.get(str(i)) is True}

        rules = {
            f"OLD absolute {OLD_THRESHOLD}": {i for i, s in sims.items() if s >= OLD_THRESHOLD},
            f"NEW floor {NEW_FLOOR} + band {NEW_MARGIN}":
                {i for i, s in sims.items() if s >= max(NEW_FLOOR, best - NEW_MARGIN)},
            f"PLAIN floor {PLAIN_FLOOR}": plain_rows,
            f"PLAIN floor {PLAIN_FLOOR} + LLM rerank": plain_llm_rows,
        }
        if unjudged:
            print(f"[warn] {ph}: {len(unjudged)}/{len(plain_rows)} plain-floor candidates "
                  f"have no cached LLM verdict — re-run evaluation/description_gt/"
                  f"evaluate_llm_rerank.py --arms plain")
        for rule, pred_rows in rules.items():
            broadened = rule.startswith("OLD") and not pred_rows
            eff_rows = set(cand) if broadened else pred_rows
            eff_nodes = base_nodes if broadened else ({node_of[i] for i in pred_rows} - {None})

            for level, m, extra in (("row", metrics(eff_rows, T_rows), len(cand)),
                                    ("node", metrics(eff_nodes, T_nodes), len(base_nodes))):
                summary.append([
                    ph, cat, rule, level, extra, m["n_gt"], m["n_sys"],
                    m["correct_ids"], m["wrong_ids"], m["missed_ids"],
                    m["pct_correct"], m["pct_found"], m["exact_match"],
                    m["f1"], m["jaccard"],
                    "YES — filter had no effect" if broadened else "",
                ])

        # per-stop view: the candidate set depends on (category, phrase) only, so
        # every stop asking the same phrase inherits the same numbers. Listed so the
        # affected queries are explicit rather than implied.
        newm = metrics({node_of[i] for i in rules[f"PLAIN floor {PLAIN_FLOOR} + LLM rerank"]} - {None},
                       T_nodes)
        for qid in sorted(stops_by_phrase.get(ph, [])):
            stop_rows.append([qid, ph, cat, len(base_nodes), newm["n_gt"], newm["n_sys"],
                              newm["correct_ids"], newm["pct_correct"],
                              newm["pct_found"], newm["jaccard"]])

        # timing
        lt = llm_timing.get(ph, {})
        llm_s = lt.get("llm_wall_s")
        stage1_s = score_s + QUERY_EMBED_S
        timing_rows.append([
            ph, cat, len(cand),
            round(QUERY_EMBED_S, 3), round(score_s, 4), round(stage1_s, 3),
            lt.get("rerank_candidates"), llm_s, lt.get("llm_s_per_item"),
            round(stage1_s + llm_s, 1) if llm_s is not None else None,
            round(llm_s / stage1_s, 1) if (llm_s is not None and stage1_s) else None,
        ])

    from openpyxl import Workbook
    from evaluation.evaluate_llm import _write_table

    wb = Workbook(); wb.remove(wb.active)
    _write_table(wb.create_sheet("summary"),
                 ["phrase", "category", "rule", "level", "n_candidates", "n_ground_truth",
                  "n_filtration", "correct_ids", "wrong_ids", "missed_ids",
                  "%_correct", "%_found", "exact_match", "f1", "jaccard", "broadened"], summary,
                 "Filtration candidate set vs LLM-labelled ground truth")
    _write_table(wb.create_sheet("per_stop"),
                 ["query_id", "phrase", "category", "n_candidates", "n_ground_truth",
                  "n_filtration", "correct_ids", "%_correct", "%_found", "jaccard"],
                 stop_rows, f"The 29 set3 stops that carry a description filter "
                            f"(PLAIN floor {PLAIN_FLOOR} + LLM rerank — the adopted rule)")
    _write_table(wb.create_sheet("timing"),
                 ["phrase", "category", "n_candidates_scored",
                  "stage1_query_embed_s", "stage1_scoring_s", "stage1_total_s",
                  "stage2_llm_candidates", "stage2_llm_s", "stage2_s_per_item",
                  "cascade_total_s", "stage2_vs_stage1_x"], timing_rows,
                 "Cost of each stage. NOTE: OLD 0.75 / NEW floor+band / PLAIN 0.50 all "
                 "cost exactly stage-1 — they differ only in which float comparison is "
                 "applied, so there is no per-rule time difference among them. The only "
                 "real time question is stage 1 alone vs. stage 1 + LLM rerank.")
    _write_table(wb.create_sheet("legend"), ["term", "meaning"], [
        [f"rule: PLAIN floor {PLAIN_FLOOR}", "THE ADOPTED RULE (stage 1). Plain absolute "
                                             "cosine floor — no relative band, no "
                                             "contrastive subtraction. Chosen because it "
                                             "keeps recall = 1.00 on all 3 phrases."],
        [f"rule: PLAIN floor {PLAIN_FLOOR} + LLM rerank",
         "THE ADOPTED CASCADE. Stage 1 above, then ONE local-LLM pass "
         "(openai/gpt-oss-20b) over just those candidates, shown the target phrase "
         "and each POI's description text ONLY — the same input the embedding had. "
         "Lifts precision from 0.20-0.50 to 0.90-1.00 for ~140s across all 3 phrases. "
         "Stage 2 can only PRUNE, so stage-1 recall caps the cascade."],
        ["rule: OLD / NEW", "superseded, kept so the progression is auditable. OLD 0.75 "
                            "admitted nothing (max attainable similarity is 0.748); NEW "
                            "floor+band and the contrastive variant both discard true "
                            "matches at stage 1 that no later stage can recover."],
        ["%_correct", "of the node ids filtration returned, the % that are in the "
                      "ground-truth set (= precision)"],
        ["%_found", "of the ground-truth node ids, the % filtration returned (= recall)"],
        ["exact_match", "the two node-id sets are identical"],
        ["correct/wrong/missed_ids", "node ids in both / returned but not true / true "
                                     "but not returned"],
        ["jaccard", "|intersection| / |union| — best single size-independent summary"],
        ["level = node", "PRIMARY. possible_targets is a node set, so this is what the "
                         "search actually consumes"],
        ["level = row", "ground truth is labelled per POI row; several POIs share a "
                        "nearest node, so the row/node gap is the node-level leakage"],
        ["broadened", "nothing cleared the cutoff, so the filter returned the FULL "
                      "candidate set — the description constraint was not applied"],
        ["ground truth", "LLM-labelled from the description text only (the same input the "
                         "filter sees); 2 passes, disagreements excluded as uncertain"],
        ["coverage limit", "48% of museums and 51% of attractions have NO description, so "
                           "they are unlabelled and unreachable by any description filter"],
        ["timing: stage1_query_embed_s", "ONE embedding API call for the query phrase, "
                                         "measured live. Per QUERY, not per candidate — "
                                         "it does not grow with dataset size."],
        ["timing: stage1_scoring_s", "cosine of the query vector against every candidate's "
                                     "CACHED description vector. In-memory, so milliseconds."],
        ["timing: POI description embeddings", "NOT charged here — they are precomputed "
                                               "offline once into poi_embeddings_*.json and "
                                               "reused by every query. A one-time build cost, "
                                               "not a per-query one."],
        ["timing: stage2_llm_s", "REUSED from results/craft_description_filter_llm_rerank_"
                                 "summary.json (plain arm), not re-measured on each export. "
                                 "Local openai/gpt-oss-20b, batches of 25."],
        ["timing caveat", "stage-2 numbers are local-LLM wall time on this machine and vary "
                          "run to run (batch spikes of 3-20x observed); treat them as an "
                          "order of magnitude, not a constant."],
    ], "Definitions and caveats")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)

    print(f"[done] {OUT}")
    print(f"{'phrase':24s} {'rule':28s} {'lvl':5s} {'gt':>4s} {'sys':>4s} "
          f"{'ok':>4s} {'%corr':>6s} {'%found':>7s} {'Jac':>5s}")
    for r in summary:
        print(f"{r[0]:24s} {r[2]:28s} {r[3]:5s} {r[5]:4d} {r[6]:4d} {r[7]:4d} "
              f"{r[10]:6.1f} {r[11]:7.1f} {str(r[15]):>5s}")


if __name__ == "__main__":
    main()
