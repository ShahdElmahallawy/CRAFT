#!/usr/bin/env python3
"""
evaluate_description_filter.py — Phase 3: score CRAFT's DESCRIPTION filter (set3).

Phase 1 covered category resolution; Phase 2 covered the structured filter cascade.
Both had a deterministic ground truth. The description filter does not — "is this POI
an art museum?" is a semantic judgement — so its reference set is LLM-labelled by
evaluation/description_gt/build_description_labels.py (two passes, disagreements kept
as `uncertain` and excluded from scoring rather than silently resolved).

TWO SEPARATE QUESTIONS, measured separately:

  A. EMBEDDING QUALITY (isolated). How well does cosine similarity separate labelled
     true from false, over the CATEGORY rows only? Reported as a threshold sweep.
     This originally existed to settle a threshold disagreement (filtration 0.75 vs
     constraint_checker 0.65); that is RESOLVED — both now use 0.50 — so the sweep's
     remaining job is to show the precision/recall curve the adopted floor sits on.

  B. END-TO-END (faithful replay). What node set does the real cascade actually return?
     This deliberately reproduces filter_nodes_by_description's behaviour, including
     the part that leaks: it scores every POI at a candidate node, not just POIs of the
     requested category, and takes the max similarity per node. At museum nodes 1110 of
     1263 description-bearing rows are NOT museums, so a neighbouring business's
     description can push a node over the threshold. Measured as `leakage_fp`.

Also reported: the broaden-on-empty fallback, detected by set identity (sys == base).
Historically filter_nodes_by_description RETURNED the unfiltered set when nothing
cleared the threshold and filtration treated that as success silently; it now returns
EMPTY and filtration logs the fallback, but the detection is kept as a regression guard.

Query-vector shape: filtration embeds the BARE phrase; constraint_checker keys its
cache "{poi_type}: {phrase}". Both are scored, since the discrepancy is unresolved.

⚠️ SCOPE: this script models the EMBEDDING STAGE only. Since 2026-08-22 production
runs a second LLM verification pass after it, so the numbers here are stage-1
figures, NOT what the live filter returns. End-to-end cascade numbers live in
results/desc.xlsx and results/craft_description_filter_llm_rerank.xlsx.

Usage:
  python evaluation/evaluate_description_filter.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "craft" / "generator" / "tools"))

GT_DIR = ROOT / "outputs" / "evaluation" / "description_gt"
LABELS = GT_DIR / "description_labels.json"
EMB_REPLAY = GT_DIR / "poi_embeddings_replay.json"
QUERY_EMB = ROOT / "data" / "socal" / "set3_query_embeddings.json"
BARE_EMB = GT_DIR / "bare_query_embeddings.json"
DEFAULT_OUT = ROOT / "results" / "craft_description_filter.xlsx"
DEFAULT_SUMMARY = ROOT / "results" / "craft_description_filter_summary.json"

# 2026-08-22: filtration and constraint_checker now BOTH use 0.50 — the
# threshold disagreement this script was written to settle (0.75 vs 0.65) is
# resolved, so the two reported points are now "the adopted floor" and "the
# historical cutoff", kept because the 0.75 row is the evidence that the old
# rule admitted nothing (max attainable similarity is 0.748).
# ⚠️ These score the EMBEDDING STAGE ONLY. Production now runs a second
# LLM verification pass on top (poi_embeddings.filter_nodes_by_description),
# which this script does not model — see
# results/craft_description_filter_llm_rerank.xlsx and results/desc.xlsx for
# the end-to-end cascade numbers.
FILTRATION_THRESHOLD = 0.50      # craft/generator/tools/filtration.py sim_threshold
CHECKER_THRESHOLD = 0.75         # historical: the pre-2026-08-17 absolute cutoff
SWEEP = [round(0.40 + 0.025 * i, 3) for i in range(23)]   # 0.400 .. 0.950


def cosine(a, b) -> float:
    import numpy as np
    va, vb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = float(np.linalg.norm(va) * np.linalg.norm(vb))
    return float(np.dot(va, vb) / d) if d else 0.0


def load_pois():
    import pyogrio
    from common_handlers import CATEGORY_COLS
    return pyogrio.read_dataframe(
        ROOT / "data" / "socal" / "road_network.gpkg", layer="pois",
        columns=CATEGORY_COLS + ["description", "name", "nearest_node_osmid"],
        read_geometry=False)


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


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def ensure_bare_embeddings(phrases: List[str]) -> Dict[str, List[float]]:
    """filtration embeds the bare phrase; the shipped cache is keyed '{type}: {phrase}'.
    Embed the bare forms once and persist them separately — never write to
    data/socal/set3_query_embeddings.json, which evaluate_craft_set3.py depends on."""
    cache: Dict[str, List[float]] = {}
    if BARE_EMB.exists():
        cache = json.loads(BARE_EMB.read_text())
    missing = [p for p in phrases if p not in cache]
    if missing:
        from poi_embeddings import embed_text
        key = base = None
        for line in (ROOT / ".env").read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
            if line.startswith("GEMINI_BASE_URL="):
                base = line.split("=", 1)[1].strip().strip("'\"")
        for ph in missing:
            cache[ph] = embed_text(ph, key, base_url=base)
            print(f"[embed] bare phrase {ph!r}")
        BARE_EMB.parent.mkdir(parents=True, exist_ok=True)
        BARE_EMB.write_text(json.dumps(cache))
    return cache


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--summary-out", default=str(DEFAULT_SUMMARY))
    args = ap.parse_args()

    labels_blob = json.loads(LABELS.read_text())
    rows_lab: Dict[str, Dict[str, Any]] = labels_blob["rows"]
    phrase_meta: Dict[str, Any] = labels_blob["phrases"]
    phrases = list(phrase_meta)

    emb = {int(k): v for k, v in json.loads(EMB_REPLAY.read_text()).items()}
    qcache = json.loads(QUERY_EMB.read_text())
    bare = ensure_bare_embeddings(phrases)
    print(f"[load] {len(emb)} POI vectors, {len(rows_lab)} labelled rows")

    p = load_pois()
    node_of = p["nearest_node_osmid"]

    sweep_rows: List[list] = []
    head_rows: List[list] = []
    leak_rows: List[list] = []
    per_phrase: Dict[str, Any] = {}

    for phrase in phrases:
        cat = phrase_meta[phrase]["category"]
        cmask = cat_mask(p, cat)
        cat_rows = set(int(i) for i in p.index[cmask])
        cat_nodes = set(node_of[cmask].dropna())

        # labelled reference, row level, uncertain excluded
        gt_true_rows, gt_false_rows, uncertain = set(), set(), 0
        for sid, per in rows_lab.items():
            if phrase not in per:
                continue
            rec = per[phrase]
            if rec.get("uncertain"):
                uncertain += 1
                continue
            (gt_true_rows if rec["label"] else gt_false_rows).add(int(sid))
        scored_rows = gt_true_rows | gt_false_rows

        for qlabel, qvec in (("bare (what filtration uses)", bare.get(phrase)),
                             ("prefixed (what the checker caches)",
                              qcache.get(f"{cat}: {phrase}"))):
            if qvec is None:
                continue
            sims = {i: cosine(qvec, emb[i]) for i in scored_rows if i in emb}

            # A. embedding quality, isolated (category rows only)
            for th in SWEEP:
                pred = {i for i, s in sims.items() if s >= th}
                tp = len(pred & gt_true_rows)
                fp = len(pred & gt_false_rows)
                fn = len(gt_true_rows) - tp
                pr, rc, f1 = prf(tp, fp, fn)
                sweep_rows.append([phrase, qlabel, th, tp, fp, fn, pr, rc, f1])

            for th, tag in ((FILTRATION_THRESHOLD, "adopted floor 0.50"),
                            (CHECKER_THRESHOLD, "historical absolute 0.75")):
                pred = {i for i, s in sims.items() if s >= th}
                tp = len(pred & gt_true_rows); fp = len(pred & gt_false_rows)
                fn = len(gt_true_rows) - tp
                pr, rc, f1 = prf(tp, fp, fn)
                head_rows.append([phrase, qlabel, tag, th, len(gt_true_rows),
                                  tp, fp, fn, pr, rc, f1])

            best = max(SWEEP, key=lambda t: prf(
                len({i for i, s in sims.items() if s >= t} & gt_true_rows),
                len({i for i, s in sims.items() if s >= t} & gt_false_rows),
                len(gt_true_rows) - len({i for i, s in sims.items() if s >= t} & gt_true_rows))[2])
            per_phrase.setdefault(phrase, {})[qlabel] = {
                "n_scored_rows": len(scored_rows),
                "n_true": len(gt_true_rows), "uncertain_excluded": uncertain,
                "best_threshold_by_f1": best,
            }

            # B. end-to-end faithful replay (all POIs at candidate nodes)
            if qlabel.startswith("bare"):
                gt_nodes = {node_of[i] for i in gt_true_rows} - {None}
                base_nodes = set(cat_nodes)
                for th, tag in ((FILTRATION_THRESHOLD, "adopted floor 0.50"),
                                (CHECKER_THRESHOLD, "historical absolute 0.75")):
                    node_best: Dict[Any, float] = {}
                    node_src: Dict[Any, bool] = {}      # True if a CATEGORY row set it
                    for i, vec in emb.items():
                        nd = node_of.get(i)
                        if nd is None or nd not in base_nodes:
                            continue
                        s = cosine(qvec, vec)
                        if nd not in node_best or s > node_best[nd]:
                            node_best[nd] = s
                            node_src[nd] = i in cat_rows
                    sysn = {n for n, s in node_best.items() if s >= th}
                    broadened = (not sysn) and bool(base_nodes)
                    effective = base_nodes if broadened else sysn   # the fallback
                    tp = len(effective & gt_nodes)
                    fp = len(effective - gt_nodes)
                    fn = len(gt_nodes - effective)
                    pr, rc, f1 = prf(tp, fp, fn)
                    cross = sum(1 for n in sysn if not node_src.get(n, True))
                    leak_rows.append([phrase, tag, th, len(base_nodes), len(gt_nodes),
                                      len(sysn), broadened, tp, fp, fn, pr, rc, f1, cross])

    # write
    from openpyxl import Workbook
    from evaluation.evaluate_llm import _write_table
    wb = Workbook(); wb.remove(wb.active)
    _write_table(wb.create_sheet("headline"),
                 ["phrase", "query_vector", "threshold_source", "threshold", "n_true",
                  "TP", "FP", "FN", "precision", "recall", "f1"], head_rows,
                 "Embedding filter at the two thresholds in the codebase (row level)")
    _write_table(wb.create_sheet("threshold_sweep"),
                 ["phrase", "query_vector", "threshold", "TP", "FP", "FN",
                  "precision", "recall", "f1"], sweep_rows,
                 "Threshold sweep — isolated embedding quality")
    _write_table(wb.create_sheet("end_to_end"),
                 ["phrase", "threshold_source", "threshold", "n_base_nodes",
                  "n_gt_nodes", "n_sys_nodes", "broadened", "TP", "FP", "FN",
                  "precision", "recall", "f1", "nodes_set_by_NON_category_poi"], leak_rows,
                 "Faithful replay of filter_nodes_by_description (node level)")
    _write_table(wb.create_sheet("legend"), ["term", "meaning"], [
        ["ground truth", "LLM-labelled, 2 passes, judged from the DESCRIPTION TEXT ONLY "
                         "— the same input the embedding filter sees"],
        ["uncertain", "the two passes disagreed; excluded from scoring, not resolved"],
        ["query_vector bare", "filtration.py embeds the bare phrase — what actually ran"],
        ["query_vector prefixed", "constraint_checker caches '{type}: {phrase}' while "
                                  "claiming to match filtration; it does not"],
        ["broadened", "nothing cleared the threshold, so filter_nodes_by_description "
                      "returns the UNFILTERED set and filtration.py treats it as success "
                      "— no warning is printed by either"],
        ["nodes_set_by_NON_category_poi", "nodes admitted because a POI of a DIFFERENT "
                                          "category at that node scored highest — the "
                                          "node-level leakage"],
    ], "Definitions")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    Path(args.summary_out).write_text(json.dumps(
        {"per_phrase": per_phrase, "headline": head_rows, "end_to_end": leak_rows},
        indent=2, default=str))

    print(f"\n[done] {out}")
    print(f"\n{'phrase':26s} {'qvec':10s} {'thr':>5s} {'P':>6s} {'R':>6s} {'F1':>6s}")
    for r in head_rows:
        print(f"{r[0]:26s} {r[1][:9]:10s} {r[3]:5} {r[8]:6} {r[9]:6} {r[10]:6}")
    print(f"\n{'phrase':26s} {'thr':>5s} {'base':>5s} {'gt':>4s} {'sys':>4s} "
          f"{'P':>6s} {'R':>6s} {'leak':>5s} broadened")
    for r in leak_rows:
        print(f"{r[0]:26s} {r[2]:5} {r[3]:5} {r[4]:4} {r[5]:4} {r[10]:6} {r[11]:6} "
              f"{r[13]:5} {r[6]}")


if __name__ == "__main__":
    main()
