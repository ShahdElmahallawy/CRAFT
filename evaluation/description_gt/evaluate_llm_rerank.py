#!/usr/bin/env python3
"""
evaluate_llm_rerank.py — does an LLM verification pass AFTER the embedding
filter fix set3's low description-filter precision, and what does it cost?

Two-stage cascade under test
  stage 1 — embedding narrows the category rows to a candidate set, using the
            same selection rule production uses (craft/generator/tools/
            poi_embeddings.py::filter_nodes_by_description). The scoring math
            is imported from that module (cosine_sim) and the neutral phrase
            from filtration.py (_neutral_phrase), so the arms below cannot
            drift from production behaviour.
  stage 2 — ONE local-LLM pass over just that candidate set, prunes false
            positives. Reuses the ground-truth judge's own prompt
            (build_description_labels.py::label_batch / INSTRUCTION) rather
            than inventing a new one — this is the same judgement task, just
            run once with a cheap local model instead of twice with the
            ground-truth model.

ARMS (production has three selection rules in one function; contrastive
overrides the others whenever contrast_text is passed, so these are the two
real production configurations, not a knob sweep):
  plain@T        cosine(query, poi) >= T                    [sim_contrastive=False]
  contrastive@C  cosine(query, poi) - cosine(neutral, poi) >= C
                                                            [sim_contrastive=True]
The point of running both: if plain thresholding + an LLM pass matches what
contrastive + an LLM pass achieves, the contrastive mechanism can be dropped
and the pipeline gets materially simpler.

⚠️ RECALL IS SET BY STAGE 1. The LLM can only PRUNE the candidate set, never
add to it, so any true positive the embedding misses is lost for good. A
stricter stage-1 threshold therefore caps the achievable recall — which is
exactly why both stage-1 recall and post-rerank recall are reported per arm.

Scored against description_labels.json (agreed label; `uncertain` excluded) —
the same reference every other description-filter eval uses. ROW level, because
that is the level the ground truth is labelled at.

Usage:
  python evaluation/description_gt/evaluate_llm_rerank.py
  python evaluation/description_gt/evaluate_llm_rerank.py --threshold 0.5
  python evaluation/description_gt/evaluate_llm_rerank.py --arms plain
  python evaluation/description_gt/evaluate_llm_rerank.py --limit 30  # smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "craft" / "generator" / "tools"))
sys.path.append(str(Path(__file__).resolve().parent))

from build_description_labels import label_batch, cat_mask, load_pois  # noqa: E402
# production scoring math + neutral-phrase rule, imported (not reimplemented)
from poi_embeddings import cosine_sim  # noqa: E402


def _neutral_phrase(category: str) -> str:
    """The bare-category phrase the contrastive arm subtracts, e.g.
    'museum' -> 'a museum'. Lived in filtration.py until 2026-08-22, when the
    contrastive rule was removed from production; kept here so this comparison
    stays reproducible without leaving dead code in the pipeline."""
    word = str(category).replace("_", " ").strip().lower()
    article = "an" if word[:1] in "aeiou" else "a"
    return f"{article} {word}"

GT_DIR = ROOT / "outputs" / "evaluation" / "description_gt"
LABELS = GT_DIR / "description_labels.json"
EMB_REPLAY = GT_DIR / "poi_embeddings_replay.json"
BARE_EMB = GT_DIR / "bare_query_embeddings.json"
# separate cache: never write to data/socal/set3_query_embeddings.json, which
# evaluate_craft_set3.py depends on.
NEUTRAL_EMB = GT_DIR / "neutral_phrase_embeddings.json"
# Persisted so export_description_eval.py can score the LLM-rerank rule without
# re-calling the model. Verdicts are ARM-INDEPENDENT: the prompt carries only
# (target phrase, POI description), so the arm decides only WHICH rows get asked,
# never what the answer is — hence arms may safely merge into one cache.
VERDICT_CACHE = GT_DIR / "llm_rerank_verdicts.json"
DEFAULT_OUT = ROOT / "results" / "craft_description_filter_llm_rerank.xlsx"
DEFAULT_SUMMARY = ROOT / "results" / "craft_description_filter_llm_rerank_summary.json"

# production defaults, craft/generator/tools/filtration.py
PROD_THRESHOLD = 0.50            # overridden by --threshold
PROD_CONTRAST_CUTOFF = 0.01      # filtration.py sim_contrast_cutoff


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def ensure_embeddings(phrases: List[str], cache_path: Path) -> Dict[str, List[float]]:
    """Embed any phrase not already cached, using the same Gemini model/route
    poi_embeddings uses. Persisted to its own file."""
    cache: Dict[str, List[float]] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
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
            print(f"[embed] {ph!r}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache))
    return cache


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--timeout-s", type=float, default=180.0)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap candidates per arm (smoke test)")
    ap.add_argument("--threshold", type=float, default=PROD_THRESHOLD,
                    help="absolute cosine floor for the plain arm")
    ap.add_argument("--contrast-cutoff", type=float, default=PROD_CONTRAST_CUTOFF)
    ap.add_argument("--arms", nargs="+", default=["plain", "contrastive"],
                    choices=["plain", "contrastive"])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--summary-out", default=str(DEFAULT_SUMMARY))
    args = ap.parse_args()

    from craft.generator.agent import LocalLLM

    labels_blob = json.loads(LABELS.read_text())
    rows_lab: Dict[str, Dict[str, Any]] = labels_blob["rows"]
    phrase_meta: Dict[str, Any] = labels_blob["phrases"]
    phrases = list(phrase_meta)

    emb = {int(k): v for k, v in json.loads(EMB_REPLAY.read_text()).items()}
    bare = json.loads(BARE_EMB.read_text())
    neutral_needed = sorted({_neutral_phrase(phrase_meta[p]["category"]) for p in phrases})
    neutral = ensure_embeddings(neutral_needed, NEUTRAL_EMB)
    p = load_pois()

    llm = LocalLLM(args.model, timeout_s=args.timeout_s)

    summary_rows: List[list] = []
    detail_rows: List[list] = []
    timing_rows: List[list] = []
    total_wall = 0.0
    verdict_cache: Dict[str, Dict[str, bool]] = {}
    if VERDICT_CACHE.exists():
        verdict_cache = json.loads(VERDICT_CACHE.read_text())

    for arm in args.arms:
        print(f"\n{'=' * 70}\nARM: {arm}"
              f"{f' @ floor {args.threshold}' if arm == 'plain' else f' @ cutoff {args.contrast_cutoff}'}"
              f"\n{'=' * 70}")
        budget = args.limit

        for phrase in phrases:
            cat = phrase_meta[phrase]["category"]

            gt_true, gt_false, uncertain = set(), set(), 0
            for sid, per in rows_lab.items():
                if phrase not in per:
                    continue
                rec = per[phrase]
                if rec.get("uncertain"):
                    uncertain += 1
                    continue
                (gt_true if rec["label"] else gt_false).add(int(sid))
            scored_rows = gt_true | gt_false

            qvec = bare.get(phrase)
            nvec = neutral.get(_neutral_phrase(cat))
            if arm == "contrastive":
                # production: score is the DIFFERENCE, compared to contrast_cutoff
                scores = {i: cosine_sim(qvec, emb[i]) - cosine_sim(nvec, emb[i])
                          for i in scored_rows if i in emb}
                cutoff = args.contrast_cutoff
            else:
                scores = {i: cosine_sim(qvec, emb[i]) for i in scored_rows if i in emb}
                cutoff = args.threshold
            embed_pred = {i for i, s in scores.items() if s >= cutoff}

            etp = len(embed_pred & gt_true); efp = len(embed_pred & gt_false)
            efn = len(gt_true) - etp
            epr, erc, ef1 = prf(etp, efp, efn)

            candidates = sorted(embed_pred)
            if budget is not None:
                candidates = candidates[:budget]
                budget -= len(candidates)
            items = [(i, str(p.loc[i, "description"])) for i in candidates]

            print(f"\n[{arm}] [{phrase}] stage1 candidates={len(items)} "
                  f"(P={epr} R={erc} F1={ef1})")

            llm_verdict: Dict[int, bool] = {}
            n_batches = 0
            phrase_wall = 0.0
            for b in range(0, len(items), args.batch):
                chunk = items[b:b + args.batch]
                t0 = time.perf_counter()
                try:
                    got = label_batch(llm, phrase, chunk)
                except Exception as e:
                    print(f"   batch {b // args.batch}: FAILED ({e})")
                    got = {}
                dt = time.perf_counter() - t0
                n_batches += 1
                phrase_wall += dt
                total_wall += dt
                timing_rows.append([arm, phrase, b // args.batch, len(chunk),
                                    round(dt, 2), round(dt / len(chunk), 3)])
                llm_verdict.update(got)

            ph_cache = verdict_cache.setdefault(phrase, {})
            for i, v in llm_verdict.items():
                ph_cache[str(i)] = bool(v)
            VERDICT_CACHE.parent.mkdir(parents=True, exist_ok=True)
            VERDICT_CACHE.write_text(json.dumps(verdict_cache, indent=1))

            final_pred = {i for i in candidates if llm_verdict.get(i) is True}
            ftp = len(final_pred & gt_true); ffp = len(final_pred & gt_false)
            ffn = len(gt_true) - ftp
            fpr, frc, ff1 = prf(ftp, ffp, ffn)

            summary_rows.append([
                arm, cutoff, phrase, len(scored_rows), len(gt_true), uncertain,
                len(embed_pred), etp, efp, efn, epr, erc, ef1,
                len(candidates), ftp, ffp, ffn, fpr, frc, ff1,
                n_batches, round(phrase_wall, 1),
                round(phrase_wall / len(candidates), 3) if candidates else None,
            ])

            for i in candidates:
                detail_rows.append([
                    arm, phrase, i, str(p.loc[i, "name"]),
                    str(p.loc[i, "description"])[:150],
                    i in gt_true, round(scores.get(i, 0.0), 4),
                    llm_verdict.get(i), i in final_pred,
                ])

            if candidates:
                print(f"[{arm}] [{phrase}] after rerank: P={fpr} R={frc} F1={ff1} "
                      f"({phrase_wall:.1f}s, {phrase_wall/len(candidates):.2f}s/item)")
            if budget is not None and budget <= 0:
                break

    # write
    from openpyxl import Workbook
    from evaluation.evaluate_llm import _write_table

    head = ["arm", "cutoff", "phrase", "n_scored_rows", "n_true", "uncertain_excluded",
            "embed_candidates", "embed_TP", "embed_FP", "embed_FN",
            "embed_P", "embed_R", "embed_F1",
            "rerank_candidates", "rerank_TP", "rerank_FP", "rerank_FN",
            "rerank_P", "rerank_R", "rerank_F1",
            "n_llm_batches", "llm_wall_s", "llm_s_per_item"]
    wb = Workbook(); wb.remove(wb.active)
    _write_table(wb.create_sheet("summary"), head, summary_rows,
                f"Production embedding rule (stage 1) vs. + one local-LLM "
                f"({args.model}) rerank pass (stage 2)")
    _write_table(wb.create_sheet("timing"),
                ["arm", "phrase", "batch_idx", "batch_size", "wall_s", "s_per_item"],
                timing_rows, "Per-batch LLM call timing")
    _write_table(wb.create_sheet("per_item"),
                ["arm", "phrase", "row_id", "name", "description", "ground_truth",
                 "score", "llm_verdict", "kept_after_rerank"],
                detail_rows, "Every stage-1 candidate: GT label vs LLM verdict")
    _write_table(wb.create_sheet("legend"), ["term", "meaning"], [
        ["arm plain", f"production rule with sim_contrastive=False: "
                      f"cosine(query, poi) >= {args.threshold}"],
        ["arm contrastive", f"production rule with sim_contrastive=True (today's default): "
                            f"cosine(query, poi) - cosine(neutral_category, poi) >= "
                            f"{args.contrast_cutoff}"],
        ["score", "the arm's own score — a raw cosine for plain, a DIFFERENCE for "
                  "contrastive; the two are not on the same scale"],
        ["embed_*", "stage 1 alone (embedding only) — its recall CAPS the arm, since "
                    "stage 2 can only prune"],
        ["rerank_*", "after the LLM verification pass"],
        ["llm_verdict", "single-pass local-LLM true/false, same prompt as the "
                        "ground-truth judge (build_description_labels.py)"],
        ["ground_truth", "agreed label from description_labels.json; uncertain excluded"],
        ["llm_s_per_item", "wall time / candidates, including batching"],
    ], "Definitions")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)
    Path(args.summary_out).write_text(json.dumps({
        "model": args.model, "threshold": args.threshold,
        "contrast_cutoff": args.contrast_cutoff, "arms": args.arms,
        "batch": args.batch, "total_wall_s": round(total_wall, 1),
        "rows": [dict(zip(head, r)) for r in summary_rows],
    }, indent=2, default=str))

    print(f"\n[done] {out_path}  (total LLM wall time: {total_wall:.1f}s)")
    print(f"\n{'arm':14s} {'phrase':26s} {'stage1 P/R/F1':22s} {'rerank P/R/F1':22s}")
    for r in summary_rows:
        print(f"{r[0]:14s} {r[2]:26s} "
              f"{r[10]:.2f}/{r[11]:.2f}/{r[12]:.2f}{'':10s} "
              f"{r[17]:.2f}/{r[18]:.2f}/{r[19]:.2f}")


if __name__ == "__main__":
    main()
