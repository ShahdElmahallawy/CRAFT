#!/usr/bin/env python3
"""
build_description_labels.py — LLM-labelled ground truth for set3's description filters.

Structured filters have a deterministic truth (column == value on the same row).
Description filters do not: "is this POI an art museum?" is a semantic judgement, so
the ground truth has to be *labelled*. This asks an LLM to label every candidate POI
T/F for each description phrase, producing the reference set the embedding filter is
then scored against.

FAIRNESS RULE: the judge is shown only the POI's description text — exactly the input
the embedding filter sees. Showing it the name would give the judge information the
filter never had, and would flatter or punish the filter for the wrong reason.

SCOPE (measured, not assumed): set3 uses 3 description phrases —
"an art museum", "a history museum" (museum), "a theme park attraction" (attraction).
534 POI rows carry a real description (153 museum, 381 attraction), so the job is
153x2 + 381x1 = 687 judgements.

RELIABILITY: two independent passes at temperature 0 with the item order SHUFFLED
between them (so a pass cannot simply be re-read off position). Items where the two
passes disagree are recorded as `uncertain` rather than silently averaged, and are
reported so they can be audited by hand. The generator's own keyword rule is also
recorded per row as a cheap, independent cross-check.

Results are cached per (phrase,row) so re-runs cost nothing and are deterministic.

Usage:
  python evaluation/description_gt/build_description_labels.py
  python evaluation/description_gt/build_description_labels.py --passes 1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "craft" / "generator" / "tools"))

OUT_DIR = ROOT / "outputs" / "evaluation" / "description_gt"
LABELS = OUT_DIR / "description_labels.json"

# phrase -> (category, generator keyword rule) — the keywords are the benchmark's own
# notion of a match, kept only as an independent cross-check on the LLM labels.
PHRASES: Dict[str, Tuple[str, List[str]]] = {
    "an art museum":            ("museum", ["art"]),
    "a history museum":         ("museum", ["history"]),
    "a theme park attraction":  ("attraction", ["theme", "amusement", "disney", "ride"]),
}

INSTRUCTION = """You are labelling points of interest for a routing benchmark.

For each item you are given an id and the POI's description text. Decide whether that
description describes the target: "{target}".

Rules:
- Judge ONLY from the description text. You are given nothing else on purpose.
- Answer true only if the description would satisfy someone asking for "{target}".
  A museum about a different subject is false. A generic museum with no subject stated
  is false. Partial or incidental mentions are false.
- If the description is empty, uninformative, or you cannot tell, answer false.

Return JSON exactly of the form {{"labels": {{"<id>": true|false, ...}}}} covering every
id you were given, and nothing else."""


def load_pois():
    import pyogrio
    from common_handlers import CATEGORY_COLS
    p = pyogrio.read_dataframe(
        ROOT / "data" / "socal" / "road_network.gpkg", layer="pois",
        columns=CATEGORY_COLS + ["description", "name", "nearest_node_osmid"],
        read_geometry=False)
    return p


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


def real_description(p):
    """Explicit null check first — the column is a str extension dtype whose NaNs pass
    straight through .str operations, so an isin() test alone silently keeps them."""
    bad = {"nan", "none", "", "–", "-", "n/a"}
    d = p["description"]
    return d.notna() & ~d.astype(str).str.strip().str.lower().isin(bad)


def keyword_hit(desc: str, keywords: List[str]) -> bool:
    """Generator's rule, with a word boundary — plain substring matching gave 7/41
    false positives on 'art' ('apartment', 'Hart', 'Park')."""
    low = str(desc).lower()
    return any(re.search(r"\b" + re.escape(k), low) for k in keywords)


def label_batch(llm, target: str, items: List[Tuple[int, str]]) -> Dict[int, bool]:
    payload = {"items": [{"id": str(i), "description": str(t)[:600]} for i, t in items]}
    out = llm.ask_json(INSTRUCTION.format(target=target), payload)
    raw = out.get("labels") or {}
    res: Dict[int, bool] = {}
    for i, _ in items:
        v = raw.get(str(i))
        if isinstance(v, bool):
            res[i] = v
        elif isinstance(v, str):
            res[i] = v.strip().lower() in ("true", "yes", "1")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--out", default=str(LABELS))
    args = ap.parse_args()

    from craft.generator.agent import LocalLLM

    p = load_pois()
    hasd = real_description(p)

    cache: Dict[str, Any] = {}
    outp = Path(args.out)
    if outp.exists():
        cache = json.loads(outp.read_text())

    llm = LocalLLM(args.model, timeout_s=180.0)
    rows_out: Dict[str, Dict[str, Any]] = cache.get("rows", {})

    for phrase, (cat, kws) in PHRASES.items():
        sel = p.index[cat_mask(p, cat) & hasd]
        items = [(int(i), str(p.loc[i, "description"])) for i in sel]
        print(f"\n[{phrase}] category={cat}  candidates={len(items)}")

        passes: List[Dict[int, bool]] = []
        for pi in range(args.passes):
            key = f"{phrase}|pass{pi}"
            if key in cache.get("passes", {}):
                passes.append({int(k): v for k, v in cache["passes"][key].items()})
                print(f"   pass {pi}: cached")
                continue
            order = list(items)
            if pi % 2 == 1:
                order = list(reversed(order))     # break positional carry-over
            got: Dict[int, bool] = {}
            for b in range(0, len(order), args.batch):
                chunk = order[b:b + args.batch]
                try:
                    got.update(label_batch(llm, phrase, chunk))
                except Exception as e:
                    print(f"   batch {b} failed: {e}")
            print(f"   pass {pi}: labelled {len(got)}/{len(items)}  true={sum(got.values())}")
            cache.setdefault("passes", {})[key] = {str(k): v for k, v in got.items()}
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(json.dumps(cache))
            passes.append(got)

        agree = dis = 0
        for i, desc in items:
            vals = [pp.get(i) for pp in passes if pp.get(i) is not None]
            if not vals:
                continue
            consistent = len(set(vals)) == 1
            agree += consistent
            dis += (not consistent)
            rows_out.setdefault(str(i), {})[phrase] = {
                "label": vals[0] if consistent else None,
                "uncertain": not consistent,
                "votes": vals,
                "keyword": keyword_hit(desc, kws),
            }
        tot = agree + dis
        print(f"   pass agreement: {agree}/{tot} ({100*agree/tot:.1f}%)  uncertain={dis}")

    cache["rows"] = rows_out
    cache["phrases"] = {k: {"category": v[0], "keywords": v[1]} for k, v in PHRASES.items()}
    outp.write_text(json.dumps(cache, indent=1))

    print(f"\n[done] {outp}")
    for phrase in PHRASES:
        lab = [v[phrase] for v in rows_out.values() if phrase in v]
        t = sum(1 for x in lab if x["label"] is True)
        u = sum(1 for x in lab if x["uncertain"])
        kw = sum(1 for x in lab if x["keyword"])
        both = sum(1 for x in lab if x["label"] is True and x["keyword"])
        print(f"  {phrase:26s} n={len(lab):4d} LLM_true={t:4d} keyword_true={kw:4d} "
              f"both={both:4d} uncertain={u}")


if __name__ == "__main__":
    main()
