#!/usr/bin/env python3
"""
annotate_labels_for_review.py — human-readable join of description_labels.json
with the POI name/description, for manually spot-checking the ground truth.

⚠️ REVIEW-only. The LLM judge in build_description_labels.py is deliberately
shown only the description text (not the name) — that's the fairness rule the
whole ground truth rests on. This script's output (with name attached) must
never be fed back into labeling or scoring; it exists purely so a person can
eyeball "does this label look right" with the name visible. Nothing in the
evaluation pipeline (evaluate_description_filter.py, export_description_eval.py)
reads this file.

Usage:
  python evaluation/description_gt/annotate_labels_for_review.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "craft" / "generator" / "tools"))

LABELS = ROOT / "outputs" / "evaluation" / "description_gt" / "description_labels.json"
GPKG = ROOT / "data" / "socal" / "road_network.gpkg"
OUT = ROOT / "outputs" / "evaluation" / "description_gt" / "description_labels_annotated.xlsx"


def load_pois():
    import pyogrio
    from common_handlers import CATEGORY_COLS
    return pyogrio.read_dataframe(
        GPKG, layer="pois",
        columns=CATEGORY_COLS + ["description", "name", "nearest_node_osmid"],
        read_geometry=False)


def main() -> None:
    data = json.loads(LABELS.read_text())
    rows, phrases = data["rows"], list(data["phrases"])
    pois = load_pois()

    out_rows = []
    for row_id_str, per_phrase in rows.items():
        row_id = int(row_id_str)
        poi = pois.iloc[row_id]
        rec = {
            "row_id": row_id,
            "name": poi.get("name"),
            "description": poi.get("description"),
            "nearest_node_osmid": poi.get("nearest_node_osmid"),
            "tourism": poi.get("tourism"),
            "amenity": poi.get("amenity"),
            "leisure": poi.get("leisure"),
            "shop": poi.get("shop"),
            "historic": poi.get("historic"),
        }
        for phrase in phrases:
            info = per_phrase.get(phrase)
            prefix = phrase.replace(" ", "_")
            if info is None:
                rec[f"{prefix}__label"] = None
                rec[f"{prefix}__uncertain"] = None
                rec[f"{prefix}__keyword"] = None
                rec[f"{prefix}__votes"] = None
            else:
                rec[f"{prefix}__label"] = info.get("label")
                rec[f"{prefix}__uncertain"] = info.get("uncertain")
                rec[f"{prefix}__keyword"] = info.get("keyword")
                rec[f"{prefix}__votes"] = str(info.get("votes"))
        out_rows.append(rec)

    out_rows.sort(key=lambda r: r["row_id"])

    from openpyxl import Workbook
    from evaluation.evaluate_llm import _write_table
    sys.path.insert(0, str(ROOT))

    headers = list(out_rows[0].keys())
    wb = Workbook(); wb.remove(wb.active)
    _write_table(wb.create_sheet("labels_annotated"), headers,
                [[r[h] for h in headers] for r in out_rows],
                "Ground truth labels + POI name/description, for manual review only "
                "(the LLM judge never saw the name — see this file's docstring)")
    _write_table(wb.create_sheet("legend"), ["term", "meaning"], [
        ["row_id", "positional index into the 'pois' layer of road_network.gpkg"],
        ["label", "true/false — does this POI satisfy the phrase, per the LLM judge"],
        ["uncertain", "the 2 labeling passes disagreed; excluded from scoring"],
        ["keyword", "whether a simple keyword match (e.g. 'art') would have agreed"],
        ["votes", "the 2 individual pass labels, in order"],
        ["fairness note", "the judge saw ONLY the description column, never name — "
                          "this file adds name/category purely for human review"],
    ], "Definitions")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"[done] {OUT}  ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
