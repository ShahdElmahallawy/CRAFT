"""Shared constants/helpers for the parser-output collection + evaluation pair
(collect_parser_outputs.py -> writes outputs/parser_outputs/{model}.json,
evaluate_parser.py -> scores those files against benchmark/ground_truth/).
Single source of truth so the two scripts can never disagree on which gold set
maps to which raw-record directory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Set

ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = ROOT / "benchmark" / "ground_truth"
OUTPUTS_DIR = ROOT / "outputs"
COLLECTED_DIR = OUTPUTS_DIR / "parser_outputs"

MODELS = ["gemini-3_1-flash-lite", "gpt-4_1", "openai_gpt-oss-20b"]

# Step-1 (parse) stages its result here, BEFORE the generate subprocess runs:
# craft/orchestrator.py::run_parser() writes {scratch_id}.json directly. The
# final craft record's own `parser_output` field is instead re-read from a
# SEPARATE copy that the generate subprocess writes — so whenever generate
# crashes before writing that copy, the record's parser_output is null even
# though the parse itself succeeded. This staging dir is the ground truth for
# "did the parser actually produce something," independent of that bug.
# CAVEAT: it is a shared, mutable scratch space keyed only by (model,
# query_id) — a later unrelated rerun of the same pair overwrites it, so a
# staging file found here is not guaranteed to be the exact parse that fed
# the specific saved record; it is the best available evidence, not a proof.
STAGING_DIR = OUTPUTS_DIR / "eval_staging_craft_socal"
_STAGING_STRIP_KEYS = ("nl_query", "gpkg", "query_id")

# gold set -> (raw craft-record subdir under outputs/, uses the SoCal graph?)
GOLD_TO_RECORDS = {
    "set1_ev_routing":     ("craft/{model}/ev_routing", True),
    "set1_ordered_poi":    ("craft/{model}/ordered_poi", True),
    "set1_poi_collection": ("craft/{model}/poi_collection", True),
    "set1_ev+ordered_poi":       ("craft/{model}/ev+ordered_poi", True),
    "set1_unordered_poi":        ("craft/{model}/unordered_poi", True),
    "poi_road_pref":             ("craft/{model}/poi_road_pref", True),
    "set3_ordered+desc":         ("craft/{model}/ordered+desc", True),
    # set2 ran on the California graph; endpoints are not comparable against
    # the SoCal node index, so evaluate_parser.py scores it without src/dst.
    "set2_ev_routing_ca":  ("craft_set2_ca/{model}/ev_routing", False),
}


def load_gold_sets() -> Dict[str, list]:
    gold_sets = {}
    for name in GOLD_TO_RECORDS:
        p = GOLD_DIR / f"{name}_gold.json"
        if p.exists():
            gold_sets[name] = json.loads(p.read_text())
    return gold_sets


def load_raw_records(dirpath: Path, valid: Set[str]) -> Dict[str, Optional[Dict[str, Any]]]:
    """query_id -> parser_output dict, or None if the record has no usable
    parser_output (parse failure). Only ids in `valid` (the gold id set) are kept."""
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    if not dirpath.is_dir():
        return out
    for f in sorted(dirpath.glob("*.json")):
        if f.stem == "summary":
            continue
        try:
            rec = json.loads(f.read_text())
        except Exception:
            continue
        qid = rec.get("query_id")
        if qid in valid:
            po = rec.get("parser_output")
            out[qid] = po if isinstance(po, dict) and po else None
    return out


def load_staging_output(model: str, query_id: str,
                        staging_dir: Path = STAGING_DIR) -> Optional[Dict[str, Any]]:
    """The Step-1 parse result for (model, query_id), read directly from the
    pre-generator staging file — used to recover parser_output for records
    where the final craft JSON shows it as null due to a downstream generate
    crash. Returns None if no staging file exists or it has no real content."""
    f = staging_dir / f"{model}__{query_id}.json"
    if not f.is_file():
        return None
    try:
        raw = json.loads(f.read_text())
    except Exception:
        return None
    if not isinstance(raw, dict) or not raw.get("src"):
        # a genuinely successful parse always has at least a resolved src node;
        # anything else (empty dict, error dump) isn't usable parser output.
        return None
    return {k: v for k, v in raw.items() if k not in _STAGING_STRIP_KEYS}


def collected_path(model: str, collected_dir: Path = COLLECTED_DIR) -> Path:
    return collected_dir / f"{model}.json"
