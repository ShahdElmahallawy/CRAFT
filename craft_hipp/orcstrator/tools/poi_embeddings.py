"""
poi_embeddings.py — Embedding utilities for POI description matching.

Loads a prebuilt POI description embedding cache and uses it to semantically
filter candidate POI nodes against a natural-language constraint, e.g.
"park with a playground" or "fast charging station".

Offline cache building lives in data/build_poi_embeddings.py.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


_GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"
_GEMINI_DEFAULT_EMBED_MODEL = "gemini-embedding-001"

# ── description filter defaults (changed 2026-11-09, evaluation/cand_stop_exp) ──
# OpenAI text-embedding-3-small replaces gemini-embedding-001 (Gemini credits ran out;
# measured mean AUC 0.752 vs 0.762 for -large, so the small model is kept).
# T_LO: lowest cosine kept; every row at or above it goes to the LLM. Set 2026-11-10 by
#       quantile calibration at a fixed 5% lost-match rate on 4 phrases disjoint from set3
#       (gpt-4.1 labels): 0.32 keeps 94.8% of matches there and 93.2% on the set3 phrases
#       (evaluation/cand_stop_exp/thresholds.py -> results/cand_stop_exp.xlsx). Gemini's
#       0.50 does NOT transfer — the two models' cosines live on different scales.
#       No accept threshold: at a 1% false-accept rate it auto-accepted 128 wrong rows
#       next to 91 right ones (precision 0.42), and no cutoff reaches a 95% lower bound
#       of 0.95 on precision.
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_T_LO = 0.32


def is_openai_embed_model(model: str) -> bool:
    return str(model).startswith("text-embedding-3")


def embedding_cache_name(model: str = DEFAULT_EMBED_MODEL) -> str:
    """Cache file per embedding model: vectors from different models never mix."""
    return f"poi_embeddings_{model}.json" if is_openai_embed_model(model) else "poi_embeddings.json"

# ── stage-2 LLM verification ─────────────────────────────────────────────────
# The embedding alone is a poor precision instrument on description filters
# (measured: 0.20-0.50 precision at the 0.50 floor), but an excellent recall
# instrument (1.00 on all three set3 phrases). So the cascade is:
#   stage 1  lenient cosine floor  -> catch everything, accept the noise
#   stage 2  one LLM pass          -> prune the noise
# Stage 2 can only REMOVE candidates, so stage-1 recall caps the cascade — which
# is exactly why the floor is lenient rather than clever. Measured end to end
# (results/desc.xlsx, results/craft_description_filter_llm_rerank.xlsx):
# node-level precision 0.20-0.42 -> 0.82-0.98 with recall held at 0.79-1.00.
#
# The prompt is the one the LLM-labelled GROUND TRUTH was built with
# (evaluation/description_gt/build_description_labels.py), reused verbatim so the
# production filter and the reference it was validated against ask the identical
# question. It deliberately shows the model ONLY the description text — the same
# input the embedding gets — never the POI name.
_LLM_VERIFY_INSTRUCTION = """You are labelling points of interest for a routing benchmark.

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

# Lowered from 100 (2026-10-16, measured live against the frozen description_labels.json
# ground truth, "an art museum", N=153 rows, gpt-oss-20b local endpoint). Batch size drives
# BOTH speed and accuracy in the SAME direction -- this is not a speed/accuracy trade-off:
#   batch=25:  recall=1.00  total wall time=3.3s  (7 calls)
#   batch=50:  recall=0.93  total wall time=10.4s
#   batch=100: recall=0.82  total wall time=9.9s  (the old default)
#   batch=150: recall=0.63  total wall time=12.1s
# Confirmed via a side-by-side run of the UNCHANGED prompt/format at the same batch sizes
# (identical tp/fp/fn/tn at batch=25 and batch=100) -- this is a property of batch size on
# this model, not something a different output format fixes. Likely cause: gpt-oss's hidden
# reasoning scales worse than linearly with the number of items judged in one call, so more,
# smaller calls are both faster in aggregate AND keep each item's judgment from degrading.
# Do not raise this back toward 100 without re-measuring against description_labels.json.
_LLM_VERIFY_BATCH = 25
_LLM_VERIFY_TIMEOUT_S = 180.0
_LLM_VERIFY_CACHE_NAME = "poi_description_llm_verdicts.json"


def _read_env_value(key: str, env_path: Optional[Path] = None) -> Optional[str]:
    """os.environ first, then the project .env — the same resolution order the
    rest of the pipeline uses, so a shell export always wins."""
    import os
    v = os.environ.get(key)
    if v:
        return v.strip().strip("'\"")
    p = env_path or (Path(__file__).resolve().parents[3] / ".env")
    try:
        for line in p.read_text().splitlines():
            line = line.strip()
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip().strip("'\"")
    except Exception:
        pass
    return None


def _resolve_llm_endpoint(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """Where to send the stage-2 verification call.

    Resolution order — explicit args, then:
      1. LOCAL_LLM_BASE_URL / LOCAL_LLM_API_KEY / LOCAL_LLM_MODEL   (preferred)
      2. QWEN_*                                                     (legacy alias)
      3. OPENAI_*                                                   (cloud fallback)

    (2) exists only because the local vLLM endpoint was originally configured
    under QWEN_* and those names are still in .env; it has served gpt-oss-20b,
    not Qwen, since 2026-07-14. Prefer LOCAL_LLM_* in new configs — the QWEN_*
    branch is a compatibility shim, not the intended name.

    Returns None when nothing is configured, which callers must treat as
    'skip stage 2', never as an error: an offline run should still produce
    candidates."""
    base = base_url or _read_env_value("LOCAL_LLM_BASE_URL") or \
        _read_env_value("QWEN_BASE_URL") or _read_env_value("OPENAI_BASE_URL")
    key = api_key or _read_env_value("LOCAL_LLM_API_KEY") or \
        _read_env_value("QWEN_API_KEY") or _read_env_value("OPENAI_API_KEY")
    mdl = model or _read_env_value("LOCAL_LLM_MODEL") or "openai/gpt-oss-20b"
    if not base or not key:
        return None
    return {"base_url": base.rstrip("/"), "api_key": key, "model": mdl}


def _llm_chat_json(endpoint: Dict[str, str], instruction: str,
                   payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    """One OpenAI-compatible chat call returning parsed JSON. urllib rather than
    the openai package, matching this module's existing dependency footprint."""
    import json as _json
    import urllib.request

    body = _json.dumps({
        "model": endpoint["model"],
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": _json.dumps(payload, ensure_ascii=True)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        # gpt-oss defaults to ~medium reasoning depth, which burns hidden
        # chain-of-thought tokens even for a flat per-item true/false judgment
        # (measured: 2058 completion tokens / 60 items at default vs 965 at
        # "low", ~2x wall-clock for identical output). This is a true/false
        # classification, not multi-step reasoning, so low is the right depth
        # — re-validate against the frozen label set if this is ever raised.
        "reasoning_effort": "low",
    }).encode("utf-8")
    req = urllib.request.Request(
        endpoint["base_url"] + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {endpoint['api_key']}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = _json.loads(resp.read())
    text = data["choices"][0]["message"]["content"]
    if text is None:
        raise ValueError("LLM returned no content")
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        text = text.lstrip("json").strip()
    return _json.loads(text)


def _llm_verify_rows(
    query_text: str,
    row_descriptions: Dict[Any, str],
    endpoint: Dict[str, str],
    cache: Dict[str, bool],
    batch_size: int = _LLM_VERIFY_BATCH,
    timeout_s: float = _LLM_VERIFY_TIMEOUT_S,
) -> Dict[Any, bool]:
    """True/false per POI row for `query_text`. Cached per (query_text, row) so a
    200-query benchmark re-asking the same phrase costs one pass, not 200.

    A batch that fails is treated as 'keep everything in it' (verdict True), not
    'drop everything': stage 2 exists to improve precision, and a transport error
    must never silently delete valid candidates and make a query unsolvable."""
    verdicts: Dict[Any, bool] = {}
    todo = []
    for row_idx, desc in row_descriptions.items():
        ck = f"{query_text}||{row_idx}"
        if ck in cache:
            verdicts[row_idx] = bool(cache[ck])
        else:
            todo.append((row_idx, desc))

    for b in range(0, len(todo), batch_size):
        chunk = todo[b:b + batch_size]
        payload = {"items": [{"id": str(i), "description": str(d)[:600]}
                             for i, d in chunk]}
        try:
            out = _llm_chat_json(endpoint,
                                 _LLM_VERIFY_INSTRUCTION.format(target=query_text),
                                 payload, timeout_s)
            raw = out.get("labels") or {}
        except Exception as e:
            print(f"[poi_embeddings] LLM verification batch failed ({e}) — "
                  f"keeping its {len(chunk)} candidates unverified")
            for i, _ in chunk:
                verdicts[i] = True
            continue
        for i, _ in chunk:
            v = raw.get(str(i))
            if isinstance(v, bool):
                keep = v
            elif isinstance(v, str):
                keep = v.strip().lower() in ("true", "yes", "1")
            else:
                # id absent from the reply: keep it, same reasoning as above.
                keep = True
            verdicts[i] = keep
            cache[f"{query_text}||{i}"] = keep
    return verdicts


def _embed_batch_gemini_native(
    texts: List[str],
    api_key: str,
    model: str = _GEMINI_DEFAULT_EMBED_MODEL,
) -> List[List[float]]:
    import json as _json
    import urllib.request

    url = _GEMINI_EMBED_URL.format(model=model) + f"?key={api_key}"
    payload = {
        "requests": [
            {"model": f"models/{model}", "content": {"parts": [{"text": t}]}}
            for t in texts
        ]
    }
    body = _json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = _json.loads(resp.read())
    return [item["values"] for item in data["embeddings"]]


def _embed_batch_openai_compat(
    texts: List[str],
    api_key: str,
    model: str = _GEMINI_DEFAULT_EMBED_MODEL,
    base_url: Optional[str] = None,
) -> List[List[float]]:
    """Embed via OpenAI-compatible endpoint (works for non-Gemini providers)."""
    try:
        import openai
    except ImportError:
        raise ImportError("openai package required: pip install openai")

    url = base_url or "https://generativelanguage.googleapis.com/v1beta/openai/"
    client = openai.OpenAI(api_key=api_key, base_url=url)
    response = client.embeddings.create(input=texts, model=model)
    items = list(response.data)
    if items and all(getattr(item, "index", None) is not None for item in items):
        items = sorted(items, key=lambda x: x.index)
    return [item.embedding for item in items]


def _embed_batch(
    texts: List[str],
    api_key: str,
    model: str = _GEMINI_DEFAULT_EMBED_MODEL,
    base_url: Optional[str] = None,
) -> List[List[float]]:
    """
    Embed a batch of texts. Strategy:
      1. Try Gemini native REST (works for text-embedding-004 with a Gemini key).
      2. Fall back to OpenAI-compatible endpoint (works for other providers).
    """
    if is_openai_embed_model(model):
        return _embed_batch_openai_compat(texts, api_key, model, base_url or "https://api.openai.com/v1")
    # If a non-Gemini base_url is explicitly given, skip native Gemini path
    gemini_native = base_url is None or "generativelanguage.googleapis.com" in base_url
    if gemini_native:
        try:
            return _embed_batch_gemini_native(texts, api_key, model)
        except Exception:
            pass  # fall through to OpenAI-compat
    return _embed_batch_openai_compat(texts, api_key, model, base_url)


def embed_text(
    text: str,
    api_key: str,
    model: str = _GEMINI_DEFAULT_EMBED_MODEL,
    base_url: Optional[str] = None,
) -> List[float]:
    """Embed a single string. Returns a float vector.

    OpenAI models always use the OPENAI_* keys: callers (template.py) pass the Gemini
    key, which would be rejected by OpenAI."""
    if is_openai_embed_model(model):
        api_key = _read_env_value("OPENAI_API_KEY") or api_key
        base_url = _read_env_value("OPENAI_BASE_URL")
    return _embed_batch([text], api_key, model, base_url)[0]


def load_embedding_cache(cache_path: str) -> Dict[int, List[float]]:
    """Load embedding cache from JSON. Returns empty dict if file missing."""
    p = Path(cache_path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def embedding_cache_path(dataset_dir: Path, model: str = DEFAULT_EMBED_MODEL) -> Path:
    """
    Return the embedding cache path for a dataset directory and embedding model.

    OpenAI models use poi_embeddings_<model>.json. Gemini uses poi_embeddings.json,
    with the old poi_embeddings_cache.json name as a fallback.
    """
    if is_openai_embed_model(model):
        return dataset_dir / embedding_cache_name(model)
    primary = dataset_dir / "poi_embeddings.json"
    if primary.exists():
        return primary
    return dataset_dir / "poi_embeddings_cache.json"

def _default_llm_cache_dir() -> Path:
    """Where stage-2 verdicts live when the caller names no directory: alongside
    the SoCal embedding cache, so it sits with the other derived POI data rather
    than in the repo root."""
    return Path(__file__).resolve().parents[3] / "data" / "socal"


def cosine_sim(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two embedding vectors."""
    import numpy as np

    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def filter_nodes_by_description(
    pois_gdf,
    candidate_nodes: Set,
    query_text: str,
    embedding_cache: Dict[int, List[float]],
    api_key: str,
    node_col: str = "nearest_node_osmid",
    description_col: str = "description",
    model: str = DEFAULT_EMBED_MODEL,
    base_url: Optional[str] = None,
    threshold: float = DEFAULT_T_LO,
    valid_rows: Optional[Set] = None,
    llm_verify: bool = True,
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_cache_dir: Optional[str] = None,
    rows_out: Optional[Set] = None,
) -> Set:
    """
    Narrow candidate_nodes to those whose description matches query_text.

    rows_out (optional, additive): if given, filled with the POI ROW indices that
    passed (stage-2 survivors, or stage-1 rows when stage 2 did not apply), so the
    caller can tell WHICH POI qualified a node. Does not change the return value.

    TWO-STAGE CASCADE (adopted 2026-08-22, measured — see results/desc.xlsx):
      stage 1  cosine >= `threshold` (plain absolute floor)
      stage 2  one LLM pass over the survivors, dropping those it judges false

    Returns an EMPTY set when nothing qualifies, so the caller can tell "no
    match" apart from "matched everything".

    WHY A LENIENT FLOOR PLUS AN LLM, RATHER THAN A CLEVERER CUTOFF
    Cosine on these descriptions is a strong RECALL instrument and a weak
    PRECISION one: at 0.50 it returns every true match on all three set3
    phrases (recall 1.00) but only 0.20-0.50 precision. Stage 2 can only
    PRUNE, never restore, so stage-1 recall is a hard ceiling on the cascade
    — making the lenient floor the correct choice and the LLM the precision
    instrument. Node-level result: precision 0.20-0.42 -> 0.82-0.98, recall
    held at 0.79-1.00.

    SUPERSEDED RULES (removed 2026-08-22, kept here only so they are not
    reinvented): a 0.75 absolute cutoff admitted NOTHING (max attainable
    similarity is 0.748); a floor-plus-relative-band and a contrastive variant
    (subtracting the bare-category similarity) both raised stage-1 precision
    but DISCARDED true matches at stage 1 — recall 0.81-0.98 — which the LLM
    could then never recover. Measured head to head, plain-floor + LLM beat
    contrastive + LLM on mean F1 (0.938 vs 0.901).

    ⚠️ Tuned on 3 phrases over 2 categories (museum, attraction) — evidence-based
    defaults, not universal ones.

    Args:
        pois_gdf:         POI GeoDataFrame.
        candidate_nodes:  Node IDs already filtered by category.
        query_text:       Filter string, e.g. "park with a playground".
        embedding_cache:  Dict of DataFrame index → embedding vector.
        api_key:          Gemini API key (for embedding the query).
        node_col:         Column mapping POI rows to graph node IDs.
        description_col:  Column with POI description text.
        model:            Embedding model for the query string.
        threshold:        Absolute cosine floor (0–1). Stage 1.
        valid_rows:       Restrict scoring to these POI rows (the requested
                          category). Without it a neighbouring POI's description
                          can carry a node over the floor.
        llm_verify:       Run stage 2. Silently skipped (stage-1 result returned
                          unchanged) when no LLM endpoint is configured, so
                          offline runs still work.
        llm_cache_dir:    Where to persist verdicts; defaults to the embedding
                          cache's directory.

    Returns:
        The qualifying subset of candidate_nodes, or an EMPTY set if nothing
        qualifies. Callers must decide what an empty result means — this function
        does not silently return the unfiltered set.
    """
    if not query_text.strip() or not embedding_cache or not candidate_nodes:
        return candidate_nodes

    if description_col not in pois_gdf.columns or node_col not in pois_gdf.columns:
        return candidate_nodes

    query_vec = embed_text(query_text, api_key, model, base_url)

    candidate_str_map = {str(n): n for n in candidate_nodes}

    # Scored at ROW level, then collapsed to nodes at the end. Stage 2 judges a
    # description, which is a property of a row — a node hosting several POIs has
    # no single description to judge.
    row_score: Dict[Any, float] = {}
    row_node: Dict[Any, Any] = {}
    row_desc: Dict[Any, str] = {}
    subset = pois_gdf[pois_gdf[node_col].astype(str).isin(candidate_str_map)]

    for df_idx, row in subset.iterrows():
        if df_idx not in embedding_cache:
            continue
        # Score ONLY POIs of the requested category. Without this the loop scores
        # every POI sharing a candidate node — ~7 per node on average — so a
        # neighbouring shop's description could push a node over the cutoff and the
        # node would be offered as e.g. "an art museum". Measured: it cut precision
        # from ~83% to ~18% on set3's description filters.
        if valid_rows is not None and df_idx not in valid_rows:
            continue
        raw_id = str(row[node_col])
        node_id = candidate_str_map.get(raw_id)
        if node_id is None:
            continue
        row_score[df_idx] = cosine_sim(query_vec, embedding_cache[df_idx])
        row_node[df_idx] = node_id
        row_desc[df_idx] = str(row.get(description_col) or "")

    node_scores: Dict[Any, float] = {}
    for df_idx, sim in row_score.items():
        nid = row_node[df_idx]
        if nid not in node_scores or sim > node_scores[nid]:
            node_scores[nid] = sim

    if not node_scores:
        print(
            f"[poi_embeddings] '{query_text}': no candidate node has a cached "
            f"description embedding — cannot filter"
        )
        return set()

    best = max(node_scores.values())

    # ── stage 1: plain absolute floor ────────────────────────────────────────
    kept_rows = {i for i, s in row_score.items() if s >= threshold}
    stage1_nodes = {row_node[i] for i in kept_rows}

    if not stage1_nodes:
        # Deliberately EMPTY, not candidate_nodes. Returning the unfiltered set here
        # made the caller read a failed filter as a successful one, so a query for
        # "a cafe with Wi-Fi" silently became "any cafe" with nothing logged.
        print(
            f"[poi_embeddings] No nodes passed floor {threshold:.3f} for '{query_text}'"
            f" (best score {best:.3f}) — returning EMPTY; the caller decides"
        )
        return set()

    # ── stage 2: LLM verification of the surviving rows ──────────────────────
    filtered = stage1_nodes
    final_rows = set(kept_rows)
    n_stage1 = len(stage1_nodes)
    verified = False
    to_verify = kept_rows
    if llm_verify and to_verify:
        endpoint = _resolve_llm_endpoint(llm_base_url, llm_api_key, llm_model)
        if endpoint is None:
            print("[poi_embeddings] stage-2 LLM verification SKIPPED (no endpoint "
                  "configured) — returning the embedding-only set, which is "
                  "high-recall but low-precision")
        else:
            cache_dir = Path(llm_cache_dir) if llm_cache_dir else _default_llm_cache_dir()
            cache_file = cache_dir / _LLM_VERIFY_CACHE_NAME
            cache: Dict[str, bool] = {}
            if cache_file.exists():
                try:
                    cache = json.loads(cache_file.read_text())
                except Exception:
                    cache = {}
            before = len(cache)
            verdicts = _llm_verify_rows(
                query_text, {i: row_desc[i] for i in to_verify}, endpoint, cache)
            if len(cache) != before:
                try:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps(cache, indent=1))
                except Exception as e:
                    print(f"[poi_embeddings] could not persist LLM verdict cache ({e})")
            passed = {i for i in to_verify if verdicts.get(i, True)}
            llm_nodes = {row_node[i] for i in passed}
            if llm_nodes:
                filtered = llm_nodes
                final_rows = set(passed)
                verified = True
            else:
                # Every candidate rejected. Keep stage 1 rather than returning
                # empty: a filter that removes everything makes the query
                # unsolvable, and stage 2 is a precision aid, not an authority.
                print(f"[poi_embeddings] stage-2 rejected ALL {n_stage1} candidates for "
                      f"'{query_text}' — keeping the stage-1 set instead")

    print(
        f"[poi_embeddings] '{query_text}': {len(candidate_nodes)} → {n_stage1}"
        f"{f' → {len(filtered)}' if verified else ''} nodes"
        f" (floor={threshold}, best={best:.3f}"
        f"{', llm-verified' if verified else ', embedding only'})"
    )
    if rows_out is not None:
        rows_out.update(final_rows)
    return filtered
