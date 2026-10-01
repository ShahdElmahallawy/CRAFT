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
    """Embed a single string. Returns a float vector."""
    return _embed_batch([text], api_key, model, base_url)[0]


def load_embedding_cache(cache_path: str) -> Dict[int, List[float]]:
    """Load embedding cache from JSON. Returns empty dict if file missing."""
    p = Path(cache_path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def embedding_cache_path(dataset_dir: Path) -> Path:
    """Embedding cache path for a dataset dir — poi_embeddings.json, falling back to the old poi_embeddings_cache.json name."""
    primary = dataset_dir / "poi_embeddings.json"
    if primary.exists():
        return primary
    return dataset_dir / "poi_embeddings_cache.json"

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
    model: str = _GEMINI_DEFAULT_EMBED_MODEL,
    base_url: Optional[str] = None,
    threshold: float = 0.75,
) -> Set:
    """Narrows candidate_nodes to those whose description matches query_text.

    Embeds query_text, scores each candidate against its cached description
    embedding via cosine similarity, and keeps nodes scoring >= threshold.
    Falls back to the original candidate_nodes if nothing passes or there's
    no cached embedding to compare against."""
    if not query_text.strip() or not embedding_cache or not candidate_nodes:
        return candidate_nodes

    if description_col not in pois_gdf.columns or node_col not in pois_gdf.columns:
        return candidate_nodes

    query_vec = embed_text(query_text, api_key, model, base_url)

    candidate_str_map = {str(n): n for n in candidate_nodes}

    node_scores: Dict[Any, float] = {}
    subset = pois_gdf[pois_gdf[node_col].astype(str).isin(candidate_str_map)]

    for df_idx, row in subset.iterrows():
        if df_idx not in embedding_cache:
            continue
        raw_id = str(row[node_col])
        node_id = candidate_str_map.get(raw_id)
        if node_id is None:
            continue
        sim = cosine_sim(query_vec, embedding_cache[df_idx])
        if node_id not in node_scores or sim > node_scores[node_id]:
            node_scores[node_id] = sim

    filtered = {nid for nid, score in node_scores.items() if score >= threshold}

    if not filtered:
        print(
            f"[poi_embeddings] No nodes passed threshold {threshold} for '{query_text}'"
            f" — keeping unfiltered set ({len(candidate_nodes)} nodes)"
        )
        return candidate_nodes

    print(
        f"[poi_embeddings] '{query_text}': {len(candidate_nodes)} → {len(filtered)} nodes"
        f" (threshold={threshold})"
    )
    return filtered
