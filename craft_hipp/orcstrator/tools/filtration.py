"""
filtration.py - Build possible_targets: the general candidate node sets for routing.

possible_targets[i] = set of valid graph nodes for the i-th required stop.

Cascade per stop i (structured → embedding → broader category set):
  1. Collect all nodes of the right POI type   (category filter)
  2. Apply structured column filters            (case-insensitive substring on any
                                                GPKG column; schema-driven, not hardcoded)
  3. Apply description embedding filter         (cosine similarity, optional)

Fallback rule: if any filter step produces an empty candidate set, the
PREVIOUS (broader) set is kept. Routing never fails silently due to an
overly narrow filter — a miss is treated as "no constraint" for that column.

Runtime POI attribute access (e.g. opening_hours) is handled by the generated
expand() code reading self.pois_gdf directly — not pre-baked here.
"""

from typing import Any, Dict, List, Optional, Set, Union

from poi_embeddings import DEFAULT_EMBED_MODEL, DEFAULT_T_LO

def _category_rows(pois_gdf, category: str) -> Set:
    """Row indices whose OWN tags carry `category`.

    Passed to the description filter so it scores only POIs of the requested
    category. It works from candidate NODES, and many POIs share a node, so
    without this restriction another POI's description can qualify a node.
    Uses the same category rule as build_poi_type_nodes (poi_categories over
    CATEGORY_COLS) so the two never disagree.
    """
    import re as _re
    from common_handlers import CATEGORY_COLS
    target = str(category).strip().lower()
    # Vectorised equivalent of poi_categories(): split_multi_value treats "|" and
    # "," as ";", strips, and compares whole values. Done with pandas string ops
    # rather than iterrows, which would add ~45s per description stop.
    pat = r"(?:^|;)\s*" + _re.escape(target) + r"\s*(?:;|$)"
    mask = None
    for col in CATEGORY_COLS:
        if col not in pois_gdf.columns:
            continue
        norm = (pois_gdf[col].astype(str).str.lower()
                .str.replace("|", ";", regex=False)
                .str.replace(",", ";", regex=False))
        hit = norm.str.contains(pat, regex=True, na=False)
        mask = hit if mask is None else (mask | hit)
    return set() if mask is None else set(pois_gdf.index[mask])


def build_possible_targets(
    G,
    pois_gdf,
    poi_types: List[str],
    node_col: str = "nearest_node_osmid",
    poi_filters: Optional[Union[Dict[str, Dict[str, str]], List[Optional[Dict[str, str]]]]] = None,
    embedding_cache: Optional[Dict[int, List[float]]] = None,
    api_key: Optional[str] = None,
    embedding_model: str = DEFAULT_EMBED_MODEL,
    embedding_base_url: Optional[str] = None,
    description_col: str = "description",
    sim_threshold: float = DEFAULT_T_LO,
    sim_llm_verify: bool = True,
    sim_llm_base_url: Optional[str] = None,
    sim_llm_api_key: Optional[str] = None,
    sim_llm_model: Optional[str] = None,
    rating_filters: Optional[List[Optional[float]]] = None,
    rating_col: str = "rating",
    target_pois: Optional[Dict[int, Dict[Any, List[int]]]] = None,
    dropped_filters: Optional[Dict[int, List[str]]] = None,
) -> Dict[int, Set]:
    """
    Build possible_targets: stop_index → set of valid candidate node IDs.

    Args:
        G:                  NetworkX road graph.
        pois_gdf:           POI GeoDataFrame (from GPKG pois layer).
        poi_types:          Ordered list of POI categories, e.g. ["fuel", "park"].
        node_col:           Column in pois_gdf that maps rows to graph node IDs.
        poi_filters:        Optional filters, either:
                            1) list aligned with poi_types by stop index:
                               [None, {"description": "playground for kids"}, ...]
                            2) legacy dict category → {column: value}
                               (or category → "description text").
                            Special key "description" triggers semantic filter.
        embedding_cache:    Pre-built cache of DataFrame index → embedding vector.
                            Required for description filtering; ignored otherwise.
        api_key:            Gemini API key for embedding the query text.
        embedding_model:    Embedding model name.
        embedding_base_url: Override for Gemini base URL.
        description_col:    Column in pois_gdf that holds POI description text.
        sim_threshold:      Absolute cosine floor for description filtering
                            (stage 1). Deliberately LENIENT: stage 2 can only
                            prune, so this floor caps the cascade's recall.
        sim_llm_verify:     Run the stage-2 LLM verification pass that prunes
                            stage 1's false positives. Skipped automatically when
                            no LLM endpoint is configured, so offline runs still
                            produce candidates (high recall, low precision).
        sim_llm_*:          Override the stage-2 endpoint; by default resolved
                            from LOCAL_LLM_* (preferred), then QWEN_* (legacy
                            alias for the same local endpoint), then OPENAI_*.

                            Rule changed 2026-08-22: the relative band and the
                            contrastive variant were REMOVED — both raised stage-1
                            precision but discarded true matches the LLM could not
                            recover. See poi_embeddings.py for the measurements.

        rating_filters:     Optional numeric floor per stop, index-aligned with
                            poi_types: [4.0, None, 4.0, ...]. A candidate is kept
                            only if its own POI row rates >= the threshold, so a
                            well-rated POI of another category sharing the node
                            does not qualify the stop. A hard filter, but it
                            broadens back on empty like the others. None at an
                            index means no rating constraint for that stop.
        rating_col:         Column in pois_gdf holding the numeric rating.

        target_pois:        Optional OUT dict, filled with stop index → {node: [POI row
                            indices that qualify that node]}. Additive: the return
                            value is unchanged, so the search is not affected.

    Returns:
        Dict mapping stop index (int) → set of valid graph node IDs.
    """
    from common_handlers import build_poi_type_nodes, resolve_graph_node_id

    # Step 1: base type → nodes map (existing logic, unchanged)
    type_nodes = build_poi_type_nodes(G, pois_gdf, poi_types, node_col)
    poi_filters = poi_filters or {}

    possible_targets: Dict[int, Set] = {}

    def _filters_for_stop(stop_idx: int, cat: str) -> Dict[str, Any]:
        # Preferred: index-aligned list
        if isinstance(poi_filters, list):
            if 0 <= stop_idx < len(poi_filters):
                v = poi_filters[stop_idx]
                if v is None:
                    return {}
                if isinstance(v, str):
                    return {"description": v}
                if isinstance(v, dict):
                    return dict(v)
            return {}

        # Legacy dict keyed by category and/or index string
        if isinstance(poi_filters, dict):
            v = poi_filters.get(str(stop_idx), poi_filters.get(cat))
            if v is None:
                return {}
            if isinstance(v, str):
                return {"description": v}
            if isinstance(v, dict):
                return dict(v)
        return {}

    def _rows_to_nodes(rows) -> Dict[Any, List[int]]:
        out: Dict[Any, List[int]] = {}
        if not rows:
            return out
        for idx, raw in pois_gdf.loc[sorted(rows), node_col].items():
            if raw is None or raw != raw:  # NaN
                continue
            nid = resolve_graph_node_id(G, raw)
            if nid is not None:
                out.setdefault(nid, []).append(int(idx))
        return out

    for i, cat in enumerate(poi_types):
        nodes: Set = set(type_nodes.get(cat, set()))
        filters = _filters_for_stop(i, cat)
        # Rating floor for this stop, if any. Read before the early exit below,
        # because a HIPP query commonly asks for a rating and nothing else.
        rating_min = (
            rating_filters[i] if (rating_filters and i < len(rating_filters)) else None
        )
        if not filters and rating_min is None:
            possible_targets[i] = nodes
            if target_pois is not None:
                target_pois[i] = _rows_to_nodes(_category_rows(pois_gdf, cat))
            continue

        # Step 2: rating floor, then structured column filters (any GPKG column
        # except description).
        # PER-POI (fixed 2026-11-07): every filter is applied to the SAME POI rows,
        # starting from rows whose own category is `cat`. The old code intersected
        # NODE sets ("nodes with a restaurant" ∩ "nodes with anything takeaway=yes"),
        # so a node passed when a plain restaurant and a different takeaway POI
        # merely shared it — e.g. Fred 62 (restaurant, no takeaway) + Guac Daddy
        # (fast_food, takeaway) qualified as "a restaurant that offers takeout".
        cat_rows: Set = _category_rows(pois_gdf, cat)
        rows: Set = set(cat_rows)

        # Rating floor (HIPP). A hard filter, not a preference: a candidate
        # survives only if its own POI row rates at or above the threshold.
        # Narrowed on ROWS for the same reason the structured filters are — a
        # well-rated POI of some other category sharing a graph node must not
        # qualify this stop. Broadens back on empty like every other filter, so
        # an unmeetable floor costs precision, never the whole query.
        if rating_min is not None:
            if rating_col not in pois_gdf.columns:
                print(f"[filtration] Column '{rating_col}' not in pois_gdf — skipping rating filter.")
            else:
                import pandas as pd
                numeric_rating = pd.to_numeric(pois_gdf[rating_col], errors="coerce")
                narrowed_rows = rows & set(pois_gdf.index[numeric_rating >= float(rating_min)])
                if not narrowed_rows:
                    print(
                        f"[filtration] Rating filter {rating_col}>={rating_min} for '{cat}'"
                        f" produced empty set — keeping broader set."
                    )
                    if dropped_filters is not None:
                        dropped_filters.setdefault(i, []).append(f"{rating_col}>={rating_min}")
                else:
                    rows = narrowed_rows

        desc_filter: Optional[str] = None
        for col, val in filters.items():
            if col == "description":
                desc_filter = val  # defer to step 3
                continue
            if col not in pois_gdf.columns:
                print(f"[filtration] Column '{col}' not in pois_gdf — skipping filter.")
                continue
            val_str = str(val).lower()
            mask = (
                pois_gdf[col]
                .astype(str)
                .str.lower()
                .str.contains(val_str, na=False, regex=False)
            )
            narrowed_rows = rows & set(pois_gdf.index[mask])
            if not narrowed_rows:
                print(
                    f"[filtration] Structured filter {col}='{val}' for '{cat}'"
                    f" produced empty set — keeping broader set."
                )
                if dropped_filters is not None:
                    dropped_filters.setdefault(i, []).append(f"{col}={val}")
            else:
                rows = narrowed_rows
        if rows != cat_rows:
            nodes = set(_rows_to_nodes(rows)) & nodes

        # Step 3: description filter — embedding only, no keyword fallback
        if desc_filter is not None:
            narrowed: Optional[Set] = None

            desc_rows: Set = set()
            if embedding_cache is not None and api_key:
                try:
                    from poi_embeddings import filter_nodes_by_description
                    # The parser often condenses the phrase to a bare word ("history",
                    # "art"). T_lo was tuned on phrases naming the place type ("a history
                    # museum"), and a bare word scores lower (best 0.30 on set3_006), so
                    # the type is appended when the phrase does not already contain it.
                    desc_query = (desc_filter if str(cat).lower() in str(desc_filter).lower()
                                  else f"{desc_filter} {cat}")
                    narrowed = filter_nodes_by_description(
                        pois_gdf=pois_gdf,
                        candidate_nodes=nodes,
                        query_text=desc_query,
                        embedding_cache=embedding_cache,
                        api_key=api_key,
                        node_col=node_col,
                        description_col=description_col,
                        model=embedding_model,
                        base_url=embedding_base_url,
                        threshold=sim_threshold,
                        # same POI rows that passed the structured filters, so the
                        # description holds on the same POI
                        valid_rows=rows,
                        llm_verify=sim_llm_verify,
                        llm_base_url=sim_llm_base_url,
                        llm_api_key=sim_llm_api_key,
                        llm_model=sim_llm_model,
                        rows_out=desc_rows,
                    )
                except Exception as emb_err:
                    print(f"[filtration] Embedding filter failed ({emb_err}) — keeping broader set.")

            if narrowed:
                nodes = narrowed
                rows = desc_rows or rows
            else:
                # Reached when the filter genuinely matched nothing. This used to be
                # unreachable: filter_nodes_by_description returned the UNFILTERED
                # set on no-match, which is truthy, so the fallback was taken with
                # no warning at all and the description constraint vanished
                # silently. It now returns an empty set, so this warning fires.
                print(f"[filtration] Description filter '{desc_filter}' matched nothing"
                      f" (or embeddings unavailable) — keeping broader set of"
                      f" {len(nodes)} nodes; the description constraint is NOT applied.")
                if dropped_filters is not None:
                    dropped_filters.setdefault(i, []).append(f"description={desc_filter}")

        possible_targets[i] = nodes
        if target_pois is not None:
            # which POI rows qualify each candidate node (fallback: the broader rows)
            target_pois[i] = {n: r for n, r in _rows_to_nodes(rows).items() if n in nodes}

    return possible_targets
