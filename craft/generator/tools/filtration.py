"""
Builds possible_targets: for each required stop, the set of graph nodes that qualify.

Per stop: start from every node of that POI type, narrow by structured column
filters (substring match, any GPKG column), then by description (embedding
similarity). If a step would leave zero candidates, keep the broader set from
before it instead — a bad filter shouldn't kill the whole route.

Attributes that depend on when you arrive (opening_hours etc.) aren't handled
here — the generated expand() checks those against self.pois_gdf at search time.
"""

from typing import Any, Dict, List, Optional, Set, Union


def build_possible_targets(
    G,
    pois_gdf,
    poi_types: List[str],
    node_col: str = "nearest_node_osmid",
    poi_filters: Optional[Union[Dict[str, Dict[str, str]], List[Optional[Dict[str, str]]]]] = None,
    embedding_cache: Optional[Dict[int, List[float]]] = None,
    api_key: Optional[str] = None,
    embedding_model: str = "gemini-embedding-001",
    embedding_base_url: Optional[str] = None,
    description_col: str = "description",
    sim_threshold: float = 0.75,
) -> Dict[int, Set]:
    """Returns {stop_index: set of candidate node IDs}.

    poi_filters is either a list aligned with poi_types (one filter dict, description
    string, or None per stop), or a legacy dict keyed by category/index. A
    "description" key routes that stop through the embedding filter instead of a
    column match; embedding_cache + api_key are only needed when that's used.
    """
    from common_handlers import build_poi_type_nodes, resolve_graph_node_id

    # Step 1: base type -> nodes map
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

    for i, cat in enumerate(poi_types):
        nodes: Set = set(type_nodes.get(cat, set()))
        filters = _filters_for_stop(i, cat)
        if not filters:
            possible_targets[i] = nodes
            continue

        # Step 2: structured column filters (any GPKG column except description)
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
            matching_rows = pois_gdf[mask]
            col_nodes: Set = set()
            for raw in matching_rows[node_col].dropna():
                nid = resolve_graph_node_id(G, raw)
                if nid is not None:
                    col_nodes.add(nid)
            narrowed = nodes & col_nodes
            if not narrowed:
                print(
                    f"[filtration] Structured filter {col}='{val}' for '{cat}'"
                    f" produced empty set — keeping broader set."
                )
            else:
                nodes = narrowed

        # Step 3: description filter — embedding only, no keyword fallback
        if desc_filter is not None:
            narrowed: Optional[Set] = None

            if embedding_cache is not None and api_key:
                try:
                    from poi_embeddings import filter_nodes_by_description
                    narrowed = filter_nodes_by_description(
                        pois_gdf=pois_gdf,
                        candidate_nodes=nodes,
                        query_text=desc_filter,
                        embedding_cache=embedding_cache,
                        api_key=api_key,
                        node_col=node_col,
                        description_col=description_col,
                        model=embedding_model,
                        base_url=embedding_base_url,
                        threshold=sim_threshold,
                    )
                except Exception as emb_err:
                    print(f"[filtration] Embedding filter failed ({emb_err}) — keeping broader set.")

            if narrowed:
                nodes = narrowed
            else:
                print(f"[filtration] Description filter '{desc_filter}' — embedding unavailable or empty — keeping broader set.")

        possible_targets[i] = nodes

    return possible_targets
