"""
Pure-code helpers for the parser agent — no LLM calls here.

CatalogResolver lists the dataset's real POI categories, so the LLM only ever
picks from valid names instead of inventing its own.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from craft.generator.tools.common_handlers import CATEGORY_COLS


class CatalogResolver:
    """Loads dataset categories from the gpkg pois layer's category columns
    (CATEGORY_COLS), which is the complete list — and optionally also from a
    pre-built catalog_pois.json, a legacy/secondary source kept for callers
    that don't have a gpkg handy. In practice the gpkg already covers
    everything catalog_pois.json has (and more), so gpkg_path alone is enough."""

    def __init__(
        self,
        catalog_path: Optional[str] = None,
        gpkg_path: Optional[str] = None,
    ) -> None:
        self._categories: List[str] = []

        if catalog_path:
            self._load_catalog(catalog_path)
        if gpkg_path:
            self._load_gpkg(gpkg_path)

        self._categories = sorted(set(self._categories))

    def _load_catalog(self, catalog_path: str) -> None:
        """Load catalog_pois.json (dict of category -> list of POI entries)."""
        data = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for cat, items in data.items():
                self._categories.append(str(cat).strip().lower())
                if isinstance(items, list):
                    for item in items:
                        item_cat = str(item.get("category", "")).strip().lower()
                        if item_cat:
                            self._categories.append(item_cat)
        elif isinstance(data, list):
            for item in data:
                cat = str(item.get("category", "")).strip().lower()
                if cat:
                    self._categories.append(cat)

    def _load_gpkg(self, gpkg_path: str) -> None:
        try:
            import geopandas as gpd
        except ImportError:
            return

        pois_gdf = gpd.read_file(gpkg_path, layer="pois")

        cat_values: List[str] = []
        for col in CATEGORY_COLS:
            if col not in pois_gdf.columns:
                continue
            for val in pois_gdf[col].dropna().unique():
                v = str(val).strip().lower()
                if v and v not in ("nan", "none"):
                    cat_values.append(v)
        self._categories.extend(cat_values)

    def list_categories(self) -> List[str]:
        return list(self._categories)

    def __len__(self) -> int:
        return len(self._categories)
