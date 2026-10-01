"""
Pure-code helpers for the parser agent — no LLM calls here.

CatalogResolver lists the dataset's real POI categories (so the LLM only ever
picks from valid names) and maps loose wording like "coffee shop" to "cafe".
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from rapidfuzz import fuzz, process as rf_process
    _HAVE_RAPIDFUZZ = True
except ImportError:
    import difflib
    _HAVE_RAPIDFUZZ = False

from craft.generator.tools.common_handlers import CATEGORY_COLS

MATCH_THRESHOLD = 70


class CatalogResolver:
    """Loads dataset categories from catalog_pois.json (and, if gpkg_path is
    given, from the gpkg pois layer too) and normalizes loose category wording."""

    def __init__(
        self,
        catalog_path: str,
        gpkg_path: Optional[str] = None,
    ) -> None:
        self._categories: List[str] = []
        self._category_aliases: Dict[str, str] = {}

        self._load_catalog(catalog_path)
        if gpkg_path:
            self._load_gpkg(gpkg_path)

        self._categories = sorted(set(self._categories))
        self._build_category_aliases()

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

    def _build_category_aliases(self) -> None:
        aliases = {
            "coffee": "cafe",
            "coffee shop": "cafe",
            "gas station": "fuel",
            "grocery": "supermarket",
            "grocery store": "supermarket",
            "fitness": "fitness",
            "gym": "fitness",
            "movie theater": "cinema",
            "pharmacy": "pharmacy",
        }
        category_set = set(self._categories)
        for alias, target in aliases.items():
            if target in category_set:
                self._category_aliases[alias] = target

    def list_categories(self) -> List[str]:
        return list(self._categories)

    def normalize_category(self, label: str) -> Optional[str]:
        if not label:
            return None
        raw = str(label).strip().lower()
        if not raw:
            return None
        if raw in self._category_aliases:
            return self._category_aliases[raw]
        if raw in self._categories:
            return raw
        if not self._categories:
            return None

        if _HAVE_RAPIDFUZZ:
            best = rf_process.extractOne(raw, self._categories, scorer=fuzz.WRatio)
            if best is None:
                return None
            matched, score, _idx = best
            return matched if score >= MATCH_THRESHOLD else None

        import difflib

        scored = [
            (difflib.SequenceMatcher(None, raw, cat).ratio() * 100.0, cat)
            for cat in self._categories
        ]
        scored.sort(reverse=True)
        if not scored:
            return None
        score, cat = scored[0]
        return cat if score >= MATCH_THRESHOLD else None

    def __len__(self) -> int:
        return len(self._categories)
