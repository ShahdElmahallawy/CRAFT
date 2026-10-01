from __future__ import annotations

from typing import Any, Dict, List


_DIST_TO_M = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "km": 1000.0,
    "kilometer": 1000.0,
    "kilometers": 1000.0,
    "mi": 1609.34,
    "mile": 1609.34,
    "miles": 1609.34,
}

_TIME_TO_SEC = {
    # canonical names the parser is instructed to use
    "sec": 1.0,
    "min": 60.0,
    "hrs": 3600.0,
    # common aliases kept for robustness
    "s": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "minute": 60.0,
    "minutes": 60.0,
    "mins": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
}


def _to_number(x: Any):
    if x is None:
        return None
    return float(x)


def _convert_distance_to_m(value: Any, unit: str):
    n = _to_number(value)
    if n is None:
        return None
    f = _DIST_TO_M.get(str(unit).strip().lower())
    return n * (f if f is not None else 1.0)


def _convert_time_to_sec(value: Any, unit: str):
    n = _to_number(value)
    if n is None:
        return None
    f = _TIME_TO_SEC.get(str(unit).strip().lower())
    return n * (f if f is not None else 1.0)


def _normalize_poi_filters(out: Dict[str, Any]) -> None:
    """
    Ensure poi_filters is index-aligned with poi_types.
    Accepts:
      - list (preferred)
      - dict by category and/or index string (legacy)
    """
    poi_types = out.get("poi_types")
    poi_filters = out.get("poi_filters")

    if not isinstance(poi_types, list) or not poi_types:
        return

    if isinstance(poi_filters, list):
        aligned = list(poi_filters[: len(poi_types)])
        if len(aligned) < len(poi_types):
            aligned.extend([None] * (len(poi_types) - len(aligned)))
        out["poi_filters"] = aligned
        return

    if isinstance(poi_filters, dict):
        aligned: List[Any] = []
        for i, cat in enumerate(poi_types):
            v = None
            if str(i) in poi_filters:
                v = poi_filters[str(i)]
            elif cat in poi_filters:
                v = poi_filters[cat]
            aligned.append(v)
        out["poi_filters"] = aligned
        return

    out["poi_filters"] = [None] * len(poi_types)


def _is_time_unit(unit: str) -> bool:
    return unit.strip().lower() in _TIME_TO_SEC


def _is_dist_unit(unit: str) -> bool:
    return unit.strip().lower() in _DIST_TO_M


def _convert_value(spec: Dict[str, Any]) -> Any:
    """Convert a {value, unit} or {values, unit} dict to base-unit float(s). Returns None if unrecognised."""
    unit = str(spec.get("unit", "")).strip().lower()
    if "values" in spec:
        vals = [v for v in (spec["values"] or []) if v is not None]
        if _is_time_unit(unit):
            return [_convert_time_to_sec(v, unit) for v in vals]
        if _is_dist_unit(unit):
            return [_convert_distance_to_m(v, unit) for v in vals]
        return vals 
    if "value" in spec:
        val = spec["value"]
        if _is_time_unit(unit):
            return _convert_time_to_sec(val, unit)
        if _is_dist_unit(unit):
            return _convert_distance_to_m(val, unit)
    return None


def normalize_constraints_for_pipeline(constraints: Dict[str, Any]) -> Dict[str, Any]:
    """Convert {value, unit} / {values, unit} pairs to base units: time -> seconds,
    distance -> meters, percent (on a battery key) -> meters via max_battery, two-pass.
    Key names stay as-is; plain numbers pass through untouched."""
    out = dict(constraints or {})

    # convert all {value/values, unit} dicts except percent
    percent_keys: List[str] = []
    for key, val in list(out.items()):
        if not isinstance(val, dict):
            continue
        unit = str(val.get("unit", "")).strip().lower()
        if unit == "percent":
            percent_keys.append(key)
            continue
        converted = _convert_value(val)
        if converted is not None:
            out[key] = converted

    # percent battery field
    for key in percent_keys:
        spec = out[key]
        pct = float(spec.get("value", 0))
        # find max battery capacity among other keys (max_range, battery_capacity, etc.) — take the largest
        _BATT_KWORDS = {"max", "battery", "range", "capacity", "full"}
        max_val = None
        for k, v in out.items():
            if k == key:
                continue
            kl = k.lower()
            if any(w in kl for w in _BATT_KWORDS) and isinstance(v, (int, float)):
                if max_val is None or float(v) > max_val:
                    max_val = float(v)
        if max_val is not None:
            out[key] = (pct / 100.0) * max_val
        else:
            out[key] = pct  # fallback: leave as raw percentage number

    _normalize_poi_filters(out)
    return out
