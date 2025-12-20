from typing import Optional

import numpy as np
import pandas as pd

_SUFFIX_MAP = {
    "k": 1_000.0,
    "m": 1_000_000.0,
    "mm": 1_000_000.0,
    "b": 1_000_000_000.0,
    "bn": 1_000_000_000.0,
    "t": 1_000_000_000_000.0,
}


def smart_numeric_value(val) -> float:
    if val is None:
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    text = str(val).strip()
    if not text:
        return np.nan
    text = text.replace(",", "").replace("$", "").replace("€", "")
    if text.endswith("%"):
        inner = text[:-1].strip()
        try:
            return float(inner) / 100.0
        except ValueError:
            return np.nan
    lower = text.lower()
    if lower in {"n/a", "na", "nan", "none", "null", "-", "--", "—"}:
        return np.nan

    for suffix in sorted(_SUFFIX_MAP.keys(), key=len, reverse=True):
        if lower.endswith(suffix):
            num_part = text[: -len(suffix)]
            try:
                return float(num_part) * _SUFFIX_MAP[suffix]
            except ValueError:
                return np.nan

    try:
        return float(text)
    except ValueError:
        return np.nan


def smart_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    converted = series.map(smart_numeric_value)
    return pd.to_numeric(converted, errors="coerce")


def choose_label_column(df: pd.DataFrame, categorical_info: list[dict]) -> Optional[str]:
    if df.empty:
        return None
    name_tokens = ("name", "title", "label")
    for col in df.columns:
        if not (
            pd.api.types.is_string_dtype(df[col])
            or pd.api.types.is_object_dtype(df[col])
        ):
            continue
        lower = col.lower()
        if any(tok in lower for tok in name_tokens) and "id" not in lower:
            return col
    best: Optional[str] = None
    best_score = -1.0
    info_map = {c.get("name"): c for c in categorical_info if isinstance(c, dict)}
    for col in df.columns:
        if not (
            pd.api.types.is_string_dtype(df[col])
            or pd.api.types.is_object_dtype(df[col])
        ):
            continue
        lower = col.lower()
        if "id" in lower:
            continue
        info = info_map.get(col, {})
        unique_ratio = float(info.get("ratio") or info.get("unique_ratio") or 0.0)
        score = unique_ratio
        if score > best_score:
            best_score = score
            best = col
    return best
def infer_axis_format(series: pd.Series) -> Optional[str]:
    """Infer a d3-format string based on raw value patterns."""
    if series.empty:
        return None
    sample = series.dropna()
    if sample.empty:
        return None

    text = sample.astype("string").str.strip()
    if text.empty:
        return None

    n = len(text)
    if n == 0:
        return None

    def ratio(mask: pd.Series) -> float:
        try:
            return float(mask.mean())
        except Exception:
            return 0.0

    dollar = ratio(text.str.contains(r"\\$"))
    euro = ratio(text.str.contains(r"€"))
    pound = ratio(text.str.contains(r"£"))
    if max(dollar, euro, pound) >= 0.2:
        if dollar >= max(euro, pound):
            return "$~s"
        if euro >= pound:
            return "€~s"
        return "£~s"

    if ratio(text.str.contains(r"%$")) >= 0.2:
        return ".0%"

    if ratio(text.str.contains(r"(?i)\\d\\s*[kmbt]\\b")) >= 0.2:
        return "~s"

    return None
