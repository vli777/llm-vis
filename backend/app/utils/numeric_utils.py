from typing import Optional

import pandas as pd

from core.utils import smart_numeric_series, smart_numeric_value  # noqa: F401


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


def compact_si_label_expr(axis_format: str) -> Optional[str]:
    """Build a Vega labelExpr to display billions as B instead of G."""
    if "~s" not in axis_format:
        return None
    prefix = ""
    if "$" in axis_format:
        prefix = "$"
    elif "€" in axis_format:
        prefix = "€"
    elif "£" in axis_format:
        prefix = "£"
    fmt = ".3~s"
    expr = f"replace(format(datum.value, '{fmt}'), 'G', 'B')"
    if prefix:
        expr = f"'{prefix}' + {expr}"
    return expr
