"""
Shared utility helpers extracted from nlq_llm.py.

Pure functions — no LLM, no I/O, no side effects.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Lowercase text and extract alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# DataFrame safety
# ---------------------------------------------------------------------------

def df_json_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Replace +/-inf -> NaN, then NaN -> None so JSON serialization works."""
    if df.empty:
        return df
    tmp = df.replace([np.inf, -np.inf], np.nan)
    tmp = tmp.astype(object)
    return tmp.where(pd.notna(tmp), None)


def df_to_records_safe(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to list of dicts with JSON-safe values."""
    return df_json_safe(df).to_dict(orient="records")


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------

def resolve_col(name: Optional[str], df: pd.DataFrame) -> Optional[str]:
    """Resolve a column name case-insensitively; tolerate spacing/underscore diffs."""
    if not name:
        return None
    candidates = list(df.columns)
    lower_map = {c.lower(): c for c in candidates}
    key = name.lower()
    if key in lower_map:
        return lower_map[key]

    def norm(s: str) -> str:
        return re.sub(r"[\s_\-]+", "", s.lower())

    target = norm(name)
    for c in candidates:
        if norm(c) == target:
            return c
    return name if name in df.columns else None


def fix_field(value, df: pd.DataFrame):
    """Resolve a single field reference against the DataFrame columns."""
    if not isinstance(value, str):
        return value
    resolved = resolve_col(value, df)
    return resolved or value


# ---------------------------------------------------------------------------
# Smart numeric parsing (currency, SI suffixes, percentages)
# ---------------------------------------------------------------------------

_SUFFIX_MAP = {
    "k": 1_000.0,
    "m": 1_000_000.0,
    "mm": 1_000_000.0,
    "b": 1_000_000_000.0,
    "bn": 1_000_000_000.0,
    "t": 1_000_000_000_000.0,
}


def smart_numeric_value(val) -> float:
    """Parse a single value that might be currency, SI-suffixed, or percentage."""
    if val is None:
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    text = str(val).strip()
    if not text:
        return np.nan
    text = text.replace(",", "").replace("$", "").replace("\u20ac", "")
    if text.endswith("%"):
        inner = text[:-1].strip()
        try:
            return float(inner) / 100.0
        except ValueError:
            return np.nan
    lower = text.lower()
    if lower in {"n/a", "na", "nan", "none", "null", "-", "--", "\u2014"}:
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
    """Coerce a Series to numeric, parsing currency/SI/percent strings."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    converted = series.map(smart_numeric_value)
    return pd.to_numeric(converted, errors="coerce")


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def pct(n: int, d: int) -> float:
    """Percentage with 2-decimal rounding; zero-safe."""
    return 0.0 if d <= 0 else round(100.0 * n / d, 2)


def nice_bounds(lo: float, hi: float) -> Optional[Tuple[float, float]]:
    """Compute aesthetically rounded axis bounds."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    if lo == hi:
        span = abs(lo) if lo else 1.0
        lo -= span * 0.05 + 1.0
        hi += span * 0.05 + 1.0

    span = hi - lo
    if span <= 0:
        span = abs(lo) if lo else 1.0
    buffer = max(span * 0.05, 1.0)
    lo_adj = lo - buffer
    hi_adj = hi + buffer
    rng = hi_adj - lo_adj
    if rng <= 0:
        rng = abs(lo_adj) if lo_adj else 1.0
    exponent = math.floor(math.log10(rng)) if rng > 0 else 0
    step = 10 ** exponent
    nice_lo = math.floor(lo_adj / step) * step
    nice_hi = math.ceil(hi_adj / step) * step
    return nice_lo, nice_hi


# ---------------------------------------------------------------------------
# Sample / top-k helpers
# ---------------------------------------------------------------------------

def example_values(s: pd.Series, k: int = 3) -> List[str]:
    """Return up to *k* unique non-null example values as short strings."""
    vals = s.dropna().unique()[:k]
    return [str(v)[:80] for v in vals]


def topk_counts(s: pd.Series, k: int = 5) -> list[dict]:
    """Top-k value counts as [{value, n}, ...]."""
    vc = s.astype("string").fillna("\u2205").value_counts().head(k)
    return [{"value": str(i)[:80], "n": int(v)} for i, v in vc.items()]
