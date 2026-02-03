"""
Dataset profiling skill.

Extracted from nlq_llm.py — builds a DataProfile from a DataFrame.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import pandas as pd

from core.models import ColumnInfo, ColumnRole, DataProfile
from core.utils import example_values, pct, smart_numeric_series, topk_counts

PROFILE_MAX_ROWS = 2000


# ---------------------------------------------------------------------------
# Column role detection
# ---------------------------------------------------------------------------

def detect_column_role(col_name: str, series: pd.Series, unique_ratio: float) -> ColumnRole:
    """Infer semantic role of a column based on name and characteristics."""
    name_lower = col_name.lower()

    # Temporal indicators (highest priority)
    if any(word in name_lower for word in ["date", "time", "year", "month", "day", "timestamp"]):
        return ColumnRole.temporal
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnRole.temporal

    # Geographic indicators
    if any(word in name_lower for word in [
        "country", "city", "state", "region", "location", "geo",
        "lat", "lon", "latitude", "longitude",
    ]):
        if unique_ratio < 0.5 or any(w in name_lower for w in ["country", "city", "state"]):
            return ColumnRole.geographic

    # Numeric columns
    if pd.api.types.is_numeric_dtype(series):
        if any(word in name_lower for word in [
            "amount", "price", "value", "revenue", "cost", "sales",
            "total", "profit", "income", "expense", "valuation", "arr",
        ]):
            return ColumnRole.measure
        if any(word in name_lower for word in ["count", "quantity", "number", "num", "qty"]):
            return ColumnRole.count
        if "id" in name_lower and unique_ratio > 0.9:
            return ColumnRole.identifier
        return ColumnRole.measure

    # String-encoded numerics (e.g. "$1.3B", "1,234.56")
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        sample = series.dropna().head(min(100, len(series)))
        if len(sample) > 0:
            parsed = smart_numeric_series(sample)
            parseable_ratio = parsed.notna().sum() / len(sample)
            if parseable_ratio > 0.5:
                if any(word in name_lower for word in [
                    "amount", "price", "value", "revenue", "cost", "sales",
                    "total", "profit", "income", "expense", "valuation", "arr",
                ]):
                    return ColumnRole.measure
                if parseable_ratio > 0.8:
                    return ColumnRole.measure

    # Categorical indicators
    if unique_ratio < 0.05:
        return ColumnRole.categorical
    if unique_ratio > 0.9:
        return ColumnRole.identifier

    return ColumnRole.nominal


# ---------------------------------------------------------------------------
# Column stats collection
# ---------------------------------------------------------------------------

def collect_column_stats(
    df: pd.DataFrame,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Separate columns into numeric (>50 % parseable) and categorical."""
    total = len(df)
    numeric_info: List[Dict[str, Any]] = []
    categorical_info: List[Dict[str, Any]] = []

    for col in df.columns:
        series = df[col]
        numeric_series = smart_numeric_series(series)
        non_na = numeric_series.notna().sum()
        parseable_ratio = non_na / total if total > 0 else 0.0

        if total > 0 and non_na >= max(3, int(total * 0.05)) and parseable_ratio > 0.5:
            variance = float(numeric_series.var(skipna=True)) if non_na >= 2 else 0.0
            if not math.isfinite(variance):
                variance = 0.0
            numeric_info.append({
                "name": col,
                "variance": variance,
                "tokens": set(col.lower().split()),
            })
            continue

        unique = series.astype("string").nunique(dropna=True)
        ratio = unique / total if total else 0.0
        categorical_info.append({
            "name": col,
            "unique": unique,
            "ratio": ratio,
            "tokens": set(col.lower().split()),
        })

    return numeric_info, categorical_info


# ---------------------------------------------------------------------------
# Chart type suggestions
# ---------------------------------------------------------------------------

def suggest_chart_types(
    numeric_info: List[Dict[str, Any]],
    categorical_info: List[Dict[str, Any]],
    has_temporal: bool,
) -> List[str]:
    """Return top-4 chart type suggestions based on column characteristics."""
    suggestions: List[str] = []

    if has_temporal:
        suggestions.append("line chart (temporal trends)")
    if len(numeric_info) >= 2:
        suggestions.append("scatter plot (correlations)")
    if categorical_info and numeric_info:
        suggestions.append("bar chart (category comparisons)")

    low_card = [c for c in categorical_info if c.get("unique", 0) <= 10]
    if low_card:
        suggestions.append("pie/donut chart (part-to-whole)")

    high_card = [c for c in categorical_info if c.get("unique", 0) > 10]
    if high_card:
        suggestions.append("horizontal bar chart (many categories)")

    if numeric_info:
        suggestions.append("histogram (distribution)")

    return suggestions[:4]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_profile(
    df: pd.DataFrame,
    table_name: str,
    *,
    max_cols: int = 30,
    include_quants: bool = True,
) -> DataProfile:
    """
    Build a DataProfile from a DataFrame.

    This is the single entry-point that replaces `dataset_profile()` from
    nlq_llm.py, but returns a typed Pydantic model.
    """
    nrows = len(df)
    work_df = df.sample(PROFILE_MAX_ROWS, random_state=0) if nrows > PROFILE_MAX_ROWS else df
    work_rows = len(work_df)

    columns: List[ColumnInfo] = []
    has_temporal = False
    numeric_cols_info: List[Dict[str, Any]] = []
    categorical_cols_info: List[Dict[str, Any]] = []

    for i, c in enumerate(df.columns):
        if i >= max_cols:
            break

        s = df[c]
        s_sample = work_df[c] if c in work_df.columns else s
        dtype = str(s.dtype)
        missing = int(s.isna().sum())
        unique = int(s_sample.nunique(dropna=True))
        unique_ratio = unique / nrows if nrows > 0 else 0.0

        role = detect_column_role(c, s, unique_ratio)
        if role == ColumnRole.temporal:
            has_temporal = True

        warnings_list: List[str] = []
        stats_dict = None
        top_vals = None

        # Numeric stats
        s_num = smart_numeric_series(s_sample)
        parseable_count = s_num.notna().sum()
        is_parseable_numeric = (parseable_count / work_rows) > 0.5 if work_rows > 0 else False

        if pd.api.types.is_numeric_dtype(s) or is_parseable_numeric:
            numeric_cols_info.append({
                "name": c,
                "variance": float(s_num.var(skipna=True)) if len(s_num.dropna()) >= 2 else 0.0,
            })
            stats_dict = {
                "min": float(s_num.min()) if parseable_count > 0 else None,
                "max": float(s_num.max()) if parseable_count > 0 else None,
                "mean": float(s_num.mean()) if parseable_count > 0 else None,
                "std": float(s_num.std()) if parseable_count > 0 else None,
            }
            if include_quants and parseable_count > 0:
                qs = s_num.quantile([0.25, 0.5, 0.75]).to_dict()
                stats_dict.update({
                    f"q{int(q * 100)}": float(v)
                    for q, v in qs.items()
                    if not math.isnan(v)
                })
            if not pd.api.types.is_numeric_dtype(s) and is_parseable_numeric:
                warnings_list.append("string-encoded numeric (will be parsed)")
        elif pd.api.types.is_datetime64_any_dtype(s):
            has_temporal = True
            s_dt = pd.to_datetime(s_sample, errors="coerce")
            stats_dict = {
                "min": s_dt.min().isoformat() if s_dt.notna().any() else None,
                "max": s_dt.max().isoformat() if s_dt.notna().any() else None,
            }
        else:
            categorical_cols_info.append({"name": c, "unique": unique, "ratio": unique_ratio})
            top_vals = topk_counts(s_sample, 5)

        col_info = ColumnInfo(
            name=c,
            dtype=dtype,
            role=role,
            cardinality=unique,
            missing_pct=pct(missing, nrows),
            stats=stats_dict,
            top_values=top_vals,
            examples=example_values(s_sample, 3),
            warnings=warnings_list or None,
        )
        columns.append(col_info)

    # Build visualization hints
    viz_hints = {
        "suggested_chart_types": suggest_chart_types(numeric_cols_info, categorical_cols_info, has_temporal),
        "summary": {
            "numeric_columns": len(numeric_cols_info),
            "categorical_columns": len(categorical_cols_info),
            "has_temporal_data": has_temporal,
            "total_columns": len(columns),
        },
    }

    # Sample rows
    sample_rows = None
    if work_rows > 0:
        sample_rows = (
            work_df.sample(min(3, work_rows), random_state=0)
            .astype(str)
            .map(lambda s: s[:80])
            .to_dict(orient="records")
        )

    return DataProfile(
        table_name=table_name,
        row_count=nrows,
        columns=columns,
        sample_rows=sample_rows,
        visualization_hints=viz_hints,
    )


# ---------------------------------------------------------------------------
# Backward-compatible wrapper (so existing tests keep working)
# ---------------------------------------------------------------------------

def dataset_profile(
    df: pd.DataFrame,
    *,
    max_cols: int = 30,
    include_quants: bool = True,
    include_viz_hints: bool = True,
) -> Dict[str, Any]:
    """
    Legacy dict-based profile matching the original nlq_llm.py signature.

    Existing tests and llm.py call this; it delegates to build_profile()
    and converts back to the expected dict shape.
    """
    profile = build_profile(df, "unknown", max_cols=max_cols, include_quants=include_quants)

    cols_dicts: List[Dict[str, Any]] = []
    for col in profile.columns:
        info: Dict[str, Any] = {
            "name": col.name,
            "dtype": col.dtype,
            "missing_pct": col.missing_pct,
            "unique": col.cardinality,
            "unique_ratio": round(col.cardinality / profile.row_count, 3) if profile.row_count else 0.0,
            "examples": col.examples or [],
        }
        if include_viz_hints:
            info["role"] = col.role.value

        if col.stats:
            # If it looks like numeric stats (has 'mean'), put under num_stats
            if "mean" in col.stats:
                info["num_stats"] = col.stats
            else:
                # datetime range
                info["datetime_range"] = col.stats

        if col.top_values is not None:
            info["top_values"] = col.top_values

        if col.warnings:
            for w in col.warnings:
                if "string-encoded numeric" in w:
                    info["note"] = w

        cols_dicts.append(info)

    result: Dict[str, Any] = {
        "row_count": profile.row_count,
        "columns": cols_dicts,
    }

    if include_viz_hints and profile.visualization_hints:
        result["visualization_hints"] = profile.visualization_hints

    if profile.sample_rows:
        result["sample_rows"] = profile.sample_rows

    return result
