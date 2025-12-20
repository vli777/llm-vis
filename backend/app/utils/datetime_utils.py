from typing import Dict, Optional

import pandas as pd


def infer_temporal_granularity(series: pd.Series) -> Optional[Dict[str, str]]:
    """Infer temporal granularity and return format + timeUnit for Vega-Lite."""
    if series.empty:
        return None

    s = series.dropna()
    if s.empty:
        return None

    if pd.api.types.is_datetime64_any_dtype(series):
        valid = s
        times = valid.dt.normalize()
        has_time = (valid != times).any()
        if has_time:
            return None
        years = valid.dt.year
        if (years < 1000).any() or (years > 3000).any():
            return None
        months = valid.dt.month
        days = valid.dt.day
        if (months == 1).all() and (days == 1).all():
            return {"format": "%Y", "timeUnit": "year"}
        if (days == 1).all():
            return {"format": "%Y-%m", "timeUnit": "yearmonth"}
        return {"format": "%Y-%m-%d", "timeUnit": "yearmonthdate"}

    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        text = s.astype("string").str.strip()
        if text.empty:
            return None
        if text.str.match(r"^\\d{4}$").all():
            return {"format": "%Y", "timeUnit": "year"}
        if text.str.match(r"^\\d{4}[-/]\\d{1,2}$").all():
            return {"format": "%Y-%m", "timeUnit": "yearmonth"}
        if text.str.match(r"^\\d{4}[-/]\\d{1,2}[-/]\\d{1,2}$").all():
            return {"format": "%Y-%m-%d", "timeUnit": "yearmonthdate"}

    return None


def maybe_coerce_year_temporal(series: pd.Series) -> Optional[pd.Series]:
    """Coerce year-like values into datetime (year precision) for temporal charts."""
    if series.empty:
        return None
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce")
        valid = vals.dropna()
        if valid.empty:
            return None
        if not (valid.round() == valid).all():
            return None
        if not ((valid >= 1000) & (valid <= 3000)).all():
            return None
        years = pd.to_numeric(series, errors="coerce").round().astype("Int64")
        return pd.to_datetime(years.astype("string"), format="%Y", errors="coerce")

    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        year_str = series.astype("string").str.extract(r"(\\d{4})", expand=False)
        dt = pd.to_datetime(year_str, format="%Y", errors="coerce")
        if dt.isna().all():
            return None
        return dt

    return None
