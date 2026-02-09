"""
View builder skill.

Takes a ViewPlan + DataFrame → ViewResult with pre-aggregated data_inline.
The frontend simply renders what it receives — no computation needed there.

Chart data contract (data_inline):
- bar: rows contain x field + y field (aggregated), plus optional color field.
- line: rows contain x field + y field, optional color field for series.
- scatter: rows contain x field + y field, optional color field.
- hist: rows contain bin_label, bin_start, bin_end, count.
- box: rows contain group, min, q1, median, q3, max, outliers.
- heatmap: rows contain x field, y field, count.
- pie: rows contain name field + value field (encoding.color/theta or inferred).
- table: rows are raw/preview records with any fields.
"""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.models import (
    ChartEncoding,
    ChartOptions,
    ChartSpec,
    ChartType,
    EncodingChannel,
    ViewPlan,
    ViewResult,
)
from core.utils import df_to_records_safe, resolve_col, smart_numeric_series

logger = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------------------------
# Encoding reconciliation — ensure field names match actual data keys
# ---------------------------------------------------------------------------

def _reconcile_encoding(
    chart_type: ChartType,
    encoding: ChartEncoding,
    data: List[Dict[str, Any]],
) -> ChartEncoding:
    """Fix encoding field references to match actual data column names.

    Builders sometimes produce columns with different names than what the
    plan's encoding declares (e.g. count aggregation creates a ``"count"``
    column while the encoding still says ``y.field = original_col``).  The
    frontend reads ``spec.encoding.*.field`` to map data keys, so a mismatch
    causes empty charts.
    """
    if not data or chart_type in (ChartType.hist, ChartType.box, ChartType.table):
        return encoding

    keys = set(data[0].keys())
    x_field = encoding.x.field if encoding.x else None

    new_y = encoding.y
    new_theta = encoding.theta
    new_color = encoding.color

    # y field: either missing from data, or same as x with count aggregation
    if encoding.y and encoding.y.field:
        y = encoding.y
        agg = (y.aggregate or "").lower()
        if y.field not in keys:
            replacement = _find_numeric_key(
                keys, exclude={x_field} if x_field else set(), data=data,
            )
            if replacement:
                new_y = EncodingChannel(
                    field=replacement, type=y.type,
                    aggregate=y.aggregate, bin=y.bin,
                )
        elif agg == "count" and y.field == x_field and "count" in keys:
            # Count aggregation output: data column is "count", not the category
            new_y = EncodingChannel(
                field="count", type=y.type,
                aggregate=y.aggregate, bin=y.bin,
            )

    # theta field: pie chart count aggregation
    if encoding.theta and encoding.theta.field:
        theta = encoding.theta
        agg = (theta.aggregate or "").lower()
        cat_field = encoding.color.field if encoding.color else None
        if theta.field not in keys:
            exclude = {cat_field} if cat_field and cat_field in keys else set()
            replacement = _find_numeric_key(keys, exclude=exclude, data=data)
            if replacement:
                new_theta = EncodingChannel(
                    field=replacement, type=theta.type,
                    aggregate=theta.aggregate, bin=theta.bin,
                )
        elif agg == "count" and theta.field == cat_field and "count" in keys:
            new_theta = EncodingChannel(
                field="count", type=theta.type,
                aggregate=theta.aggregate, bin=theta.bin,
            )

    # color (quantitative) field not in data — heatmap count
    if (encoding.color and encoding.color.field
            and encoding.color.type == "quantitative"
            and encoding.color.field not in keys):
        if "count" in keys:
            new_color = EncodingChannel(
                field="count", type=encoding.color.type,
                aggregate=encoding.color.aggregate,
            )

    if new_y is encoding.y and new_theta is encoding.theta and new_color is encoding.color:
        return encoding

    return ChartEncoding(
        x=encoding.x, y=new_y, color=new_color,
        theta=new_theta, facet=encoding.facet, size=encoding.size,
    )


def _find_numeric_key(
    keys: set, exclude: set, data: List[Dict[str, Any]],
) -> Optional[str]:
    """Find a numeric column in data, preferring ``"count"``."""
    if "count" in keys and "count" not in exclude:
        return "count"
    for k in keys:
        if k not in exclude and isinstance(data[0].get(k), (int, float)):
            return k
    return None


# ---------------------------------------------------------------------------
# Per-type data builders
# ---------------------------------------------------------------------------

def _resolve(field: Optional[str], df: pd.DataFrame) -> Optional[str]:
    """Resolve a field name against the DataFrame, returning None if not found."""
    if field is None:
        return None
    r = resolve_col(field, df)
    return r if r and r in df.columns else None


def _coerce_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if pd.api.types.is_numeric_dtype(df[col]):
        return pd.to_numeric(df[col], errors="coerce")
    return smart_numeric_series(df[col])


def _safe_stats(series: pd.Series) -> Dict[str, Any]:
    valid = series.dropna()
    if valid.empty:
        return {}
    return {
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "median": float(valid.median()),
    }


def _aggregate_bar(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Group-by aggregation for bar charts."""
    x_enc = plan.encoding.x
    y_enc = plan.encoding.y
    if not x_enc or not y_enc:
        return []

    x_field = _resolve(x_enc.field, df)
    y_field = _resolve(y_enc.field, df)
    if not x_field:
        return []

    agg = (y_enc.aggregate or "").lower() if y_enc else ""

    if agg == "count" or y_field is None or y_field == x_field:
        # value counts
        grouped = df[x_field].value_counts().reset_index()
        grouped.columns = [x_field, "count"]
        y_col = "count"
    else:
        # real aggregation
        y_series = _coerce_numeric(df, y_field)
        tmp = df[[x_field]].copy()
        tmp["__y__"] = y_series
        agg_fn = agg if agg in ("sum", "mean", "median", "min", "max") else "mean"
        grouped = tmp.groupby(x_field, dropna=False)["__y__"].agg(agg_fn).reset_index()
        grouped.columns = [x_field, y_field]
        y_col = y_field

    # Sort / top-n
    if plan.options.sort == "descending":
        grouped = grouped.sort_values(y_col, ascending=False)
    elif plan.options.sort == "ascending":
        grouped = grouped.sort_values(y_col, ascending=True)
    else:
        grouped = grouped.sort_values(y_col, ascending=False)

    if plan.options.top_n:
        grouped = grouped.head(plan.options.top_n)

    return df_to_records_safe(grouped)


def _aggregate_line(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Time-series or sequential line data."""
    x_enc = plan.encoding.x
    y_enc = plan.encoding.y
    if not x_enc or not y_enc:
        return []

    x_field = _resolve(x_enc.field, df)
    y_field = _resolve(y_enc.field, df)
    if not x_field or not y_field:
        return []

    tmp = df[[x_field]].copy()
    tmp[y_field] = _coerce_numeric(df, y_field)
    tmp = tmp.dropna(subset=[y_field])

    # Include color/group field if present
    color_enc = plan.encoding.color
    c_field = None
    if color_enc:
        c_field = _resolve(color_enc.field, df)
        if c_field and c_field in df.columns:
            tmp[c_field] = df.loc[tmp.index, c_field]

    # Group if temporal: aggregate by the temporal unit
    if x_enc.type == "temporal":
        try:
            x_series = tmp[x_field]
            # Detect numeric year-like values (e.g. 2020, 2021) — pd.to_datetime
            # treats bare integers as nanosecond timestamps, producing wrong dates.
            if pd.api.types.is_numeric_dtype(x_series):
                valid_x = x_series.dropna()
                if (not valid_x.empty
                        and valid_x.min() >= 1900 and valid_x.max() <= 2100):
                    # Treat as year labels — group by year
                    tmp[x_field] = x_series.astype(int)
                else:
                    # Non-year numeric → try datetime parsing
                    tmp[x_field] = pd.to_datetime(x_series, unit="s", errors="coerce")
                    tmp = tmp.dropna(subset=[x_field])
                    tmp[x_field] = tmp[x_field].dt.strftime("%Y-%m-%dT%H:%M:%S")
            else:
                tmp[x_field] = pd.to_datetime(tmp[x_field], errors="coerce")
                tmp = tmp.dropna(subset=[x_field])
                tmp[x_field] = tmp[x_field].dt.strftime("%Y-%m-%dT%H:%M:%S")
        except Exception:
            tmp = tmp.sort_values(x_field)
        # Aggregate by temporal x (and optional color group)
        agg = (y_enc.aggregate or "mean").lower()
        if agg not in {"sum", "mean", "min", "max", "median", "count"}:
            agg = "mean"
        group_cols = [x_field]
        if c_field:
            group_cols.append(c_field)
        tmp = tmp.groupby(group_cols, dropna=False)[y_field].agg(agg).reset_index()
        tmp = tmp.sort_values(x_field)
    else:
        tmp = tmp.sort_values(x_field)

    if len(tmp) > 2000:
        tmp = tmp.head(2000)

    return df_to_records_safe(tmp)


def _build_scatter(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Scatter data — coerce numeric, drop NaN."""
    x_enc = plan.encoding.x
    y_enc = plan.encoding.y
    if not x_enc or not y_enc:
        return []

    x_field = _resolve(x_enc.field, df)
    y_field = _resolve(y_enc.field, df)
    if not x_field or not y_field:
        return []

    cols = [x_field, y_field]
    tmp = df[[]].copy()
    tmp[x_field] = _coerce_numeric(df, x_field)
    tmp[y_field] = _coerce_numeric(df, y_field)

    # Optional color
    color_enc = plan.encoding.color
    if color_enc:
        c_field = _resolve(color_enc.field, df)
        if c_field and c_field in df.columns:
            tmp[c_field] = df[c_field]
            cols.append(c_field)

    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_field, y_field])

    if plan.options.log:
        tmp = tmp[(tmp[x_field] > 0) & (tmp[y_field] > 0)]
        tmp[x_field] = np.log10(tmp[x_field])
        tmp[y_field] = np.log10(tmp[y_field])

    if len(tmp) > 5000:
        tmp = tmp.sample(5000, random_state=0)

    return df_to_records_safe(tmp)


def _build_histogram(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Histogram — bin numeric data into percentiles/deciles."""
    x_enc = plan.encoding.x
    if not x_enc:
        return []

    x_field = _resolve(x_enc.field, df)
    if not x_field:
        return []

    series = _coerce_numeric(df, x_field).dropna()
    if series.empty:
        return []

    total = len(series)
    if total == 0:
        return []

    bin_count = plan.options.bin_count or 20
    records: List[Dict[str, Any]] = []

    try:
        p1 = float(np.percentile(series, 1))
        p99 = float(np.percentile(series, 99))
    except Exception:
        p1, p99 = float(series.min()), float(series.max())

    # Tail bins
    below = series[series < p1]
    above = series[series > p99]
    mid = series[(series >= p1) & (series <= p99)]

    if len(below) > 0:
        records.append({
            "bin_start": float(series.min()),
            "bin_end": p1,
            "bin_label": "<P1",
            "count": int(len(below)),
            "percent": round(100.0 * len(below) / total, 2),
        })

    mid_bins = max(1, bin_count - 2)
    counts, edges = np.histogram(mid, bins=mid_bins)
    for i, count in enumerate(counts):
        records.append({
            "bin_start": float(edges[i]),
            "bin_end": float(edges[i + 1]),
            "bin_label": f"B{i+1}: {edges[i]:.6g}-{edges[i+1]:.6g}",
            "count": int(count),
            "percent": round(100.0 * count / total, 2),
        })

    if len(above) > 0:
        records.append({
            "bin_start": p99,
            "bin_end": float(series.max()),
            "bin_label": ">P99",
            "count": int(len(above)),
            "percent": round(100.0 * len(above) / total, 2),
        })

    # Add percentile markers (used by frontend for reference lines)
    try:
        percentiles = [10, 25, 50, 75, 90]
        markers = []
        for p in percentiles:
            markers.append({
                "value": float(np.percentile(series, p)),
                "label": f"P{p}",
            })
        if markers:
            plan.options.markers = markers
    except Exception:
        pass

    return records


def _build_box(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Box plot stats — precompute quartiles, whiskers."""
    y_enc = plan.encoding.y
    x_enc = plan.encoding.x
    if not y_enc:
        return []

    y_field = _resolve(y_enc.field, df)
    if not y_field:
        return []

    series = _coerce_numeric(df, y_field)

    groups: List[tuple] = []
    if x_enc:
        x_field = _resolve(x_enc.field, df)
        if x_field and x_field in df.columns:
            for name, group in df.groupby(x_field, dropna=False):
                g_series = _coerce_numeric(group, y_field).dropna()
                if not g_series.empty:
                    groups.append((str(name), g_series))

    if not groups:
        valid = series.dropna()
        if not valid.empty:
            groups = [("all", valid)]

    records = []
    for label, g in groups:
        q1 = float(g.quantile(0.25))
        median = float(g.quantile(0.5))
        q3 = float(g.quantile(0.75))
        iqr = q3 - q1
        whisker_lo = float(g[g >= q1 - 1.5 * iqr].min())
        whisker_hi = float(g[g <= q3 + 1.5 * iqr].max())
        outliers = g[(g < whisker_lo) | (g > whisker_hi)].tolist()

        records.append({
            "group": label,
            "min": whisker_lo,
            "q1": q1,
            "median": median,
            "q3": q3,
            "max": whisker_hi,
            "outliers": [float(o) for o in outliers[:50]],  # cap outlier list
        })

    return records


def _build_table(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Summary table — df.describe()-style stats when tagged as summary."""
    if "summary" in plan.tags:
        return _build_describe_table(df)

    fields = plan.fields_used or list(df.columns[:10])
    resolved = [_resolve(f, df) for f in fields]
    cols = [c for c in resolved if c and c in df.columns]
    if not cols:
        cols = list(df.columns[:10])

    subset = df[cols].head(100)
    return df_to_records_safe(subset)


def _tag_value(tags: List[str], key: str) -> Optional[str]:
    prefix = f"{key}="
    for t in tags:
        if t.startswith(prefix):
            return t[len(prefix):]
    return None


def _parse_percentile_ranges(tag_value: Optional[str]) -> List[tuple[int, int]]:
    if not tag_value:
        return []
    ranges = []
    for part in tag_value.split(";"):
        part = part.strip()
        if not part or "-" not in part:
            continue
        lo_str, hi_str = part.split("-", 1)
        try:
            lo = int(lo_str)
            hi = int(hi_str)
        except ValueError:
            continue
        if lo < 0 or hi > 100:
            continue
        if lo > hi:
            lo, hi = hi, lo
        ranges.append((lo, hi))
    return ranges


def _coerce_time(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    try:
        return pd.to_datetime(series, errors="coerce")
    except Exception:
        return series


def _build_percentile_compare(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Compare percentile groups for a metric, optionally across time."""
    metric = resolve_col(_tag_value(plan.tags, "metric"), df)
    temporal = resolve_col(_tag_value(plan.tags, "temporal"), df)
    entity = resolve_col(_tag_value(plan.tags, "entity"), df)
    compare = _tag_value(plan.tags, "compare") or ""
    ranges = _parse_percentile_ranges(_tag_value(plan.tags, "percentiles"))

    if not metric or metric not in df.columns or not ranges:
        return []

    base = df[[metric]].copy()
    base[metric] = smart_numeric_series(base[metric])
    base = base.dropna(subset=[metric])
    if base.empty:
        return []

    total = len(base)
    values = base[metric]

    def group_stats(series: pd.Series, label: str) -> Dict[str, Any]:
        s = series.dropna()
        if s.empty:
            return {"group": label, "count": 0, "mean": None, "median": None, "pct_of_rows": 0.0}
        return {
            "group": label,
            "count": int(len(s)),
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "pct_of_rows": round(100.0 * len(s) / max(1, total), 2),
        }

    # If we can compute change across time, do so
    if compare == "change" and temporal and temporal in df.columns:
        work_cols = [metric, temporal]
        if entity and entity in df.columns:
            work_cols.append(entity)
        work = df[work_cols].copy()
        work[metric] = smart_numeric_series(work[metric])
        work = work.dropna(subset=[metric])
        if work.empty:
            return []
        work["_time"] = _coerce_time(work[temporal])
        work = work.dropna(subset=["_time"])
        if work.empty:
            return []

        records: List[Dict[str, Any]] = []
        if entity and entity in work.columns:
            idx_min = work.groupby(entity)["_time"].idxmin()
            idx_max = work.groupby(entity)["_time"].idxmax()
            base_df = work.loc[idx_min, [entity, metric]].set_index(entity)
            last_df = work.loc[idx_max, [entity, metric]].set_index(entity)
            joined = base_df.join(last_df, lsuffix="_base", rsuffix="_last", how="inner")
            if joined.empty:
                return []
            joined["change"] = joined[f"{metric}_last"] - joined[f"{metric}_base"]
            metric_series = joined[f"{metric}_base"]
            for lo, hi in ranges:
                lo_v = float(np.percentile(metric_series, lo))
                hi_v = float(np.percentile(metric_series, hi))
                mask = (metric_series >= lo_v) & (metric_series <= hi_v)
                subset = joined.loc[mask]
                stats = group_stats(subset["change"], f"P{lo}-{hi}")
                stats["mean_baseline"] = round(float(subset[f"{metric}_base"].mean()), 4) if len(subset) else None
                stats["mean_latest"] = round(float(subset[f"{metric}_last"].mean()), 4) if len(subset) else None
                stats["mean_change"] = stats.pop("mean")
                stats["median_change"] = stats.pop("median")
                records.append(stats)
            return records

        # No entity: compare first vs last time using same percentile bounds
        time_vals = work["_time"].dropna().sort_values()
        if time_vals.empty:
            return []
        first_t = time_vals.iloc[0]
        last_t = time_vals.iloc[-1]
        base_t = work[work["_time"] == first_t][metric].dropna()
        last_t_series = work[work["_time"] == last_t][metric].dropna()
        if base_t.empty or last_t_series.empty:
            return []
        for lo, hi in ranges:
            lo_v = float(np.percentile(base_t, lo))
            hi_v = float(np.percentile(base_t, hi))
            base_subset = base_t[(base_t >= lo_v) & (base_t <= hi_v)]
            last_subset = last_t_series[(last_t_series >= lo_v) & (last_t_series <= hi_v)]
            stats = group_stats(last_subset - base_subset.mean(), f"P{lo}-{hi}")
            stats["mean_baseline"] = round(float(base_subset.mean()), 4) if len(base_subset) else None
            stats["mean_latest"] = round(float(last_subset.mean()), 4) if len(last_subset) else None
            stats["mean_change"] = stats.pop("mean")
            stats["median_change"] = stats.pop("median")
            records.append(stats)
        return records

    # Fallback: compare levels without change
    records = []
    for lo, hi in ranges:
        lo_v = float(np.percentile(values, lo))
        hi_v = float(np.percentile(values, hi))
        subset = values[(values >= lo_v) & (values <= hi_v)]
        records.append(group_stats(subset, f"P{lo}-{hi}"))
    return records


def _build_linear_regression(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    x_field = _tag_value(plan.tags, "x") or (plan.encoding.x.field if plan.encoding.x else None)
    y_field = _tag_value(plan.tags, "y") or (plan.encoding.y.field if plan.encoding.y else None)
    x_field = resolve_col(x_field, df)
    y_field = resolve_col(y_field, df)
    if not x_field or not y_field:
        return []
    if x_field not in df.columns or y_field not in df.columns:
        return []

    x = smart_numeric_series(df[x_field])
    y = smart_numeric_series(df[y_field])
    mask = (~x.isna()) & (~y.isna())
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return []

    X = np.column_stack([np.ones(len(x)), x.to_numpy()])
    beta, *_ = np.linalg.lstsq(X, y.to_numpy(), rcond=None)
    y_pred = X @ beta
    ss_res = float(np.sum((y.to_numpy() - y_pred) ** 2))
    ss_tot = float(np.sum((y.to_numpy() - y.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return [
        {"type": "coef", "term": "intercept", "value": round(float(beta[0]), 6)},
        {"type": "coef", "term": x_field, "value": round(float(beta[1]), 6)},
        {"type": "metric", "term": "r2", "value": round(r2, 6)},
        {"type": "metric", "term": "n", "value": int(len(x))},
    ]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _build_logistic_regression(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    x_field = _tag_value(plan.tags, "x") or (plan.encoding.x.field if plan.encoding.x else None)
    y_field = _tag_value(plan.tags, "y") or (plan.encoding.y.field if plan.encoding.y else None)
    x_field = resolve_col(x_field, df)
    y_field = resolve_col(y_field, df)
    if not x_field or not y_field:
        return []
    if x_field not in df.columns or y_field not in df.columns:
        return []

    x = smart_numeric_series(df[x_field])
    y_raw = df[y_field]
    y_unique = y_raw.dropna().unique().tolist()
    if len(y_unique) != 2:
        return []
    y_map = {y_unique[0]: 0, y_unique[1]: 1}
    y = y_raw.map(y_map)

    mask = (~x.isna()) & (~y.isna())
    x = x[mask]
    y = y[mask]
    if len(x) < 5:
        return []

    # Standardize predictor
    x_mean = float(x.mean())
    x_std = float(x.std()) or 1.0
    xz = (x - x_mean) / x_std

    # Simple gradient descent
    w0, w1 = 0.0, 0.0
    lr = 0.2
    for _ in range(400):
        z = w0 + w1 * xz.to_numpy()
        p = _sigmoid(z)
        grad0 = float((p - y.to_numpy()).mean())
        grad1 = float(((p - y.to_numpy()) * xz.to_numpy()).mean())
        w0 -= lr * grad0
        w1 -= lr * grad1

    z = w0 + w1 * xz.to_numpy()
    p = _sigmoid(z)
    preds = (p >= 0.5).astype(int)
    acc = float((preds == y.to_numpy()).mean())

    return [
        {"type": "coef", "term": "intercept", "value": round(float(w0), 6)},
        {"type": "coef", "term": x_field, "value": round(float(w1), 6), "note": "standardized x"},
        {"type": "metric", "term": "accuracy", "value": round(acc, 6)},
        {"type": "metric", "term": "n", "value": int(len(x))},
        {"type": "metric", "term": "positive_label", "value": str(y_unique[1])[:50]},
    ]


def _build_segmentation(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    metric = resolve_col(_tag_value(plan.tags, "metric"), df)
    group = resolve_col(_tag_value(plan.tags, "group"), df)
    if not metric or not group:
        return []
    if metric not in df.columns or group not in df.columns:
        return []

    work = df[[metric, group]].copy()
    work[metric] = smart_numeric_series(work[metric])
    work = work.dropna(subset=[metric])
    if work.empty:
        return []

    total = len(work)
    agg = (
        work.groupby(group)[metric]
        .agg(["count", "mean", "median", "sum"])
        .reset_index()
    )
    agg["pct_of_rows"] = agg["count"].apply(lambda v: round(100.0 * v / max(1, total), 2))
    agg = agg.sort_values("mean", ascending=False)

    limit = plan.options.top_n or 15
    if len(agg) > limit:
        agg = agg.head(limit)

    return df_to_records_safe(agg)


def _build_cohort_change(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    metric = resolve_col(_tag_value(plan.tags, "metric"), df)
    temporal = resolve_col(_tag_value(plan.tags, "temporal"), df)
    entity = resolve_col(_tag_value(plan.tags, "entity"), df)
    if not metric or not temporal or not entity:
        return []
    if metric not in df.columns or temporal not in df.columns or entity not in df.columns:
        return []

    work = df[[metric, temporal, entity]].copy()
    work[metric] = smart_numeric_series(work[metric])
    work = work.dropna(subset=[metric])
    if work.empty:
        return []

    work["_time"] = _coerce_time(work[temporal])
    work = work.dropna(subset=["_time"])
    if work.empty:
        return []

    # Cohort by first observed period (year-month)
    work["_period"] = work["_time"].dt.to_period("M").dt.to_timestamp()
    first_period = work.groupby(entity)["_period"].min().rename("cohort")
    work = work.join(first_period, on=entity)

    # Compute change between first and last period per entity
    idx_min = work.groupby(entity)["_period"].idxmin()
    idx_max = work.groupby(entity)["_period"].idxmax()
    base = work.loc[idx_min, [entity, metric, "cohort"]].set_index(entity)
    last = work.loc[idx_max, [entity, metric]].set_index(entity)
    joined = base.join(last, lsuffix="_base", rsuffix="_last", how="inner")
    if joined.empty:
        return []
    joined["change"] = joined[f"{metric}_last"] - joined[f"{metric}_base"]

    agg = (
        joined.groupby("cohort")["change"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .sort_values("cohort")
    )
    agg["cohort"] = agg["cohort"].astype(str)
    return df_to_records_safe(agg)


def _build_group_comparison(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    metric = resolve_col(_tag_value(plan.tags, "metric"), df)
    group = resolve_col(_tag_value(plan.tags, "group"), df)
    if not metric or not group:
        return []
    if metric not in df.columns or group not in df.columns:
        return []

    work = df[[metric, group]].copy()
    work[metric] = smart_numeric_series(work[metric])
    work = work.dropna(subset=[metric, group])
    if work.empty:
        return []

    counts = work[group].value_counts().head(2)
    if len(counts) < 2:
        return []

    g1, g2 = counts.index[0], counts.index[1]
    s1 = work[work[group] == g1][metric]
    s2 = work[work[group] == g2][metric]
    if s1.empty or s2.empty:
        return []

    mean1 = float(s1.mean())
    mean2 = float(s2.mean())
    diff = mean1 - mean2
    pooled = float(np.sqrt((s1.var(ddof=1) + s2.var(ddof=1)) / 2.0)) if (len(s1) > 1 and len(s2) > 1) else 0.0
    d = diff / pooled if pooled else 0.0

    return [
        {"group": str(g1), "mean": round(mean1, 6), "n": int(len(s1))},
        {"group": str(g2), "mean": round(mean2, 6), "n": int(len(s2))},
        {"metric": "diff_in_means", "value": round(diff, 6), "note": "naive (not causal)"},
        {"metric": "cohens_d", "value": round(d, 6)},
    ]


def _build_describe_table(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Build df.describe()-style summary with skew and kurtosis."""
    records: List[Dict[str, Any]] = []

    for col in df.columns:
        row: Dict[str, Any] = {"column": col, "dtype": str(df[col].dtype)}
        total = len(df)
        missing = int(df[col].isna().sum())
        row["count"] = total - missing
        row["missing"] = missing

        # Numeric stats
        num = smart_numeric_series(df[col])
        valid = num.dropna()

        if len(valid) >= 2:
            row["mean"] = round(float(valid.mean()), 4)
            row["std"] = round(float(valid.std()), 4)
            row["min"] = round(float(valid.min()), 4)
            row["25%"] = round(float(valid.quantile(0.25)), 4)
            row["50%"] = round(float(valid.quantile(0.50)), 4)
            row["75%"] = round(float(valid.quantile(0.75)), 4)
            row["max"] = round(float(valid.max()), 4)
            row["skew"] = round(float(valid.skew()), 4)
            row["kurtosis"] = round(float(valid.kurtosis()), 4)
        else:
            # Categorical: show unique count and top value
            nunique = df[col].nunique(dropna=True)
            row["unique"] = nunique
            top = df[col].value_counts().head(1)
            if not top.empty:
                row["top"] = str(top.index[0])[:60]
                row["freq"] = int(top.iloc[0])

        records.append(row)

    return records


def _build_heatmap(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Cross-tab heatmap between two categoricals."""
    x_enc = plan.encoding.x
    y_enc = plan.encoding.y
    if not x_enc or not y_enc:
        return []

    x_field = _resolve(x_enc.field, df)
    y_field = _resolve(y_enc.field, df)
    if not x_field or not y_field:
        return []

    ct = pd.crosstab(df[y_field], df[x_field])
    records = []
    for row_label in ct.index:
        for col_label in ct.columns:
            records.append({
                x_field: str(col_label),
                y_field: str(row_label),
                "count": int(ct.loc[row_label, col_label]),
            })

    return records


def _build_pie(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Pie chart — aggregate category frequencies or use a value column."""
    color_enc = plan.encoding.color
    theta_enc = plan.encoding.theta

    cat_field = None
    if color_enc:
        cat_field = _resolve(color_enc.field, df)
    if not cat_field and plan.encoding.x:
        cat_field = _resolve(plan.encoding.x.field, df)

    if not cat_field:
        return []

    agg = (theta_enc.aggregate or "").lower() if theta_enc else ""

    if agg == "count" or not theta_enc or not theta_enc.field or theta_enc.field == cat_field:
        counts = df[cat_field].value_counts().reset_index()
        counts.columns = [cat_field, "count"]
        return df_to_records_safe(counts)
    else:
        val_field = _resolve(theta_enc.field, df)
        if val_field and val_field in df.columns:
            tmp = df[[cat_field]].copy()
            tmp["__val__"] = _coerce_numeric(df, val_field)
            agg_fn = agg if agg in ("sum", "mean") else "sum"
            grouped = tmp.groupby(cat_field, dropna=False)["__val__"].agg(agg_fn).reset_index()
            grouped.columns = [cat_field, val_field]
            return df_to_records_safe(grouped)

    return []


def _build_area(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Area chart — same logic as line."""
    return _aggregate_line(plan, df)


def _build_missingness(plan: ViewPlan, df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Special builder for the missingness overview chart."""
    records = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        total = len(df)
        pct = round(100.0 * missing / total, 2) if total > 0 else 0.0
        if pct > 0:
            records.append({
                "__column__": col,
                "__missing_pct__": pct,
            })
    records.sort(key=lambda r: r["__missing_pct__"], reverse=True)
    return records


# ---------------------------------------------------------------------------
# Builder dispatch
# ---------------------------------------------------------------------------

_BUILDERS = {
    ChartType.bar: _aggregate_bar,
    ChartType.line: _aggregate_line,
    ChartType.scatter: _build_scatter,
    ChartType.hist: _build_histogram,
    ChartType.box: _build_box,
    ChartType.table: _build_table,
    ChartType.heatmap: _build_heatmap,
    ChartType.pie: _build_pie,
    ChartType.area: _build_area,
}


def build_view(plan: ViewPlan, df: pd.DataFrame) -> ViewResult:
    """
    Build a ViewResult from a ViewPlan and a DataFrame.

    All data is pre-aggregated: the frontend just renders.
    """
    # Special cases: skill-specific table builders
    if "percentile_compare" in plan.tags:
        data = _build_percentile_compare(plan, df)
    elif "segmentation" in plan.tags:
        data = _build_segmentation(plan, df)
    elif "cohort_change" in plan.tags:
        data = _build_cohort_change(plan, df)
    elif "group_comparison" in plan.tags:
        data = _build_group_comparison(plan, df)
    elif "regression_linear" in plan.tags:
        data = _build_linear_regression(plan, df)
    elif "regression_logistic" in plan.tags:
        data = _build_logistic_regression(plan, df)
    elif "__missing_pct__" in plan.fields_used or "missingness" in plan.tags:
        data = _build_missingness(plan, df)
    else:
        builder = _BUILDERS.get(plan.chart_type, _build_table)
        data = builder(plan, df)

    encoding = _reconcile_encoding(plan.chart_type, plan.encoding, data)

    spec = ChartSpec(
        chart_type=plan.chart_type,
        encoding=encoding,
        options=plan.options,
        data_inline=data,
        title=plan.intent or f"{plan.chart_type.value} chart",
    )

    explanation = _auto_explanation(plan, data, df)

    if not data:
        logger.warning(
            "View data empty: chart=%s intent=%s fields=%s",
            plan.chart_type.value, plan.intent, plan.fields_used,
        )
    else:
        logger.info(
            "View built: chart=%s keys=%s intent=%s",
            plan.chart_type.value, list(data[0].keys()), plan.intent,
        )

    return ViewResult(
        plan=plan,
        spec=spec,
        data_inline=data,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Auto-explanation
# ---------------------------------------------------------------------------

def _auto_explanation(plan: ViewPlan, data: List[Dict[str, Any]], df: pd.DataFrame) -> str:
    """Generate a simple textual explanation of what the chart shows."""
    if not data:
        return "No data available for this view."

    ct = plan.chart_type.value
    fields = ", ".join(plan.fields_used) if plan.fields_used else "selected columns"

    parts = []

    if plan.chart_type == ChartType.bar and plan.encoding.y:
        y_field = plan.encoding.y.field
        if y_field and any(y_field in d for d in data[:1]):
            vals = [d.get(y_field) for d in data if d.get(y_field) is not None]
            if vals:
                top_val = data[0]
                x_field = plan.encoding.x.field if plan.encoding.x else None
                if x_field and x_field in top_val:
                    parts.append(f"To compare categories and identify leaders (top: {top_val.get(x_field)}).")

    if plan.chart_type == ChartType.scatter and plan.encoding.x and plan.encoding.y:
        parts.append(f"To assess relationship between {plan.encoding.x.field} and {plan.encoding.y.field}.")

    if plan.chart_type == ChartType.line and plan.encoding.x and plan.encoding.y:
        parts.append(f"To evaluate trends over {plan.encoding.x.field}.")

    if plan.chart_type == ChartType.hist:
        parts.append("To assess distribution shape and skew.")

    if "percentile_compare" in plan.tags:
        parts.append("To compare percentile groups and quantify differences.")
    elif "segmentation" in plan.tags:
        parts.append("To summarize the metric by group and spot segment differences.")
    elif "cohort_change" in plan.tags:
        parts.append("To track how cohorts change over time.")
    elif "group_comparison" in plan.tags:
        parts.append("To compare two groups (naive difference, not causal).")
    elif "regression_linear" in plan.tags:
        parts.append("To estimate a linear relationship between the selected variables.")
    elif "regression_logistic" in plan.tags:
        parts.append("To estimate classification likelihood from the selected predictor.")

    if not parts:
        parts.append(f"Selected {ct} chart to examine {fields}.")

    return " ".join(parts)
