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
                    # Treat as year labels — convert to string for display
                    tmp[x_field] = x_series.astype(int).astype(str)
                    tmp = tmp.sort_values(x_field)
                else:
                    # Non-year numeric → try datetime parsing
                    tmp[x_field] = pd.to_datetime(x_series, unit="s", errors="coerce")
                    tmp = tmp.dropna(subset=[x_field])
                    tmp = tmp.sort_values(x_field)
                    tmp[x_field] = tmp[x_field].dt.strftime("%Y-%m-%dT%H:%M:%S")
            else:
                tmp[x_field] = pd.to_datetime(tmp[x_field], errors="coerce")
                tmp = tmp.dropna(subset=[x_field])
                tmp = tmp.sort_values(x_field)
                tmp[x_field] = tmp[x_field].dt.strftime("%Y-%m-%dT%H:%M:%S")
        except Exception:
            tmp = tmp.sort_values(x_field)
    else:
        tmp = tmp.sort_values(x_field)

    # Include color/group field if present
    color_enc = plan.encoding.color
    if color_enc:
        c_field = _resolve(color_enc.field, df)
        if c_field and c_field in df.columns:
            tmp[c_field] = df.loc[tmp.index, c_field]

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
    """Histogram — bin numeric data."""
    x_enc = plan.encoding.x
    if not x_enc:
        return []

    x_field = _resolve(x_enc.field, df)
    if not x_field:
        return []

    series = _coerce_numeric(df, x_field).dropna()
    if series.empty:
        return []

    bin_count = plan.options.bin_count or 20
    counts, edges = np.histogram(series, bins=bin_count)

    records = []
    for i, count in enumerate(counts):
        records.append({
            "bin_start": float(edges[i]),
            "bin_end": float(edges[i + 1]),
            "bin_label": f"{edges[i]:.4g}-{edges[i+1]:.4g}",
            "count": int(count),
        })

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
    # Special case: missingness chart uses synthetic fields
    if "__missing_pct__" in plan.fields_used or "missingness" in plan.tags:
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
                    parts.append(f"Top: {top_val.get(x_field)} ({top_val.get(y_field)}).")

    if plan.chart_type == ChartType.scatter:
        parts.append("Selected scatter chart.")

    if not parts:
        parts.append(f"Selected {ct} chart.")

    return " ".join(parts)
