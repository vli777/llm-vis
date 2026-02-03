"""
View builder skill.

Takes a ViewPlan + DataFrame → ViewResult with pre-aggregated data_inline.
The frontend simply renders what it receives — no computation needed there.
"""

from __future__ import annotations

import math
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
            tmp[x_field] = pd.to_datetime(tmp[x_field], errors="coerce")
            tmp = tmp.dropna(subset=[x_field])
            tmp = tmp.sort_values(x_field)
            # Convert datetime to ISO string for JSON
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
    """Summary table — return top rows for the fields used."""
    fields = plan.fields_used or list(df.columns[:10])
    resolved = [_resolve(f, df) for f in fields]
    cols = [c for c in resolved if c and c in df.columns]
    if not cols:
        cols = list(df.columns[:10])

    subset = df[cols].head(100)
    return df_to_records_safe(subset)


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

    spec = ChartSpec(
        chart_type=plan.chart_type,
        encoding=plan.encoding,
        options=plan.options,
        data_inline=data,
        title=plan.intent or f"{plan.chart_type.value} chart",
    )

    explanation = _auto_explanation(plan, data, df)

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

    n = len(data)
    ct = plan.chart_type.value
    fields = ", ".join(plan.fields_used) if plan.fields_used else "selected columns"

    parts = [f"Shows {ct} chart with {n} data points."]

    if plan.chart_type == ChartType.bar and plan.encoding.y:
        y_field = plan.encoding.y.field
        if y_field and any(y_field in d for d in data[:1]):
            vals = [d.get(y_field) for d in data if d.get(y_field) is not None]
            if vals:
                top_val = data[0]
                x_field = plan.encoding.x.field if plan.encoding.x else None
                if x_field and x_field in top_val:
                    parts.append(f"Top: {top_val.get(x_field)} ({top_val.get(y_field)}).")

    if plan.chart_type == ChartType.hist:
        parts.append(f"Distribution across {n} bins.")

    if plan.chart_type == ChartType.scatter and plan.encoding.x and plan.encoding.y:
        parts.append(f"Comparing {plan.encoding.x.field} vs {plan.encoding.y.field}.")

    return " ".join(parts)
