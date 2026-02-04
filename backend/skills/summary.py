"""
Summary statistics skill — computes dataset summary without charts.
"""

from __future__ import annotations

from typing import Dict, List

from core.models import DataProfile
from core.models import ChartSpec, ChartType, ChartEncoding, ChartOptions, ViewPlan, ViewResult


def summarize_dataset(profile: DataProfile) -> Dict[str, List[str]]:
    """Return headline + findings strings for summary statistics step."""
    findings: List[str] = []

    findings.append(
        f"{profile.row_count:,} rows across {len(profile.columns)} columns."
    )

    missing_cols = [
        c for c in profile.columns
        if c.missing_pct and c.missing_pct >= 5.0
    ]
    if missing_cols:
        top = sorted(missing_cols, key=lambda c: c.missing_pct, reverse=True)[:5]
        cols = ", ".join(f"{c.name} ({c.missing_pct:.1f}%)" for c in top)
        findings.append(f"Columns with notable missingness: {cols}.")

    role_counts = {}
    for c in profile.columns:
        role_counts[c.role.value] = role_counts.get(c.role.value, 0) + 1
    role_bits = ", ".join(f"{k}: {v}" for k, v in role_counts.items())
    findings.append(f"Role summary: {role_bits}.")

    headline = "Dataset summary statistics"
    return {"headline": headline, "findings": findings}


def build_summary_stats_view(df, profile: DataProfile) -> ViewResult:
    """Build a table view with summary statistics for each column."""
    rows: List[Dict[str, object]] = []
    for c in profile.columns:
        s = df[c.name] if c.name in df.columns else None
        dtype = c.dtype
        count = int(s.notna().sum()) if s is not None else 0
        missing = int(s.isna().sum()) if s is not None else 0

        row: Dict[str, object] = {
            "column": c.name,
            "dtype": dtype,
            "count": count,
            "missing": missing,
            "mean": None,
            "std": None,
            "min": None,
            "25%": None,
            "50%": None,
            "75%": None,
            "max": None,
            "skew": None,
            "kurtosis": None,
        }

        if s is not None and (s.dtype.kind in ("i", "u", "f")):
            s_num = s.dropna()
            if len(s_num) > 0:
                desc = s_num.describe()
                row["mean"] = float(desc.get("mean"))
                row["std"] = float(desc.get("std"))
                row["min"] = float(desc.get("min"))
                row["25%"] = float(desc.get("25%"))
                row["50%"] = float(desc.get("50%"))
                row["75%"] = float(desc.get("75%"))
                row["max"] = float(desc.get("max"))
                row["skew"] = float(s_num.skew())
                row["kurtosis"] = float(s_num.kurt())
        rows.append(row)

    spec = ChartSpec(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(),
        data_inline=rows,
        title="Dataset summary statistics",
    )
    plan = ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(),
        intent="Summary statistics table",
        fields_used=[],
        tags=["summary", "stats", "table"],
    )
    return ViewResult(
        plan=plan,
        spec=spec,
        data_inline=rows,
        explanation="Summary statistics by column.",
    )
