"""
Validation skill for ViewPlans and ViewResults.

Catches common issues (missing fields, empty data, cardinality mismatches)
before they reach the frontend.
"""

from __future__ import annotations

from typing import List, Tuple

from core.models import (
    ChartType,
    ColumnRole,
    DataProfile,
    EncodingChannel,
    ViewPlan,
    ViewResult,
)


# ---------------------------------------------------------------------------
# Plan validation
# ---------------------------------------------------------------------------

def validate_plan(plan: ViewPlan, profile: DataProfile) -> Tuple[bool, List[str]]:
    """
    Check that the plan references valid fields and types.

    Returns (is_valid, warnings).
    """
    warnings: List[str] = []
    col_names = {c.name for c in profile.columns}
    col_map = {c.name: c for c in profile.columns}

    # Skip synthetic fields used in missingness charts
    synthetic = {"__column__", "__missing_pct__"}

    for field_name in plan.fields_used:
        if field_name in synthetic:
            continue
        if field_name not in col_names:
            warnings.append(f"Field '{field_name}' not found in dataset.")

    # Validate encoding channels reference real columns
    for channel_name in ("x", "y", "color", "theta", "size", "facet"):
        enc: EncodingChannel | None = getattr(plan.encoding, channel_name, None)
        if enc is None or not enc.field:
            continue
        if enc.field in synthetic:
            continue
        if enc.field not in col_names:
            warnings.append(f"Encoding {channel_name} references unknown field '{enc.field}'.")

        # Type-role mismatch checks
        col = col_map.get(enc.field)
        if col and enc.type:
            if enc.type == "temporal" and col.role != ColumnRole.temporal:
                warnings.append(
                    f"Channel {channel_name} typed as temporal but '{enc.field}' role is {col.role.value}."
                )
            if enc.type == "quantitative" and col.role in (ColumnRole.identifier, ColumnRole.categorical):
                warnings.append(
                    f"Channel {channel_name} typed as quantitative but '{enc.field}' role is {col.role.value}."
                )

    # Cardinality guard for bar/pie on x-axis
    if plan.chart_type in (ChartType.bar, ChartType.pie):
        x_enc = plan.encoding.x or plan.encoding.color
        if x_enc and x_enc.field and x_enc.field in col_map:
            card = col_map[x_enc.field].cardinality
            if card > 50 and not plan.options.top_n:
                warnings.append(
                    f"High cardinality ({card}) on {plan.chart_type.value} without top_n limit."
                )

    is_valid = not any("not found" in w for w in warnings)
    return is_valid, warnings


# ---------------------------------------------------------------------------
# View validation
# ---------------------------------------------------------------------------

def validate_view(view: ViewResult) -> Tuple[bool, List[str]]:
    """
    Check the built view for problems: empty data, NaN-heavy, oversized.

    Returns (is_valid, warnings).
    """
    warnings: List[str] = []

    if not view.data_inline:
        warnings.append("View produced no data.")
        return False, warnings

    n = len(view.data_inline)
    if n > 10_000:
        warnings.append(f"Very large dataset ({n} rows) — may slow rendering.")

    # Check for all-null columns in first row
    first = view.data_inline[0]
    null_fields = [k for k, v in first.items() if v is None]
    if null_fields and len(null_fields) == len(first):
        warnings.append("All fields in first row are null.")

    is_valid = len(view.data_inline) > 0
    return is_valid, warnings


# ---------------------------------------------------------------------------
# Fallback / repair
# ---------------------------------------------------------------------------

def deterministic_fallback(plan: ViewPlan, profile: DataProfile) -> ViewPlan:
    """
    Fix an invalid plan by adjusting fields to ones that exist.

    Returns a repaired copy.
    """
    col_names = {c.name for c in profile.columns}
    measures = [c for c in profile.columns if c.role in (ColumnRole.measure, ColumnRole.count)]
    cats = [c for c in profile.columns if c.role in (ColumnRole.categorical, ColumnRole.nominal)]

    repaired = plan.model_copy(deep=True)

    # Fix encoding fields that don't exist
    for channel_name in ("x", "y", "color", "theta", "size", "facet"):
        enc: EncodingChannel | None = getattr(repaired.encoding, channel_name, None)
        if enc is None or not enc.field:
            continue
        if enc.field in col_names or enc.field.startswith("__"):
            continue

        # Try to substitute
        if enc.type == "quantitative" and measures:
            enc.field = measures[0].name
        elif enc.type in ("nominal", "ordinal") and cats:
            enc.field = cats[0].name
        else:
            # Just pick the first column
            enc.field = profile.columns[0].name if profile.columns else enc.field

    # Fix fields_used
    repaired.fields_used = [f for f in repaired.fields_used if f in col_names or f.startswith("__")]
    if not repaired.fields_used:
        repaired.fields_used = [profile.columns[0].name] if profile.columns else []

    return repaired
