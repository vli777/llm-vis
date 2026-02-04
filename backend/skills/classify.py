"""
Column classification skill — uses LLM to refine column roles.

The deterministic `detect_column_role()` in profile.py uses keyword matching
which is brittle for unfamiliar naming conventions.  An LLM can reason about
column name + dtype + cardinality + sample values together.

Follows the same LLM-with-fallback pattern as narrate.py.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from core.models import ColumnRole, DataProfile

logger = logging.getLogger("uvicorn.error")

# Track whether we've already warned about LLM unavailability this session
_llm_warn_logged = False

# All valid role strings for validation
_VALID_ROLES = {r.value for r in ColumnRole}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM = """You are a data-profiling expert.  Given a list of columns
with their dtype, cardinality, unique_ratio, sample values, and the current
heuristic guess, return a JSON object mapping ONLY the column names whose role
should change to their corrected role.

Allowed roles (use these exact strings):
- "temporal"    — dates, timestamps, years, months
- "geographic"  — countries, cities, states, regions, coordinates
- "measure"     — continuous numeric values meant for aggregation (revenue, price, score)
- "count"       — discrete counts (number of items, frequency)
- "categorical" — low-cardinality labels / groups (status, type, category)
- "identifier"  — unique or near-unique keys (user_id, order_id, SKU)
- "nominal"     — free-text or high-cardinality strings that are not identifiers

Disambiguation tips:
- A column named "Item Code" with int64 dtype and high cardinality → identifier, not measure.
- A column named "Year" with int64 and only ~20 unique values → temporal, not measure.
- A column named "Zip Code" with int64 → categorical or geographic, not measure.
- A numeric "Rating" column with values 1-5 → categorical, not measure.
- Only reclassify a column if you are confident the heuristic is wrong.

Return a JSON object like {"col_name": "role", ...}.
If all guesses are correct, return {}.
Return ONLY valid JSON. No prose, no code fences."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_columns(profile: DataProfile) -> DataProfile:
    """
    Use LLM to refine column roles in a DataProfile.

    On any failure the profile is returned unchanged (deterministic roles
    from detect_column_role are preserved).
    """
    try:
        from app.llm import chat_json

        payload = _build_payload(profile)
        result = chat_json(_CLASSIFY_SYSTEM, json.dumps(payload, default=str),
                           max_tokens=1024)
        return _apply_corrections(profile, result)

    except Exception as e:
        global _llm_warn_logged
        if not _llm_warn_logged:
            logger.warning(
                "LLM column classification unavailable, keeping deterministic roles: %s", e,
            )
            _llm_warn_logged = True
        return profile


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _build_payload(profile: DataProfile) -> List[Dict[str, Any]]:
    """Build a compact JSON payload describing each column."""
    items: List[Dict[str, Any]] = []
    for col in profile.columns:
        unique_ratio = (
            col.cardinality / profile.row_count
            if profile.row_count > 0 else 0.0
        )
        items.append({
            "name": col.name,
            "dtype": col.dtype,
            "cardinality": col.cardinality,
            "unique_ratio": round(unique_ratio, 3),
            "examples": (col.examples or [])[:3],
            "current_guess": col.role.value,
        })
    return items


def _apply_corrections(
    profile: DataProfile,
    corrections: Dict[str, str],
) -> DataProfile:
    """Validate LLM corrections and apply them to the profile."""
    if not corrections:
        return profile

    col_lookup = {col.name: col for col in profile.columns}
    changed = False

    for col_name, new_role_str in corrections.items():
        if col_name not in col_lookup:
            logger.debug("LLM suggested role for unknown column %r, skipping", col_name)
            continue
        if not isinstance(new_role_str, str):
            logger.debug(
                "LLM suggested non-string role %r for %r, skipping", new_role_str, col_name,
            )
            continue
        if new_role_str not in _VALID_ROLES:
            logger.debug("LLM suggested invalid role %r for %r, skipping", new_role_str, col_name)
            continue

        new_role = ColumnRole(new_role_str)
        col = col_lookup[col_name]
        if col.role != new_role:
            logger.info("LLM reclassified %r: %s -> %s", col_name, col.role.value, new_role.value)
            col.role = new_role
            changed = True

    if changed:
        _refresh_viz_hints(profile)

    return profile


def _refresh_viz_hints(profile: DataProfile) -> None:
    """Recompute visualization_hints summary counters after role changes."""
    if profile.visualization_hints is None:
        return

    from skills.profile import suggest_chart_types

    numeric_info: List[Dict[str, Any]] = []
    categorical_info: List[Dict[str, Any]] = []

    for col in profile.columns:
        if col.role in (ColumnRole.measure, ColumnRole.count):
            numeric_info.append({"name": col.name, "variance": 0.0})
        elif col.role in (ColumnRole.categorical, ColumnRole.nominal, ColumnRole.geographic):
            ratio = (
                col.cardinality / profile.row_count
                if profile.row_count > 0 else 0.0
            )
            categorical_info.append({
                "name": col.name,
                "unique": col.cardinality,
                "ratio": ratio,
            })

    numeric_count = sum(
        1 for c in profile.columns
        if c.role in (ColumnRole.measure, ColumnRole.count)
    )
    categorical_count = sum(
        1 for c in profile.columns
        if c.role in (ColumnRole.categorical, ColumnRole.nominal, ColumnRole.geographic)
    )
    has_temporal = any(c.role == ColumnRole.temporal for c in profile.columns)

    summary = profile.visualization_hints.get("summary", {})
    summary["numeric_columns"] = numeric_count
    summary["categorical_columns"] = categorical_count
    summary["has_temporal_data"] = has_temporal
    profile.visualization_hints["summary"] = summary
    profile.visualization_hints["suggested_chart_types"] = suggest_chart_types(
        numeric_info, categorical_info, has_temporal,
    )
