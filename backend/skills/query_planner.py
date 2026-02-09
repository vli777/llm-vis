"""
LLM-driven tool selection for query-focused analysis.

The agent chooses which query tools to run; each tool then builds a ViewPlan
from the profile + query. This keeps selection flexible and generalized.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from core.models import DataProfile

logger = logging.getLogger("uvicorn.error")

_TOOL_NAMES = {
    "percentile_compare",
    "linear_regression",
    "logistic_regression",
    "segmentation",
    "cohort_change",
    "group_comparison",
    "generic_query_charts",
}

_SYSTEM = """You are a data analysis agent. Choose which tools to use to answer
a user's query given the dataset profile and analysis context.

Available tools:
- percentile_compare: Compare metric values across percentile groups (e.g., top 10% vs 50-90%), optionally by time.
- linear_regression: Fit a simple linear regression between two numeric variables.
- logistic_regression: Fit a simple logistic regression for a binary target vs a numeric predictor.
- segmentation: Segment a metric by a categorical field (group summaries).
- cohort_change: Cohort analysis using first-seen time per entity and change over time.
- group_comparison: Compare a metric between two groups (naive difference in means).
- generic_query_charts: Fall back to standard query-driven charts (trends, comparisons, distributions).

Return a JSON object with:
- "tools": string[] — ordered tool names to run, from most important to least.

Rules:
- Pick at most 2 tools.
- If the query is about percentiles/deciles/top X%/bottom X% or group comparisons, include percentile_compare.
- If the query requests regression, include linear_regression or logistic_regression.
- If the query mentions cohorts/retention or change over time, consider cohort_change.
- If the query mentions segment/breakdown/by-group, consider segmentation.
- If the query asks for impact/effect/difference between groups, consider group_comparison.
- If none apply, return ["generic_query_charts"].

Return ONLY valid JSON. No prose, no code fences."""


def plan_query_tools(
    profile: DataProfile,
    query: str,
    *,
    context: Optional[Dict[str, Any]] = None,
) -> List[str]:
    if not query:
        return []

    try:
        from app.llm import chat_json

        payload: Dict[str, Any] = {
            "table": profile.table_name,
            "row_count": profile.row_count,
            "columns": [c.name for c in profile.columns[:15]],
            "query": query,
        }
        if context:
            payload["analysis_context"] = context

        result = chat_json(_SYSTEM, json.dumps(payload, default=str), max_tokens=512)
        tools = result.get("tools", [])
        if not isinstance(tools, list):
            return ["generic_query_charts"]

        cleaned: List[str] = []
        for t in tools:
            if not isinstance(t, str):
                continue
            if t not in _TOOL_NAMES:
                continue
            if t not in cleaned:
                cleaned.append(t)

        if not cleaned:
            return ["generic_query_charts"]

        return cleaned[:2]

    except Exception as e:
        logger.warning("Query tool planner unavailable, using heuristic fallback: %s", e)
        return ["generic_query_charts"]
