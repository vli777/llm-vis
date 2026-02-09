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
    "lgbm_regression",
    "lgbm_classification",
    "quantile_regression",
    "segmentation",
    "cohort_change",
    "group_comparison",
    "matched_comparison",
    "diff_in_diff",
    "uplift_check",
    "seasonality_test",
    "autocorrelation_test",
    "lag_feature_search",
    "rolling_stats",
    "trend_breaks",
    "numeric_transforms",
    "interaction_scan",
    "binning_optimizer",
    "date_part_features",
    "target_encoding",
    "shap_summary",
    "shap_dependence",
    "partial_dependence",
    "leakage_scan",
    "drift_check",
    "missingness_mechanism",
    "hypothesis_generator",
    "test_selector",
    "result_validator",
    "generic_query_charts",
}

_SYSTEM = """You are a data analysis agent. Choose which tools to use to answer
a user's query given the dataset profile and analysis context.

Available tools:
- percentile_compare: Compare metric values across percentile groups (e.g., top 10% vs 50-90%), optionally by time.
- linear_regression: Fit a simple linear regression between two numeric variables.
- logistic_regression: Fit a simple logistic regression for a binary target vs a numeric predictor.
- lgbm_regression: Train LightGBM regressor, return metrics + feature importances.
- lgbm_classification: Train LightGBM classifier, return metrics + feature importances.
- quantile_regression: Fit quantile regression for multiple quantiles.
- segmentation: Segment a metric by a categorical field (group summaries).
- cohort_change: Cohort analysis using first-seen time per entity and change over time.
- group_comparison: Compare a metric between two groups (naive difference in means).
- matched_comparison: Match treated/control units on covariates and estimate effect.
- diff_in_diff: Pre/post comparison with treated vs control groups over time.
- uplift_check: Treatment effect directionality across segments.
- seasonality_test: Detect seasonality strength in a time series.
- autocorrelation_test: ACF/PACF summary for temporal structure.
- lag_feature_search: Evaluate predictive lag candidates.
- rolling_stats: Rolling mean/volatility feature summaries.
- trend_breaks: Detect structural breaks / regime shifts.
- numeric_transforms: Evaluate log/sqrt transforms for skew reduction.
- interaction_scan: Check interaction features for predictive lift.
- binning_optimizer: Quantile binning to capture nonlinear effects.
- date_part_features: Date-part feature testing (month/dow/holiday proxy).
- target_encoding: Target-encode categorical features (summary).
- shap_summary: SHAP global feature importance for model.
- shap_dependence: SHAP dependence summary for top feature.
- partial_dependence: Partial dependence for a feature.
- leakage_scan: Detect near-leakage predictors.
- drift_check: Distribution shift across time splits.
- missingness_mechanism: Missingness correlated with target.
- hypothesis_generator: Propose testable hypotheses from profile + context.
- test_selector: Map hypothesis to a tool.
- result_validator: Validate if results support hypothesis.
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
- If the query mentions matching, control/treatment, or causal language, consider matched_comparison or diff_in_diff.
- If the query mentions uplift, treatment effect, or conversion lift, consider uplift_check.
- If the query mentions seasonality/lag/autocorr, choose time series tools.
- If the query mentions feature engineering, consider numeric_transforms, interaction_scan, binning_optimizer, date_part_features.
- If the query mentions SHAP, partial dependence, or feature importance, choose shap_summary/shap_dependence/partial_dependence.
- If the query mentions leakage, drift, or missingness, choose the corresponding diagnostics.
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
