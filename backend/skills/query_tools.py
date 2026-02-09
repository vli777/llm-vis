"""
Query tool selection and specialized query-driven plans.

These are "skills" the agent can invoke for targeted questions.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from core.models import (
    ChartEncoding,
    ChartOptions,
    ChartType,
    ColumnRole,
    DataProfile,
    EncodingChannel,
    ViewPlan,
)
from core.utils import tokenize
from skills.target import infer_task_and_target


def _tokens(query: str) -> set[str]:
    return set(tokenize(query or ""))


def _match_columns(profile: DataProfile, tokens: set[str]) -> List[str]:
    scored: List[tuple[int, str]] = []
    for col in profile.columns:
        col_tokens = set(tokenize(col.name))
        overlap = len(tokens & col_tokens)
        if overlap:
            scored.append((overlap, col.name))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [name for _, name in scored]


def _pick_metric(profile: DataProfile, tokens: set[str]) -> Optional[str]:
    matched = _match_columns(profile, tokens)
    if matched:
        return matched[0]
    for col in profile.columns:
        if col.role in (ColumnRole.measure, ColumnRole.count):
            return col.name
    return profile.columns[0].name if profile.columns else None


def _pick_temporal(profile: DataProfile) -> Optional[str]:
    for col in profile.columns:
        if col.role == ColumnRole.temporal:
            return col.name
    return None


def _pick_identifier(profile: DataProfile) -> Optional[str]:
    for col in profile.columns:
        if col.role == ColumnRole.identifier:
            return col.name
    return None


def _pick_categorical(profile: DataProfile, tokens: set[str]) -> Optional[str]:
    matched = _match_columns(profile, tokens)
    for name in matched:
        col = next((c for c in profile.columns if c.name == name), None)
        if col and col.role in (ColumnRole.categorical, ColumnRole.nominal, ColumnRole.geographic):
            return col.name
    for col in profile.columns:
        if col.role in (ColumnRole.categorical, ColumnRole.nominal, ColumnRole.geographic):
            return col.name
    return None


def _pick_target(profile: DataProfile, tokens: set[str]) -> Optional[str]:
    matched = _match_columns(profile, tokens)
    if matched:
        return matched[0]
    spec = infer_task_and_target(profile)
    return spec.column


def _pick_treatment(profile: DataProfile, tokens: set[str]) -> Optional[str]:
    matched = _match_columns(profile, tokens)
    for name in matched:
        col = next((c for c in profile.columns if c.name == name), None)
        if col and col.cardinality == 2:
            return col.name
    for col in profile.columns:
        if col.cardinality == 2 and col.role in (ColumnRole.categorical, ColumnRole.nominal, ColumnRole.count):
            return col.name
    return None


def _parse_percentile_groups(query: str) -> List[Tuple[str, int, int]]:
    q = query.lower()
    groups: List[Tuple[str, int, int]] = []

    # top/bottom X%
    m = re.findall(r"(top|bottom)\s+(\d{1,2})\s*%?", q)
    for direction, num in m:
        p = int(num)
        if p <= 0 or p >= 100:
            continue
        if direction == "top":
            groups.append((f"Top {p}%", 100 - p, 100))
        else:
            groups.append((f"Bottom {p}%", 0, p))

    # explicit ranges: 50-90, 50 to 90, 50th-90th percentile
    m2 = re.findall(r"(\d{1,2})\s*(?:-|to|–)\s*(\d{1,2})", q)
    for a, b in m2:
        lo = int(a)
        hi = int(b)
        if lo == hi or lo < 0 or hi > 100:
            continue
        if lo > hi:
            lo, hi = hi, lo
        groups.append((f"P{lo}-{hi}", lo, hi))

    # dedupe by bounds
    seen = set()
    uniq: List[Tuple[str, int, int]] = []
    for label, lo, hi in groups:
        key = (lo, hi)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((label, lo, hi))
    return uniq


def percentile_compare_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens:
        return None

    if not any(t in tokens for t in {"percentile", "percentiles", "top", "bottom", "decile"}):
        return None

    groups = _parse_percentile_groups(query)
    if len(groups) < 2:
        return None

    metric = _pick_metric(profile, tokens)
    if not metric:
        return None

    temporal = _pick_temporal(profile)
    entity = _pick_identifier(profile)

    tags = [
        "percentile_compare",
        f"metric={metric}",
        f"percentiles={';'.join([f'{lo}-{hi}' for _, lo, hi in groups])}",
    ]
    if temporal:
        tags.append(f"temporal={temporal}")
    if entity:
        tags.append(f"entity={entity}")
    if any(t in tokens for t in {"increase", "growth", "change", "delta", "marginal"}):
        tags.append("compare=change")

    label_a, lo_a, hi_a = groups[0]
    label_b, lo_b, hi_b = groups[1]
    intent = f"Compare {metric} for {label_a} vs {label_b}"

    fields_used = [metric]
    if temporal:
        fields_used.append(temporal)
    if entity:
        fields_used.append(entity)

    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(),
        intent=intent,
        fields_used=fields_used,
        tags=tags,
    )


def segmentation_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"segment", "segmentation", "group", "breakdown", "by"}):
        return None

    metric = _pick_metric(profile, tokens)
    if not metric:
        return None

    # pick a categorical grouping column
    group = None
    matched = _match_columns(profile, tokens)
    for name in matched:
        col = next((c for c in profile.columns if c.name == name), None)
        if col and col.role in (ColumnRole.categorical, ColumnRole.nominal, ColumnRole.geographic):
            group = col.name
            break
    if not group:
        for col in profile.columns:
            if col.role in (ColumnRole.categorical, ColumnRole.nominal, ColumnRole.geographic):
                group = col.name
                break
    if not group:
        return None

    tags = ["segmentation", f"metric={metric}", f"group={group}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(sort="descending", top_n=15),
        intent=f"Segment {metric} by {group}",
        fields_used=[metric, group],
        tags=tags,
    )


def cohort_change_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"cohort", "retention", "churn", "over", "time", "trend", "change"}):
        return None

    metric = _pick_metric(profile, tokens)
    temporal = _pick_temporal(profile)
    entity = _pick_identifier(profile)
    if not metric or not temporal or not entity:
        return None

    tags = ["cohort_change", f"metric={metric}", f"temporal={temporal}", f"entity={entity}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(),
        intent=f"Cohort change for {metric} over {temporal}",
        fields_used=[metric, temporal, entity],
        tags=tags,
    )


def group_comparison_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"impact", "effect", "difference", "compare", "versus", "vs", "treatment", "control"}):
        return None

    metric = _pick_metric(profile, tokens)
    if not metric:
        return None

    group = None
    matched = _match_columns(profile, tokens)
    for name in matched:
        col = next((c for c in profile.columns if c.name == name), None)
        if col and col.role in (ColumnRole.categorical, ColumnRole.nominal):
            group = col.name
            break
    if not group:
        for col in profile.columns:
            if col.role in (ColumnRole.categorical, ColumnRole.nominal):
                group = col.name
                break
    if not group:
        return None

    tags = ["group_comparison", f"metric={metric}", f"group={group}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(),
        intent=f"Compare {metric} across {group}",
        fields_used=[metric, group],
        tags=tags,
    )


def linear_regression_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"regression", "linear", "predict"}):
        return None

    # Try to pick two numeric columns matching tokens
    matched = _match_columns(profile, tokens)
    numeric = [c.name for c in profile.columns if c.role in (ColumnRole.measure, ColumnRole.count)]
    x = None
    y = None
    for name in matched:
        if name in numeric:
            if not y:
                y = name
            elif not x and name != y:
                x = name
                break
    if not y and numeric:
        y = numeric[0]
    if not x and len(numeric) > 1:
        x = numeric[1]

    if not x or not y or x == y:
        return None

    tags = [f"regression_linear", f"x={x}", f"y={y}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(
            x=EncodingChannel(field=x, type="quantitative"),
            y=EncodingChannel(field=y, type="quantitative"),
        ),
        options=ChartOptions(),
        intent=f"Linear regression: {y} ~ {x}",
        fields_used=[x, y],
        tags=tags,
    )


def logistic_regression_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"logistic", "classification"}):
        return None

    # pick one numeric predictor and a likely binary target
    numeric = [c for c in profile.columns if c.role in (ColumnRole.measure, ColumnRole.count)]
    target = None
    for c in profile.columns:
        if c.role in (ColumnRole.categorical, ColumnRole.nominal) and c.cardinality == 2:
            target = c.name
            break
    if not target:
        for c in profile.columns:
            if c.role in (ColumnRole.measure, ColumnRole.count) and c.cardinality == 2:
                target = c.name
                break

    x = None
    if numeric:
        x = numeric[0].name

    if not target or not x:
        return None

    tags = [f"regression_logistic", f"x={x}", f"y={target}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(
            x=EncodingChannel(field=x, type="quantitative"),
            y=EncodingChannel(field=target, type="nominal"),
        ),
        options=ChartOptions(),
        intent=f"Logistic regression: {target} ~ {x}",
        fields_used=[x, target],
        tags=tags,
    )


def lgbm_regression_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"lgbm", "lightgbm", "gbm", "boost", "xgboost", "nonlinear"}):
        return None
    target = _pick_target(profile, tokens)
    if not target:
        return None
    tags = [f"lgbm_regression", f"target={target}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(),
        intent=f"LGBM regression for {target}",
        fields_used=[target],
        tags=tags,
    )


def lgbm_classification_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"lgbm", "lightgbm", "gbm", "boost", "classify", "classification"}):
        return None
    target = _pick_target(profile, tokens)
    if not target:
        return None
    tags = [f"lgbm_classification", f"target={target}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(),
        intent=f"LGBM classification for {target}",
        fields_used=[target],
        tags=tags,
    )


def quantile_regression_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"quantile", "quantiles"}):
        return None
    y = _pick_target(profile, tokens) or _pick_metric(profile, tokens)
    x = _pick_metric(profile, tokens)
    if not y or not x or x == y:
        return None
    tags = [f"quantile_regression", f"x={x}", f"y={y}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(
            x=EncodingChannel(field=x, type="quantitative"),
            y=EncodingChannel(field=y, type="quantitative"),
        ),
        options=ChartOptions(),
        intent=f"Quantile regression: {y} ~ {x}",
        fields_used=[x, y],
        tags=tags,
    )


def seasonality_test_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"seasonality", "seasonal", "periodic"}):
        return None
    metric = _pick_metric(profile, tokens)
    temporal = _pick_temporal(profile)
    if not metric or not temporal:
        return None
    tags = ["seasonality_test", f"metric={metric}", f"temporal={temporal}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(),
        intent=f"Seasonality test for {metric}",
        fields_used=[metric, temporal],
        tags=tags,
    )


def autocorrelation_test_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"autocorrelation", "acf", "pacf"}):
        return None
    metric = _pick_metric(profile, tokens)
    temporal = _pick_temporal(profile)
    if not metric or not temporal:
        return None
    tags = ["autocorrelation_test", f"metric={metric}", f"temporal={temporal}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(),
        intent=f"Autocorrelation test for {metric}",
        fields_used=[metric, temporal],
        tags=tags,
    )


def lag_feature_search_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"lag", "lagged", "autoregressive"}):
        return None
    metric = _pick_metric(profile, tokens)
    temporal = _pick_temporal(profile)
    if not metric or not temporal:
        return None
    tags = ["lag_feature_search", f"metric={metric}", f"temporal={temporal}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(),
        intent=f"Lag feature search for {metric}",
        fields_used=[metric, temporal],
        tags=tags,
    )


def rolling_stats_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"rolling", "moving", "volatility"}):
        return None
    metric = _pick_metric(profile, tokens)
    temporal = _pick_temporal(profile)
    if not metric or not temporal:
        return None
    tags = ["rolling_stats", f"metric={metric}", f"temporal={temporal}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(),
        intent=f"Rolling stats for {metric}",
        fields_used=[metric, temporal],
        tags=tags,
    )


def trend_breaks_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"break", "regime", "changepoint", "shift"}):
        return None
    metric = _pick_metric(profile, tokens)
    temporal = _pick_temporal(profile)
    if not metric or not temporal:
        return None
    tags = ["trend_breaks", f"metric={metric}", f"temporal={temporal}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(),
        intent=f"Trend breaks for {metric}",
        fields_used=[metric, temporal],
        tags=tags,
    )


def numeric_transforms_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"transform", "log", "sqrt", "skew"}):
        return None
    metric = _pick_metric(profile, tokens)
    if not metric:
        return None
    tags = ["numeric_transforms", f"metric={metric}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(),
        intent=f"Numeric transform check for {metric}",
        fields_used=[metric],
        tags=tags,
    )


def interaction_scan_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"interaction", "nonlinear", "feature pair"}):
        return None
    target = _pick_target(profile, tokens)
    if not target:
        return None
    tags = ["interaction_scan", f"target={target}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(),
        intent=f"Interaction scan for {target}",
        fields_used=[target],
        tags=tags,
    )


def binning_optimizer_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"bin", "bucket", "nonlinear", "quantile"}):
        return None
    target = _pick_target(profile, tokens)
    feature = _pick_metric(profile, tokens)
    if not target or not feature:
        return None
    tags = ["binning_optimizer", f"target={target}", f"feature={feature}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(),
        intent=f"Quantile binning for {feature} vs {target}",
        fields_used=[feature, target],
        tags=tags,
    )


def date_part_features_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"date", "time", "month", "weekday", "dow", "season"}):
        return None
    metric = _pick_metric(profile, tokens)
    temporal = _pick_temporal(profile)
    if not metric or not temporal:
        return None
    tags = ["date_part_features", f"metric={metric}", f"temporal={temporal}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(),
        intent=f"Date-part features for {metric}",
        fields_used=[metric, temporal],
        tags=tags,
    )


def target_encoding_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"target", "encode", "encoding"}):
        return None
    target = _pick_target(profile, tokens)
    cat = _pick_categorical(profile, tokens)
    if not target or not cat:
        return None
    tags = ["target_encoding", f"target={target}", f"feature={cat}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(top_n=15),
        intent=f"Target encoding for {cat} -> {target}",
        fields_used=[cat, target],
        tags=tags,
    )


def matched_comparison_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"match", "matching", "treated", "control"}):
        return None
    metric = _pick_metric(profile, tokens)
    treatment = _pick_treatment(profile, tokens)
    if not metric or not treatment:
        return None
    tags = ["matched_comparison", f"metric={metric}", f"treatment={treatment}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(),
        intent=f"Matched comparison on {metric} by {treatment}",
        fields_used=[metric, treatment],
        tags=tags,
    )


def diff_in_diff_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"diff", "difference", "pre", "post"}):
        return None
    metric = _pick_metric(profile, tokens)
    treatment = _pick_treatment(profile, tokens)
    temporal = _pick_temporal(profile)
    if not metric or not treatment or not temporal:
        return None
    tags = ["diff_in_diff", f"metric={metric}", f"treatment={treatment}", f"temporal={temporal}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(),
        intent=f"Diff-in-diff for {metric}",
        fields_used=[metric, treatment, temporal],
        tags=tags,
    )


def uplift_check_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"uplift", "lift", "treatment", "conversion"}):
        return None
    treatment = _pick_treatment(profile, tokens)
    target = _pick_target(profile, tokens)
    segment = _pick_categorical(profile, tokens)
    if not treatment or not target:
        return None
    tags = ["uplift_check", f"treatment={treatment}", f"target={target}"]
    if segment:
        tags.append(f"segment={segment}")
    fields = [treatment, target] + ([segment] if segment else [])
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(top_n=10),
        intent=f"Uplift check for {target} by {treatment}",
        fields_used=fields,
        tags=tags,
    )


def shap_summary_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"shap", "importance"}):
        return None
    target = _pick_target(profile, tokens)
    if not target:
        return None
    tags = ["shap_summary", f"target={target}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(top_n=15),
        intent=f"SHAP summary for {target}",
        fields_used=[target],
        tags=tags,
    )


def shap_dependence_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"dependence", "shap"}):
        return None
    target = _pick_target(profile, tokens)
    if not target:
        return None
    tags = ["shap_dependence", f"target={target}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(top_n=200),
        intent=f"SHAP dependence for {target}",
        fields_used=[target],
        tags=tags,
    )


def partial_dependence_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"partial", "dependence", "pdp"}):
        return None
    target = _pick_target(profile, tokens)
    feature = _pick_metric(profile, tokens)
    if not target or not feature:
        return None
    tags = ["partial_dependence", f"target={target}", f"feature={feature}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(top_n=50),
        intent=f"Partial dependence of {feature} on {target}",
        fields_used=[feature, target],
        tags=tags,
    )


def leakage_scan_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"leakage", "leak"}):
        return None
    target = _pick_target(profile, tokens)
    if not target:
        return None
    tags = ["leakage_scan", f"target={target}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(top_n=15),
        intent=f"Leakage scan for {target}",
        fields_used=[target],
        tags=tags,
    )


def drift_check_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"drift", "shift"}):
        return None
    temporal = _pick_temporal(profile)
    if not temporal:
        return None
    tags = ["drift_check", f"temporal={temporal}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(),
        intent="Drift check across time",
        fields_used=[temporal],
        tags=tags,
    )


def missingness_mechanism_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"missing", "missingness"}):
        return None
    target = _pick_target(profile, tokens)
    if not target:
        return None
    tags = ["missingness_mechanism", f"target={target}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(top_n=15),
        intent=f"Missingness mechanism vs {target}",
        fields_used=[target],
        tags=tags,
    )


def hypothesis_generator_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"hypothesis", "hypotheses"}):
        return None
    q = (query or "")[:120]
    tags = ["hypothesis_generator", f"query={q}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(),
        intent="Generate hypotheses",
        fields_used=[],
        tags=tags,
    )


def test_selector_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"test", "select", "choose"}):
        return None
    q = (query or "")[:120]
    tags = ["test_selector", f"query={q}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(),
        intent="Select tests for hypothesis",
        fields_used=[],
        tags=tags,
    )


def result_validator_plan(profile: DataProfile, query: str) -> Optional[ViewPlan]:
    tokens = _tokens(query)
    if not tokens or not any(t in tokens for t in {"validate", "confirm", "reject"}):
        return None
    q = (query or "")[:120]
    tags = ["result_validator", f"query={q}"]
    return ViewPlan(
        chart_type=ChartType.table,
        encoding=ChartEncoding(),
        options=ChartOptions(),
        intent="Validate results vs hypothesis",
        fields_used=[],
        tags=tags,
    )


def query_tool_plans(profile: DataProfile, query: str, *, tools: Optional[List[str]] = None) -> List[ViewPlan]:
    plans: List[ViewPlan] = []
    tool_set = set(tools or [])

    def want(name: str) -> bool:
        return not tool_set or name in tool_set

    if want("segmentation"):
        seg = segmentation_plan(profile, query)
        if seg:
            plans.append(seg)

    if want("seasonality_test"):
        plan = seasonality_test_plan(profile, query)
        if plan:
            plans.append(plan)

    if want("autocorrelation_test"):
        plan = autocorrelation_test_plan(profile, query)
        if plan:
            plans.append(plan)

    if want("lag_feature_search"):
        plan = lag_feature_search_plan(profile, query)
        if plan:
            plans.append(plan)

    if want("rolling_stats"):
        plan = rolling_stats_plan(profile, query)
        if plan:
            plans.append(plan)

    if want("trend_breaks"):
        plan = trend_breaks_plan(profile, query)
        if plan:
            plans.append(plan)

    if want("numeric_transforms"):
        plan = numeric_transforms_plan(profile, query)
        if plan:
            plans.append(plan)

    if want("interaction_scan"):
        plan = interaction_scan_plan(profile, query)
        if plan:
            plans.append(plan)

    if want("binning_optimizer"):
        plan = binning_optimizer_plan(profile, query)
        if plan:
            plans.append(plan)

    if want("date_part_features"):
        plan = date_part_features_plan(profile, query)
        if plan:
            plans.append(plan)

    if want("target_encoding"):
        plan = target_encoding_plan(profile, query)
        if plan:
            plans.append(plan)

    if want("matched_comparison"):
        plan = matched_comparison_plan(profile, query)
        if plan:
            plans.append(plan)

    if want("diff_in_diff"):
        plan = diff_in_diff_plan(profile, query)
        if plan:
            plans.append(plan)

    if want("uplift_check"):
        plan = uplift_check_plan(profile, query)
        if plan:
            plans.append(plan)

    if want("lgbm_regression"):
        plan = lgbm_regression_plan(profile, query)
        if plan:
            plans.append(plan)

    if want("lgbm_classification"):
        plan = lgbm_classification_plan(profile, query)
        if plan:
            plans.append(plan)

    if want("quantile_regression"):
        plan = quantile_regression_plan(profile, query)
        if plan:
            plans.append(plan)

    if want("shap_summary"):
        plan = shap_summary_plan(profile, query)
        if plan:
            plans.append(plan)

    if want("shap_dependence"):
        plan = shap_dependence_plan(profile, query)
        if plan:
            plans.append(plan)

    if want("partial_dependence"):
        plan = partial_dependence_plan(profile, query)
        if plan:
            plans.append(plan)

    if want("leakage_scan"):
        plan = leakage_scan_plan(profile, query)
        if plan:
            plans.append(plan)

    if want("drift_check"):
        plan = drift_check_plan(profile, query)
        if plan:
            plans.append(plan)

    if want("missingness_mechanism"):
        plan = missingness_mechanism_plan(profile, query)
        if plan:
            plans.append(plan)

    if want("hypothesis_generator"):
        plan = hypothesis_generator_plan(profile, query)
        if plan:
            plans.append(plan)

    if want("test_selector"):
        plan = test_selector_plan(profile, query)
        if plan:
            plans.append(plan)

    if want("result_validator"):
        plan = result_validator_plan(profile, query)
        if plan:
            plans.append(plan)

    if want("cohort_change"):
        coh = cohort_change_plan(profile, query)
        if coh:
            plans.append(coh)

    if want("group_comparison"):
        comp = group_comparison_plan(profile, query)
        if comp:
            plans.append(comp)

    if want("percentile_compare"):
        pct = percentile_compare_plan(profile, query)
        if pct:
            plans.append(pct)

    if want("linear_regression"):
        lin = linear_regression_plan(profile, query)
        if lin:
            plans.append(lin)

    if want("logistic_regression"):
        logit = logistic_regression_plan(profile, query)
        if logit:
            plans.append(logit)

    return plans
