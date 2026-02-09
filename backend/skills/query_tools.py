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


def query_tool_plans(profile: DataProfile, query: str, *, tools: Optional[List[str]] = None) -> List[ViewPlan]:
    plans: List[ViewPlan] = []
    tool_set = set(tools or [])

    def want(name: str) -> bool:
        return not tool_set or name in tool_set

    if want("segmentation"):
        seg = segmentation_plan(profile, query)
        if seg:
            plans.append(seg)

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
