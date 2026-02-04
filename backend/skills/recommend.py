"""
Deterministic chart recommendation — no LLM.

generate_candidates() produces ViewPlan candidates for a given step type.
score_and_select() scores and deduplicates them.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import re

from core.models import (
    ChartEncoding,
    ChartOptions,
    ChartType,
    ColumnInfo,
    ColumnRole,
    DataProfile,
    EncodingChannel,
    StepType,
    ViewPlan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _measures(profile: DataProfile) -> List[ColumnInfo]:
    return [c for c in profile.columns if c.role in (ColumnRole.measure, ColumnRole.count)]


def _categoricals(profile: DataProfile) -> List[ColumnInfo]:
    return [c for c in profile.columns if c.role in (ColumnRole.categorical, ColumnRole.nominal, ColumnRole.geographic)]


def _temporals(profile: DataProfile) -> List[ColumnInfo]:
    return [c for c in profile.columns if c.role == ColumnRole.temporal]


def _low_card_cats(profile: DataProfile, threshold: int = 10) -> List[ColumnInfo]:
    return [c for c in _categoricals(profile) if c.cardinality <= threshold]


def _high_card_cats(profile: DataProfile, threshold: int = 10) -> List[ColumnInfo]:
    return [c for c in _categoricals(profile) if c.cardinality > threshold]


def _missing_cols(profile: DataProfile, threshold: float = 1.0) -> List[ColumnInfo]:
    return [c for c in profile.columns if c.missing_pct >= threshold]


# ---------------------------------------------------------------------------
# Candidate generation per step type
# ---------------------------------------------------------------------------

def _quality_candidates(profile: DataProfile) -> List[ViewPlan]:
    """Generate overview/quality charts: distributions, missingness, top-values."""
    plans: List[ViewPlan] = []
    measures = _measures(profile)
    cats = _categoricals(profile)
    missing = _missing_cols(profile)

    # Histogram for each measure (up to 3)
    for m in measures[:3]:
        plans.append(ViewPlan(
            chart_type=ChartType.hist,
            encoding=ChartEncoding(
                x=EncodingChannel(field=m.name, type="quantitative", bin=True),
                y=EncodingChannel(field=m.name, type="quantitative", aggregate="count"),
            ),
            intent=f"Distribution of {m.name}",
            fields_used=[m.name],
            tags=["overview", "distribution"],
        ))

    # Bar chart for top categorical values (up to 2)
    for cat in cats[:2]:
        top_n = min(cat.cardinality, 15)
        plans.append(ViewPlan(
            chart_type=ChartType.bar,
            encoding=ChartEncoding(
                x=EncodingChannel(field=cat.name, type="nominal"),
                y=EncodingChannel(field=cat.name, type="quantitative", aggregate="count"),
            ),
            options=ChartOptions(top_n=top_n, sort="descending"),
            intent=f"Frequency of {cat.name}",
            fields_used=[cat.name],
            tags=["overview", "frequency"],
        ))

    # Missingness bar chart
    if missing:
        plans.append(ViewPlan(
            chart_type=ChartType.bar,
            encoding=ChartEncoding(
                x=EncodingChannel(field="__column__", type="nominal"),
                y=EncodingChannel(field="__missing_pct__", type="quantitative"),
            ),
            intent="Missing data by column",
            fields_used=[c.name for c in missing],
            tags=["overview", "quality", "missingness"],
        ))

    return plans


def _relationship_candidates(profile: DataProfile) -> List[ViewPlan]:
    """Generate relationship/correlation charts."""
    plans: List[ViewPlan] = []
    measures = _measures(profile)
    cats = _categoricals(profile)
    temporals = _temporals(profile)

    # Time series: temporal x measure
    for t in temporals[:1]:
        for m in measures[:3]:
            plans.append(ViewPlan(
                chart_type=ChartType.line,
                encoding=ChartEncoding(
                    x=EncodingChannel(field=t.name, type="temporal"),
                    y=EncodingChannel(field=m.name, type="quantitative"),
                ),
                intent=f"{m.name} over {t.name}",
                fields_used=[t.name, m.name],
                tags=["relationship", "trend"],
            ))

    # Scatter: measure vs measure
    if len(measures) >= 2:
        pairs_done: set = set()
        for i, m1 in enumerate(measures):
            for m2 in measures[i + 1 :]:
                if len(pairs_done) >= 3:
                    break
                key = tuple(sorted([m1.name, m2.name]))
                if key in pairs_done:
                    continue
                pairs_done.add(key)
                color_channel = None
                if cats:
                    best_cat = min(cats, key=lambda c: abs(c.cardinality - 5))
                    if best_cat.cardinality <= 20:
                        color_channel = EncodingChannel(field=best_cat.name, type="nominal")
                plans.append(ViewPlan(
                    chart_type=ChartType.scatter,
                    encoding=ChartEncoding(
                        x=EncodingChannel(field=m1.name, type="quantitative"),
                        y=EncodingChannel(field=m2.name, type="quantitative"),
                        color=color_channel,
                    ),
                    intent=f"{m1.name} vs {m2.name}",
                    fields_used=[m1.name, m2.name] + ([color_channel.field] if color_channel else []),
                    tags=["relationship", "correlation"],
                ))

    # Grouped bar: categorical x measure
    for cat in cats[:2]:
        for m in measures[:2]:
            plans.append(ViewPlan(
                chart_type=ChartType.bar,
                encoding=ChartEncoding(
                    x=EncodingChannel(field=cat.name, type="nominal"),
                    y=EncodingChannel(field=m.name, type="quantitative", aggregate="mean"),
                ),
                options=ChartOptions(sort="descending", top_n=min(cat.cardinality, 15)),
                intent=f"Average {m.name} by {cat.name}",
                fields_used=[cat.name, m.name],
                tags=["relationship", "comparison"],
            ))

    # Heatmap: two categoricals
    if len(cats) >= 2:
        c1, c2 = cats[0], cats[1]
        if c1.cardinality <= 20 and c2.cardinality <= 20:
            plans.append(ViewPlan(
                chart_type=ChartType.heatmap,
                encoding=ChartEncoding(
                    x=EncodingChannel(field=c1.name, type="nominal"),
                    y=EncodingChannel(field=c2.name, type="nominal"),
                    color=EncodingChannel(field="count", type="quantitative", aggregate="count"),
                ),
                intent=f"{c1.name} vs {c2.name} co-occurrence",
                fields_used=[c1.name, c2.name],
                tags=["relationship", "heatmap"],
            ))

    return plans


def _outlier_candidates(profile: DataProfile) -> List[ViewPlan]:
    """Generate outlier/segment charts: box plots, pie for low-card."""
    plans: List[ViewPlan] = []
    measures = _measures(profile)
    cats = _categoricals(profile)
    low_card = _low_card_cats(profile)

    # Box plots: measure grouped by categorical
    for m in measures[:2]:
        for cat in cats[:1]:
            plans.append(ViewPlan(
                chart_type=ChartType.box,
                encoding=ChartEncoding(
                    x=EncodingChannel(field=cat.name, type="nominal"),
                    y=EncodingChannel(field=m.name, type="quantitative"),
                ),
                intent=f"Distribution of {m.name} by {cat.name}",
                fields_used=[cat.name, m.name],
                tags=["outliers", "distribution"],
            ))

    # Box plot for each measure (ungrouped) — up to 2
    for m in measures[:2]:
        plans.append(ViewPlan(
            chart_type=ChartType.box,
            encoding=ChartEncoding(
                y=EncodingChannel(field=m.name, type="quantitative"),
            ),
            intent=f"Outlier detection for {m.name}",
            fields_used=[m.name],
            tags=["outliers"],
        ))

    # Pie for low-cardinality categoricals
    for cat in low_card[:2]:
        plans.append(ViewPlan(
            chart_type=ChartType.pie,
            encoding=ChartEncoding(
                theta=EncodingChannel(field=cat.name, type="quantitative", aggregate="count"),
                color=EncodingChannel(field=cat.name, type="nominal"),
            ),
            intent=f"Segment breakdown by {cat.name}",
            fields_used=[cat.name],
            tags=["segments", "composition"],
        ))

    return plans


def _query_driven_candidates(
    profile: DataProfile, *, query: str = "",
) -> List[ViewPlan]:
    """
    Generate candidates guided by a user's natural language query.

    Tokenises the query and matches against column names / roles to find
    the most relevant columns, then builds chart plans around them.
    """
    plans: List[ViewPlan] = []
    if not query:
        return plans

    tokens = set(re.findall(r"[a-z0-9]+", query.lower()))

    # Score each column by how many query tokens appear in its name
    scored: List[tuple[int, ColumnInfo]] = []
    for col in profile.columns:
        col_tokens = set(re.findall(r"[a-z0-9]+", col.name.lower()))
        overlap = len(tokens & col_tokens)
        if overlap > 0:
            scored.append((overlap, col))
    scored.sort(key=lambda t: t[0], reverse=True)

    matched = [col for _, col in scored]

    # Fallback: if no column matched, use the top measures + categoricals
    if not matched:
        matched = _measures(profile)[:2] + _categoricals(profile)[:2]

    matched_measures = [c for c in matched if c.role in (ColumnRole.measure, ColumnRole.count)]
    matched_cats = [c for c in matched if c.role in (ColumnRole.categorical, ColumnRole.nominal, ColumnRole.geographic)]
    matched_temporals = [c for c in matched if c.role == ColumnRole.temporal]

    # If no measures matched, supplement from profile
    if not matched_measures:
        matched_measures = _measures(profile)[:2]
    if not matched_cats:
        matched_cats = _categoricals(profile)[:1]

    # Keyword hints for chart types
    wants_trend = bool(tokens & {"trend", "over", "time", "growth", "change", "timeline"})
    wants_compare = bool(tokens & {"compare", "comparison", "vs", "versus", "by", "per", "across"})
    wants_dist = bool(tokens & {"distribution", "histogram", "spread", "range", "outlier", "outliers"})
    wants_breakdown = bool(tokens & {"breakdown", "composition", "share", "proportion", "percentage", "pie"})
    wants_corr = bool(tokens & {"correlation", "scatter", "relationship", "correlate"})

    temporals = matched_temporals or _temporals(profile)

    # Trend: line chart over temporal
    if (wants_trend or temporals) and matched_measures:
        t = temporals[0] if temporals else None
        if t:
            for m in matched_measures[:2]:
                plans.append(ViewPlan(
                    chart_type=ChartType.line,
                    encoding=ChartEncoding(
                        x=EncodingChannel(field=t.name, type="temporal"),
                        y=EncodingChannel(field=m.name, type="quantitative"),
                    ),
                    intent=f"{m.name} over {t.name}",
                    fields_used=[t.name, m.name],
                    tags=["query_driven", "trend"],
                ))

    # Comparison: bar chart of measure by category
    if wants_compare or (matched_cats and matched_measures):
        for cat in matched_cats[:2]:
            for m in matched_measures[:2]:
                plans.append(ViewPlan(
                    chart_type=ChartType.bar,
                    encoding=ChartEncoding(
                        x=EncodingChannel(field=cat.name, type="nominal"),
                        y=EncodingChannel(field=m.name, type="quantitative", aggregate="mean"),
                    ),
                    options=ChartOptions(sort="descending", top_n=min(cat.cardinality, 15)),
                    intent=f"{m.name} by {cat.name}",
                    fields_used=[cat.name, m.name],
                    tags=["query_driven", "comparison"],
                ))

    # Distribution: histogram
    if wants_dist and matched_measures:
        for m in matched_measures[:2]:
            plans.append(ViewPlan(
                chart_type=ChartType.hist,
                encoding=ChartEncoding(
                    x=EncodingChannel(field=m.name, type="quantitative", bin=True),
                    y=EncodingChannel(field=m.name, type="quantitative", aggregate="count"),
                ),
                intent=f"Distribution of {m.name}",
                fields_used=[m.name],
                tags=["query_driven", "distribution"],
            ))

    # Breakdown: pie chart
    if wants_breakdown:
        low_card = _low_card_cats(profile)
        for cat in (low_card or matched_cats)[:1]:
            plans.append(ViewPlan(
                chart_type=ChartType.pie,
                encoding=ChartEncoding(
                    theta=EncodingChannel(field=cat.name, type="quantitative", aggregate="count"),
                    color=EncodingChannel(field=cat.name, type="nominal"),
                ),
                intent=f"Breakdown by {cat.name}",
                fields_used=[cat.name],
                tags=["query_driven", "composition"],
            ))

    # Correlation: scatter plot
    if wants_corr and len(matched_measures) >= 2:
        m1, m2 = matched_measures[0], matched_measures[1]
        plans.append(ViewPlan(
            chart_type=ChartType.scatter,
            encoding=ChartEncoding(
                x=EncodingChannel(field=m1.name, type="quantitative"),
                y=EncodingChannel(field=m2.name, type="quantitative"),
            ),
            intent=f"{m1.name} vs {m2.name}",
            fields_used=[m1.name, m2.name],
            tags=["query_driven", "correlation"],
        ))

    # If nothing matched the keywords, generate a mixed set using matched columns
    if not plans:
        if matched_measures and matched_cats:
            m = matched_measures[0]
            cat = matched_cats[0]
            plans.append(ViewPlan(
                chart_type=ChartType.bar,
                encoding=ChartEncoding(
                    x=EncodingChannel(field=cat.name, type="nominal"),
                    y=EncodingChannel(field=m.name, type="quantitative", aggregate="mean"),
                ),
                options=ChartOptions(sort="descending", top_n=min(cat.cardinality, 15)),
                intent=f"{m.name} by {cat.name}",
                fields_used=[cat.name, m.name],
                tags=["query_driven"],
            ))
        if matched_measures:
            m = matched_measures[0]
            plans.append(ViewPlan(
                chart_type=ChartType.hist,
                encoding=ChartEncoding(
                    x=EncodingChannel(field=m.name, type="quantitative", bin=True),
                    y=EncodingChannel(field=m.name, type="quantitative", aggregate="count"),
                ),
                intent=f"Distribution of {m.name}",
                fields_used=[m.name],
                tags=["query_driven", "distribution"],
            ))

    return plans


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

_GENERATORS = {
    StepType.quality_overview: _quality_candidates,
    StepType.relationships: _relationship_candidates,
    StepType.outliers_segments: _outlier_candidates,
}


def generate_candidates(
    profile: DataProfile,
    step_type: StepType,
    *,
    query: str = "",
    intents: Optional[List[str]] = None,
) -> List[ViewPlan]:
    """Generate candidate ViewPlans for a given analysis step."""
    if step_type == StepType.query_driven:
        return _query_driven_candidates(profile, query=query)
    if step_type == StepType.intent_views:
        all_candidates = (
            _quality_candidates(profile)
            + _relationship_candidates(profile)
            + _outlier_candidates(profile)
        )
        return _intent_filter(all_candidates, intents or [])
    gen = _GENERATORS.get(step_type)
    if gen is None:
        return []
    candidates = gen(profile)
    if intents:
        return _intent_filter(candidates, intents)
    return candidates


def _intent_filter(candidates: List[ViewPlan], intents: List[str]) -> List[ViewPlan]:
    intent_text = " ".join(intents).lower()
    filtered: List[ViewPlan] = []
    for plan in candidates:
        intent_match = any(
            field.lower() in intent_text or field.lower() in (plan.intent or "").lower()
            for field in plan.fields_used
        )
        tag_match = any(tag in intent_text for tag in plan.tags)
        if intent_match or tag_match:
            filtered.append(plan)
    return filtered


def score_and_select(
    candidates: List[ViewPlan],
    views_done: List[ViewPlan],
    budget: int = 4,
) -> List[ViewPlan]:
    """
    Score candidates and return top *budget* views, avoiding redundancy.

    Scoring criteria:
      - Field coverage (reward fields not yet visualized)
      - Chart type diversity (reward types not yet used)
      - Penalize redundancy (exact same fields_used + chart_type)
    """
    if not candidates:
        return []

    done_fields: Set[str] = set()
    done_types: Set[ChartType] = set()
    done_keys: Set[str] = set()

    for v in views_done:
        done_fields.update(v.fields_used)
        done_types.add(v.chart_type)
        done_keys.add(_plan_key(v))

    scored: List[tuple] = []
    for plan in candidates:
        key = _plan_key(plan)
        if key in done_keys:
            continue  # exact duplicate

        new_fields = len(set(plan.fields_used) - done_fields)
        type_bonus = 2.0 if plan.chart_type not in done_types else 0.0
        field_coverage = new_fields * 1.5
        score = field_coverage + type_bonus
        scored.append((score, plan))

    scored.sort(key=lambda t: t[0], reverse=True)
    selected: List[ViewPlan] = []
    selected_keys: Set[str] = set(done_keys)

    for _score, plan in scored:
        if len(selected) >= budget:
            break
        key = _plan_key(plan)
        if key in selected_keys:
            continue
        selected_keys.add(key)
        selected.append(plan)

    return selected


def _plan_key(plan: ViewPlan) -> str:
    """Unique key for deduplication: chart_type + sorted fields."""
    fields = ",".join(sorted(plan.fields_used))
    return f"{plan.chart_type.value}:{fields}"
