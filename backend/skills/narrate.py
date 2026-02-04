"""
Narration skill — uses LLM to summarize steps and plan next actions.

Phase C: Agent as narrator + planner.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from core.models import (
    DataProfile,
    DecisionRecord,
    StepResult,
    StepType,
    ViewResult,
)

logger = logging.getLogger("uvicorn.error")

# Track whether we've already warned about LLM unavailability this session
_llm_warn_logged = False


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SUMMARIZE_SYSTEM = """You are a data analyst summarizing findings from an exploratory data analysis.

Given the dataset profile and a set of generated charts, produce a concise summary.

Return a JSON object with:
- "headline": string — a one-sentence summary with concrete numbers (e.g., "Revenue ranges from $50K to $2.1M with most companies under $500K")
- "findings": string[] — 2-4 bullet-point findings, each referencing specific values or patterns from the data

Rules:
- Do NOT mention charts, chart types, bins, or number of data points.
- Do NOT restate schema or generic dataset descriptions.
- Avoid planning language like "analysis priorities" or "next".
- Focus on concrete insights only (ranges, trends, top categories, relationships).
- Each finding should include at least one numeric value.
- Keep it factual and grounded in values shown.

Return ONLY valid JSON. No prose, no code fences."""

_PLAN_SYSTEM = """You are a data analysis planner deciding what to explore next.

Given the dataset profile, steps completed so far, and remaining budget, decide what analysis step to run next.

Available step types:
- quality_overview: data distributions, missing values, summary statistics
- relationships: correlations, trends, comparisons between columns
- outliers_segments: outlier detection, box plots, segment breakdowns

Return a JSON object with:
- "hypothesis": string — what you want to investigate
- "decision": string — which step_type to run ("quality_overview", "relationships", or "outliers_segments")
- "next_actions": string[] — specific things to look for

Return ONLY valid JSON. No prose, no code fences."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize_step(
    profile: DataProfile,
    step: StepResult,
    views: List[ViewResult],
    *,
    context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Use LLM to produce a headline + findings for a completed step.

    Falls back to a deterministic summary if the LLM is unavailable.
    """
    try:
        from app.llm import chat_json

        # Build compact context (keep payload small for reasoning models)
        view_summaries = []
        for v in views:
            summary: Dict[str, Any] = {
                "title": v.spec.title,
                "fields": v.plan.fields_used,
            }
            stats = _view_stats(v)
            if stats:
                summary["stats"] = stats
            view_summaries.append(summary)

        payload = {
            "step_type": step.step_type.value,
            "table": profile.table_name,
            "row_count": profile.row_count,
            "columns": [c.name for c in profile.columns[:15]],
            "views": view_summaries,
        }
        if context:
            payload["analysis_context"] = context
        user_msg = json.dumps(payload, default=str)

        result = chat_json(_SUMMARIZE_SYSTEM, user_msg)
        return {
            "headline": result.get("headline", step.headline),
            "findings": result.get("findings", step.findings),
        }

    except Exception as e:
        global _llm_warn_logged
        if not _llm_warn_logged:
            logger.warning("LLM narration unavailable, using deterministic fallback: %s", e)
            _llm_warn_logged = True
        return _fallback_summary(step, views, profile)


def plan_next_actions(
    profile: DataProfile,
    steps_done: List[StepResult],
    views_done: List[ViewResult],
    budget: int,
    *,
    context: Dict[str, Any] | None = None,
) -> DecisionRecord:
    """
    Use LLM to decide what analysis step to run next.

    Falls back to the default policy if the LLM is unavailable.
    """
    try:
        from app.llm import chat_json

        done_types = [s.step_type.value for s in steps_done]
        payload = {
            "table": profile.table_name,
            "row_count": profile.row_count,
            "columns": [c.name for c in profile.columns[:15]],
            "steps_done": done_types,
            "views_count": len(views_done),
            "budget_remaining": budget,
        }
        if context:
            payload["analysis_context"] = context
        user_msg = json.dumps(payload, default=str)

        result = chat_json(_PLAN_SYSTEM, user_msg)

        # Validate the decision is a valid step type
        decision = result.get("decision", "relationships")
        valid_types = {st.value for st in StepType}
        if decision not in valid_types:
            decision = "relationships"

        return DecisionRecord(
            hypothesis=result.get("hypothesis", ""),
            decision=decision,
            next_actions=result.get("next_actions", []),
        )

    except Exception as e:
        global _llm_warn_logged
        if not _llm_warn_logged:
            logger.warning("LLM planning unavailable, using deterministic fallback: %s", e)
            _llm_warn_logged = True
        return _fallback_plan(steps_done)


# ---------------------------------------------------------------------------
# Deterministic fallbacks
# ---------------------------------------------------------------------------

def _fallback_summary(
    step: StepResult,
    views: List[ViewResult],
    profile: DataProfile,
) -> Dict[str, Any]:
    """Generate a simple deterministic summary without LLM."""
    findings: List[str] = []

    for v in views:
        if v.data_inline:
            n = len(v.data_inline)
            findings.append(f"{v.spec.title}: {n} data points")

    headline = f"{step.step_type.value.replace('_', ' ').title()}: analyzed {len(views)} views across {profile.row_count} rows"

    return {"headline": headline, "findings": findings}


def _view_stats(view: ViewResult) -> Dict[str, Any]:
    """Compute light-weight stats from view data for LLM grounding."""
    data = view.data_inline or []
    if not data:
        return {}

    ct = view.spec.chart_type.value
    stats: Dict[str, Any] = {"chart_type": ct}

    if ct in {"bar", "hist"}:
        y = view.plan.encoding.y.field if view.plan.encoding.y else None
        if y and y in data[0]:
            vals = [d.get(y) for d in data if isinstance(d.get(y), (int, float))]
            if vals:
                stats.update({
                    "min": min(vals),
                    "max": max(vals),
                    "mean": sum(vals) / len(vals),
                })
        if ct == "bar":
            x = view.plan.encoding.x.field if view.plan.encoding.x else None
            if x and x in data[0]:
                stats["top"] = {"label": data[0].get(x), "value": data[0].get(y)}

    if ct == "line":
        x = view.plan.encoding.x.field if view.plan.encoding.x else None
        y = view.plan.encoding.y.field if view.plan.encoding.y else None
        if x and y and x in data[0] and y in data[0]:
            vals = [d.get(y) for d in data if isinstance(d.get(y), (int, float))]
            if vals:
                stats["start"] = vals[0]
                stats["end"] = vals[-1]
                stats["delta"] = vals[-1] - vals[0]

    if ct == "scatter":
        x = view.plan.encoding.x.field if view.plan.encoding.x else None
        y = view.plan.encoding.y.field if view.plan.encoding.y else None
        if x and y and x in data[0] and y in data[0]:
            xs = [d.get(x) for d in data if isinstance(d.get(x), (int, float))]
            ys = [d.get(y) for d in data if isinstance(d.get(y), (int, float))]
            n = min(len(xs), len(ys))
            if n >= 3:
                import numpy as np
                corr = float(np.corrcoef(xs[:n], ys[:n])[0, 1])
                stats["corr"] = round(corr, 4)

    if ct == "pie":
        theta = view.plan.encoding.theta.field if view.plan.encoding.theta else None
        if theta and theta in data[0]:
            vals = [d.get(theta) for d in data if isinstance(d.get(theta), (int, float))]
            if vals:
                stats["total"] = sum(vals)
                stats["top_share"] = max(vals) / max(1.0, sum(vals))

    if ct == "heatmap":
        if "count" in data[0]:
            vals = [d.get("count") for d in data if isinstance(d.get("count"), (int, float))]
            if vals:
                stats["max"] = max(vals)
                stats["min"] = min(vals)

    return stats


def _fallback_plan(steps_done: List[StepResult]) -> DecisionRecord:
    """Pick the next step type that hasn't been done yet."""
    done_types = {s.step_type for s in steps_done}

    priority = [
        StepType.quality_overview,
        StepType.relationships,
        StepType.outliers_segments,
    ]

    for st in priority:
        if st not in done_types:
            return DecisionRecord(
                hypothesis=f"Run {st.value} analysis",
                decision=st.value,
                next_actions=[f"Generate {st.value} views"],
            )

    return DecisionRecord(
        hypothesis="All standard steps completed",
        decision=StepType.relationships.value,
        next_actions=["Look for deeper relationships"],
    )
