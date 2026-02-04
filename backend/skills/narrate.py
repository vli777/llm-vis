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

Keep it factual. Reference actual numbers from the profile and chart data.
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
                "chart_type": v.spec.chart_type.value,
                "data_points": len(v.data_inline),
                "fields": v.plan.fields_used,
            }
            # Include top/bottom value for bar charts
            if v.data_inline and v.spec.chart_type.value == "bar":
                summary["top"] = v.data_inline[0]
            view_summaries.append(summary)

        user_msg = json.dumps({
            "step_type": step.step_type.value,
            "table": profile.table_name,
            "row_count": profile.row_count,
            "columns": [c.name for c in profile.columns[:15]],
            "views": view_summaries,
        }, default=str)

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
) -> DecisionRecord:
    """
    Use LLM to decide what analysis step to run next.

    Falls back to the default policy if the LLM is unavailable.
    """
    try:
        from app.llm import chat_json

        done_types = [s.step_type.value for s in steps_done]
        user_msg = json.dumps({
            "table": profile.table_name,
            "row_count": profile.row_count,
            "columns": [c.name for c in profile.columns[:15]],
            "steps_done": done_types,
            "views_count": len(views_done),
            "budget_remaining": budget,
        }, default=str)

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
