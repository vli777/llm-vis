"""
EDA orchestrator — runs the deterministic analysis pipeline.

Phase A: synchronous run_eda_sync
Phase B: async run_eda_async with SSE channel
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import pandas as pd

from core.models import (
    DataProfile,
    EDAReport,
    StepResult,
    StepType,
    ViewPlan,
    ViewResult,
)
from core.storage import append_view, create_run, save_run
from server.sse import (
    SSEChannel,
    EVT_ERROR,
    EVT_PROGRESS,
    EVT_RUN_COMPLETE,
    EVT_RUN_STARTED,
    EVT_STEP_STARTED,
    EVT_STEP_SUMMARY,
    EVT_ANALYSIS_INTENTS,
    EVT_VIEW_PLANNED,
    EVT_VIEW_READY,
    EVT_WARNING,
)
from skills.build_view import build_view
from skills.classify import classify_columns
from skills.narrate import plan_next_actions, summarize_step
from skills.profile import build_profile
from skills.recommend import generate_candidates, score_and_select
from skills.intent import infer_analysis_intents
from skills.validate import deterministic_fallback, validate_plan, validate_view

logger = logging.getLogger("uvicorn.error")

# Default analysis pipeline
DEFAULT_POLICY: List[StepType] = [
    StepType.quality_overview,
    StepType.relationships,
    StepType.outliers_segments,
]

# Budget per step (max charts generated per step)
STEP_BUDGET = 4


# ---------------------------------------------------------------------------
# Synchronous pipeline (Phase A)
# ---------------------------------------------------------------------------

def run_eda_sync(
    df: pd.DataFrame,
    table_name: str,
    session_id: str,
    *,
    query: Optional[str] = None,
    budget: int = 12,
) -> EDAReport:
    """
    Run the full deterministic EDA pipeline synchronously.

    1. Profile the dataset
    2. For each step in DEFAULT_POLICY:
       a. Generate candidate views
       b. Score & select top views (respecting budget)
       c. Validate each plan
       d. Build each view (pre-aggregate data)
       e. Validate the result
       f. Append to the report
    3. Return the complete EDAReport
    """
    # Create run
    report = create_run(session_id, table_name)
    logger.info("EDA run %s started for table '%s' (%d rows)", report.run_id, table_name, len(df))

    # Step 1: Profile
    profile = build_profile(df, table_name)
    profile = classify_columns(profile)
    report.profile = profile
    logger.info("Profile complete: %d columns, %d rows", len(profile.columns), profile.row_count)

    # Track all views for deduplication
    views_done: List[ViewPlan] = []
    all_view_results: List[ViewResult] = []
    total_views = 0

    # Build the step pipeline: prepend query_driven if a query was given
    pipeline = list(DEFAULT_POLICY)
    if query:
        pipeline.insert(0, StepType.query_driven)

    # Step 2: Run each analysis step
    for step_type in pipeline:
        if total_views >= budget:
            break

        step_budget = min(STEP_BUDGET, budget - total_views)
        logger.info("Step: %s (budget=%d)", step_type.value, step_budget)

        # Generate candidates
        candidates = generate_candidates(profile, step_type, query=query or "")
        logger.info("  Generated %d candidates", len(candidates))

        # Select best
        selected = score_and_select(candidates, views_done, budget=step_budget)
        logger.info("  Selected %d views", len(selected))

        step_views: List[str] = []
        step_findings: List[str] = []
        step_warnings: List[str] = []
        step_view_results: List[ViewResult] = []

        for plan in selected:
            # Validate plan
            is_valid, plan_warnings = validate_plan(plan, profile)
            step_warnings.extend(plan_warnings)

            if not is_valid:
                logger.warning("  Plan invalid: %s — attempting fallback", plan_warnings)
                plan = deterministic_fallback(plan, profile)
                is_valid, plan_warnings = validate_plan(plan, profile)
                if not is_valid:
                    logger.warning("  Fallback also invalid, skipping: %s", plan_warnings)
                    continue

            # Build view
            try:
                view = build_view(plan, df)
            except Exception as e:
                logger.warning("  Build failed for '%s': %s", plan.intent, e)
                step_warnings.append(f"Build failed: {e}")
                continue

            # Validate view
            view_valid, view_warnings = validate_view(view)
            step_warnings.extend(view_warnings)

            if not view_valid:
                logger.warning("  View invalid: %s", view_warnings)
                continue

            # Accept the view
            append_view(report.run_id, view)
            step_views.append(view.id)
            step_view_results.append(view)
            views_done.append(plan)
            all_view_results.append(view)
            total_views += 1

            step_findings.append(view.explanation)
            logger.info("  View built: %s (%s, %d rows)",
                         view.spec.title, view.spec.chart_type.value, len(view.data_inline))

        # Record step result
        step_result = StepResult(
            step_type=step_type,
            headline=f"{step_type.value.replace('_', ' ').title()}: {len(step_views)} views",
            views=step_views,
            findings=step_findings,
            warnings=step_warnings,
        )

        # Narrate: enrich headline/findings via LLM (falls back to deterministic)
        if step_view_results:
            narration = summarize_step(profile, step_result, step_view_results)
            step_result.headline = narration.get("headline", step_result.headline)
            step_result.findings = narration.get("findings", step_result.findings)

        report.steps.append(step_result)

        if step_type == StepType.quality_overview and report.analysis_insights is None:
            report.analysis_insights = infer_analysis_intents(profile)

        # Record decision in timeline
        decision = plan_next_actions(profile, report.steps, all_view_results, budget - total_views)
        report.timeline.append(decision)

    logger.info("EDA run %s complete: %d views across %d steps",
                report.run_id, total_views, len(report.steps))

    # Persist
    save_run(report)
    return report


# ---------------------------------------------------------------------------
# Thread pool for CPU-bound view building
# ---------------------------------------------------------------------------

_executor = ThreadPoolExecutor(max_workers=2)


# ---------------------------------------------------------------------------
# Async pipeline (Phase B) — same logic, emits SSE events
# ---------------------------------------------------------------------------

async def run_eda_async(
    df: pd.DataFrame,
    table_name: str,
    session_id: str,
    channel: SSEChannel,
    *,
    query: Optional[str] = None,
    budget: int = 12,
) -> EDAReport:
    """
    Async EDA pipeline that streams progress via SSE.

    Same logic as run_eda_sync but emits events at each milestone.
    CPU-bound view building runs in a thread pool executor.
    """
    loop = asyncio.get_event_loop()

    report = create_run(session_id, table_name)
    await channel.emit(EVT_RUN_STARTED, {
        "run_id": report.run_id,
        "table_name": table_name,
        "row_count": len(df),
    })

    # Profile (CPU-bound)
    profile = await loop.run_in_executor(_executor, build_profile, df, table_name)
    profile = await loop.run_in_executor(_executor, classify_columns, profile)
    report.profile = profile
    await channel.emit(EVT_PROGRESS, {
        "stage": "profile_complete",
        "columns": len(profile.columns),
        "row_count": profile.row_count,
    })

    views_done: List[ViewPlan] = []
    all_view_results: List[ViewResult] = []
    total_views = 0

    # Build the step pipeline: prepend query_driven if a query was given
    pipeline = list(DEFAULT_POLICY)
    if query:
        pipeline.insert(0, StepType.query_driven)

    for step_type in pipeline:
        if total_views >= budget:
            break

        step_budget = min(STEP_BUDGET, budget - total_views)
        await channel.emit(EVT_STEP_STARTED, {
            "step_type": step_type.value,
            "budget": step_budget,
        })

        candidates = generate_candidates(profile, step_type, query=query or "")
        selected = score_and_select(candidates, views_done, budget=step_budget)

        step_views: List[str] = []
        step_findings: List[str] = []
        step_warnings: List[str] = []
        step_view_results: List[ViewResult] = []

        for plan in selected:
            is_valid, plan_warnings = validate_plan(plan, profile)
            step_warnings.extend(plan_warnings)

            if not is_valid:
                plan = deterministic_fallback(plan, profile)
                is_valid, plan_warnings = validate_plan(plan, profile)
                if not is_valid:
                    for w in plan_warnings:
                        await channel.emit(EVT_WARNING, {"message": w})
                    continue

            await channel.emit(EVT_VIEW_PLANNED, {
                "intent": plan.intent,
                "chart_type": plan.chart_type.value,
            })

            # Build view (CPU-bound — run in executor)
            try:
                view = await loop.run_in_executor(_executor, build_view, plan, df)
            except Exception as e:
                logger.warning("Build failed: %s", e)
                await channel.emit(EVT_WARNING, {"message": f"Build failed: {e}"})
                continue

            view_valid, view_warnings = validate_view(view)
            step_warnings.extend(view_warnings)
            if not view_valid:
                for w in view_warnings:
                    await channel.emit(EVT_WARNING, {"message": w})
                continue

            append_view(report.run_id, view)
            step_views.append(view.id)
            step_view_results.append(view)
            views_done.append(plan)
            all_view_results.append(view)
            total_views += 1

            step_findings.append(view.explanation)

            # Stream the view to the client
            await channel.emit(EVT_VIEW_READY, view.model_dump())

        step_result = StepResult(
            step_type=step_type,
            headline=f"{step_type.value.replace('_', ' ').title()}: {len(step_views)} views",
            views=step_views,
            findings=step_findings,
            warnings=step_warnings,
        )

        # Narrate: enrich headline/findings (runs in executor to avoid blocking SSE)
        if step_view_results:
            narration = await loop.run_in_executor(
                _executor, summarize_step, profile, step_result, step_view_results,
            )
            step_result.headline = narration.get("headline", step_result.headline)
            step_result.findings = narration.get("findings", step_result.findings)

        report.steps.append(step_result)

        if step_type == StepType.quality_overview and report.analysis_insights is None:
            report.analysis_insights = await loop.run_in_executor(_executor, infer_analysis_intents, profile)
            await channel.emit(EVT_ANALYSIS_INTENTS, report.analysis_insights.model_dump())

        # Record decision in timeline
        decision = plan_next_actions(profile, report.steps, all_view_results, budget - total_views)
        report.timeline.append(decision)

        await channel.emit(EVT_STEP_SUMMARY, {
            "step_type": step_type.value,
            "headline": step_result.headline,
            "view_count": len(step_views),
            "findings": step_result.findings,
        })

    save_run(report)

    await channel.emit(EVT_RUN_COMPLETE, {
        "run_id": report.run_id,
        "total_views": total_views,
        "total_steps": len(report.steps),
    })
    await channel.close()

    return report
