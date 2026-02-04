"""
In-memory session + run/view storage.

Extends the original session_store.py with EDA run tracking.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from .models import EDAReport, ViewResult


# ---------------------------------------------------------------------------
# Original session store (unchanged API)
# ---------------------------------------------------------------------------

SESSIONS: Dict[str, Dict[str, pd.DataFrame]] = {}
SESS_HASHES: Dict[str, Dict[str, str]] = {}
SESS_META: Dict[str, Dict[str, dict]] = {}
SAMPLE_TABLES: Dict[str, pd.DataFrame] = {}


def get_session(session_id: str) -> Dict[str, pd.DataFrame]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {}
    return SESSIONS[session_id]


def get_session_hashes(session_id: str) -> Dict[str, str]:
    if session_id not in SESS_HASHES:
        SESS_HASHES[session_id] = {}
    return SESS_HASHES[session_id]


def get_session_meta(session_id: str) -> Dict[str, dict]:
    if session_id not in SESS_META:
        SESS_META[session_id] = {}
    return SESS_META[session_id]


# ---------------------------------------------------------------------------
# EDA run store
# ---------------------------------------------------------------------------

# run_id -> EDAReport
RUNS: Dict[str, EDAReport] = {}

# run_id -> [ViewResult, ...]
RUN_VIEWS: Dict[str, List[ViewResult]] = {}

# session_id -> [run_id, ...]
SESSION_RUNS: Dict[str, List[str]] = {}


def create_run(session_id: str, table_name: str) -> EDAReport:
    """Create a new EDA run and associate it with the session."""
    report = EDAReport(
        run_id=str(uuid.uuid4()),
        table_name=table_name,
    )
    RUNS[report.run_id] = report
    RUN_VIEWS[report.run_id] = []
    SESSION_RUNS.setdefault(session_id, []).append(report.run_id)
    return report


def get_run(run_id: str) -> Optional[EDAReport]:
    return RUNS.get(run_id)


def save_run(report: EDAReport) -> None:
    """Persist (overwrite) the report in the store."""
    RUNS[report.run_id] = report


def append_view(run_id: str, view: ViewResult) -> None:
    """Append a view to the run's view list and to the report."""
    RUN_VIEWS.setdefault(run_id, []).append(view)
    report = RUNS.get(run_id)
    if report is not None:
        report.views.append(view)


def get_views(run_id: str) -> List[ViewResult]:
    return RUN_VIEWS.get(run_id, [])


def set_run_views(run_id: str, views: List[ViewResult]) -> None:
    """Replace the stored views for a run (used when reusing prior results)."""
    RUN_VIEWS[run_id] = list(views)


def get_session_runs(session_id: str) -> List[str]:
    return SESSION_RUNS.get(session_id, [])


def get_latest_run_for_table(session_id: str, table_name: str) -> Optional[EDAReport]:
    """Return most recent run for a given table in the session."""
    run_ids = SESSION_RUNS.get(session_id, [])
    for run_id in reversed(run_ids):
        report = RUNS.get(run_id)
        if report and report.table_name == table_name:
            return report
    return None
