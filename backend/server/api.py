"""
EDA API routes — mounted as a sub-router on the main FastAPI app.

Phase A: synchronous POST /api/runs, GET /api/runs/{run_id}
Phase B: SSE streaming via GET /api/runs/{run_id}/events
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from core.models import RunRequest
from core.storage import (
    get_run,
    get_session,
    get_session_meta,
    get_session_runs,
    create_run as storage_create_run,
)
from server.orchestrator import run_eda_async, run_eda_sync
from server.sse import SSEChannel

logger = logging.getLogger("uvicorn.error")

router = APIRouter(prefix="/api", tags=["eda"])

# Track active SSE channels for streaming runs
_active_channels: dict[str, SSEChannel] = {}


def _require_session_id(request: Request) -> str:
    sid = request.headers.get("X-Session-Id")
    if not sid:
        raise HTTPException(status_code=400, detail="Missing X-Session-Id header.")
    return sid


def _parse_created_at(meta: dict) -> float:
    created = meta.get("created_at") if isinstance(meta, dict) else None
    if isinstance(created, str) and created:
        try:
            return datetime.fromisoformat(created.replace("Z", "")).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def _pick_table_name(sess: dict, meta_store: dict, requested: Optional[str] = None) -> str:
    """Pick the most recent table or the explicitly requested one."""
    if not sess:
        raise HTTPException(status_code=400, detail="No tables uploaded.")

    if requested and requested in sess:
        return requested

    if meta_store:
        def sort_key(name: str):
            return _parse_created_at(meta_store.get(name, {}))
        return max(sess.keys(), key=sort_key)

    return next(reversed(sess.keys()))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/runs")
async def create_eda_run(request: Request, body: RunRequest = RunRequest()):
    """
    Start a new EDA run.

    If 'stream' query param is set, returns {run_id} immediately and starts
    background processing; connect to /api/runs/{run_id}/events for SSE.
    Otherwise returns the complete EDAReport synchronously.
    """
    sid = _require_session_id(request)
    sess = get_session(sid)
    meta_store = get_session_meta(sid)

    table_name = _pick_table_name(sess, meta_store, body.table_name)
    df = sess[table_name]

    # Check for streaming mode
    stream = request.query_params.get("stream", "").lower() in ("1", "true", "yes")

    if stream:
        # Create a run ID and channel, start background task
        channel = SSEChannel()
        run_id = str(uuid.uuid4())

        # Pre-create the run in storage so GET /runs/{id} works immediately
        from core.storage import RUNS, SESSION_RUNS, RUN_VIEWS
        from core.models import EDAReport
        report = EDAReport(run_id=run_id, table_name=table_name)
        RUNS[run_id] = report
        RUN_VIEWS[run_id] = []
        SESSION_RUNS.setdefault(sid, []).append(run_id)

        _active_channels[run_id] = channel

        async def _bg():
            try:
                await run_eda_async(df, table_name, sid, channel, query=body.query)
            except Exception as e:
                logger.exception("Async EDA run failed")
                await channel.emit("error", {"message": str(e)})
                await channel.close()
            finally:
                _active_channels.pop(run_id, None)

        asyncio.create_task(_bg())
        return {"run_id": run_id, "streaming": True}

    # Synchronous mode (Phase A)
    try:
        report = run_eda_sync(df, table_name, sid, query=body.query)
    except Exception as e:
        logger.exception("EDA run failed")
        raise HTTPException(status_code=500, detail=f"EDA run failed: {e}")

    return report.model_dump()


@router.get("/runs/{run_id}/events")
async def stream_eda_events(
    request: Request,
    run_id: str,
    session_id: str = Query(None, alias="session_id"),
):
    """
    SSE endpoint — streams events for an active run.

    EventSource doesn't support custom headers, so session_id is passed
    as a query parameter.
    """
    # Validate session (from query param since EventSource can't set headers)
    sid = session_id or request.headers.get("X-Session-Id")
    if not sid:
        raise HTTPException(status_code=400, detail="Missing session_id query parameter.")

    channel = _active_channels.get(run_id)
    if channel is None:
        # Run may have already completed — return the full report as a single event
        report = get_run(run_id)
        if report is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

        async def _completed():
            from server.sse import SSEEvent
            yield SSEEvent(
                event="run_complete",
                data=report.model_dump(),
            ).format()

        return StreamingResponse(
            _completed(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    async def _stream():
        async for event_str in channel:
            yield event_str

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/runs/{run_id}")
async def get_eda_run(request: Request, run_id: str):
    """Retrieve a previously completed EDA run."""
    _require_session_id(request)
    report = get_run(run_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return report.model_dump()


@router.get("/runs")
async def list_eda_runs(request: Request):
    """List all EDA runs for this session."""
    sid = _require_session_id(request)
    run_ids = get_session_runs(sid)
    runs = []
    for rid in run_ids:
        report = get_run(rid)
        if report:
            runs.append({
                "run_id": report.run_id,
                "table_name": report.table_name,
                "views_count": len(report.views),
                "steps_count": len(report.steps),
            })
    return {"runs": runs}
