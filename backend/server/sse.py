"""
Server-Sent Events (SSE) infrastructure.

Provides SSEEvent formatting and SSEChannel (async queue wrapper).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, AsyncIterator, Optional

from pydantic import BaseModel


class SSEEvent(BaseModel):
    """A single SSE message."""
    event: str
    data: Any = None
    id: Optional[str] = None

    def format(self) -> str:
        """Serialize to SSE wire format."""
        lines: list[str] = []
        if self.id:
            lines.append(f"id: {self.id}")
        lines.append(f"event: {self.event}")

        if self.data is not None:
            if isinstance(self.data, str):
                payload = self.data
            else:
                payload = json.dumps(self.data, default=str)
            for line in payload.split("\n"):
                lines.append(f"data: {line}")
        else:
            lines.append("data: {}")

        return "\n".join(lines) + "\n\n"


class SSEChannel:
    """
    Async queue wrapper for streaming SSE events.

    Usage:
        channel = SSEChannel()

        # Producer (in background task):
        await channel.emit("view_ready", {"id": "abc", ...})
        await channel.close()

        # Consumer (in SSE endpoint):
        async for event_str in channel:
            yield event_str
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Optional[SSEEvent]] = asyncio.Queue()
        self._closed = False

    async def emit(self, event: str, data: Any = None, event_id: Optional[str] = None) -> None:
        """Put an event onto the channel."""
        if self._closed:
            return
        sse = SSEEvent(
            event=event,
            data=data,
            id=event_id or str(uuid.uuid4())[:8],
        )
        await self._queue.put(sse)

    async def close(self) -> None:
        """Signal the consumer that no more events will arrive."""
        self._closed = True
        await self._queue.put(None)  # sentinel

    async def __aiter__(self) -> AsyncIterator[str]:
        """Yield formatted SSE strings until the channel is closed."""
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event.format()


# ---------------------------------------------------------------------------
# Standard event types (constants for consistency)
# ---------------------------------------------------------------------------

EVT_RUN_STARTED = "run_started"
EVT_PROGRESS = "progress"
EVT_STEP_STARTED = "step_started"
EVT_STEP_SUMMARY = "step_summary"
EVT_VIEW_PLANNED = "view_planned"
EVT_VIEW_READY = "view_ready"
EVT_WARNING = "warning"
EVT_ANALYSIS_INTENTS = "analysis_intents"
EVT_RUN_COMPLETE = "run_complete"
EVT_ERROR = "error"
EVT_QUESTION_RECEIVED = "question_received"
EVT_RETRIEVAL_RESULTS = "retrieval_results"
EVT_ANALYSIS_EXTENSION = "analysis_extension_started"
