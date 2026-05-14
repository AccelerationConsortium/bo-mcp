"""MCP progress-bridge wiring (TODO 1.48).

Background: bo-engine emits :class:`bo_engine.progress.ProgressEvent`
via a synchronous callback because most of its work runs under
``asyncio.to_thread``. The MCP session's ``report_progress`` is async,
so the bridge marshals events back to the loop via
``asyncio.run_coroutine_threadsafe``.

The tests deliberately do not bring in an MCP server — they call
:func:`build_progress_callback` with a stub async sink so the contract
is exercised in isolation.
"""

from __future__ import annotations

import asyncio

import pytest
from bo_engine.progress import ProgressEvent

from bo_mcp_server.progress_bridge import build_progress_callback


@pytest.mark.asyncio
async def test_callback_forwards_event_to_async_sink() -> None:
    """A single event is delivered to the async sink with the right shape."""
    received: list[tuple[float, float | None, str | None]] = []

    async def sink(progress: float, total: float | None, message: str | None) -> None:
        received.append((progress, total, message))

    loop = asyncio.get_running_loop()
    callback = build_progress_callback(sink, loop)

    callback(ProgressEvent(phase="gp_fit_start", message="fitting GP", total=10.0))

    # Allow the scheduled coroutine to run.
    for _ in range(10):
        await asyncio.sleep(0)
        if received:
            break

    assert received == [(0.0, 10.0, "gp_fit_start: fitting GP")]


@pytest.mark.asyncio
async def test_callback_with_explicit_progress_value() -> None:
    """``ProgressEvent.progress`` propagates verbatim when set."""
    received: list[tuple[float, float | None, str | None]] = []

    async def sink(progress: float, total: float | None, message: str | None) -> None:
        received.append((progress, total, message))

    loop = asyncio.get_running_loop()
    callback = build_progress_callback(sink, loop)

    callback(
        ProgressEvent(
            phase="acq_step",
            message="optimizing acquisition",
            progress=3.0,
            total=5.0,
        )
    )

    for _ in range(10):
        await asyncio.sleep(0)
        if received:
            break

    assert received == [(3.0, 5.0, "acq_step: optimizing acquisition")]


@pytest.mark.asyncio
async def test_callback_runs_in_worker_thread() -> None:
    """The synchronous callback returns immediately; the sink runs on the loop."""
    received: list[str] = []

    async def sink(progress: float, total: float | None, message: str | None) -> None:
        _ = progress, total
        received.append(message or "")

    loop = asyncio.get_running_loop()
    callback = build_progress_callback(sink, loop)

    def worker() -> None:
        # Synchronous worker (the bo-engine path runs under
        # ``asyncio.to_thread``); the callback must not block on the
        # loop.
        callback(ProgressEvent(phase="x", message="from-thread"))

    await asyncio.to_thread(worker)
    for _ in range(10):
        await asyncio.sleep(0)
        if received:
            break

    assert received == ["x: from-thread"]
