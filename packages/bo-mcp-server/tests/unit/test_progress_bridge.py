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
import logging

import pytest
from bo_engine.progress import ProgressEvent

from bo_mcp_server.metrics import PROGRESS_NOTIFY_FAILURES
from bo_mcp_server.progress_bridge import ProgressStatus, build_progress_callback


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


def _get_failure_count() -> float:
    """Return the current value of the loop_closed counter."""
    return PROGRESS_NOTIFY_FAILURES.labels("loop_closed")._value.get()


@pytest.mark.asyncio
async def test_callback_logs_warning_and_bumps_counter_when_loop_closed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A dropped event surfaces at WARNING, on the metric, and on the status snapshot.

    Reference: the audit (TODO 8.59) flags the original ``DEBUG`` swallow
    as silently hiding a stuck ETA/cancellation surface from operators.
    """

    async def sink(progress: float, total: float | None, message: str | None) -> None:
        _ = progress, total, message

    # Build a loop, capture it, then close it so the next coroutine
    # submission raises ``RuntimeError``. The bridge must promote that
    # failure to WARNING + counter without re-raising.
    closed_loop = asyncio.new_event_loop()
    closed_loop.close()

    status = ProgressStatus()
    callback = build_progress_callback(sink, closed_loop, status=status)

    before = _get_failure_count()
    with caplog.at_level(logging.WARNING, logger="bo_mcp_server.progress_bridge"):
        callback(ProgressEvent(phase="acq_step", message="optimizing"))

    after = _get_failure_count()
    assert after - before == pytest.approx(1.0)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "loop-closed failure must log at WARNING"
    assert "Progress event dropped" in warnings[-1].getMessage()

    snapshot = status.snapshot()
    assert snapshot["failures"] == 1
    assert snapshot["last_failure_reason"] == "loop_closed"
    # The status snapshot still records the latest event for the
    # poll fallback even though the push delivery failed.
    assert snapshot["phase"] == "acq_step"
    assert snapshot["message"] == "optimizing"


def _get_send_failed_count() -> float:
    """Return the current value of the send_failed counter."""
    return PROGRESS_NOTIFY_FAILURES.labels("send_failed")._value.get()


@pytest.mark.asyncio
async def test_callback_records_async_sink_exception(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A sink that raises *after* successful submission still bumps the counter.

    ``asyncio.run_coroutine_threadsafe`` returns a ``Future`` once the
    coroutine has been scheduled; a sink that raises inside the
    coroutine body would otherwise drop silently. The bridge attaches
    a ``done`` callback that surfaces the exception through the same
    WARNING + counter + snapshot path the synchronous-failure case
    uses.
    """

    async def raising_sink(progress: float, total: float | None, message: str | None) -> None:
        _ = progress, total, message
        raise RuntimeError("transport hangup")

    loop = asyncio.get_running_loop()
    status = ProgressStatus()
    callback = build_progress_callback(raising_sink, loop, status=status)

    before = _get_send_failed_count()
    with caplog.at_level(logging.WARNING, logger="bo_mcp_server.progress_bridge"):
        callback(ProgressEvent(phase="acq_step", message="optimizing"))
        # Yield so the scheduled coroutine runs, raises, and the
        # ``done`` callback fires.
        for _ in range(20):
            await asyncio.sleep(0)
            failures = status.snapshot()["failures"]
            assert isinstance(failures, int)
            if failures > 0:
                break

    after = _get_send_failed_count()
    assert after - before == pytest.approx(1.0)

    snapshot = status.snapshot()
    assert snapshot["failures"] == 1
    assert snapshot["last_failure_reason"] == "send_failed"

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "async send failure must log at WARNING"
    last_message = warnings[-1].getMessage()
    assert "Progress event dropped" in last_message
    assert "send_failed" in last_message


@pytest.mark.asyncio
async def test_status_snapshot_records_latest_event() -> None:
    """``ProgressStatus.snapshot`` reflects the most recent event."""
    received: list[tuple[float, float | None, str | None]] = []

    async def sink(progress: float, total: float | None, message: str | None) -> None:
        received.append((progress, total, message))

    loop = asyncio.get_running_loop()
    status = ProgressStatus()
    callback = build_progress_callback(sink, loop, status=status)

    callback(ProgressEvent(phase="acq_step", message="step-1", progress=1.0, total=3.0))
    callback(ProgressEvent(phase="acq_step", message="step-2", progress=2.0, total=3.0))

    for _ in range(10):
        await asyncio.sleep(0)
        if len(received) >= 2:
            break

    snapshot = status.snapshot()
    assert snapshot["progress"] == pytest.approx(2.0)
    assert snapshot["total"] == pytest.approx(3.0)
    assert snapshot["message"] == "step-2"
    assert snapshot["failures"] == 0
