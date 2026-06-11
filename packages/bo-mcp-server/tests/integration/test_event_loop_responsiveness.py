"""Integration tests for event-loop responsiveness under BO workloads.

GP fitting, Sobol sampling, acquisition optimization, and SAASBO MCMC are all
CPU-bound synchronous calls. When invoked directly from an ``async def``
operation they freeze the single FastAPI / MCP event loop: a 30 s SAASBO fit
stalls every concurrent request — health checks, status polls, unrelated
campaigns. In production this surfaces as random timeouts.

The remediation wraps every heavy BO-engine entry point in
``asyncio.to_thread`` so the synchronous work runs on a worker thread and the
event loop is free to schedule other coroutines.

These tests deterministically simulate a slow backend by monkey-patching the
``BoTorchBackend.generate_initial_design`` entry point (which bypasses the DB
state needed for the model-fitting path, making the tests fast and hermetic)
to block for a known duration with ``time.sleep``. ``time.sleep`` releases the
GIL, so when it runs inside ``to_thread`` other coroutines keep running; if
the call were still on the event loop, the loop would be frozen for the full
sleep.

Reference:
    - Python docs on offloading blocking calls:
      https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread
    - FastAPI concurrency guidance: CPU-bound synchronous work must be moved
      off the event loop (https://fastapi.tiangolo.com/async/#very-technical-details).
"""

import asyncio
import time
from uuid import uuid4

import pytest
from bo_engine.botorch_backend import BoTorchBackend

from bo_mcp_server import backend as backend_module

# Chosen so the test stays hermetic under CI load: long enough that any
# scheduling slack in ``asyncio.sleep`` is negligible against it, short enough
# that the whole test runs in well under a second.
_WORK_SECONDS = 0.3
# Upper bound for a probe that yields to the event loop; generous enough to
# absorb CI jitter without making the contrast with _WORK_SECONDS ambiguous.
_RESPONSIVENESS_LIMIT = _WORK_SECONDS / 2


async def _create_campaign() -> str:
    """Create a minimal single-objective campaign and return its ID."""
    from bo_mcp_server.tools.create_campaign import create_campaign

    owner_id = str(uuid4())
    intake = {
        "name": "Event-loop responsiveness",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }
    created = await create_campaign(intake, owner_id)
    return created["campaign_id"]


def _patch_slow_initial_design(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the initial-design backend call block for ``_WORK_SECONDS`` seconds.

    This substitutes a synchronous ``time.sleep`` for the real Sobol draw —
    the point is to prove the call is offloaded, not to exercise Sobol. The
    original method is still invoked so the caller observes realistic output.
    """
    original = BoTorchBackend.generate_initial_design

    def slow_generate_initial_design(self, spec, n_points):  # type: ignore[no-untyped-def]
        time.sleep(_WORK_SECONDS)
        return original(self, spec, n_points)

    monkeypatch.setattr(
        BoTorchBackend,
        "generate_initial_design",
        slow_generate_initial_design,
    )


@pytest.mark.usefixtures("setup_database")
class TestEventLoopResponsiveness:
    """Heavy BO calls must not block the asyncio event loop."""

    @pytest.mark.asyncio
    async def test_concurrent_probe_runs_while_backend_is_busy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A lightweight probe must finish while the backend is mid-computation.

        Launches a ``generate_suggestions_operation`` whose backend call is
        patched to block for ``_WORK_SECONDS`` seconds. While that task is
        running, a probe coroutine does a small ``asyncio.sleep`` — if the
        event loop were frozen by the backend call, the probe's elapsed
        wall-clock would be close to ``_WORK_SECONDS``. With the
        ``asyncio.to_thread`` offload in place the probe finishes in a small
        multiple of its own sleep duration.
        """
        from bo_mcp_server.operations.generate_suggestions import (
            generate_suggestions_operation,
        )

        campaign_id = await _create_campaign()
        _patch_slow_initial_design(monkeypatch)

        gen_task = asyncio.create_task(
            generate_suggestions_operation(campaign_id=campaign_id, batch_size=1)
        )
        # Give the task a turn to dispatch the blocking call into the worker
        # thread; otherwise the probe could finish before the "slow" path even
        # starts and the assertion would pass for the wrong reason.
        await asyncio.sleep(0.05)

        probe_start = time.monotonic()
        await asyncio.sleep(0.01)
        probe_elapsed = time.monotonic() - probe_start

        result = await gen_task

        assert result["success"] is True
        assert probe_elapsed < _RESPONSIVENESS_LIMIT, (
            "Event loop was blocked during generate_suggestions — the backend "
            f"call did not offload to a worker thread (probe took {probe_elapsed:.3f}s, "
            f"limit was {_RESPONSIVENESS_LIMIT:.3f}s)."
        )

    @pytest.mark.asyncio
    async def test_concurrent_generate_calls_overlap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """N concurrent generate_suggestions calls must overlap in wall time.

        With the synchronous backend call offloaded to a worker thread, three
        concurrent requests should finish in roughly the duration of a single
        call plus scheduling overhead. Without the offload they serialize on
        the event loop and total wall time is ~N×single. The bound below
        (2×single) catches the regression while tolerating CI scheduling
        jitter and any per-call overhead the test fixtures introduce.
        """
        from bo_mcp_server.operations.generate_suggestions import (
            generate_suggestions_operation,
        )

        # Each concurrent call needs its own campaign — generate_suggestions
        # updates campaign state under optimistic locking, so parallel calls
        # on the same campaign would collide on the version check (and that
        # race is covered by test_concurrent_modification.py, not here).
        campaign_ids = [await _create_campaign() for _ in range(3)]
        _patch_slow_initial_design(monkeypatch)

        start = time.monotonic()
        results = await asyncio.gather(
            *(generate_suggestions_operation(campaign_id=cid, batch_size=1) for cid in campaign_ids)
        )
        elapsed = time.monotonic() - start

        assert all(r["success"] is True for r in results)
        overlap_limit = 2 * _WORK_SECONDS
        assert elapsed < overlap_limit, (
            f"Three concurrent generate_suggestions calls took {elapsed:.3f}s "
            f"(limit {overlap_limit:.3f}s). Either the backend call is still "
            "serialized on the event loop or the thread pool is exhausted."
        )


def _patch_slow_backend_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty the backend cache and make every entry-point load slow.

    Simulates the first-use scenario on a fresh server process: the
    backend class import (torch / baybe at module level) takes seconds.
    The stub substitutes ``time.sleep`` for the import and returns a real
    ``BoTorchBackend`` regardless of the requested name, so downstream
    capability validation still behaves. Replacing the ``_backends`` dict
    via monkeypatch restores the session's warm cache afterwards.
    """

    def slow_load_backend(name: str) -> BoTorchBackend:
        del name
        time.sleep(_WORK_SECONDS)
        return BoTorchBackend()

    monkeypatch.setattr(backend_module, "_backends", {})
    monkeypatch.setattr(backend_module, "_load_backend", slow_load_backend)


# Probe sleep used by the continuous max-gap sampler below.
_PROBE_SLEEP_SECONDS = 0.01


async def _max_probe_gap_until_done(task: "asyncio.Task[object]") -> float:
    """Worst extra latency of a short sleep probe while ``task`` runs.

    The backend load blocks almost immediately after the task's first
    turn, so a single probe after a fixed dispatch delay can miss the
    blocking window entirely (the delay itself absorbs the freeze and
    the probe then measures an idle loop). Sampling continuously for the
    task's full lifetime makes the measurement insensitive to where in
    the task the synchronous block occurs. Returns the largest observed
    overshoot beyond the probe's own sleep duration.
    """
    max_gap = 0.0
    while not task.done():
        start = time.monotonic()
        await asyncio.sleep(_PROBE_SLEEP_SECONDS)
        gap = time.monotonic() - start - _PROBE_SLEEP_SECONDS
        max_gap = max(max_gap, gap)
    return max_gap


@pytest.mark.usefixtures("setup_database")
class TestBackendLoadOffload:
    """A cold backend load must not block the asyncio event loop.

    Entry-point loading was historically synchronous inside async
    handlers: the first campaign-create or health check on a fresh
    process froze every concurrent session for the full torch import.
    These tests pin the ``asyncio.to_thread`` offload of the load itself
    (the tests above cover offloading of the *computation* that follows).
    """

    @pytest.mark.asyncio
    async def test_probe_runs_while_create_campaign_loads_backends(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Campaign creation resolves and loads backends off the loop.

        ``backend="auto"`` resolution loads every installed candidate
        backend, so a fresh process pays several sequential slow loads
        here — the worst first-call case.
        """
        _patch_slow_backend_load(monkeypatch)

        create_task = asyncio.create_task(_create_campaign())
        max_gap = await _max_probe_gap_until_done(create_task)
        campaign_id = await create_task

        assert campaign_id
        assert max_gap < _RESPONSIVENESS_LIMIT, (
            "Event loop was blocked during backend loading in create_campaign — "
            f"the entry-point load did not offload to a worker thread (worst probe "
            f"gap {max_gap:.3f}s, limit was {_RESPONSIVENESS_LIMIT:.3f}s)."
        )

    @pytest.mark.asyncio
    async def test_probe_runs_while_health_check_loads_backends(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The health check loads *all* discovered backends in a loop.

        On a fresh process this is the single heaviest cold path, and it
        is exactly the endpoint orchestrators poll concurrently with
        live traffic.
        """
        from bo_mcp_server.tools.health_check import health_check

        _patch_slow_backend_load(monkeypatch)

        health_task = asyncio.create_task(health_check())
        max_gap = await _max_probe_gap_until_done(health_task)
        result = await health_task

        assert result["healthy"] is True
        assert max_gap < _RESPONSIVENESS_LIMIT, (
            "Event loop was blocked during backend loading in health_check — "
            f"get_backend_capabilities did not offload to a worker thread (worst "
            f"probe gap {max_gap:.3f}s, limit was {_RESPONSIVENESS_LIMIT:.3f}s)."
        )
