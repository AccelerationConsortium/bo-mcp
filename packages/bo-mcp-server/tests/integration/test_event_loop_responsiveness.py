"""Integration tests for event-loop responsiveness under BO workloads (TODO 1.4).

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
    - TODO.md §1.4 "BoTorch blocks the FastAPI / MCP event loop"
      (packages/bo-mcp-server/src/bo_mcp_server/operations/generate_suggestions.py,
      packages/bo-engine/src/bo_engine/botorch_backend.py).
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


class TestEventLoopResponsiveness:
    """Heavy BO calls must not block the asyncio event loop."""

    @pytest.mark.asyncio
    async def test_concurrent_probe_runs_while_backend_is_busy(
        self, setup_database, monkeypatch: pytest.MonkeyPatch
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
    async def test_concurrent_generate_calls_overlap(
        self, setup_database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
