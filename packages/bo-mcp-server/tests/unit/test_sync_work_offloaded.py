"""Synchronous per-row / per-section compute runs off the event loop.

Two operations promised (module docstrings) that engine-touching work is
offloaded via ``asyncio.to_thread`` but ran parts of it inline:

* ``submit_results`` ran ``detect_duplicates`` synchronously once (later
  twice) per submitted row — O(batch x n_results) distance computations
  on the loop, stalling every concurrent session during bulk uploads.
* ``get_diagnostics`` ran the outcome-constraint calibration (per-
  constraint feasibility-GP fits) and the constraint-satisfaction
  section inline.

These tests pin the offload by recording the thread each computation
runs on: under ``asyncio.to_thread`` it must never be the thread that
drives the event loop.

Reference: https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread
— "asynchronously run function *func* in a separate thread".
"""

from __future__ import annotations

import threading
from typing import Any
from uuid import uuid4

import pytest

from bo_mcp_server.domain import ResultSubmissionInput
from bo_mcp_server.operations import get_diagnostics as get_diagnostics_module
from bo_mcp_server.operations import submit_results_pipeline
from bo_mcp_server.operations.get_diagnostics import get_diagnostics_operation
from bo_mcp_server.operations.submit_results import submit_results_operation


def _row(x: float, y: float) -> ResultSubmissionInput:
    return ResultSubmissionInput(parameter_values={"x": x}, objective_values={"y": y})


pytestmark = pytest.mark.usefixtures("setup_database")


async def _create_campaign() -> str:
    from bo_mcp_server.tools.create_campaign import create_campaign

    intake = {
        "name": "Offload contract",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }
    created = await create_campaign(intake, str(uuid4()))
    assert created["success"] is True
    return created["campaign_id"]


@pytest.mark.asyncio
async def test_detect_duplicates_passes_run_off_the_loop_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both duplicate passes (stored + in-batch) run on worker threads."""
    loop_thread = threading.current_thread()
    seen_threads: list[threading.Thread] = []
    original = submit_results_pipeline._check_duplicates_for_result

    def _recording_check(*args: Any, **kwargs: Any) -> Any:
        seen_threads.append(threading.current_thread())
        return original(*args, **kwargs)

    monkeypatch.setattr(submit_results_pipeline, "_check_duplicates_for_result", _recording_check)

    campaign_id = await _create_campaign()
    # First submit seeds the stored baseline; the second exercises the
    # stored-duplicate pass per row plus the in-batch pass.
    first = await submit_results_operation(
        campaign_id=campaign_id,
        results=[_row(0.1, 1.0)],
        submitted_by=str(uuid4()),
    )
    assert first["success"] is True
    second = await submit_results_operation(
        campaign_id=campaign_id,
        results=[_row(0.4, 2.0), _row(0.7, 3.0)],
        submitted_by=str(uuid4()),
    )
    assert second["success"] is True

    assert seen_threads, "duplicate detection never ran"
    on_loop = [t for t in seen_threads if t is loop_thread]
    assert not on_loop, (
        f"{len(on_loop)}/{len(seen_threads)} duplicate scans ran on the event-loop thread"
    )


@pytest.mark.asyncio
async def test_constraint_diagnostics_sections_run_off_the_loop_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop_thread = threading.current_thread()
    seen_threads: list[threading.Thread] = []

    def _record(*_args: Any, **_kwargs: Any) -> None:
        seen_threads.append(threading.current_thread())

    monkeypatch.setattr(get_diagnostics_module, "compute_constraint_satisfaction_metrics", _record)
    monkeypatch.setattr(
        get_diagnostics_module, "compute_outcome_constraint_calibration_metrics", _record
    )

    campaign_id = await _create_campaign()
    result = await get_diagnostics_operation(campaign_id, sections=["constraints"])

    assert result["success"] is True
    assert len(seen_threads) == 2
    assert all(t is not loop_thread for t in seen_threads)


@pytest.mark.asyncio
async def test_partial_section_diagnostics_hit_the_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeated single-section read is served from the cache.

    Pre-fix only the full-section envelope was cached, so every
    ``sections=["constraints"]`` request re-ran the section (including
    the feasibility-GP fits it now offloads).
    """
    calls = [0]

    def _count(*_args: Any, **_kwargs: Any) -> None:
        calls[0] += 1

    monkeypatch.setattr(get_diagnostics_module, "compute_constraint_satisfaction_metrics", _count)
    monkeypatch.setattr(
        get_diagnostics_module, "compute_outcome_constraint_calibration_metrics", _count
    )

    campaign_id = await _create_campaign()
    first = await get_diagnostics_operation(campaign_id, sections=["constraints"])
    computed_after_first = calls[0]
    second = await get_diagnostics_operation(campaign_id, sections=["constraints"])

    assert first["success"] is True
    assert second["success"] is True
    assert computed_after_first == 2
    assert calls[0] == computed_after_first, "second partial-section read recomputed"
