"""Synchronous per-section compute runs off the event loop.

``get_diagnostics`` promised (module docstring) that engine-touching
work is offloaded via ``asyncio.to_thread`` but ran parts of it inline:
the outcome-constraint calibration (per-constraint feasibility-GP fits)
and the constraint-satisfaction section.

``submit_results`` used to be covered here too, for its per-row
``detect_duplicates`` scan. That scan is gone — parameter equality no
longer rejects a submission — so there is no longer any engine-touching
per-row compute on that path to pin.

These tests pin the offload by recording the thread each computation
runs on: under ``asyncio.to_thread`` it must never be the thread that
drives the event loop.

Reference: https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread
— "asynchronously run function *func* in a separate thread".
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from bo_mcp_server.operations import get_diagnostics as get_diagnostics_module
from bo_mcp_server.operations.get_diagnostics import get_diagnostics_operation
from tests.factories import seed_owner

pytestmark = pytest.mark.usefixtures("setup_database")


async def _create_campaign() -> str:
    from bo_mcp_server.tools.create_campaign import create_campaign

    intake = {
        "name": "Offload contract",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }
    created = await create_campaign(intake, await seed_owner())
    assert created["success"] is True
    return created["campaign_id"]


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
