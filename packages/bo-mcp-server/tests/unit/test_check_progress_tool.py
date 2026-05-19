"""Poll-fallback progress lookup.

These tests cover the read side of the progress-bridge contract:
``bo_check_progress`` returns the snapshot a polling client would
have received via the push channel, or signals that no operation is
tracked. Push delivery itself is exercised by
``test_progress_bridge.py``.
"""

from __future__ import annotations

import pytest
from bo_engine.progress import ProgressEvent

from bo_mcp_server.progress_bridge import (
    ProgressStatus,
    register_progress_status,
    unregister_progress_status,
)
from bo_mcp_server.tools.check_progress import check_progress


async def test_check_progress_returns_not_tracked_for_unknown_id() -> None:
    """An id with no registered operation reports ``tracked: False``."""
    from bo_mcp_server.response_formatter import RESPONSE_SCHEMA_VERSION

    result = await check_progress(campaign_id="unknown-campaign")
    assert result["tracked"] is False
    assert result["campaign_id"] == "unknown-campaign"
    # Tool envelope must carry the contract version even for the
    # not-tracked path (so older clients can detect a schema bump
    # without parsing the body).
    assert result["schema_version"] == RESPONSE_SCHEMA_VERSION


async def test_check_progress_returns_latest_snapshot() -> None:
    """A registered status object exposes its latest event."""
    status = ProgressStatus()
    status.update_event(
        ProgressEvent(phase="acq_step", message="optimizing", progress=2.0, total=5.0)
    )
    register_progress_status("camp-1", status)
    try:
        result = await check_progress(campaign_id="camp-1")
    finally:
        unregister_progress_status("camp-1")

    from bo_mcp_server.response_formatter import RESPONSE_SCHEMA_VERSION

    assert result["tracked"] is True
    assert result["campaign_id"] == "camp-1"
    assert result["phase"] == "acq_step"
    assert result["message"] == "optimizing"
    assert result["progress"] == pytest.approx(2.0)
    assert result["total"] == pytest.approx(5.0)
    assert result["failures"] == 0
    assert result["schema_version"] == RESPONSE_SCHEMA_VERSION


async def test_check_progress_surfaces_failure_counter() -> None:
    """A push failure visible via the snapshot tells the agent to poll."""
    status = ProgressStatus()
    status.update_event(ProgressEvent(phase="gp_fit_start", message="fit"))
    status.record_failure("loop_closed")
    register_progress_status("camp-2", status)
    try:
        result = await check_progress(campaign_id="camp-2")
    finally:
        unregister_progress_status("camp-2")

    assert result["tracked"] is True
    assert result["failures"] == 1
    assert result["last_failure_reason"] == "loop_closed"
