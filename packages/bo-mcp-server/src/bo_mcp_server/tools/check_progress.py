"""Poll-fallback progress lookup (TODO 8.59).

The primary progress channel is MCP push (``Context.report_progress``).
When the transport stalls — closed loop, dropped connection, hung
client — the push bridge promotes the failure to ``WARNING`` and
bumps ``bo_mcp_progress_notify_failures_total``. Clients that depend
on progress for ETAs or cancellation can call ``bo_check_progress``
to read the same status snapshot the bridge would have pushed.

The snapshot is process-local: a client polling through a different
replica will not see in-flight state. Push remains the primary
channel; poll is the safety net.
"""

from __future__ import annotations

from typing import Any

from bo_mcp_server.progress_bridge import get_progress_status
from bo_mcp_server.response_formatter import attach_response_metadata
from bo_mcp_server.server import mcp
from bo_mcp_server.tools.annotations import READ_ONLY


@mcp.tool(name="bo_check_progress", annotations=READ_ONLY)
async def check_progress(campaign_id: str) -> dict[str, Any]:
    """Return the latest progress snapshot for a long-running campaign.

    Use this when the MCP push channel for progress has gone silent —
    typically signalled by an absence of ``ProgressNotification``
    deliveries past your usual cadence — to confirm whether the
    operation is still making progress, has stalled, or has dropped.

    Args:
        campaign_id: UUID of the campaign whose progress you want to
            read. Must match the ``campaign_id`` passed to the
            currently-running ``bo_generate_suggestions`` call.

    Returns:
        A dictionary with:
            - tracked: True iff a snapshot is registered for this id.
              False means either the operation never started, has
              already completed, or is running on a different replica.
            - phase / message / progress / total: the most recent
              :class:`bo_engine.progress.ProgressEvent` fields the
              push bridge observed. ``None`` until the first event
              fires.
            - failures: count of push deliveries that failed since the
              bridge was constructed. A non-zero value tells the agent
              that progress events have been dropped and polling is the
              authoritative readout.
            - last_failure_reason: short tag (``loop_closed``, etc.)
              describing the most recent failure mode, or ``None``.
    """
    status = get_progress_status(campaign_id)
    if status is None:
        response: dict[str, Any] = {
            "tracked": False,
            "campaign_id": campaign_id,
        }
    else:
        snapshot = status.snapshot()
        response = {
            "tracked": True,
            "campaign_id": campaign_id,
            **snapshot,
        }
    # Tools that bypass the per-verbosity Pydantic formatter still need
    # the top-level ``schema_version`` (and ``_metadata.trace_id`` when
    # bound) so transport contracts stay uniform. ``attach_response_metadata``
    # handles both fields without forcing each raw-dict tool to repeat
    # the boilerplate.
    return attach_response_metadata(response)
