"""Generate suggestions tool wrapper for MCP."""

from typing import Any, Literal

from mcp.server.fastmcp import Context
from sqlalchemy.ext.asyncio import AsyncSession

from bo_mcp_server.idempotency import apply_idempotency
from bo_mcp_server.operations.generate_suggestions import (
    generate_suggestions_operation,
)
from bo_mcp_server.progress_bridge import make_progress_callback_from_context
from bo_mcp_server.server import mcp
from bo_mcp_server.tools.annotations import NON_IDEMPOTENT_MUTATION
from bo_mcp_server.trace_context import bind_trace_id


@mcp.tool(name="bo_generate_suggestions", annotations=NON_IDEMPOTENT_MUTATION)
async def generate_suggestions(
    campaign_id: str,
    batch_size: int | None = None,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
    idempotency_key: str | None = None,
    ctx: Context | None = None,
    dry_run: bool = False,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Generate next batch of experiment suggestions for a campaign.

    Workflow: Call after bo_create_campaign (first batch) or after
    bo_submit_results (subsequent batches). Check results with
    bo_get_diagnostics afterward.

    Args:
        campaign_id: UUID of the campaign.
        batch_size: Number of suggestions (default: campaign's batch_size).
        verbosity: Response verbosity level (minimal, standard, detailed).
        idempotency_key: Optional client-supplied key (recommended: UUIDv7
            per logical generation request). Replays the prior response
            with ``idempotency_replay: True`` if the same key + payload
            was seen in the last 24 hours, instead of producing a fresh
            batch of suggestions.
        dry_run: If True, validate the campaign is generation-ready and
            return a cheap preview (next iteration + planned batch
            size) without running the BO algorithm or persisting any
            suggestion. Dry-runs bypass the idempotency cache so the
            slot stays free for a real generation request.
        trace_id: Optional workflow trace id. See ``bo_create_campaign``.

    Returns:
        Dictionary with success, suggestions, iteration, errors.
    """
    with bind_trace_id(trace_id):
        if dry_run:
            return await generate_suggestions_operation(
                campaign_id=campaign_id,
                batch_size=batch_size,
                verbosity=verbosity,
                dry_run=True,
            )

        request_payload = {
            "campaign_id": campaign_id,
            "batch_size": batch_size,
            "verbosity": verbosity,
        }

        progress_callback = make_progress_callback_from_context(ctx)

        async def run(session: AsyncSession) -> dict[str, Any]:
            return await generate_suggestions_operation(
                campaign_id=campaign_id,
                batch_size=batch_size,
                verbosity=verbosity,
                progress_callback=progress_callback,
                session=session,
            )

        return await apply_idempotency(
            tool_name="bo_generate_suggestions",
            idempotency_key=idempotency_key,
            request_payload=request_payload,
            executor=run,
        )
