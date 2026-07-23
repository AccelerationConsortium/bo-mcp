"""Generate suggestions tool wrapper for MCP."""

from typing import Any, Literal, cast
from uuid import uuid4

from mcp.server.fastmcp import Context
from sqlalchemy.ext.asyncio import AsyncSession

from bo_mcp_server.idempotency import apply_idempotency
from bo_mcp_server.operations.generate_suggestions import (
    generate_suggestions_operation,
)
from bo_mcp_server.operations.idempotency_wrapper import (
    canonical_generate_suggestions_payload,
)
from bo_mcp_server.progress_bridge import (
    ProgressStatus,
    make_progress_callback_from_context,
    register_progress_status,
    unregister_progress_status,
)
from bo_mcp_server.server import mcp
from bo_mcp_server.tools.annotations import NON_IDEMPOTENT_MUTATION
from bo_mcp_server.tools.response_models import GenerateSuggestionsResponse
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
) -> GenerateSuggestionsResponse:
    """Generate next batch of experiment suggestions for a campaign.

    Workflow: Call after bo_create_campaign (first batch) or after
    bo_submit_results (subsequent batches). Check results with
    bo_get_diagnostics afterward.

    Each suggestion carries its identity under ``suggestion_id`` — the
    same key ``bo_list_suggestions`` emits and ``bo_submit_results``
    consumes, so copy its value into the ``suggestion_id`` field when
    submitting results.

    Args:
        campaign_id: UUID of the campaign.
        batch_size: Number of suggestions (default: campaign's batch_size).
        verbosity: Response verbosity level (minimal, standard, detailed).
        idempotency_key: Optional client-supplied key (recommended: UUIDv7
            per logical generation request). Replays the prior response
            with ``idempotency_replay: True`` if the same key + payload
            was seen in the last 24 hours, instead of producing a fresh
            batch of suggestions.
        ctx: FastMCP request context used for progress notifications and
            cancellation. Supplied automatically by the MCP runtime.
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
            return cast(
                GenerateSuggestionsResponse,
                await generate_suggestions_operation(
                    campaign_id=campaign_id,
                    batch_size=batch_size,
                    verbosity=verbosity,
                    dry_run=True,
                ),
            )

        # Shared canonical builder — the REST generate route hashes the
        # same shape, so a retry on either transport replays the other's
        # cached response.
        request_payload = canonical_generate_suggestions_payload(
            campaign_id=campaign_id,
            batch_size=batch_size,
            verbosity=verbosity,
        )

        progress_status = ProgressStatus()
        progress_callback = make_progress_callback_from_context(ctx, status=progress_status)
        # Register the snapshot under a per-call key (campaign id +
        # unique suffix) so a polling client (``bo_check_progress``,
        # which prefix-matches on the campaign id) can read the latest
        # event even when the push channel has dropped. The unique
        # suffix keeps two concurrent generate calls for the same
        # campaign from clobbering each other's registration — and the
        # finally below from deleting the other call's still-live
        # entry.
        registration_token = f"{campaign_id}:{uuid4().hex}"
        register_progress_status(registration_token, progress_status)

        async def run(session: AsyncSession) -> dict[str, Any]:
            return await generate_suggestions_operation(
                campaign_id=campaign_id,
                batch_size=batch_size,
                verbosity=verbosity,
                progress_callback=progress_callback,
                session=session,
            )

        try:
            return cast(
                GenerateSuggestionsResponse,
                await apply_idempotency(
                    tool_name="bo_generate_suggestions",
                    idempotency_key=idempotency_key,
                    request_payload=request_payload,
                    executor=run,
                ),
            )
        finally:
            unregister_progress_status(registration_token)
