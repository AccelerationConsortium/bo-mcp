"""Update suggestion status tool wrapper for MCP."""

from typing import Any, Literal

from sqlalchemy.ext.asyncio import AsyncSession

from bo_mcp_server.idempotency import apply_idempotency
from bo_mcp_server.operations.update_suggestion_status import (
    update_suggestion_status_operation,
)
from bo_mcp_server.server import mcp
from bo_mcp_server.tools.annotations import NON_IDEMPOTENT_MUTATION
from bo_mcp_server.trace_context import bind_trace_id

# ``completed`` is intentionally excluded -- it is set automatically by
# ``bo_submit_results`` (manual transitions live in
# :data:`bo_mcp_server.operations.update_suggestion_status.ALLOWED_TARGET_STATUSES`).
ManualSuggestionStatus = Literal["accepted", "rejected", "expired"]


@mcp.tool(name="bo_update_suggestion_status", annotations=NON_IDEMPOTENT_MUTATION)
async def update_suggestion_status(
    suggestion_id: str,
    status: ManualSuggestionStatus,
    idempotency_key: str | None = None,
    dry_run: bool = False,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Update the status of a suggestion.

    Workflow: Call after reviewing suggestions from bo_list_suggestions to
    accept, reject, or expire them.

    Use this to accept, reject, or expire a suggestion. The "completed"
    status is set automatically when results are submitted via bo_submit_results.

    Valid transitions:
        - pending -> accepted (mark for execution)
        - pending -> rejected (skip this suggestion)
        - pending -> expired  (suggestion no longer relevant)
        - accepted -> rejected (changed mind before executing)
        - accepted -> expired  (suggestion no longer relevant)

    Args:
        suggestion_id: UUID of the suggestion to update.
        status: New status. One of: "accepted", "rejected", "expired".
        idempotency_key: Optional client-supplied key (recommended: UUIDv7
            per logical transition). Replays the prior response with
            ``idempotency_replay: True`` if the same key + payload was
            seen in the last 24 hours.
        dry_run: If True, validate the transition and return a preview
            without committing. The response carries ``dry_run: True``
            and a ``preview`` block. Dry-runs bypass the idempotency
            cache so they never reserve a slot.

    Returns:
        Dictionary with:
            - success: Boolean
            - suggestion_id: UUID of the updated suggestion
            - status: New status
            - previous_status: Status before the update
            - errors: List of error messages
    """
    with bind_trace_id(trace_id):
        if dry_run:
            return await update_suggestion_status_operation(
                suggestion_id=suggestion_id,
                status=status,
                dry_run=True,
            )

        request_payload = {
            "suggestion_id": suggestion_id,
            "status": status,
        }

        async def run(session: AsyncSession) -> dict[str, Any]:
            return await update_suggestion_status_operation(
                suggestion_id=suggestion_id,
                status=status,
                session=session,
            )

        return await apply_idempotency(
            tool_name="bo_update_suggestion_status",
            idempotency_key=idempotency_key,
            request_payload=request_payload,
            executor=run,
        )
