"""Update suggestion status tool wrapper for MCP."""

from typing import Any

from bo_mcp_server.operations.update_suggestion_status import (
    update_suggestion_status_operation,
)
from bo_mcp_server.server import mcp


@mcp.tool(name="bo_update_suggestion_status")
async def update_suggestion_status(
    suggestion_id: str,
    status: str,
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

    Returns:
        Dictionary with:
            - success: Boolean
            - suggestion_id: UUID of the updated suggestion
            - status: New status
            - previous_status: Status before the update
            - errors: List of error messages
    """
    return await update_suggestion_status_operation(
        suggestion_id=suggestion_id,
        status=status,
    )
