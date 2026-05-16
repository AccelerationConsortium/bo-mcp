"""Batch status tool wrapper for MCP."""

from typing import Annotated, Any

from pydantic import Field

from bo_mcp_server.operations.batch_status import MAX_BATCH_SIZE, batch_get_status_operation
from bo_mcp_server.server import mcp
from bo_mcp_server.tools.annotations import READ_ONLY


@mcp.tool(name="bo_batch_get_status", annotations=READ_ONLY)
async def batch_get_status(
    campaign_ids: Annotated[
        list[str],
        Field(
            min_length=1,
            max_length=MAX_BATCH_SIZE,
            description=(
                f"Campaign UUIDs to fetch in one call. The server caps the "
                f"batch at {MAX_BATCH_SIZE} to bound per-request work and DB "
                "fan-out; advertising the limit in the JSON schema lets agents "
                "split the request locally instead of discovering the limit "
                "via an error response."
            ),
        ),
    ],
    verbosity: str = "minimal",
) -> dict[str, Any]:
    """Get status of multiple campaigns in one call.

    Workflow: Call to efficiently monitor many campaigns at once instead
    of calling bo_get_diagnostics for each one individually.

    Args:
        campaign_ids: List of campaign UUIDs to check. Up to
            ``MAX_BATCH_SIZE`` per call; the limit is enforced in the
            tool JSON schema (``maxItems``) so well-behaved agents see
            it without trial-and-error.
        verbosity: Response verbosity level (default "minimal" for efficiency).

    Returns:
        Dictionary with:
            - success: Boolean
            - campaigns: Dict mapping campaign_id to status summary
            - failed_ids: List of campaign IDs that could not be retrieved
            - errors: List of error messages
    """
    return await batch_get_status_operation(campaign_ids=campaign_ids, verbosity=verbosity)
