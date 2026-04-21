"""Batch status tool wrapper for MCP."""

from typing import Any

from bo_mcp_server.operations.batch_status import batch_get_status_operation
from bo_mcp_server.server import mcp


@mcp.tool(name="bo_batch_get_status")
async def batch_get_status(
    campaign_ids: list[str],
    verbosity: str = "minimal",
) -> dict[str, Any]:
    """Get status of multiple campaigns in one call.

    Workflow: Call to efficiently monitor many campaigns at once instead
    of calling bo_get_diagnostics for each one individually.

    Args:
        campaign_ids: List of campaign UUIDs to check.
        verbosity: Response verbosity level (default "minimal" for efficiency).

    Returns:
        Dictionary with:
            - success: Boolean
            - campaigns: Dict mapping campaign_id to status summary
            - failed_ids: List of campaign IDs that could not be retrieved
            - errors: List of error messages
    """
    return await batch_get_status_operation(campaign_ids=campaign_ids, verbosity=verbosity)
