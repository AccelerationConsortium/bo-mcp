"""List campaigns tool wrapper for MCP.

This tool wraps the list_campaigns operation to provide a tool-based
interface for agents that prefer tools over MCP resources.

Reference: MCP Tool Best Practices - Agents prefer tools for consistent workflow.
https://modelcontextprotocol.io/docs/concepts/tools
"""

from typing import Any
from uuid import UUID

from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.operations.list_campaigns import list_campaigns_operation
from bo_mcp_server.server import mcp


@mcp.tool(name="bo_list_campaigns")
async def list_campaigns(
    owner_id: str | None = None,
    status: str | None = None,
    limit: int = 20,
    offset: int = 0,
    verbosity: str = "standard",
) -> dict[str, Any]:
    """List optimization campaigns with optional filtering and pagination.

    Workflow: Call anytime to find campaign IDs or check campaign statuses.

    Args:
        owner_id: Optional UUID of the owner to filter campaigns by.
        status: Optional status filter. Valid values:
            - "created": Campaigns that have been created but not yet started
            - "running": Active campaigns
            - "paused": Paused campaigns
            - "completed": Finished campaigns
            - "failed": Failed campaigns
        limit: Maximum number of campaigns to return (default 20, max 100).
        offset: Number of campaigns to skip for pagination (default 0).
        verbosity: Response verbosity level. Options:
            - "minimal": ~50 tokens - campaign_id, name, status only
            - "standard": ~200 tokens - includes iteration, n_results, created_at
            - "detailed": ~500+ tokens - includes full spec summary and metrics

    Returns:
        Dictionary with:
            - success: Boolean indicating if retrieval succeeded
            - campaigns: List of campaign summaries
            - total_count: Total number of campaigns matching filters
            - limit: Applied limit
            - offset: Applied offset
            - errors: List of error messages (if any)
    """
    # Parse owner_id string to UUID in the transport layer
    owner_uuid: UUID | None = None
    if owner_id is not None:
        try:
            owner_uuid = UUID(owner_id)
        except ValueError:
            return make_error_response(
                ErrorCode.VALIDATION_FAILED,
                message="Invalid owner_id format",
                details={"owner_id": owner_id},
            )

    return await list_campaigns_operation(
        owner_id=owner_uuid,
        status=status,
        limit=limit,
        offset=offset,
        verbosity=verbosity,
    )
