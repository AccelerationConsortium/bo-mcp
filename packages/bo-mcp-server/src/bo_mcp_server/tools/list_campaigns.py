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
from bo_mcp_server.tools.annotations import READ_ONLY


@mcp.tool(name="bo_list_campaigns", annotations=READ_ONLY)
async def list_campaigns(
    owner_id: str | None = None,
    status: str | None = None,
    limit: int = 20,
    offset: int = 0,
    verbosity: str = "standard",
    cursor: str | None = None,
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
        offset: **Deprecated**. Number of campaigns to skip for
            pagination. Unsafe under concurrent inserts — use ``cursor``
            instead. Kept for backward compatibility.
        verbosity: Response verbosity level. Options:
            - "minimal": ~50 tokens - campaign_id, name, status only
            - "standard": ~200 tokens - includes iteration, n_results, created_at
            - "detailed": ~500+ tokens - includes full spec summary and metrics
        cursor: Opaque cursor from a previous call's ``next_cursor``
            field. When supplied, pagination walks the keyset on
            ``(created_at, id)`` so concurrent inserts cannot duplicate
            or skip rows. Server-signed; treat the value as opaque.

    Returns:
        Dictionary with:
            - success: Boolean indicating if retrieval succeeded
            - campaigns: List of campaign summaries
            - total_count: Total number of campaigns matching filters
            - limit: Applied limit
            - offset: Applied offset (echo of input)
            - next_cursor: Opaque cursor for the next page, or null when
              there are no more rows
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
        cursor=cursor,
    )
