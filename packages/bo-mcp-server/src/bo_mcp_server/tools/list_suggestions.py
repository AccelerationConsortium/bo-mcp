"""List suggestions tool wrapper for MCP."""

from typing import Any, Literal

from bo_mcp_server.operations.list_suggestions import list_suggestions_operation
from bo_mcp_server.server import mcp


@mcp.tool(name="bo_list_suggestions")
async def list_suggestions(
    campaign_id: str,
    status_filter: str | None = None,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
) -> dict[str, Any]:
    """List suggestions for a campaign with optional status filtering.

    Workflow: Call to review pending, accepted, or completed suggestions.

    Args:
        campaign_id: UUID of the campaign.
        status_filter: Optional status to filter by. Valid values:
            "pending", "accepted", "rejected", "completed", "expired".
        verbosity: Response verbosity level. Options:
            - "minimal": suggestion_id, status only
            - "standard": includes parameter_values, iteration, created_at
            - "detailed": includes full provenance (acquisition value, model info)

    Returns:
        Dictionary with:
            - success: Boolean
            - suggestions: List of suggestion dictionaries
            - total_count: Total suggestions matching the filter
            - errors: List of error messages
    """
    return await list_suggestions_operation(
        campaign_id=campaign_id,
        status_filter=status_filter,
        verbosity=verbosity,
    )
