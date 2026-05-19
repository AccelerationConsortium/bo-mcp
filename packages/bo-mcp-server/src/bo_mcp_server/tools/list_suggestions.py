"""List suggestions tool wrapper for MCP."""

from typing import Any, Literal

from bo_mcp_server.operations.list_suggestions import list_suggestions_operation
from bo_mcp_server.server import mcp
from bo_mcp_server.tools.annotations import READ_ONLY

# ``Literal`` mirrors :class:`bo_mcp_server.domain.SuggestionStatus` so the
# generated MCP tool schema declares an ``enum`` constraint on
# ``status_filter``. Agents see the valid values directly
# from the schema instead of failing requests to discover them. Keep
# this list aligned with ``SuggestionStatus``.
SuggestionStatusFilter = Literal[
    "pending",
    "accepted",
    "rejected",
    "completed",
    "expired",
]


@mcp.tool(name="bo_list_suggestions", annotations=READ_ONLY)
async def list_suggestions(
    campaign_id: str,
    status_filter: SuggestionStatusFilter | None = None,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
    limit: int | None = None,
    offset: int = 0,
    cursor: str | None = None,
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
        limit: Maximum suggestions to return. Defaults to the server cap
            (500). Combine with ``cursor`` to page through long lists.
        offset: **Deprecated**. Use ``cursor`` instead.
        cursor: Opaque cursor from a previous response's ``next_cursor``
            field. Stable under concurrent suggestion generation.

    Returns:
        Dictionary with:
            - success: Boolean
            - suggestions: List of suggestion dictionaries
            - total_count: Total suggestions matching the filter
            - next_cursor: Cursor for the next page (null when finished)
            - errors: List of error messages
    """
    return await list_suggestions_operation(
        campaign_id=campaign_id,
        status_filter=status_filter,
        limit=limit,
        offset=offset,
        verbosity=verbosity,
        cursor=cursor,
    )
