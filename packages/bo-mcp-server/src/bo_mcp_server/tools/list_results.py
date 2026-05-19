"""List results and export campaign tool wrappers for MCP."""

from typing import Any, Literal

from bo_mcp_server.operations.export_campaign import export_campaign_operation
from bo_mcp_server.operations.list_results import list_results_operation
from bo_mcp_server.server import mcp
from bo_mcp_server.tools.annotations import READ_ONLY


@mcp.tool(name="bo_list_results", annotations=READ_ONLY)
async def list_results(
    campaign_id: str,
    limit: int = 50,
    offset: int = 0,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
    cursor: str | None = None,
) -> dict[str, Any]:
    """List experimental results for a campaign.

    Workflow: Call to review submitted results or audit past submissions.

    Returns structured result data for review, audit, or correction workflows.

    Args:
        campaign_id: UUID of the campaign.
        limit: Maximum number of results to return (default 50, max 500).
        offset: **Deprecated**. Number of results to skip for pagination.
            Unsafe under concurrent inserts — use ``cursor`` instead.
        verbosity: Response verbosity level. Options:
            - "minimal": result_id, objective_values only
            - "standard": includes parameter_values, suggestion_id, created_at
            - "detailed": includes metadata, measurement_uncertainty, source
        cursor: Opaque cursor from a previous response's ``next_cursor``
            field. When supplied, pagination walks the keyset on
            ``(created_at, id)`` so concurrent submissions cannot
            duplicate or skip results.

    Returns:
        Dictionary with:
            - success: Boolean
            - results: List of result dictionaries
            - total_count: Total results for this campaign
            - next_cursor: Cursor for the next page (null when finished)
            - errors: List of error messages
    """
    return await list_results_operation(
        campaign_id=campaign_id,
        limit=limit,
        offset=offset,
        verbosity=verbosity,
        cursor=cursor,
    )


@mcp.tool(name="bo_export_campaign", annotations=READ_ONLY)
async def export_campaign(
    campaign_id: str,
    output_format: str = "csv",
) -> dict[str, Any]:
    """Export all results for a campaign as CSV.

    Workflow: Call to export all campaign data for offline analysis.

    Returns the full dataset (parameters + objectives) in a format suitable
    for downstream analysis or archival.

    Args:
        campaign_id: UUID of the campaign.
        output_format: Export format. Currently only ``"csv"`` is supported.

    Returns:
        Dictionary with:
            - success: Boolean
            - format: The export format used
            - content: CSV string with all results
            - n_results: Number of results exported
            - errors: List of error messages
    """
    return await export_campaign_operation(
        campaign_id=campaign_id,
        output_format=output_format,
    )
