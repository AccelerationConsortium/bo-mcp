"""List results and export campaign tool wrappers for MCP."""

from typing import Any, Literal

from bo_mcp_server.operations.export_campaign import export_campaign_operation
from bo_mcp_server.operations.list_results import list_results_operation
from bo_mcp_server.server import mcp


@mcp.tool(name="bo_list_results")
async def list_results(
    campaign_id: str,
    limit: int = 50,
    offset: int = 0,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
) -> dict[str, Any]:
    """List experimental results for a campaign.

    Workflow: Call to review submitted results or audit past submissions.

    Returns structured result data for review, audit, or correction workflows.

    Args:
        campaign_id: UUID of the campaign.
        limit: Maximum number of results to return (default 50, max 500).
        offset: Number of results to skip for pagination (default 0).
        verbosity: Response verbosity level. Options:
            - "minimal": result_id, objective_values only
            - "standard": includes parameter_values, suggestion_id, created_at
            - "detailed": includes metadata, measurement_uncertainty, source

    Returns:
        Dictionary with:
            - success: Boolean
            - results: List of result dictionaries
            - total_count: Total results for this campaign
            - errors: List of error messages
    """
    return await list_results_operation(
        campaign_id=campaign_id,
        limit=limit,
        offset=offset,
        verbosity=verbosity,
    )


@mcp.tool(name="bo_export_campaign")
async def export_campaign(
    campaign_id: str,
    format: str = "csv",
) -> dict[str, Any]:
    """Export all results for a campaign as CSV.

    Workflow: Call to export all campaign data for offline analysis.

    Returns the full dataset (parameters + objectives) in a format suitable
    for downstream analysis or archival.

    Args:
        campaign_id: UUID of the campaign.
        format: Export format. Currently only "csv" is supported.

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
        format=format,
    )
