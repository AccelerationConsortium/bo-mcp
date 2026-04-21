"""Campaign comparison tool wrapper for MCP."""

from typing import Any

from bo_mcp_server.operations.compare_campaigns import compare_campaigns_operation
from bo_mcp_server.server import mcp


@mcp.tool(name="bo_compare_campaigns")
async def compare_campaigns(
    campaign_ids: list[str],
    verbosity: str = "standard",
) -> dict[str, Any]:
    """Compare multiple optimization campaigns side by side.

    Workflow: Call with 2-10 campaign IDs to compare performance metrics,
    convergence, and sample efficiency across campaigns.

    Args:
        campaign_ids: List of 2-10 campaign UUIDs to compare.
        verbosity: Response verbosity level (minimal, standard, detailed).

    Returns:
        Dictionary with:
            - success: Boolean
            - campaigns: List of per-campaign metrics
            - comparison: Cross-campaign comparison with best performer
            - errors: List of error messages
    """
    return await compare_campaigns_operation(campaign_ids=campaign_ids, verbosity=verbosity)
