"""Campaign comparison tool wrapper for MCP."""

from typing import Annotated, cast

from pydantic import Field

from bo_mcp_server.operations.compare_campaigns import (
    MAX_COMPARE_CAMPAIGNS,
    MIN_COMPARE_CAMPAIGNS,
    compare_campaigns_operation,
)
from bo_mcp_server.response_formatter import CompareCampaignsResponse
from bo_mcp_server.server import mcp
from bo_mcp_server.tools.annotations import READ_ONLY
from bo_mcp_server.tools.common import VerbosityLiteral


@mcp.tool(name="bo_compare_campaigns", annotations=READ_ONLY)
async def compare_campaigns(
    campaign_ids: Annotated[
        list[str],
        Field(
            min_length=MIN_COMPARE_CAMPAIGNS,
            max_length=MAX_COMPARE_CAMPAIGNS,
            description=(
                f"{MIN_COMPARE_CAMPAIGNS}-{MAX_COMPARE_CAMPAIGNS} campaign "
                "UUIDs to compare. The upper bound is enforced in the JSON "
                "schema so agents see it without first failing a request."
            ),
        ),
    ],
    verbosity: VerbosityLiteral = "standard",
) -> CompareCampaignsResponse:
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
    return cast(
        CompareCampaignsResponse,
        await compare_campaigns_operation(campaign_ids=campaign_ids, verbosity=verbosity),
    )
