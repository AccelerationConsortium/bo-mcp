"""Campaign comparison tool wrapper for MCP."""

from typing import Any

from bo_mcp_server.operations.compare_campaigns import compare_campaigns_operation
from bo_mcp_server.server import mcp


@mcp.tool(name="bo_compare_campaigns")
async def compare_campaigns(
    campaign_ids: list[str],
    verbosity: str = "standard",
) -> dict[str, Any]:
    """Compare multiple optimization campaigns."""
    return await compare_campaigns_operation(campaign_ids=campaign_ids, verbosity=verbosity)
