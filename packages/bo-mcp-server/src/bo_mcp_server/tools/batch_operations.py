"""Batch status tool wrapper for MCP."""

from typing import Any

from bo_mcp_server.operations.batch_status import batch_get_status_operation
from bo_mcp_server.server import mcp


@mcp.tool(name="bo_batch_get_status")
async def batch_get_status(
    campaign_ids: list[str],
    verbosity: str = "minimal",
) -> dict[str, Any]:
    """Get status of multiple campaigns in one call."""
    return await batch_get_status_operation(campaign_ids=campaign_ids, verbosity=verbosity)
