"""Campaign lifecycle tool wrapper for MCP."""

from typing import Any

from bo_mcp_server.operations.campaign_lifecycle import (
    LifecycleAction,
    manage_campaign_lifecycle_operation,
)
from bo_mcp_server.server import mcp


@mcp.tool()
async def manage_campaign_lifecycle(
    campaign_id: str,
    action: LifecycleAction,
) -> dict[str, Any]:
    """Manage campaign lifecycle with a single consolidated tool."""
    return await manage_campaign_lifecycle_operation(campaign_id=campaign_id, action=action)
