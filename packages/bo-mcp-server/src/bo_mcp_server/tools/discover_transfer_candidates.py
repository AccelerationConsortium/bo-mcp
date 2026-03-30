"""Transfer candidate discovery tool wrapper for MCP."""

from typing import Any

from bo_mcp_server.operations.transfer_candidates import (
    discover_transfer_candidates_operation,
)
from bo_mcp_server.server import mcp


@mcp.tool(name="bo_discover_transfer_candidates")
async def discover_transfer_candidates(
    campaign_id: str,
    similarity_threshold: float = 0.5,
    max_candidates: int = 5,
    verbosity: str = "standard",
) -> dict[str, Any]:
    """Discover campaigns suitable for transfer learning."""
    return await discover_transfer_candidates_operation(
        campaign_id=campaign_id,
        similarity_threshold=similarity_threshold,
        max_candidates=max_candidates,
        verbosity=verbosity,
    )
