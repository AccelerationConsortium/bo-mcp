"""Generate suggestions tool wrapper for MCP."""

from typing import Any, Literal

from bo_mcp_server.operations.generate_suggestions import (
    generate_suggestions_operation,
)
from bo_mcp_server.server import mcp


@mcp.tool(name="bo_generate_suggestions")
async def generate_suggestions(
    campaign_id: str,
    batch_size: int | None = None,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
) -> dict[str, Any]:
    """Generate next batch of experiment suggestions for a campaign.

    Workflow: Call after bo_create_campaign (first batch) or after
    bo_submit_results (subsequent batches). Check results with
    bo_get_diagnostics afterward.

    Args:
        campaign_id: UUID of the campaign.
        batch_size: Number of suggestions (default: campaign's batch_size).
        verbosity: Response verbosity level (minimal, standard, detailed).

    Returns:
        Dictionary with success, suggestions, iteration, errors.
    """
    return await generate_suggestions_operation(
        campaign_id=campaign_id,
        batch_size=batch_size,
        verbosity=verbosity,
    )
