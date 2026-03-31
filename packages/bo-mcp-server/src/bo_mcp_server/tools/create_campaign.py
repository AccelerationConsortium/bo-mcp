"""Create campaign tool wrapper for MCP."""

from typing import Any, Literal

from bo_mcp_server.domain import CampaignIntakeInput
from bo_mcp_server.operations.create_campaign import create_campaign_operation
from bo_mcp_server.server import mcp


@mcp.tool(name="bo_create_campaign")
async def create_campaign(
    intake_data: CampaignIntakeInput,
    owner_id: str,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
) -> dict[str, Any]:
    """Create a new optimization campaign from validated intake data.

    Args:
        intake_data: Campaign intake payload validated via CampaignIntakeInput.
        owner_id: UUID of the user creating the campaign.
        verbosity: Response verbosity level (minimal, standard, detailed).

    Returns:
        Dictionary with success, campaign_id, spec_id, errors.
    """
    return await create_campaign_operation(
        intake_data=intake_data,
        owner_id=owner_id,
        verbosity=verbosity,
    )
