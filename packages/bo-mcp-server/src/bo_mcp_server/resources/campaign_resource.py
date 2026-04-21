"""Campaign resource for MCP."""

from uuid import UUID

from bo_mcp_server.domain import CampaignSpec
from bo_mcp_server.server import mcp
from bo_mcp_server.storage import CampaignRepository, CampaignSpecRepository, get_session


def _format_parameters_section(spec: CampaignSpec) -> list[str]:
    """Format the parameters section of a campaign resource."""
    lines: list[str] = []
    for param in spec.parameters:
        if param.bounds:
            bounds_str = f"[{param.bounds.lower}, {param.bounds.upper}]"
            lines.append(f"- **{param.name}** ({param.type.value}): {bounds_str}")
        elif param.categories:
            lines.append(f"- **{param.name}** ({param.type.value}): {param.categories}")
        elif param.values:
            lines.append(f"- **{param.name}** ({param.type.value}): {param.values}")
    return lines


def _format_objectives_section(spec: CampaignSpec) -> list[str]:
    """Format the objectives section of a campaign resource."""
    lines: list[str] = []
    for obj in spec.objectives:
        target = f" (target: {obj.target})" if obj.target else ""
        lines.append(f"- **{obj.name}**: {obj.direction}{target}")
    return lines


def _format_constraints_section(spec: CampaignSpec) -> list[str]:
    """Format the constraints section of a campaign resource."""
    if not spec.constraints:
        return []
    lines = ["", "## Constraints", ""]
    for constraint in spec.constraints:
        lines.append(f"- {constraint.type.value}: {constraint.parameters} = {constraint.value}")
    return lines


@mcp.resource("campaign://{campaign_id}")
async def get_campaign(campaign_id: str) -> str:
    """Get campaign details as a resource.

    Args:
        campaign_id: UUID of the campaign

    Returns:
        Campaign details as formatted text
    """
    try:
        campaign_uuid = UUID(campaign_id)
    except ValueError:
        return f"Error: Invalid campaign_id format: {campaign_id}"

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        spec_repo = CampaignSpecRepository(session)

        campaign = await campaign_repo.get(campaign_uuid)
        if campaign is None:
            return f"Error: Campaign {campaign_id} not found"

        spec = await spec_repo.get(campaign.spec_id)
        if spec is None:
            return f"Error: Campaign spec not found for campaign {campaign_id}"

        lines = [
            f"# Campaign: {spec.name}",
            "",
            f"**ID:** {campaign.id}",
            f"**Status:** {campaign.status.value}",
            f"**Iteration:** {campaign.iteration}",
            f"**Created:** {campaign.created_at.isoformat()}",
            "",
            "## Parameters",
            "",
            *_format_parameters_section(spec),
            "",
            "## Objectives",
            "",
            *_format_objectives_section(spec),
            *_format_constraints_section(spec),
        ]

        return "\n".join(lines)


@mcp.resource("campaigns://list")
async def list_campaigns() -> str:
    """List all campaigns.

    Returns:
        Formatted list of all campaigns
    """
    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        spec_repo = CampaignSpecRepository(session)

        campaigns = await campaign_repo.list_all()

        if not campaigns:
            return "No campaigns found."

        lines = ["# Campaigns", ""]

        for campaign in campaigns:
            spec = await spec_repo.get(campaign.spec_id)
            name = spec.name if spec else "Unknown"
            lines.append(
                f"- **{name}** (ID: {campaign.id}): {campaign.status.value}, "
                f"iteration {campaign.iteration}"
            )

        return "\n".join(lines)
