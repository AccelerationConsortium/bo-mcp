"""Events resource for MCP — provides audit trail as context."""

from uuid import UUID

from bo_mcp_server.errors import ErrorCode, render_resource_error
from bo_mcp_server.server import mcp
from bo_mcp_server.storage import CampaignRepository, EventRepository, get_session


@mcp.resource("events://{campaign_id}")
async def get_campaign_events(campaign_id: str) -> str:
    """Get audit trail for a campaign.

    Returns recent tool invocations as Markdown context on success.
    Failure modes are surfaced via the standard MCP-tool error
    envelope: ``INVALID_CAMPAIGN_ID`` when the id is malformed, and
    ``CAMPAIGN_NOT_FOUND`` when it is a well-formed UUID but no such
    campaign exists. The empty-but-existing-campaign case still returns
    Markdown, so callers can distinguish "campaign has no audit trail
    yet" from "campaign does not exist" by checking whether the response
    starts with ``{`` (TODO 1.9 review pass).
    """
    try:
        campaign_uuid = UUID(campaign_id)
    except ValueError:
        return render_resource_error(
            ErrorCode.INVALID_CAMPAIGN_ID,
            details={"campaign_id": campaign_id},
        )

    async with get_session() as session:
        # Verify the campaign exists before listing events so that an
        # empty event log cannot be confused with a missing campaign.
        if await CampaignRepository(session).get(campaign_uuid) is None:
            return render_resource_error(
                ErrorCode.CAMPAIGN_NOT_FOUND,
                message=f"Campaign {campaign_id} not found",
                details={"campaign_id": campaign_id},
            )

        repo = EventRepository(session)
        events = await repo.list_by_campaign(campaign_uuid, limit=50)

    if not events:
        return f"No events recorded for campaign {campaign_id}."

    lines = [f"# Audit Trail for Campaign {campaign_id}", ""]
    for event in events:
        ts = event.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")
        success = event.output_summary.get("success", "?")
        lines.append(f"- **{ts}** `{event.tool_name}` — success={success}")
        if event.input_summary:
            details = ", ".join(f"{k}={v}" for k, v in event.input_summary.items())
            lines.append(f"  - Input: {details}")

    return "\n".join(lines)
