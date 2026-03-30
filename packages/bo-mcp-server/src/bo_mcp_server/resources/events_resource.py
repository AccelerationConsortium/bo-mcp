"""Events resource for MCP — provides audit trail as context."""

from uuid import UUID

from bo_mcp_server.server import mcp
from bo_mcp_server.storage import EventRepository, get_session


@mcp.resource("events://{campaign_id}")
async def get_campaign_events(campaign_id: str) -> str:
    """Get audit trail for a campaign.

    Returns recent tool invocations as markdown context.
    """
    try:
        campaign_uuid = UUID(campaign_id)
    except ValueError:
        return f"Error: Invalid campaign_id format: {campaign_id}"

    async with get_session() as session:
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
