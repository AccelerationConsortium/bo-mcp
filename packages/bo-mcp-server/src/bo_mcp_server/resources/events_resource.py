"""Events resource for MCP — provides audit trail as context.

Errors raise :class:`ResourceOperationError`; the MCP read path
(via the wrapper installed by :mod:`bo_mcp_server.resource_boundary`)
surfaces them as :class:`McpError` JSON-RPC errors with the
structured envelope on ``error.data``. Direct in-process callers
catch the typed exception and read the envelope off ``exc.envelope``
or ``str(exc)``.
"""

from uuid import UUID

from bo_mcp_server.errors import ErrorCode, raise_resource_error
from bo_mcp_server.server import mcp
from bo_mcp_server.storage import CampaignRepository, EventRepository, get_session

# Most-recent events rendered per read. The resource is meant as quick
# audit-trail context, not a paginated export, so this stays a small
# fixed window rather than growing a cursor-based API of its own.
_EVENTS_RESOURCE_LIMIT = 50


@mcp.resource("events://{campaign_id}")
async def get_campaign_events(campaign_id: str) -> str:
    """Get audit trail for a campaign.

    Returns recent tool invocations as Markdown context on success.

    Raises:
        ResourceOperationError: ``INVALID_CAMPAIGN_ID`` when the id is
            malformed and ``CAMPAIGN_NOT_FOUND`` when it is a well-
            formed UUID but no such campaign exists. The empty-but-
            existing-campaign case still returns Markdown so callers
            can distinguish "campaign has no audit trail yet" from
            "campaign does not exist".
    """
    try:
        campaign_uuid = UUID(campaign_id)
    except ValueError:
        raise_resource_error(
            ErrorCode.INVALID_CAMPAIGN_ID,
            details={"campaign_id": campaign_id},
        )

    async with get_session() as session:
        # Verify the campaign exists before listing events so that an
        # empty event log cannot be confused with a missing campaign.
        if await CampaignRepository(session).get(campaign_uuid) is None:
            raise_resource_error(
                ErrorCode.CAMPAIGN_NOT_FOUND,
                message=f"Campaign {campaign_id} not found",
                details={"campaign_id": campaign_id},
            )

        repo = EventRepository(session)
        events = await repo.list_by_campaign(campaign_uuid, limit=_EVENTS_RESOURCE_LIMIT)

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
