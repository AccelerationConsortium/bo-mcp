"""Diagnostics routes."""

from typing import Any

from bo_mcp_server.tools.get_diagnostics import get_diagnostics
from fastapi import APIRouter, HTTPException, status

from api.deps import CurrentUser, get_authorized_campaign

router = APIRouter()


@router.get("/{campaign_id}")
async def get_campaign_diagnostics(
    campaign_id: str,
    current_user: CurrentUser,
) -> dict[str, Any]:
    """Get diagnostic information for a campaign.

    This is a thin proxy to the MCP bo_get_diagnostics tool.
    """
    await get_authorized_campaign(campaign_id, current_user)

    # Get diagnostics via MCP tool
    result = await get_diagnostics(campaign_id)

    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("errors", ["Unknown error"])[0],
        )

    return result
