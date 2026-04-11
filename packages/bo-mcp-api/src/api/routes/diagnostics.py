"""Diagnostics routes."""

from typing import Annotated, Any

from bo_mcp_server.operations.get_diagnostics import get_diagnostics_operation
from fastapi import APIRouter, HTTPException, Query, status

from api.deps import CurrentUser, get_authorized_campaign

router = APIRouter()


@router.get("/{campaign_id}")
async def get_campaign_diagnostics(
    campaign_id: str,
    current_user: CurrentUser,
    verbosity: Annotated[str, Query()] = "standard",
    use_cache: Annotated[bool, Query()] = True,
    sections: Annotated[list[str] | None, Query()] = None,
) -> dict[str, Any]:
    """Get diagnostic information for a campaign."""
    await get_authorized_campaign(campaign_id, current_user)

    result = await get_diagnostics_operation(
        campaign_id=campaign_id,
        verbosity=verbosity,
        use_cache=use_cache,
        sections=sections,
    )

    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("errors", ["Unknown error"])[0],
        )

    return result
