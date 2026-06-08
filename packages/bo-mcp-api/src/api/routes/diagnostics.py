"""Diagnostics routes."""

from typing import Annotated

from bo_mcp_server.client import (
    VerbosityLevel,
    get_diagnostics_operation,
    http_status_for_error,
)
from fastapi import APIRouter, HTTPException, Query

from api.deps import CurrentUser, get_authorized_campaign
from api.schemas.diagnostics import DiagnosticsResponse
from api.schemas.errors import COMMON_HTTP_ERROR_RESPONSES

router = APIRouter(responses=COMMON_HTTP_ERROR_RESPONSES)


@router.get("/{campaign_id}", response_model_exclude_unset=True)
async def get_campaign_diagnostics(
    campaign_id: str,
    current_user: CurrentUser,
    verbosity: Annotated[VerbosityLevel, Query()] = VerbosityLevel.STANDARD,
    use_cache: Annotated[bool, Query()] = True,
    sections: Annotated[list[str] | None, Query()] = None,
) -> DiagnosticsResponse:
    """Get diagnostic information for a campaign.

    The deep metric blocks vary by backend and verbosity, so
    :class:`DiagnosticsResponse` pins the stable top-level keys and lets
    the rest pass through (``extra="allow"``). The route serializes with
    ``response_model_exclude_unset=True`` so a declared key the operation
    omitted is **not** re-introduced as a default — keeping the body
    byte-equal to the MCP ``bo_get_diagnostics`` projection at every
    verbosity (the ``minimal`` projection drops ``campaign_status`` /
    ``n_pending_suggestions`` / ``warnings``, and so does this route).
    """
    await get_authorized_campaign(campaign_id, current_user)

    result = await get_diagnostics_operation(
        campaign_id=campaign_id,
        verbosity=verbosity.value,
        use_cache=use_cache,
        sections=sections,
    )

    if not result["success"]:
        status_code = http_status_for_error(result)
        raise HTTPException(
            status_code=status_code,
            detail=result.get("errors", ["Unknown error"])[0],
        )

    return DiagnosticsResponse(**result)
