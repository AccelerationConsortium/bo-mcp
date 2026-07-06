"""Capabilities routes."""

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_current_user
from api.schemas.campaign import CapabilitiesResponse
from api.schemas.errors import AUTH_ERROR_RESPONSES
from bo_mcp_server.client import list_capabilities_operation

router = APIRouter(responses=AUTH_ERROR_RESPONSES)


@router.get("", dependencies=[Depends(get_current_user)])
async def list_capabilities(
    backend: Annotated[
        str | None,
        Query(
            description="Backend to report on (e.g. 'baybe', 'botorch'). "
            "Omit for the default backend.",
        ),
    ] = None,
) -> CapabilitiesResponse:
    """List the capabilities of a BO backend.

    The auth check runs as a route-level dependency rather than a
    parameter so the body does not have to accept an unused user.
    """
    try:
        # Off-thread: a cold backend lookup imports its module (seconds).
        result = await asyncio.to_thread(list_capabilities_operation, backend)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return CapabilitiesResponse(**result)
