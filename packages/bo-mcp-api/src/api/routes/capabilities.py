"""Capabilities routes."""

from bo_mcp_server.client import list_capabilities_operation
from fastapi import APIRouter, Depends

from api.deps import get_current_user
from api.schemas.campaign import CapabilitiesResponse

router = APIRouter()


@router.get("", dependencies=[Depends(get_current_user)])
async def list_capabilities() -> CapabilitiesResponse:
    """List the capabilities of the active BO backend.

    The auth check runs as a route-level dependency rather than a
    parameter so the body does not have to accept an unused user.
    """
    result = list_capabilities_operation()
    return CapabilitiesResponse(**result)
