"""Capabilities routes."""

from bo_mcp_server.client import list_capabilities_operation
from fastapi import APIRouter

from api.deps import CurrentUser
from api.schemas.campaign import CapabilitiesResponse

router = APIRouter()


@router.get("", response_model=CapabilitiesResponse)
async def list_capabilities(
    current_user: CurrentUser,
) -> CapabilitiesResponse:
    """List the capabilities of the active BO backend."""
    result = list_capabilities_operation()
    return CapabilitiesResponse(**result)
