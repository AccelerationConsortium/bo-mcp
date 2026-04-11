"""Capabilities routes."""

from bo_mcp_server import __version__
from bo_mcp_server.backend import get_backend
from fastapi import APIRouter

from api.deps import CurrentUser
from api.schemas.campaign import CapabilitiesResponse

router = APIRouter()


@router.get("", response_model=CapabilitiesResponse)
async def list_capabilities(
    current_user: CurrentUser,
) -> CapabilitiesResponse:
    """List the capabilities of the active BO backend."""
    backend = get_backend()
    return CapabilitiesResponse(
        backend=backend.name,
        supported_features=sorted(backend.supported_features),
        server_version=__version__,
    )
