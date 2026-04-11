"""List capabilities operation - protocol-neutral business logic."""

from typing import Any

from bo_mcp_server import __version__
from bo_mcp_server.backend import get_backend


def list_capabilities_operation() -> dict[str, Any]:
    """List the capabilities of the active BO backend.

    Returns:
        Dictionary with backend name, supported features, and server version.
    """
    backend = get_backend()
    return {
        "backend": backend.name,
        "supported_features": sorted(backend.supported_features),
        "server_version": __version__,
    }
