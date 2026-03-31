"""List backend capabilities MCP tool.

Returns the active backend's name, supported features, and server version
so agents and API consumers know what functionality is available.
"""

from typing import Any

from bo_mcp_server import __version__
from bo_mcp_server.backend import get_backend
from bo_mcp_server.server import mcp


@mcp.tool(name="bo_list_capabilities")
async def list_capabilities() -> dict[str, Any]:
    """List the capabilities of the active BO backend.

    Use this tool to discover which features are supported by the
    current backend before configuring a campaign. For example, check
    whether multi-fidelity or transfer learning is available.

    Returns:
        Dictionary with:
            - backend: Name of the active backend (e.g. "botorch")
            - supported_features: List of feature names this backend supports
            - server_version: Server version string
    """
    backend = get_backend()
    return {
        "backend": backend.name,
        "supported_features": sorted(backend.supported_features),
        "server_version": __version__,
    }
