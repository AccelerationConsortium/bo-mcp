"""List backend capabilities MCP tool wrapper.

Returns a backend's name, supported features, and server version
so agents and API consumers know what functionality is available.
"""

import asyncio
from typing import Annotated, cast

from pydantic import Field

from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.operations.list_capabilities import list_capabilities_operation
from bo_mcp_server.server import mcp
from bo_mcp_server.tools.annotations import READ_ONLY
from bo_mcp_server.tools.response_models import CapabilitiesResponse


@mcp.tool(name="bo_list_capabilities", annotations=READ_ONLY)
async def list_capabilities(
    backend: Annotated[
        str | None,
        Field(
            description=(
                "Backend to report on (e.g. 'baybe', 'botorch'). Omit for the default backend."
            ),
        ),
    ] = None,
) -> CapabilitiesResponse:
    """List the capabilities of a BO backend.

    Use this tool to discover which features are supported by a
    backend before configuring a campaign. For example, check
    whether multi-fidelity or transfer learning is available.
    The response also lists every installed backend and the server
    default, so per-backend follow-up queries need no other tool.

    Returns:
        Dictionary with:
            - backend: Name of the reported backend (e.g. "baybe")
            - supported_features: List of feature names this backend supports
            - conditional_features: Feature -> precondition description map
            - available_backends: All installed backend names
            - default_backend: The server's default backend
            - server_version: Server version string
    """
    try:
        # Offloaded to a worker thread: a cache miss would otherwise
        # block the event loop on the backend module import.
        return cast(
            CapabilitiesResponse,
            await asyncio.to_thread(list_capabilities_operation, backend),
        )
    except ValueError as e:
        return cast(
            CapabilitiesResponse,
            make_error_response(
                ErrorCode.VALIDATION_FAILED,
                message=str(e),
                details={"backend": backend},
            ),
        )
