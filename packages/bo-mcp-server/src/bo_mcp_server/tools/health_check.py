"""Health check tool for MCP server verification.

This tool allows agents to verify MCP server connectivity and status
before starting optimization operations.
"""

import asyncio
import logging
import time
from typing import cast

from sqlalchemy import text

from bo_mcp_server import __version__
from bo_mcp_server.backend import get_backend_capabilities
from bo_mcp_server.backend_context import campaign_backend_scope
from bo_mcp_server.response_formatter import attach_response_metadata
from bo_mcp_server.server import create_mcp_server, mcp
from bo_mcp_server.storage.database import get_session
from bo_mcp_server.tools.annotations import READ_ONLY
from bo_mcp_server.tools.response_models import HealthCheckResponse

logger = logging.getLogger(__name__)

# Track server start time for uptime calculation
_server_start_time: float | None = None


def _get_server_start_time() -> float:
    """Get or initialize server start time."""
    global _server_start_time
    if _server_start_time is None:
        _server_start_time = time.time()
    return _server_start_time


@mcp.tool(name="bo_health_check", annotations=READ_ONLY)
async def health_check() -> HealthCheckResponse:
    """Check MCP server health and connectivity.

    Use this tool to verify the server is running and responsive before
    starting optimization workflows. This is especially useful for agents
    to confirm connectivity after initial setup.

    Returns:
        Dictionary with:
            - healthy: Boolean indicating overall server health
            - version: Server version string
            - database: Database connectivity status ("connected" or "error")
            - tools_available: Number of available MCP tools
            - uptime_seconds: Approximate server uptime in seconds
            - backends: Mapping of discovered backend name to capability
              metadata (loaded flag, declared features, or load error).
              Reported so agents can confirm the backend they intend to
              use is actually installed before issuing a campaign create.
    """
    logger.debug("Health check requested")

    # Check database connectivity
    db_status = "error"
    try:
        async with get_session() as session:
            await session.execute(text("SELECT 1"))
            db_status = "connected"
    except Exception as e:  # noqa: BLE001 - health checks must never crash
        logger.warning("Database health check failed: %s", e)
        db_status = "error"

    # Calculate uptime
    start_time = _get_server_start_time()
    uptime = int(time.time() - start_time)

    create_mcp_server()
    tools_count = len(mcp._tool_manager.list_tools())

    # Offloaded to a worker thread: this loads every discovered backend,
    # and a cache miss (e.g. baybe not yet requested by any campaign)
    # would otherwise block the event loop on the module import.
    backends = await asyncio.to_thread(get_backend_capabilities)

    healthy = db_status == "connected" and any(info.get("loaded") for info in backends.values())

    logger.info(
        "Health check completed: healthy=%s, database=%s, uptime=%ds, backends=%s",
        healthy,
        db_status,
        uptime,
        sorted(backends),
    )

    # Campaign-agnostic stamp: shield ``_metadata`` from a binding leaked
    # by a previous same-task operation — in-process callers may bypass
    # the transports' campaign_backend_scope (issue #82).
    with campaign_backend_scope():
        return cast(
            HealthCheckResponse,
            attach_response_metadata(
                {
                    "healthy": healthy,
                    "version": __version__,
                    "database": db_status,
                    "tools_available": tools_count,
                    "uptime_seconds": uptime,
                    "backends": backends,
                }
            ),
        )
