"""Health check tool for MCP server verification.

This tool allows agents to verify MCP server connectivity and status
before starting optimization operations.
"""

import logging
import time
from typing import Any

from sqlalchemy import text

from bo_mcp_server import __version__
from bo_mcp_server.server import mcp
from bo_mcp_server.storage.database import get_session

logger = logging.getLogger(__name__)

# Track server start time for uptime calculation
_server_start_time: float | None = None


def _get_server_start_time() -> float:
    """Get or initialize server start time."""
    global _server_start_time
    if _server_start_time is None:
        _server_start_time = time.time()
    return _server_start_time


@mcp.tool()
async def health_check() -> dict[str, Any]:
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
    """
    logger.debug("Health check requested")

    # Check database connectivity
    db_status = "error"
    try:
        async with get_session() as session:
            await session.execute(text("SELECT 1"))
            db_status = "connected"
    except Exception as e:
        logger.warning("Database health check failed: %s", e)
        db_status = "error"

    # Calculate uptime
    start_time = _get_server_start_time()
    uptime = int(time.time() - start_time)

    # Count available tools (update when adding new tools)
    # Tools: validate_intake, create_campaign, generate_suggestions, submit_results,
    #        get_diagnostics, upload_results_file, get_suggestion_explanation,
    #        pause_campaign, resume_campaign, terminate_campaign,
    #        compare_campaigns, discover_transfer_candidates, health_check
    tools_count = 13

    healthy = db_status == "connected"

    logger.info(
        "Health check completed: healthy=%s, database=%s, uptime=%ds",
        healthy,
        db_status,
        uptime,
    )

    return {
        "healthy": healthy,
        "version": __version__,
        "database": db_status,
        "tools_available": tools_count,
        "uptime_seconds": uptime,
    }
