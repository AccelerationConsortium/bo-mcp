"""MCP Server setup using FastMCP.

Architecture note (1.7):
    The module-level `mcp` instance is required by FastMCP's decorator-based
    registration model — tool and resource modules import it to apply
    ``@mcp.tool()``. ``create_mcp_server()`` is the sole public entry point;
    it triggers tool/resource registration by importing the relevant modules
    and returns the fully-configured instance.

    Direct use of the module-level ``mcp`` outside of decorator registration
    is discouraged — always go through ``create_mcp_server()``.
"""

import logging

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

logger = logging.getLogger(__name__)

# Module-level instance for decorator-based tool/resource registration.
# ONLY import this in tool/resource modules for @mcp.tool()/@mcp.resource().
# For all other uses, call create_mcp_server().
mcp = FastMCP(
    "bo-mcp",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*", "mcp:*"],
        allowed_origins=[
            "http://127.0.0.1:*",
            "http://localhost:*",
            "http://[::1]:*",
            "http://mcp:*",
        ],
    ),
)


def create_mcp_server() -> FastMCP:
    """Create and return the fully-configured MCP server.

    This is the sole public entry point. It imports all tool and resource
    modules, which triggers decorator-based registration on the module-level
    ``mcp`` instance. Returns that instance.

    The imports are inside the function to avoid circular imports, since
    tool modules import ``mcp`` from this module.
    """
    logger.info("Creating MCP server...")

    from bo_mcp_server.resources import (  # noqa: F401
        campaign_resource,
        events_resource,
        suggestion_resource,
    )
    from bo_mcp_server.tools import (  # noqa: F401
        batch_operations,
        campaign_lifecycle,
        compare_campaigns,
        create_campaign,
        discover_transfer_candidates,
        generate_suggestions,
        get_diagnostics,
        get_suggestion_explanation,
        health_check,
        list_campaigns,
        list_results,
        list_suggestions,
        submit_results,
        update_suggestion_status,
        upload_results_file,
        validate_intake,
    )

    logger.info("MCP server created with 18 tools and 3 resources")
    return mcp
