"""MCP Server setup using FastMCP."""

import logging

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

logger = logging.getLogger(__name__)


# Create MCP server instance
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
    """Create and configure the MCP server.

    This imports all tools and resources to register them.
    The imports are kept inside the function to avoid circular imports,
    as the resources/tools modules need to import `mcp` from this module.
    """
    logger.info("Creating MCP server...")

    # Import tools and resources to register them
    # These imports must be inside the function to avoid circular imports
    from bo_mcp_server.resources import (  # noqa: F401
        campaign_resource,
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
        search_tools,
        submit_results,
        upload_results_file,
        validate_intake,
    )

    logger.info("MCP server created with 17 tools and 2 resources")
    return mcp
