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
import os

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

logger = logging.getLogger(__name__)

# Default allowed hosts/origins for DNS rebinding protection.
# Extend via MCP_ALLOWED_HOSTS env var (comma-separated, e.g. "proxy:*,gateway:*").
_DEFAULT_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*", "mcp:*"]
_DEFAULT_ORIGINS = [
    "http://127.0.0.1:*",
    "http://localhost:*",
    "http://[::1]:*",
    "http://mcp:*",
]


def _build_allowed_hosts() -> list[str]:
    extra = os.environ.get("MCP_ALLOWED_HOSTS", "")
    hosts = list(_DEFAULT_HOSTS)
    if extra:
        hosts.extend(h.strip() for h in extra.split(",") if h.strip())
    return hosts


def _build_allowed_origins() -> list[str]:
    extra = os.environ.get("MCP_ALLOWED_ORIGINS", "")
    origins = list(_DEFAULT_ORIGINS)
    if extra:
        origins.extend(o.strip() for o in extra.split(",") if o.strip())
    return origins


# Module-level instance for decorator-based tool/resource registration.
# ONLY import this in tool/resource modules for @mcp.tool()/@mcp.resource().
# For all other uses, call create_mcp_server().
mcp = FastMCP(
    "bo-mcp",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_build_allowed_hosts(),
        allowed_origins=_build_allowed_origins(),
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

    from bo_mcp_server.resources import (  # noqa: F401, PLC0415
        campaign_resource,
        events_resource,
        suggestion_resource,
    )
    from bo_mcp_server.tools import (  # noqa: F401, PLC0415
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
