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

    # Wire the enum-aware ``completion/complete`` handler so agents
    # discover valid values for ``status`` / ``acquisition_method`` /
    # ``backend`` / ``action`` / etc. without resorting to trial-and-
    # error retries. Lazy import keeps the dependency graph clean for
    # tests that only exercise tools.
    from bo_mcp_server.completion import (  # noqa: PLC0415
        register_completion_handler,
    )
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
        list_capabilities,
        list_results,
        list_suggestions,
        submit_results,
        update_suggestion_status,
        upload_results_file,
        validate_intake,
    )

    register_completion_handler(mcp)

    # Wire ``resources/subscribe`` + ``resources/unsubscribe`` so
    # long-running agents can stop polling ``campaign://{id}`` for
    # state transitions. Notifications are emitted by the lifecycle
    # operations themselves (see :mod:`bo_mcp_server.subscriptions`).
    from bo_mcp_server.subscriptions import (  # noqa: PLC0415
        register_subscription_handlers,
    )

    register_subscription_handlers(mcp)

    # Convert FastMCP's argument-validation ``ToolError`` (raised when
    # callers send missing / wrongly-typed scalars like ``owner_id``,
    # ``campaign_id``, ``submitted_by``, or boolean flags) into our
    # structured ``field_errors`` envelope. Without this, FastMCP
    # would intercept those failures before the tool body runs and
    # agents would see opaque ToolError text instead of an
    # addressable per-field response.
    from bo_mcp_server.tool_boundary import (  # noqa: PLC0415
        assert_all_tools_routed_through_wrapper,
        install_validation_envelope_wrapper,
    )

    install_validation_envelope_wrapper(mcp)
    # Startup invariant (TODO 8.45): every registered tool must route
    # through the envelope wrapper so payload-validation failures
    # always surface as the structured ``field_errors`` envelope. Pin
    # the contract here so a future code path that bypasses the
    # wrapper fails at boot instead of leaking ``ToolError`` text on
    # the first failing call.
    assert_all_tools_routed_through_wrapper(mcp)

    # TODO 8.43 follow-up: surface ``ResourceOperationError`` at the
    # FastMCP read-resource boundary so the structured envelope is
    # not double-wrapped in ``"Error creating resource from template:
    # ..."`` strings before the lowlevel JSON-RPC dispatcher sees it.
    from bo_mcp_server.resource_boundary import (  # noqa: PLC0415
        install_resource_envelope_wrapper,
    )

    install_resource_envelope_wrapper(mcp)

    n_tools = len(mcp._tool_manager._tools)
    logger.info("MCP server created with %d tools", n_tools)
    return mcp
