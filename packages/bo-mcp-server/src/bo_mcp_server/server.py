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
from collections.abc import Iterable, Sequence
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ContentBlock
from pydantic import AnyUrl
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from bo_mcp_server.backend_context import campaign_backend_scope

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


class _ScopedFastMCP(FastMCP):
    """FastMCP with a per-dispatch campaign-backend scope.

    Each tool call / resource read runs inside
    :func:`bo_mcp_server.backend_context.campaign_backend_scope`, so a
    campaign backend bound during one dispatch can never leak into the
    ``_metadata.backend`` stamp of a later dispatch that happens to run
    in the same context (issue #57 review follow-up).
    """

    async def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> Sequence[ContentBlock] | dict[str, Any]:
        with campaign_backend_scope():
            return await super().call_tool(name, arguments)

    async def read_resource(self, uri: AnyUrl | str) -> Iterable[ReadResourceContents]:
        with campaign_backend_scope():
            return await super().read_resource(uri)


# Module-level instance for decorator-based tool/resource registration.
# ONLY import this in tool/resource modules for @mcp.tool()/@mcp.resource().
# For all other uses, call create_mcp_server().
mcp = _ScopedFastMCP(
    "bo-mcp",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=_build_allowed_hosts(),
        allowed_origins=_build_allowed_origins(),
    ),
)


@mcp.custom_route("/health", methods=["GET"])
async def _health_endpoint(request: Request) -> Response:
    """Non-streaming liveness probe for orchestrators (docker, k8s).

    Docker's healthcheck has no notion of a streaming endpoint, so
    pointing ``curl`` at the SSE ``/sse`` route causes the probe to
    block until docker's healthcheck timeout fires — even though the
    server is healthy. This route returns a single small response
    that orchestrators can poll without keeping a connection open.

    Liveness-only by design: a 200 here means the FastMCP uvicorn
    process is serving HTTP. It does NOT probe downstream
    dependencies (database, BO backend). Those failures are surfaced
    per-tool-call (the api container's own ``/health`` probes the DB)
    rather than by yanking the whole MCP container out of rotation,
    where a transient DB blip would cascade into MCP being marked
    unhealthy and break the docker-compose ``depends_on``
    ``service_healthy`` chain.

    Registered at module level (not inside ``create_mcp_server()``)
    so the route is appended to ``mcp._custom_starlette_routes``
    exactly once at import time. ``create_mcp_server()`` is called
    repeatedly from tests; an inline ``@mcp.custom_route`` there
    would register duplicate routes on each call.
    """
    del request  # Starlette handler signature requires it; we do not read it.
    return JSONResponse({"status": "ok"})


def create_mcp_server() -> FastMCP:
    """Create and return the fully-configured MCP server.

    This is the sole public entry point. It imports all tool and resource
    modules, which triggers decorator-based registration on the module-level
    ``mcp`` instance. Returns that instance.

    The imports are inside the function to avoid circular imports, since
    tool modules import ``mcp`` from this module.
    """
    logger.info("Creating MCP server...")

    # ``tools/__init__.py`` is the single source of truth for which tool
    # modules get registered (it is what pins ``bo_check_progress`` in,
    # for example) -- importing the package here, rather than repeating
    # its member list, means there is exactly one place to update when a
    # tool module is added or removed.
    from bo_mcp_server import tools  # noqa: F401

    # Wire the enum-aware ``completion/complete`` handler so agents
    # discover valid values for ``status`` / ``acquisition_method`` /
    # ``backend`` / ``action`` / etc. without resorting to trial-and-
    # error retries. Lazy import keeps the dependency graph clean for
    # tests that only exercise tools.
    from bo_mcp_server.completion import (
        register_completion_handler,
    )
    from bo_mcp_server.resources import (  # noqa: F401
        campaign_resource,
        events_resource,
        suggestion_resource,
    )

    register_completion_handler(mcp)

    # Wire ``resources/subscribe`` + ``resources/unsubscribe`` so
    # long-running agents can stop polling ``campaign://{id}`` for
    # state transitions. Notifications are emitted by the lifecycle
    # operations themselves (see :mod:`bo_mcp_server.subscriptions`).
    from bo_mcp_server.subscriptions import (
        register_subscription_handlers,
    )

    register_subscription_handlers(mcp)

    # Convert FastMCP's argument-validation ``ToolError`` (raised when
    # callers send missing / wrongly-typed scalars like ``campaign_id``
    # or boolean flags) into our
    # structured ``field_errors`` envelope. Without this, FastMCP
    # would intercept those failures before the tool body runs and
    # agents would see opaque ToolError text instead of an
    # addressable per-field response.
    from bo_mcp_server.tool_boundary import (
        assert_all_tools_routed_through_wrapper,
        install_validation_envelope_wrapper,
    )

    install_validation_envelope_wrapper(mcp)
    # Startup invariant: every registered tool must route
    # through the envelope wrapper so payload-validation failures
    # always surface as the structured ``field_errors`` envelope. Pin
    # the contract here so a future code path that bypasses the
    # wrapper fails at boot instead of leaking ``ToolError`` text on
    # the first failing call.
    assert_all_tools_routed_through_wrapper(mcp)

    # Follow-up: surface ``ResourceOperationError`` at the
    # FastMCP read-resource boundary so the structured envelope is
    # not double-wrapped in ``"Error creating resource from template:
    # ..."`` strings before the lowlevel JSON-RPC dispatcher sees it.
    from bo_mcp_server.resource_boundary import (
        install_resource_envelope_wrapper,
    )

    install_resource_envelope_wrapper(mcp)

    n_tools = len(mcp._tool_manager._tools)
    logger.info("MCP server created with %d tools", n_tools)
    return mcp
