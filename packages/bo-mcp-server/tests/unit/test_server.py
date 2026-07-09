"""Tests for MCP server configuration."""

from starlette.testclient import TestClient

from bo_mcp_server.server import create_mcp_server, mcp


def test_transport_security_allows_docker_service_hostname() -> None:
    settings = mcp.settings.transport_security

    assert settings is not None
    assert settings.enable_dns_rebinding_protection is True
    assert "mcp:*" in settings.allowed_hosts
    assert "http://mcp:*" in settings.allowed_origins


def test_all_registered_tool_names_use_bo_prefix() -> None:
    tool_names = [tool.name for tool in create_mcp_server()._tool_manager.list_tools()]

    assert tool_names
    assert all(name.startswith("bo_") for name in tool_names)


def test_registered_tool_names_match_expected_set() -> None:
    """Pin the exact tool set so a silently-dropped registration fails CI.

    ``create_mcp_server`` registers tools by importing the
    ``bo_mcp_server.tools`` package rather than repeating its member
    list; ``tools/__init__.py`` is the single source of truth for
    membership. This test is the guard against that source of truth
    silently losing a tool (e.g. a future refactor that imports
    submodules individually again and forgets one).
    """
    expected = {
        "bo_batch_get_status",
        "bo_check_progress",
        "bo_compare_campaigns",
        "bo_create_campaign",
        "bo_discover_transfer_candidates",
        "bo_generate_suggestions",
        "bo_get_diagnostics",
        "bo_get_suggestion_explanation",
        "bo_health_check",
        "bo_list_campaigns",
        "bo_list_capabilities",
        "bo_list_results",
        "bo_export_campaign",
        "bo_list_suggestions",
        "bo_pause_campaign",
        "bo_resume_campaign",
        "bo_terminate_campaign",
        "bo_reopen_campaign",
        "bo_submit_results",
        "bo_update_suggestion_status",
        "bo_upload_results_file",
        "bo_validate_intake",
    }
    tool_names = {tool.name for tool in create_mcp_server()._tool_manager.list_tools()}

    assert tool_names == expected


def test_health_endpoint_returns_200_with_status_ok() -> None:
    """``/health`` is a non-streaming JSON probe for docker / k8s.

    Regression for the unhealthy-MCP-container bug:
    docker-compose's healthcheck used to point at ``/sse``, FastMCP's
    streaming endpoint. ``curl -sf http://.../sse`` blocks until
    docker's 10s healthcheck timeout fires (the SSE connection never
    closes), so the container was always marked unhealthy even when
    the process was serving requests normally. The fix is a tiny
    non-streaming ``/health`` route that returns immediately.

    Pins both the path (``/health``) and the body shape
    (``{"status": "ok"}``) so a future rename in either layer
    (route handler or docker-compose probe) fails this test.

    Reference: FastMCP ``custom_route`` decorator usage —
    https://github.com/modelcontextprotocol/python-sdk (see
    ``mcp.server.fastmcp.FastMCP.custom_route`` for the supported
    "non-MCP HTTP routes" pattern).
    """
    app = mcp.sse_app()
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_endpoint_response_is_not_chunked_stream() -> None:
    """``/health`` must return a finite, non-streaming body.

    The whole point of this route is to be safe for ``curl`` without
    ``--max-time``: docker's healthcheck has no way to bail on a
    stream, so any future change that converts the handler to a
    streaming response (``StreamingResponse``, generator body) would
    silently reintroduce the original bug. Asserting on a present
    ``Content-Length`` header is the cheapest way to pin
    "non-chunked, finite body": Starlette's ``JSONResponse`` sets it
    from the encoded body length; ``StreamingResponse`` does not.
    """
    app = mcp.sse_app()
    with TestClient(app) as client:
        response = client.get("/health")

    assert "content-length" in {k.lower() for k in response.headers}, (
        "/health must return a finite body with Content-Length set, not a "
        "stream — docker's healthcheck has no way to bail on a stream and "
        "would block until the 10s timeout fires, reproducing the original "
        "unhealthy-MCP-container bug"
    )


def test_health_route_registered_exactly_once_across_create_calls() -> None:
    """``create_mcp_server()`` must not register duplicate ``/health`` routes.

    Guards against a refactor that moves the ``@mcp.custom_route``
    registration from module level into ``create_mcp_server()``.
    FastMCP's ``custom_route`` decorator appends to
    ``_custom_starlette_routes`` on every call, so an in-function
    registration would duplicate the route on each
    ``create_mcp_server()`` invocation. Tests in this suite call
    ``create_mcp_server()`` more than once across the session
    (e.g. ``test_all_registered_tool_names_use_bo_prefix``), which
    would trip such a regression here.
    """
    create_mcp_server()
    create_mcp_server()

    health_routes = [r for r in mcp._custom_starlette_routes if r.path == "/health"]
    assert len(health_routes) == 1, (
        f"expected /health to be registered exactly once at module import, "
        f"got {len(health_routes)} registrations — a likely cause is the "
        f"@mcp.custom_route decorator being moved inside create_mcp_server()"
    )
