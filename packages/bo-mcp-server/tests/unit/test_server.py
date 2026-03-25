"""Tests for MCP server configuration."""

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
