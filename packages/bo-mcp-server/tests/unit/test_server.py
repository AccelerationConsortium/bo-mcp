"""Tests for MCP server transport security configuration."""

from bo_mcp_server.server import mcp


def test_transport_security_allows_docker_service_hostname() -> None:
    settings = mcp.settings.transport_security

    assert settings is not None
    assert settings.enable_dns_rebinding_protection is True
    assert "mcp:*" in settings.allowed_hosts
    assert "http://mcp:*" in settings.allowed_origins
