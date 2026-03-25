"""Tests for the health_check tool.

These tests verify that the health check tool provides correct
server status information for agents to verify connectivity.

References:
- Implementation Plan Section 12.3: Health Check Tool
- MCP Specification: https://modelcontextprotocol.io/specification/2025-11-25
"""

import time
from unittest.mock import AsyncMock, patch

import pytest

from bo_mcp_server import __version__
from bo_mcp_server.server import create_mcp_server
from bo_mcp_server.tools.health_check import (
    _get_server_start_time,
    health_check,
)


class TestHealthCheckResponse:
    """Tests for health_check response structure."""

    @pytest.mark.asyncio
    async def test_health_check_returns_required_fields(self, setup_database: None) -> None:
        """Verify health_check returns all required fields.

        Reference: Section 12.3 specifies the response schema.
        """
        result = await health_check()

        assert "healthy" in result
        assert "version" in result
        assert "database" in result
        assert "tools_available" in result
        assert "uptime_seconds" in result

    @pytest.mark.asyncio
    async def test_health_check_healthy_when_database_connected(self, setup_database: None) -> None:
        """Verify healthy=True when database is connected.

        This is the primary success indicator for agents.
        """
        result = await health_check()

        assert result["healthy"] is True
        assert result["database"] == "connected"

    @pytest.mark.asyncio
    async def test_health_check_returns_correct_version(self, setup_database: None) -> None:
        """Verify version matches package version."""
        result = await health_check()

        assert result["version"] == __version__

    @pytest.mark.asyncio
    async def test_health_check_returns_correct_tool_count(self, setup_database: None) -> None:
        """Verify tools_available matches actual tool count."""
        expected_tools = len(create_mcp_server()._tool_manager.list_tools())
        result = await health_check()

        assert result["tools_available"] == expected_tools

    @pytest.mark.asyncio
    async def test_health_check_uptime_is_non_negative(self, setup_database: None) -> None:
        """Verify uptime is a non-negative integer."""
        result = await health_check()

        assert isinstance(result["uptime_seconds"], int)
        assert result["uptime_seconds"] >= 0


class TestHealthCheckDatabaseError:
    """Tests for health_check behavior when database is unavailable."""

    @pytest.mark.asyncio
    async def test_health_check_unhealthy_on_database_error(self) -> None:
        """Verify healthy=False when database connection fails.

        This test mocks the database session to simulate connection failure.
        """
        with patch("bo_mcp_server.tools.health_check.get_session") as mock_get_session:
            # Simulate database connection error
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.execute = AsyncMock(side_effect=Exception("Connection refused"))
            mock_get_session.return_value = mock_session

            result = await health_check()

            assert result["healthy"] is False
            assert result["database"] == "error"

    @pytest.mark.asyncio
    async def test_health_check_still_returns_version_on_db_error(self) -> None:
        """Verify version is returned even when database fails.

        Agents should still be able to identify the server version.
        """
        with patch("bo_mcp_server.tools.health_check.get_session") as mock_get_session:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session.execute = AsyncMock(side_effect=Exception("Connection refused"))
            mock_get_session.return_value = mock_session

            result = await health_check()

            assert result["version"] == __version__
            assert result["tools_available"] == len(create_mcp_server()._tool_manager.list_tools())


class TestServerStartTime:
    """Tests for server start time tracking."""

    def test_get_server_start_time_initializes_on_first_call(self) -> None:
        """Verify start time is initialized on first call."""
        # Reset the global variable for testing
        import bo_mcp_server.tools.health_check as hc_module

        hc_module._server_start_time = None

        before = time.time()
        start_time = _get_server_start_time()
        after = time.time()

        assert before <= start_time <= after

    def test_get_server_start_time_returns_same_value(self) -> None:
        """Verify start time is consistent across calls."""
        # Reset for testing
        import bo_mcp_server.tools.health_check as hc_module

        hc_module._server_start_time = None

        first_call = _get_server_start_time()
        time.sleep(0.01)  # Small delay
        second_call = _get_server_start_time()

        assert first_call == second_call


class TestHealthCheckIntegration:
    """Integration tests for health_check usage patterns."""

    @pytest.mark.asyncio
    async def test_agent_can_verify_server_before_operations(self, setup_database: None) -> None:
        """Test typical agent verification workflow.

        Agents should call health_check before starting optimization
        to verify the server is ready.
        """
        result = await health_check()

        # Agent decision logic
        if result["healthy"]:
            # Server is ready, can proceed with create_campaign, etc.
            assert result["database"] == "connected"
            assert result["tools_available"] > 0
        else:
            # Server not ready, agent should wait or report error
            pytest.fail("Server should be healthy in test environment")

    @pytest.mark.asyncio
    async def test_health_check_is_lightweight(self, setup_database: None) -> None:
        """Verify health_check completes quickly.

        Health checks should be fast for agent pre-flight checks.
        """
        start = time.time()
        await health_check()
        elapsed = time.time() - start

        # Should complete in under 1 second (generous for CI environments)
        assert elapsed < 1.0, f"Health check took {elapsed:.2f}s, expected <1s"

    @pytest.mark.asyncio
    async def test_multiple_health_checks_are_idempotent(self, setup_database: None) -> None:
        """Verify multiple health checks return consistent results.

        Agents may call health_check multiple times; results should be stable.
        """
        result1 = await health_check()
        result2 = await health_check()
        result3 = await health_check()

        # Core fields should be identical
        assert result1["healthy"] == result2["healthy"] == result3["healthy"]
        assert result1["version"] == result2["version"] == result3["version"]
        assert result1["database"] == result2["database"] == result3["database"]
        assert (
            result1["tools_available"] == result2["tools_available"] == result3["tools_available"]
        )

        # Uptime may increase slightly but should all be valid
        assert all(r["uptime_seconds"] >= 0 for r in [result1, result2, result3])
