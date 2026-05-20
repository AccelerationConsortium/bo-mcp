"""Tests for API health endpoints."""

import pytest


class TestAPIHealth:
    @pytest.mark.asyncio
    async def test_health_reports_api_readiness(self, api_client) -> None:
        response = await api_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["healthy"] is True
        assert data["service"] == "api"
        assert data["database"] == "connected"
        # ``database_error`` is null on the healthy path so the shape
        # is uniform — clients can always address the key.
        assert data["database_error"] is None
        assert isinstance(data["version"], str)
        assert data["uptime_seconds"] >= 0
        assert "tools_available" not in data

    @pytest.mark.asyncio
    async def test_health_surfaces_db_error_class(
        self, api_client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the probe fails the response carries the exception class name.

        Operators reading the response should be able to tell a
        ``OperationalError`` (connectivity) apart from a
        ``ProgrammingError`` (auth/permissions) without grepping logs.
        """
        from bo_mcp_server import client as client_module
        from bo_mcp_server.client.lifecycle import DatabasePingResult

        async def fake_ping() -> DatabasePingResult:
            return DatabasePingResult(healthy=False, error_class="OperationalError")

        # Patch both the original symbol and the api.main import site so
        # the route sees the failing probe.
        monkeypatch.setattr(client_module, "ping_database_detailed", fake_ping)
        from api import main as api_main

        monkeypatch.setattr(api_main, "ping_database_detailed", fake_ping)

        response = await api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["healthy"] is False
        assert data["database"] == "error"
        assert data["database_error"] == "OperationalError"
