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
        assert isinstance(data["version"], str)
        assert data["uptime_seconds"] >= 0
        assert "tools_available" not in data
