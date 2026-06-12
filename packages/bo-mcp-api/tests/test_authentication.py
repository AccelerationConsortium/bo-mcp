"""End-to-end authentication tests for the REST transport.

Covers the real API-key flow that replaced the dev-auth bypass:

* Missing / blank / unknown ``X-API-Key`` headers must yield HTTP 401
  with a generic challenge that does not leak which keys are merely
  malformed.
* A valid key resolves the persisted user and the request continues.
* Cross-tenant ownership remains enforced (a valid key for user A
  cannot reach user B's campaign).

References
----------
* OWASP API Security Top 10, ``API2: Broken Authentication`` —
  https://owasp.org/API-Security/editions/2023/en/0xa2-broken-authentication/
* RFC 7235 §3.1 ``401 Unauthorized`` — the challenge header is
  ``WWW-Authenticate``.
"""

from __future__ import annotations

import pytest

from bo_mcp_server.tools.create_campaign import create_campaign

pytestmark = pytest.mark.usefixtures("persisted_user")


async def _create_campaign_for_owner(owner_id: str, name: str) -> str:
    """Create a minimal campaign via the MCP operation path.

    Mirrors the helper used in the parity-route tests so we can stand up
    realistic state without making an HTTP round-trip — keeps the auth
    test focused on the auth boundary.
    """
    result = await create_campaign(
        {
            "name": name,
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        },
        owner_id,
    )
    return result["campaign_id"]


class TestApiKeyChallenge:
    """Missing or unknown API keys must yield a generic 401 challenge."""

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_401(self, api_client) -> None:
        response = await api_client.get("/api/campaigns")

        assert response.status_code == 401
        assert response.headers.get("www-authenticate", "").lower().startswith("apikey")
        assert "Authentication required" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_blank_api_key_returns_401(self, api_client) -> None:
        """A whitespace-only header must not be treated as a valid principal."""
        response = await api_client.get("/api/campaigns", headers={"X-API-Key": "   "})

        assert response.status_code == 401
        assert response.headers.get("www-authenticate", "").lower().startswith("apikey")

    @pytest.mark.asyncio
    async def test_unknown_api_key_returns_401(self, api_client) -> None:
        response = await api_client.get(
            "/api/campaigns", headers={"X-API-Key": "no-such-key-exists"}
        )

        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]


class TestApiKeyOpenApi:
    """OpenAPI must advertise API-key auth as a security scheme."""

    @pytest.mark.asyncio
    async def test_openapi_documents_api_key_security_scheme(self, api_client) -> None:
        response = await api_client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()
        assert schema["components"]["securitySchemes"]["ApiKeyAuth"] == {
            "type": "apiKey",
            "description": "BO-MCP API key. Send this header on all authenticated API requests.",
            "in": "header",
            "name": "X-API-Key",
        }

        list_campaigns = schema["paths"]["/api/v1/campaigns"]["get"]
        assert {"ApiKeyAuth": []} in list_campaigns["security"]
        assert all(
            parameter["name"].lower() != "x-api-key"
            for parameter in list_campaigns.get("parameters", [])
        )


class TestApiKeyAcceptance:
    """A valid API key must resolve the matching persisted user."""

    @pytest.mark.asyncio
    async def test_valid_api_key_returns_200(self, api_client, auth_headers) -> None:
        response = await api_client.get("/api/campaigns", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "campaigns" in data
        assert "total" in data

    @pytest.mark.asyncio
    async def test_secondary_user_key_returns_only_own_campaigns(
        self,
        api_client,
        auth_headers,
        other_auth_headers,
        persisted_user,
        persisted_another_user,
    ) -> None:
        """Each key must resolve to its own user, not a shared bypass."""
        await _create_campaign_for_owner(str(persisted_user.id), "Owned by primary user")
        await _create_campaign_for_owner(str(persisted_another_user.id), "Owned by secondary user")

        primary = await api_client.get("/api/campaigns", headers=auth_headers)
        secondary = await api_client.get("/api/campaigns", headers=other_auth_headers)

        assert primary.status_code == 200
        assert secondary.status_code == 200
        assert primary.json()["total"] == 1
        assert secondary.json()["total"] == 1
        assert primary.json()["campaigns"][0]["name"] == "Owned by primary user"
        assert secondary.json()["campaigns"][0]["name"] == "Owned by secondary user"


class TestCrossTenantAccess:
    """Ownership checks must remain enforced for authenticated callers."""

    @pytest.mark.asyncio
    async def test_cross_tenant_campaign_get_returns_403(
        self,
        api_client,
        auth_headers,
        persisted_another_user,
    ) -> None:
        foreign_id = await _create_campaign_for_owner(
            str(persisted_another_user.id), "Owned by another user"
        )

        response = await api_client.get(f"/api/campaigns/{foreign_id}", headers=auth_headers)

        assert response.status_code == 403


class TestDeactivatedUser:
    """A deactivated user's API key must be rejected.

    Reference: OWASP ASVS V2.5 (credential lifecycle) — disabling an
    account must invalidate its credentials. Without filtering on
    ``is_active`` the hash lookup keeps working after offboarding.
    """

    @pytest.mark.asyncio
    async def test_inactive_user_key_returns_401(
        self, api_client, auth_headers, persisted_user
    ) -> None:
        from bo_mcp_server.storage import UserRepository, get_session

        async with get_session() as session:
            repo = UserRepository(session)
            await repo.save(persisted_user.deactivate())
            await session.commit()

        response = await api_client.get("/api/campaigns", headers=auth_headers)

        assert response.status_code == 401
        # Reuses the same "Invalid API key" wording as unknown keys so
        # the response cannot be used to enumerate deactivated accounts.
        assert "Invalid API key" in response.json()["detail"]
