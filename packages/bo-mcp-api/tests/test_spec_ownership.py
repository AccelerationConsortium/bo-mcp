"""Ownership checks for the ``GET /api/campaigns/spec/{spec_id}`` route.

The route previously resolved specs purely by id, so any authenticated
caller could read another tenant's parameter, objective, and constraint
shape just by knowing or guessing a spec UUID. The fix routes the lookup
through the owning campaign — a spec is only visible to the user who
owns at least one campaign referencing it.

References
----------
* OWASP API Security Top 10, ``API1: Broken Object Level Authorization``
  https://owasp.org/API-Security/editions/2023/en/0xa1-broken-object-level-authorization/
* OWASP Access Control Cheat Sheet (object-level checks)
  https://cheatsheetseries.owasp.org/cheatsheets/Access_Control_Cheat_Sheet.html
"""

from __future__ import annotations

import pytest
from bo_mcp_server.tools.create_campaign import create_campaign

pytestmark = pytest.mark.usefixtures("persisted_user")


async def _create_campaign_returning_spec(owner_id: str, name: str) -> tuple[str, str]:
    """Create a campaign and return ``(campaign_id, spec_id)`` for the caller."""
    result = await create_campaign(
        {
            "name": name,
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        },
        owner_id,
    )
    return result["campaign_id"], result["spec_id"]


class TestSpecLookupOwnership:
    @pytest.mark.asyncio
    async def test_owner_can_read_spec(self, api_client, auth_headers, persisted_user) -> None:
        _, spec_id = await _create_campaign_returning_spec(str(persisted_user.id), "Owner Spec")

        response = await api_client.get(f"/api/campaigns/spec/{spec_id}", headers=auth_headers)

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == spec_id
        assert body["name"] == "Owner Spec"
        assert body["parameters"]
        assert body["objectives"]

    @pytest.mark.asyncio
    async def test_cross_tenant_returns_404(
        self,
        api_client,
        auth_headers,
        persisted_another_user,
    ) -> None:
        """A foreign caller must not be able to read another tenant's spec.

        The response is 404 (not 403) so the route does not leak the
        existence of specs the caller does not own.
        """
        _, foreign_spec_id = await _create_campaign_returning_spec(
            str(persisted_another_user.id), "Foreign Spec"
        )

        response = await api_client.get(
            f"/api/campaigns/spec/{foreign_spec_id}", headers=auth_headers
        )

        assert response.status_code == 404
        # Sanity: the foreign owner can still see it, so the 404 above is
        # specifically due to the caller not owning a campaign with this spec.
        from bo_mcp_server.client import get_spec_for_user

        owner_view = await get_spec_for_user(foreign_spec_id, persisted_another_user.id)
        assert owner_view is not None

    @pytest.mark.asyncio
    async def test_missing_spec_returns_404(self, api_client, auth_headers) -> None:
        import uuid

        response = await api_client.get(f"/api/campaigns/spec/{uuid.uuid4()}", headers=auth_headers)

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_invalid_spec_id_returns_400(self, api_client, auth_headers) -> None:
        response = await api_client.get("/api/campaigns/spec/not-a-uuid", headers=auth_headers)

        assert response.status_code == 400
