"""Synchronous POST creates return 201 + ``Location``.

The contract: every endpoint that *creates* a resource synchronously
must (a) return ``201 Created`` and (b) carry a ``Location`` header
whose GET resolves the freshly-created entity (or, for batch creates
that have no per-row GET, the collection that contains it).

References
----------
* RFC 9110 §15.3.2 (201 Created) — the ``Location`` header is the
  primary pointer at the new resource.
* RFC 9110 §10.2.2 — ``Location`` for synchronous creates.
"""

from __future__ import annotations

import pytest

from bo_mcp_server.tools.create_campaign import create_campaign


async def _create_campaign_for_owner(owner_id: str, name: str = "Status Codes") -> str:
    result = await create_campaign(
        {
            "name": name,
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        },
        owner_id,
    )
    return result["campaign_id"]


class TestCreateCampaignStatusCode:
    @pytest.mark.asyncio
    async def test_post_campaigns_returns_201_with_location(
        self, api_client, auth_headers, persisted_user
    ) -> None:
        _ = persisted_user
        payload = {
            "intake": {
                "name": "Status Code Create",
                "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
                "objectives": [{"name": "y", "direction": "minimize"}],
            }
        }
        response = await api_client.post("/api/campaigns", json=payload, headers=auth_headers)

        assert response.status_code == 201, response.text
        body = response.json()
        location = response.headers["Location"]
        assert location == f"/api/v1/campaigns/{body['campaign_id']}"

        # Contract test: the Location URL must resolve via GET back
        # to the same campaign.
        follow_up = await api_client.get(location, headers=auth_headers)
        assert follow_up.status_code == 200
        assert follow_up.json()["id"] == body["campaign_id"]


class TestSubmitResultsStatusCode:
    @pytest.mark.asyncio
    async def test_post_results_returns_201_with_collection_location(
        self, api_client, auth_headers, persisted_user
    ) -> None:
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign_for_owner(owner_id, "Submit Status")
        body = {
            "results": [
                {
                    "parameter_values": {"x": 0.5},
                    "objective_values": {"y": 0.7},
                }
            ],
            "source": "api",
        }

        response = await api_client.post(
            f"/api/results/{campaign_id}", json=body, headers=auth_headers
        )

        assert response.status_code == 201, response.text
        location = response.headers["Location"]
        assert location == f"/api/v1/results/{campaign_id}"

        # The collection URL must resolve and include the new row.
        follow_up = await api_client.get(location, headers=auth_headers)
        assert follow_up.status_code == 200
        rows = follow_up.json()
        assert any(row["id"] == response.json()["result_ids"][0] for row in rows)


class TestGenerateSuggestionsStatusCode:
    @pytest.mark.asyncio
    async def test_post_generate_returns_201_with_collection_location(
        self, api_client, auth_headers, persisted_user
    ) -> None:
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign_for_owner(owner_id, "Generate Status")

        response = await api_client.post(
            f"/api/suggestions/{campaign_id}/generate",
            headers=auth_headers,
        )

        assert response.status_code == 201, response.text
        location = response.headers["Location"]
        assert location == f"/api/v1/suggestions/{campaign_id}"

        follow_up = await api_client.get(location, headers=auth_headers)
        assert follow_up.status_code == 200
        listed_ids = {row["suggestion_id"] for row in follow_up.json()}
        produced_ids = {s["suggestion_id"] for s in response.json()["suggestions"]}
        assert produced_ids.issubset(listed_ids)


class TestUploadResultsStatusCode:
    @pytest.mark.asyncio
    async def test_post_upload_returns_201_with_collection_location(
        self, api_client, auth_headers, persisted_user
    ) -> None:
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign_for_owner(owner_id, "Upload Status")
        csv_payload = b"x,y\n0.4,0.5\n"

        response = await api_client.post(
            f"/api/results/{campaign_id}/upload",
            files={"file": ("rows.csv", csv_payload, "text/csv")},
            headers=auth_headers,
        )

        assert response.status_code == 201, response.text
        assert response.headers["Location"] == f"/api/v1/results/{campaign_id}"
