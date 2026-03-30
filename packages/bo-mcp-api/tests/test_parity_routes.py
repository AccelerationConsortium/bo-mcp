"""Tests for REST endpoints that mirror MCP-only functionality."""

import pytest
from bo_mcp_server.domain import ResultSubmissionInput
from bo_mcp_server.tools.create_campaign import create_campaign
from bo_mcp_server.tools.generate_suggestions import generate_suggestions
from bo_mcp_server.tools.submit_results import submit_results


def _to_result_inputs(rows: list[dict]) -> list[ResultSubmissionInput]:
    return [
        ResultSubmissionInput(
            parameter_values=row["parameter_values"],
            objective_values=row["objective_values"],
            suggestion_id=row.get("suggestion_id"),
            metadata=row.get("metadata", {}),
        )
        for row in rows
    ]


async def _create_campaign_for_owner(owner_id: str, name: str) -> str:
    result = await create_campaign(
        {
            "name": name,
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        },
        owner_id,
    )
    return result["campaign_id"]


class TestCampaignParityRoutes:
    @pytest.mark.asyncio
    async def test_lifecycle_route_pauses_campaign(self, api_client, auth_headers, persisted_user):
        campaign_id = await _create_campaign_for_owner(str(persisted_user.id), "Lifecycle API Test")
        await generate_suggestions(campaign_id)

        response = await api_client.post(
            f"/api/campaigns/{campaign_id}/lifecycle",
            json={"action": "pause"},
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "paused"
        assert data["previous_status"] == "running"

    @pytest.mark.asyncio
    async def test_batch_status_route_keeps_invalid_ids_in_payload(
        self,
        api_client,
        auth_headers,
        persisted_user,
    ):
        campaign_id = await _create_campaign_for_owner(str(persisted_user.id), "Batch API Test")

        response = await api_client.post(
            "/api/campaigns/status/batch",
            json={"campaign_ids": [campaign_id, "not-a-uuid"], "verbosity": "minimal"},
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert campaign_id in data["campaigns"]
        assert "not-a-uuid" in data["failed_ids"]
        assert any("invalid uuid format" in error.lower() for error in data["errors"])

    @pytest.mark.asyncio
    async def test_batch_status_route_rejects_foreign_campaigns(
        self,
        api_client,
        auth_headers,
        persisted_user,
        persisted_another_user,
    ):
        foreign_campaign_id = await _create_campaign_for_owner(
            str(persisted_another_user.id),
            "Foreign Batch API Test",
        )

        response = await api_client.post(
            "/api/campaigns/status/batch",
            json={"campaign_ids": [foreign_campaign_id], "verbosity": "minimal"},
            headers=auth_headers,
        )

        assert response.status_code == 403
        assert "not authorized" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_compare_route_returns_comparison(
        self,
        api_client,
        auth_headers,
        persisted_user,
    ):
        owner_id = str(persisted_user.id)
        campaign_a = await _create_campaign_for_owner(owner_id, "Compare Campaign A")
        campaign_b = await _create_campaign_for_owner(owner_id, "Compare Campaign B")

        await generate_suggestions(campaign_a)
        await generate_suggestions(campaign_b)
        await submit_results(
            campaign_a,
            _to_result_inputs([{"parameter_values": {"x": 0.4}, "objective_values": {"y": 1.0}}]),
            owner_id,
        )
        await submit_results(
            campaign_b,
            _to_result_inputs([{"parameter_values": {"x": 0.3}, "objective_values": {"y": 0.8}}]),
            owner_id,
        )

        response = await api_client.post(
            "/api/campaigns/compare",
            json={"campaign_ids": [campaign_a, campaign_b], "verbosity": "standard"},
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["campaigns"]) == 2
        assert data["comparison"] is not None
        assert "recommendation" in data["comparison"]

    @pytest.mark.asyncio
    async def test_transfer_candidates_route_returns_candidates(
        self,
        api_client,
        auth_headers,
        persisted_user,
    ):
        owner_id = str(persisted_user.id)
        source_campaign = await create_campaign(
            {
                "name": "Source Transfer Campaign",
                "parameters": [
                    {"name": "temperature", "type": "continuous", "bounds": [20.0, 100.0]},
                    {"name": "pressure", "type": "continuous", "bounds": [1.0, 10.0]},
                ],
                "objectives": [{"name": "yield", "direction": "maximize"}],
            },
            owner_id,
        )
        target_campaign = await create_campaign(
            {
                "name": "Target Transfer Campaign",
                "parameters": [
                    {"name": "temperature", "type": "continuous", "bounds": [30.0, 90.0]},
                    {"name": "pressure", "type": "continuous", "bounds": [2.0, 8.0]},
                ],
                "objectives": [{"name": "yield", "direction": "maximize"}],
            },
            owner_id,
        )

        await generate_suggestions(source_campaign["campaign_id"])
        await submit_results(
            source_campaign["campaign_id"],
            _to_result_inputs(
                [
                    {
                        "parameter_values": {"temperature": 50.0, "pressure": 5.0},
                        "objective_values": {"yield": 0.8},
                    },
                    {
                        "parameter_values": {"temperature": 60.0, "pressure": 6.0},
                        "objective_values": {"yield": 0.85},
                    },
                    {
                        "parameter_values": {"temperature": 70.0, "pressure": 7.0},
                        "objective_values": {"yield": 0.9},
                    },
                ]
            ),
            owner_id,
        )

        response = await api_client.post(
            f"/api/campaigns/{target_campaign['campaign_id']}/transfer-candidates",
            json={
                "similarity_threshold": 0.3,
                "max_candidates": 5,
                "verbosity": "standard",
            },
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["target_campaign"]["name"] == "Target Transfer Campaign"
        assert data["candidates"]
        assert data["candidates"][0]["name"] == "Source Transfer Campaign"


class TestSuggestionParityRoutes:
    @pytest.mark.asyncio
    async def test_suggestion_explanation_route_returns_provenance(
        self,
        api_client,
        auth_headers,
        persisted_user,
    ):
        campaign_id = await _create_campaign_for_owner(
            str(persisted_user.id),
            "Explanation API Test",
        )
        generated = await generate_suggestions(campaign_id)
        suggestion_id = generated["suggestions"][0]["id"]

        response = await api_client.get(
            f"/api/suggestions/{suggestion_id}/explanation",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["explanation"] is not None
        assert data["provenance"]["generation_method"] is not None

    @pytest.mark.asyncio
    async def test_suggestion_explanation_route_enforces_ownership(
        self,
        api_client,
        auth_headers,
        persisted_user,
        persisted_another_user,
    ):
        campaign_id = await _create_campaign_for_owner(
            str(persisted_another_user.id),
            "Foreign Explanation API Test",
        )
        generated = await generate_suggestions(campaign_id)
        suggestion_id = generated["suggestions"][0]["id"]

        response = await api_client.get(
            f"/api/suggestions/{suggestion_id}/explanation",
            headers=auth_headers,
        )

        assert response.status_code == 403
        assert "not authorized" in response.json()["detail"].lower()
