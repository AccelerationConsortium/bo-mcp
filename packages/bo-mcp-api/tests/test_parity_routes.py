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


class TestValidateIntakeRoute:
    @pytest.mark.asyncio
    async def test_validate_valid_intake(self, api_client, auth_headers, persisted_user):
        response = await api_client.post(
            "/api/campaigns/validate",
            json={
                "intake": {
                    "name": "Validate Test",
                    "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
                    "objectives": [{"name": "y", "direction": "minimize"}],
                },
            },
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["errors"] == []
        assert data["spec_summary"] is not None
        assert data["spec_summary"]["name"] == "Validate Test"

    @pytest.mark.asyncio
    async def test_validate_invalid_intake(self, api_client, auth_headers, persisted_user):
        response = await api_client.post(
            "/api/campaigns/validate",
            json={
                "intake": {
                    "name": "Bad Campaign",
                    "parameters": [{"name": "x", "type": "invalid_type", "bounds": [0.0, 1.0]}],
                    "objectives": [{"name": "y", "direction": "minimize"}],
                },
            },
            headers=auth_headers,
        )

        # FastAPI validates the schema — invalid type pattern rejects before operation
        assert response.status_code == 422


class TestCapabilitiesRoute:
    @pytest.mark.asyncio
    async def test_list_capabilities(self, api_client, auth_headers, persisted_user):
        response = await api_client.get(
            "/api/capabilities",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "backend" in data
        assert "supported_features" in data
        assert "server_version" in data
        assert isinstance(data["supported_features"], list)


class TestExportCampaignRoute:
    @pytest.mark.asyncio
    async def test_export_campaign_csv(self, api_client, auth_headers, persisted_user):
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign_for_owner(owner_id, "Export Route Test")
        await generate_suggestions(campaign_id)
        await submit_results(
            campaign_id,
            _to_result_inputs([{"parameter_values": {"x": 0.5}, "objective_values": {"y": 1.0}}]),
            owner_id,
        )

        response = await api_client.get(
            f"/api/campaigns/{campaign_id}/export?format=csv",
            headers=auth_headers,
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"
        assert "attachment" in response.headers["content-disposition"]
        assert "param_x" in response.text
        assert "obj_y" in response.text

    @pytest.mark.asyncio
    async def test_export_campaign_enforces_ownership(
        self, api_client, auth_headers, persisted_user, persisted_another_user
    ):
        foreign_campaign_id = await _create_campaign_for_owner(
            str(persisted_another_user.id), "Foreign Export Test"
        )

        response = await api_client.get(
            f"/api/campaigns/{foreign_campaign_id}/export",
            headers=auth_headers,
        )

        assert response.status_code == 403


class TestSuggestionStatusRoute:
    @pytest.mark.asyncio
    async def test_update_suggestion_status_accept(self, api_client, auth_headers, persisted_user):
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign_for_owner(owner_id, "Status Route Test")
        generated = await generate_suggestions(campaign_id)
        suggestion_id = generated["suggestions"][0]["id"]

        response = await api_client.post(
            f"/api/suggestions/{suggestion_id}/status",
            json={"status": "accepted"},
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["status"] == "accepted"
        assert data["previous_status"] == "pending"

    @pytest.mark.asyncio
    async def test_update_suggestion_status_enforces_ownership(
        self, api_client, auth_headers, persisted_user, persisted_another_user
    ):
        foreign_campaign_id = await _create_campaign_for_owner(
            str(persisted_another_user.id), "Foreign Status Test"
        )
        generated = await generate_suggestions(foreign_campaign_id)
        suggestion_id = generated["suggestions"][0]["id"]

        response = await api_client.post(
            f"/api/suggestions/{suggestion_id}/status",
            json={"status": "accepted"},
            headers=auth_headers,
        )

        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_update_suggestion_invalid_status(self, api_client, auth_headers, persisted_user):
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign_for_owner(owner_id, "Invalid Status Test")
        generated = await generate_suggestions(campaign_id)
        suggestion_id = generated["suggestions"][0]["id"]

        response = await api_client.post(
            f"/api/suggestions/{suggestion_id}/status",
            json={"status": "completed"},
            headers=auth_headers,
        )

        # FastAPI validates the regex pattern — "completed" is not in the allowed set
        assert response.status_code == 422


class TestCampaignQueryRoute:
    @pytest.mark.asyncio
    async def test_query_campaigns_returns_envelope(self, api_client, auth_headers, persisted_user):
        owner_id = str(persisted_user.id)
        await _create_campaign_for_owner(owner_id, "Query Test A")
        await _create_campaign_for_owner(owner_id, "Query Test B")

        response = await api_client.post(
            "/api/campaigns/query",
            json={},
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total_count"] == 2
        assert len(data["campaigns"]) == 2
        assert "limit" in data
        assert "offset" in data

    @pytest.mark.asyncio
    async def test_query_campaigns_pagination(self, api_client, auth_headers, persisted_user):
        owner_id = str(persisted_user.id)
        for i in range(3):
            await _create_campaign_for_owner(owner_id, f"Page Test {i}")

        response = await api_client.post(
            "/api/campaigns/query",
            json={"limit": 2, "offset": 0},
            headers=auth_headers,
        )

        data = response.json()
        assert data["total_count"] == 3
        assert len(data["campaigns"]) == 2

        response2 = await api_client.post(
            "/api/campaigns/query",
            json={"limit": 2, "offset": 2},
            headers=auth_headers,
        )

        data2 = response2.json()
        assert data2["total_count"] == 3
        assert len(data2["campaigns"]) == 1

    @pytest.mark.asyncio
    async def test_query_campaigns_only_returns_own(
        self, api_client, auth_headers, persisted_user, persisted_another_user
    ):
        """Query must only return campaigns owned by the authenticated user."""
        await _create_campaign_for_owner(str(persisted_user.id), "My Campaign")
        await _create_campaign_for_owner(str(persisted_another_user.id), "Foreign Campaign")

        response = await api_client.post(
            "/api/campaigns/query",
            json={},
            headers=auth_headers,
        )

        data = response.json()
        assert data["total_count"] == 1
        assert data["campaigns"][0]["name"] == "My Campaign"


class TestResultQueryRoute:
    @pytest.mark.asyncio
    async def test_query_results_returns_envelope(self, api_client, auth_headers, persisted_user):
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign_for_owner(owner_id, "Result Query Test")
        await generate_suggestions(campaign_id)
        await submit_results(
            campaign_id,
            _to_result_inputs([{"parameter_values": {"x": 0.5}, "objective_values": {"y": 1.0}}]),
            owner_id,
        )

        response = await api_client.post(
            f"/api/results/{campaign_id}/query",
            json={},
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total_count"] == 1
        assert len(data["results"]) == 1
        assert "limit" in data
        assert "offset" in data

    @pytest.mark.asyncio
    async def test_query_results_enforces_ownership(
        self, api_client, auth_headers, persisted_user, persisted_another_user
    ):
        foreign_campaign_id = await _create_campaign_for_owner(
            str(persisted_another_user.id), "Foreign Result Query"
        )

        response = await api_client.post(
            f"/api/results/{foreign_campaign_id}/query",
            json={},
            headers=auth_headers,
        )

        assert response.status_code == 403


class TestSuggestionQueryRoute:
    @pytest.mark.asyncio
    async def test_query_suggestions_returns_envelope(
        self, api_client, auth_headers, persisted_user
    ):
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign_for_owner(owner_id, "Suggestion Query Test")
        await generate_suggestions(campaign_id)

        response = await api_client.post(
            f"/api/suggestions/{campaign_id}/query",
            json={},
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total_count"] > 0
        assert len(data["suggestions"]) > 0

    @pytest.mark.asyncio
    async def test_query_suggestions_with_status_filter(
        self, api_client, auth_headers, persisted_user
    ):
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign_for_owner(owner_id, "Sugg Filter Query")
        await generate_suggestions(campaign_id)

        # All suggestions are "pending" after generation
        response = await api_client.post(
            f"/api/suggestions/{campaign_id}/query",
            json={"status_filter": "pending"},
            headers=auth_headers,
        )

        data = response.json()
        assert data["success"] is True
        assert data["total_count"] > 0

        # No "accepted" suggestions yet
        response2 = await api_client.post(
            f"/api/suggestions/{campaign_id}/query",
            json={"status_filter": "accepted"},
            headers=auth_headers,
        )

        data2 = response2.json()
        assert data2["total_count"] == 0

    @pytest.mark.asyncio
    async def test_query_suggestions_enforces_ownership(
        self, api_client, auth_headers, persisted_user, persisted_another_user
    ):
        foreign_campaign_id = await _create_campaign_for_owner(
            str(persisted_another_user.id), "Foreign Sugg Query"
        )

        response = await api_client.post(
            f"/api/suggestions/{foreign_campaign_id}/query",
            json={},
            headers=auth_headers,
        )

        assert response.status_code == 403
