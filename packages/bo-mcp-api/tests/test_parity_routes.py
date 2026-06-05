"""Tests for REST endpoints that mirror MCP-only functionality."""

import pytest
from bo_mcp_server.domain import ResultSubmissionInput
from bo_mcp_server.operations.export_campaign import export_campaign_operation
from bo_mcp_server.operations.list_capabilities import list_capabilities_operation
from bo_mcp_server.operations.update_suggestion_status import (
    update_suggestion_status_operation,
)
from bo_mcp_server.operations.validate_intake import validate_intake_operation
from bo_mcp_server.tools.create_campaign import create_campaign
from bo_mcp_server.tools.generate_suggestions import generate_suggestions
from bo_mcp_server.tools.submit_results import submit_results

pytestmark = pytest.mark.usefixtures("persisted_user")


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

    @pytest.mark.asyncio
    async def test_transfer_candidates_route_honors_parameter_aliases(
        self,
        api_client,
        auth_headers,
        persisted_user,
    ):
        """The REST transfer endpoint passes ``parameter_aliases`` through to the operation.

        Without the alias map, ``temperature`` vs ``temp_c`` look like
        disjoint parameter sets and the Jaccard intersection collapses
        to zero. With the alias, the canonical name unifies the two
        and the source surfaces as a candidate.
        """
        owner_id = str(persisted_user.id)
        source = await create_campaign(
            {
                "name": "Aliased REST Source",
                "parameters": [
                    {"name": "temperature", "type": "continuous", "bounds": [20.0, 100.0]},
                ],
                "objectives": [{"name": "yield", "direction": "maximize"}],
            },
            owner_id,
        )
        await generate_suggestions(source["campaign_id"])
        await submit_results(
            source["campaign_id"],
            _to_result_inputs(
                [
                    {"parameter_values": {"temperature": 25.0}, "objective_values": {"yield": 0.5}},
                    {"parameter_values": {"temperature": 50.0}, "objective_values": {"yield": 0.7}},
                    {"parameter_values": {"temperature": 75.0}, "objective_values": {"yield": 0.6}},
                ]
            ),
            owner_id,
        )
        target = await create_campaign(
            {
                "name": "Aliased REST Target",
                "parameters": [
                    {"name": "temp_c", "type": "continuous", "bounds": [30.0, 90.0]},
                ],
                "objectives": [{"name": "yield", "direction": "maximize"}],
            },
            owner_id,
        )

        response = await api_client.post(
            f"/api/campaigns/{target['campaign_id']}/transfer-candidates",
            json={
                "similarity_threshold": 0.0,
                "max_candidates": 5,
                "verbosity": "detailed",
                "parameter_aliases": {"temperature": ["temp_c"]},
            },
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        candidates = data["candidates"]
        assert candidates, "aliases must reveal the renamed source campaign"
        top = candidates[0]
        assert top["name"] == "Aliased REST Source"
        # ``parameter_similarity`` is part of ``component_scores`` at detailed verbosity.
        if "component_scores" in top:
            assert top["component_scores"]["parameter_similarity"] > 0.0


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
    async def test_validate_valid_intake(self, api_client, auth_headers):
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
    async def test_validate_invalid_intake(self, api_client, auth_headers):
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
    async def test_list_capabilities(self, api_client, auth_headers):
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
        self, api_client, auth_headers, persisted_another_user
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
        self, api_client, auth_headers, persisted_another_user
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

    @pytest.mark.asyncio
    async def test_query_campaigns_cursor_offset_mutual_exclusion_surfaces_envelope(
        self, api_client, auth_headers, persisted_user
    ):
        """REST surfaces the structured ``VALIDATION_FAILED`` envelope.

        Pre-fix the route constructed
        ``CampaignQueryResponse`` from the operation's response dict.
        That schema has no ``error`` field, so Pydantic silently dropped
        the structured envelope and clients saw an inconsistent body for
        a 200 OK response. The route now promotes the
        ``success=False`` envelope to an ``HTTPException`` whose
        ``detail`` carries the original ``error`` payload (code,
        message, recovery_action, retryable, details), routed through
        ``http_status_for_error`` for the correct HTTP status code.
        """
        # Seed two campaigns and walk one page via cursor so the request
        # combines a real cursor token with a non-zero offset.
        owner_id = str(persisted_user.id)
        await _create_campaign_for_owner(owner_id, "Campaign A")
        await _create_campaign_for_owner(owner_id, "Campaign B")

        first = await api_client.post(
            "/api/campaigns/query",
            json={"limit": 1, "offset": 0},
            headers=auth_headers,
        )
        cursor = first.json().get("next_cursor")
        assert cursor, first.json()

        # Both cursor and a non-zero offset: structured rejection.
        response = await api_client.post(
            "/api/campaigns/query",
            json={"cursor": cursor, "offset": 1, "limit": 1},
            headers=auth_headers,
        )

        assert response.status_code == 400, response.text
        detail = response.json()["detail"]
        # Structured envelope is fully preserved in the HTTPException
        # detail; the code lets clients route on the failure without
        # parsing free-form text.
        assert detail["code"] == "E005"
        assert "mutually exclusive" in detail["message"]
        assert detail["details"]["cursor"] == cursor
        assert detail["details"]["offset"] == 1


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
        self, api_client, auth_headers, persisted_another_user
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
        self, api_client, auth_headers, persisted_another_user
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


# --- Backward-compat: existing GET endpoints must keep their response shapes ---


class TestGetEndpointBackwardCompat:
    """Ensure existing simple GET endpoints return unchanged response shapes.

    The frontend relies on these specific shapes. Adding /query endpoints must
    not alter the simple GET response bodies.
    """

    @pytest.mark.asyncio
    async def test_get_campaigns_returns_expected_shape(
        self, api_client, auth_headers, persisted_user
    ):
        owner_id = str(persisted_user.id)
        await _create_campaign_for_owner(owner_id, "Compat Campaign")

        response = await api_client.get("/api/campaigns", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        # Must have 'campaigns' array and 'total' int — NOT envelope with success/errors
        assert "campaigns" in data
        assert "total" in data
        assert isinstance(data["campaigns"], list)
        assert isinstance(data["total"], int)
        assert len(data["campaigns"]) == 1
        # Each campaign must have the expected fields
        campaign = data["campaigns"][0]
        for field in ("id", "spec_id", "name", "status", "iteration", "created_at"):
            assert field in campaign, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_get_results_returns_bare_list(self, api_client, auth_headers, persisted_user):
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign_for_owner(owner_id, "Compat Results")
        await generate_suggestions(campaign_id)
        await submit_results(
            campaign_id,
            _to_result_inputs([{"parameter_values": {"x": 0.5}, "objective_values": {"y": 1.0}}]),
            owner_id,
        )

        response = await api_client.get(f"/api/results/{campaign_id}", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        # Must be a bare array, NOT an envelope
        assert isinstance(data, list)
        assert len(data) == 1
        result = data[0]
        for field in ("id", "campaign_id", "parameter_values", "objective_values", "source"):
            assert field in result, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_get_suggestions_returns_bare_list(
        self, api_client, auth_headers, persisted_user
    ):
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign_for_owner(owner_id, "Compat Suggestions")
        await generate_suggestions(campaign_id)

        response = await api_client.get(f"/api/suggestions/{campaign_id}", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        # Must be a bare array, NOT an envelope
        assert isinstance(data, list)
        assert len(data) > 0
        suggestion = data[0]
        for field in ("id", "campaign_id", "parameter_values", "status", "provenance"):
            assert field in suggestion, f"Missing field: {field}"


# --- MCP-vs-HTTP parity: same scenario through both transports, compare fields ---


class TestMcpHttpParity:
    """Run the same scenario through MCP operations and HTTP, compare meaningful fields."""

    @pytest.mark.asyncio
    async def test_validate_intake_parity(self, api_client, auth_headers):
        intake = {
            "name": "Parity Validate",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }

        # MCP path (operation directly)
        mcp_result = validate_intake_operation(intake)

        # HTTP path
        http_response = await api_client.post(
            "/api/campaigns/validate",
            json={"intake": intake},
            headers=auth_headers,
        )
        http_result = http_response.json()

        assert mcp_result["valid"] == http_result["valid"]
        assert mcp_result["errors"] == http_result["errors"]
        assert mcp_result["warnings"] == http_result["warnings"]

    @pytest.mark.asyncio
    async def test_list_capabilities_parity(self, api_client, auth_headers):
        # MCP path (operation directly)
        mcp_result = list_capabilities_operation()

        # HTTP path
        http_response = await api_client.get("/api/capabilities", headers=auth_headers)
        http_result = http_response.json()

        assert mcp_result["backend"] == http_result["backend"]
        assert mcp_result["supported_features"] == http_result["supported_features"]
        assert mcp_result["server_version"] == http_result["server_version"]

    @pytest.mark.asyncio
    async def test_export_campaign_parity(self, api_client, auth_headers, persisted_user):
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign_for_owner(owner_id, "Parity Export")
        await generate_suggestions(campaign_id)
        await submit_results(
            campaign_id,
            _to_result_inputs([{"parameter_values": {"x": 0.5}, "objective_values": {"y": 1.0}}]),
            owner_id,
        )

        # MCP path (operation directly)
        mcp_result = await export_campaign_operation(campaign_id)

        # HTTP path — returns CSV as file download
        http_response = await api_client.get(
            f"/api/campaigns/{campaign_id}/export", headers=auth_headers
        )

        assert mcp_result["success"] is True
        assert http_response.status_code == 200
        # The CSV content should be identical
        assert mcp_result["content"] == http_response.text

    @pytest.mark.asyncio
    async def test_update_suggestion_status_parity(self, api_client, auth_headers, persisted_user):
        owner_id = str(persisted_user.id)

        # Create two independent campaigns for independent transitions
        campaign_a = await _create_campaign_for_owner(owner_id, "Parity Status A")
        campaign_b = await _create_campaign_for_owner(owner_id, "Parity Status B")
        gen_a = await generate_suggestions(campaign_a)
        gen_b = await generate_suggestions(campaign_b)
        suggestion_a = gen_a["suggestions"][0]["id"]
        suggestion_b = gen_b["suggestions"][0]["id"]

        # MCP path (operation directly)
        mcp_result = await update_suggestion_status_operation(suggestion_a, "accepted")

        # HTTP path
        http_response = await api_client.post(
            f"/api/suggestions/{suggestion_b}/status",
            json={"status": "accepted"},
            headers=auth_headers,
        )
        http_result = http_response.json()

        assert mcp_result["success"] == http_result["success"]
        assert mcp_result["status"] == http_result["status"]
        assert mcp_result["previous_status"] == http_result["previous_status"]

    @pytest.mark.asyncio
    async def test_diagnostics_with_params_parity(self, api_client, auth_headers, persisted_user):
        """Diagnostics HTTP params should mirror MCP semantics for verbosity and sections."""
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign_for_owner(owner_id, "Parity Diag")
        await generate_suggestions(campaign_id)
        await submit_results(
            campaign_id,
            _to_result_inputs([{"parameter_values": {"x": 0.5}, "objective_values": {"y": 1.0}}]),
            owner_id,
        )

        from bo_mcp_server.operations.get_diagnostics import get_diagnostics_operation

        # MCP path — request only "health" section with minimal verbosity
        mcp_result = await get_diagnostics_operation(
            campaign_id=campaign_id,
            verbosity="minimal",
            sections=["health"],
        )

        # HTTP path — same params via query string
        http_response = await api_client.get(
            f"/api/diagnostics/{campaign_id}?verbosity=minimal&sections=health",
            headers=auth_headers,
        )
        http_result = http_response.json()

        assert mcp_result["success"] == http_result["success"]
        # Both should have health-related fields
        assert mcp_result.get("health_status") == http_result.get("health_status")

    @pytest.mark.parametrize("verbosity", ["minimal", "standard", "detailed"])
    @pytest.mark.asyncio
    async def test_diagnostics_body_matches_mcp_exactly(
        self, api_client, auth_headers, persisted_user, verbosity
    ):
        """REST diagnostics body is byte-equal to the MCP operation output at every verbosity.

        The route returns a typed ``DiagnosticsResponse`` (an
        ``extra="allow"`` envelope inheriting ``ResponseEnvelope``) and
        serializes it with ``response_model_exclude_unset=True``. This
        pins two things at once:

        * the deep, verbosity-dependent metric blocks and the
          ``_metadata`` envelope survive FastAPI serialization, and
        * the model does **not** inject declared defaults for keys the
          MCP projection omits — at ``minimal`` verbosity the projection
          drops ``campaign_status`` / ``n_pending_suggestions`` /
          ``warnings``, so the REST body must drop them too.

        Without ``exclude_unset`` the ``minimal`` case re-adds those three
        keys and this test fails — exactly the REST/MCP parity gap the
        review flagged.
        """
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign_for_owner(owner_id, f"Diag {verbosity}")
        await generate_suggestions(campaign_id)
        await submit_results(
            campaign_id,
            _to_result_inputs([{"parameter_values": {"x": 0.5}, "objective_values": {"y": 1.0}}]),
            owner_id,
        )

        from bo_mcp_server.operations.get_diagnostics import get_diagnostics_operation

        # Compute first so the full-sections payload is cached; the HTTP
        # route then serves the identical cached result (GP fitting is
        # otherwise not byte-stable run to run).
        mcp_result = await get_diagnostics_operation(campaign_id=campaign_id, verbosity=verbosity)

        http_response = await api_client.get(
            f"/api/diagnostics/{campaign_id}?verbosity={verbosity}",
            headers=auth_headers,
        )
        assert http_response.status_code == 200
        http_result = http_response.json()

        # Envelope contract: schema_version present (route inherits ResponseEnvelope).
        assert "schema_version" in http_result
        assert http_result["success"] is True
        assert "_metadata" in http_result
        # No dropped passthrough, no injected defaults — exact parity.
        assert http_result == mcp_result

    @pytest.mark.parametrize("verbosity", ["minimal", "standard", "detailed"])
    @pytest.mark.asyncio
    async def test_compare_body_matches_mcp_exactly(
        self, api_client, auth_headers, persisted_user, verbosity
    ):
        """REST compare body is byte-equal to the MCP operation at every verbosity.

        ``CompareCampaignsResponse`` now sets ``extra="allow"`` and the
        route serializes with ``response_model_exclude_unset=True``. This
        pins that the response neither drops ``_metadata`` nor injects the
        other tier's declared defaults — without the fix the ``minimal``
        case gains ``campaigns=[]`` / ``comparison=None`` and loses
        ``_metadata``.
        """
        owner_id = str(persisted_user.id)
        campaign_a = await _create_campaign_for_owner(owner_id, f"Cmp A {verbosity}")
        campaign_b = await _create_campaign_for_owner(owner_id, f"Cmp B {verbosity}")
        for campaign, y in ((campaign_a, 1.0), (campaign_b, 0.8)):
            await generate_suggestions(campaign)
            await submit_results(
                campaign,
                _to_result_inputs([{"parameter_values": {"x": 0.4}, "objective_values": {"y": y}}]),
                owner_id,
            )

        from bo_mcp_server.operations.compare_campaigns import compare_campaigns_operation

        mcp_result = await compare_campaigns_operation(
            campaign_ids=[campaign_a, campaign_b], verbosity=verbosity
        )
        http_response = await api_client.post(
            "/api/campaigns/compare",
            json={"campaign_ids": [campaign_a, campaign_b], "verbosity": verbosity},
            headers=auth_headers,
        )
        assert http_response.status_code == 200
        http_result = http_response.json()

        assert "schema_version" in http_result
        assert "_metadata" in http_result
        assert http_result == mcp_result

    @pytest.mark.parametrize("verbosity", ["minimal", "standard", "detailed"])
    @pytest.mark.asyncio
    async def test_transfer_body_matches_mcp_exactly(
        self, api_client, auth_headers, persisted_user, verbosity
    ):
        """REST transfer body is byte-equal to the MCP operation at every verbosity.

        ``TransferCandidatesResponse`` now sets ``extra="allow"`` and the
        route serializes with ``response_model_exclude_unset=True``. This
        pins that the minimal-tier keys (``n_candidates`` /
        ``top_candidate_id`` / ``top_similarity`` / ``recommendation``)
        and ``_metadata`` ride through instead of being dropped, and the
        standard-tier defaults are not injected.
        """
        owner_id = str(persisted_user.id)
        source = await create_campaign(
            {
                "name": f"Xfer Source {verbosity}",
                "parameters": [
                    {"name": "temperature", "type": "continuous", "bounds": [20.0, 100.0]}
                ],
                "objectives": [{"name": "yield", "direction": "maximize"}],
            },
            owner_id,
        )
        target = await create_campaign(
            {
                "name": f"Xfer Target {verbosity}",
                "parameters": [
                    {"name": "temperature", "type": "continuous", "bounds": [30.0, 90.0]}
                ],
                "objectives": [{"name": "yield", "direction": "maximize"}],
            },
            owner_id,
        )
        await generate_suggestions(source["campaign_id"])
        await submit_results(
            source["campaign_id"],
            _to_result_inputs(
                [
                    {"parameter_values": {"temperature": 50.0}, "objective_values": {"yield": 0.8}},
                    {"parameter_values": {"temperature": 70.0}, "objective_values": {"yield": 0.9}},
                ]
            ),
            owner_id,
        )

        from bo_mcp_server.operations.transfer_candidates import (
            discover_transfer_candidates_operation,
        )

        mcp_result = await discover_transfer_candidates_operation(
            campaign_id=target["campaign_id"],
            similarity_threshold=0.3,
            max_candidates=5,
            verbosity=verbosity,
        )
        http_response = await api_client.post(
            f"/api/campaigns/{target['campaign_id']}/transfer-candidates",
            json={"similarity_threshold": 0.3, "max_candidates": 5, "verbosity": verbosity},
            headers=auth_headers,
        )
        assert http_response.status_code == 200
        http_result = http_response.json()

        assert "schema_version" in http_result
        assert "_metadata" in http_result
        assert http_result == mcp_result

    @pytest.mark.parametrize("verbosity", ["minimal", "standard", "detailed"])
    @pytest.mark.asyncio
    async def test_batch_status_body_matches_mcp_exactly(
        self, api_client, auth_headers, persisted_user, verbosity
    ):
        """REST batch-status body is byte-equal to the MCP operation at every verbosity.

        ``batch_get_status_operation`` now carries ``with_response_metadata``
        so both transports emit ``_metadata`` + ``schema_version`` (the
        envelope contract ``ResponseEnvelope`` advertises for batch
        status), and ``BatchStatusResponse`` sets ``extra="allow"`` so the
        REST model forwards ``_metadata`` rather than dropping it. Before
        this fix the REST body carried a ``schema_version`` the MCP tool
        lacked, breaking exact parity.
        """
        owner_id = str(persisted_user.id)
        campaign_a = await _create_campaign_for_owner(owner_id, f"Batch A {verbosity}")
        campaign_b = await _create_campaign_for_owner(owner_id, f"Batch B {verbosity}")

        from bo_mcp_server.operations.batch_status import batch_get_status_operation

        mcp_result = await batch_get_status_operation(
            campaign_ids=[campaign_a, campaign_b], verbosity=verbosity
        )
        http_response = await api_client.post(
            "/api/campaigns/status/batch",
            json={"campaign_ids": [campaign_a, campaign_b], "verbosity": verbosity},
            headers=auth_headers,
        )
        assert http_response.status_code == 200
        http_result = http_response.json()

        assert "schema_version" in http_result
        assert "_metadata" in http_result
        assert http_result == mcp_result
