"""REST contract tests for the suggestion identity key.

``suggestion_id`` is the only identity key: the generate endpoint, the
list endpoint, and the query endpoint all emit it, and result
submission consumes it, so its value read from any of them can be
copied into a ``POST /api/results/{campaign_id}`` request without
renaming. The retired ``id`` alias must not reappear on any surface.
"""

from __future__ import annotations

import pytest

from bo_mcp_server.client import list_suggestions_operation
from bo_mcp_server.tools.create_campaign import create_campaign

UNLINKED_WARNING_MARKER = "without suggestion_id"


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


class TestSuggestionIdentityKeyRest:
    @pytest.mark.asyncio
    async def test_generate_emits_only_the_canonical_key(
        self, api_client, auth_headers, persisted_user
    ) -> None:
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign_for_owner(owner_id, "Identity Key REST Generate")

        response = await api_client.post(
            f"/api/suggestions/{campaign_id}/generate", headers=auth_headers
        )

        assert response.status_code == 201, response.text
        suggestions = response.json()["suggestions"]
        assert suggestions
        for suggestion in suggestions:
            assert suggestion["suggestion_id"]
            assert "id" not in suggestion

    @pytest.mark.asyncio
    async def test_list_and_query_agree_with_generate(
        self, api_client, auth_headers, persisted_user
    ) -> None:
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign_for_owner(owner_id, "Identity Key REST Query")
        generated = await api_client.post(
            f"/api/suggestions/{campaign_id}/generate", headers=auth_headers
        )
        generated_ids = {s["suggestion_id"] for s in generated.json()["suggestions"]}

        listed = await api_client.get(f"/api/suggestions/{campaign_id}", headers=auth_headers)
        assert listed.status_code == 200, listed.text
        listed_ids = {s["suggestion_id"] for s in listed.json()}

        queried = await api_client.post(
            f"/api/suggestions/{campaign_id}/query",
            json={"verbosity": "standard"},
            headers=auth_headers,
        )
        assert queried.status_code == 200, queried.text
        queried_ids = {s["suggestion_id"] for s in queried.json()["suggestions"]}

        assert generated_ids == listed_ids == queried_ids

    @pytest.mark.asyncio
    async def test_generate_payload_round_trips_into_result_submission(
        self, api_client, auth_headers, persisted_user
    ) -> None:
        """The key read from a generate response must submit verbatim.

        This pins the failure mode the canonical key exists to prevent:
        copying ``suggestion_id`` from a generated suggestion into a
        result submission without renaming must link the result and
        complete the originating suggestion.
        """
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign_for_owner(owner_id, "Identity Key REST Round Trip")
        generated = await api_client.post(
            f"/api/suggestions/{campaign_id}/generate", headers=auth_headers
        )
        suggestion = generated.json()["suggestions"][0]

        submitted = await api_client.post(
            f"/api/results/{campaign_id}",
            json={
                "results": [
                    {
                        "parameter_values": suggestion["parameter_values"],
                        "objective_values": {"y": 0.5},
                        "suggestion_id": suggestion["suggestion_id"],
                    }
                ],
                "source": "api",
            },
            headers=auth_headers,
        )

        assert submitted.status_code == 201, submitted.text
        body = submitted.json()
        assert not any(UNLINKED_WARNING_MARKER in w for w in body["warnings"])

        completed = await api_client.post(
            f"/api/suggestions/{campaign_id}/query",
            json={"status_filter": "completed"},
            headers=auth_headers,
        )
        completed_ids = {s["suggestion_id"] for s in completed.json()["suggestions"]}
        assert suggestion["suggestion_id"] in completed_ids

    @pytest.mark.asyncio
    async def test_unlinked_api_submission_with_open_suggestions_warns(
        self, api_client, auth_headers, persisted_user
    ) -> None:
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign_for_owner(owner_id, "Identity Key REST Unlinked")
        await api_client.post(f"/api/suggestions/{campaign_id}/generate", headers=auth_headers)

        submitted = await api_client.post(
            f"/api/results/{campaign_id}",
            json={
                "results": [
                    {
                        "parameter_values": {"x": 0.123},
                        "objective_values": {"y": 1.0},
                    }
                ],
                "source": "api",
            },
            headers=auth_headers,
        )

        assert submitted.status_code == 201, submitted.text
        assert any(UNLINKED_WARNING_MARKER in w for w in submitted.json()["warnings"])

    @pytest.mark.asyncio
    async def test_query_items_keep_exact_operation_key_sets(
        self, api_client, auth_headers, persisted_user
    ) -> None:
        """The REST query must not inject fields the operation omitted.

        Each verbosity has an exact wire shape; the typed response
        model must serialize precisely the keys the shared operation
        produced — no ``null`` backfill for declared-but-absent
        optional fields.
        """
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign_for_owner(owner_id, "Identity Key REST Key Sets")
        await api_client.post(f"/api/suggestions/{campaign_id}/generate", headers=auth_headers)

        for verbosity in ("minimal", "standard", "detailed"):
            operation_result = await list_suggestions_operation(campaign_id, verbosity=verbosity)
            expected_key_sets = [set(s.keys()) for s in operation_result["suggestions"]]

            response = await api_client.post(
                f"/api/suggestions/{campaign_id}/query",
                json={"verbosity": verbosity},
                headers=auth_headers,
            )
            assert response.status_code == 200, response.text
            actual_key_sets = [set(s.keys()) for s in response.json()["suggestions"]]

            assert actual_key_sets == expected_key_sets, verbosity

    @pytest.mark.asyncio
    async def test_openapi_requires_identity_fields(self, api_client) -> None:
        """OpenAPI must pin the identity contract, not merely allow it."""
        schema = (await api_client.get("/openapi.json")).json()
        components = schema["components"]["schemas"]

        summary = components["SuggestionSummary"]
        assert "suggestion_id" in summary["required"]
        assert "status" in summary["required"]

        response_model = components["SuggestionResponse"]
        assert "suggestion_id" in response_model["required"]
        assert "id" not in response_model["properties"]
