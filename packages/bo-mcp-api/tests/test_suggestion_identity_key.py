"""REST contract tests for the suggestion identity key.

``suggestion_id`` is the canonical identity key: the generate endpoint,
the list endpoint, and the query endpoint all emit it, and result
submission consumes it, so its value read from any of them can be
copied into a ``POST /api/results/{campaign_id}`` request without
renaming. ``id`` carries the same value on the typed endpoints and is
deprecated.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from bo_mcp_server.client import (
    canonical_generate_suggestions_payload,
    list_suggestions_operation,
)
from bo_mcp_server.idempotency import apply_idempotency
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
    async def test_generate_emits_canonical_key_and_deprecated_alias(
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
            assert suggestion["suggestion_id"] == suggestion["id"]

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
        assert response_model["properties"]["id"].get("deprecated") is True

    @pytest.mark.asyncio
    async def test_legacy_idempotency_replay_gains_suggestion_id(
        self, api_client, auth_headers, persisted_user
    ) -> None:
        """Pre-migration cache entries must replay with both keys.

        The idempotency cache is DB-backed and outlives a deployment;
        a retry within the cache lifetime replays a response whose
        suggestions only carry ``id``. The route must backfill
        ``suggestion_id`` before responding.
        """
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign_for_owner(owner_id, "Identity Key REST Replay")
        idempotency_key = str(uuid4())
        legacy_response = {
            "success": True,
            "suggestions": [
                {
                    "id": str(uuid4()),
                    "parameter_values": {"x": 0.5},
                    "provenance": {
                        "iteration": 1,
                        "batch_index": 0,
                        "generation_method": "sobol",
                    },
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            ],
            "iteration": 1,
            "errors": [],
            "warnings": [],
            "method_selection": {},
        }

        async def seed_legacy(_session):
            return legacy_response

        await apply_idempotency(
            tool_name="bo_generate_suggestions",
            idempotency_key=idempotency_key,
            request_payload=canonical_generate_suggestions_payload(
                campaign_id=campaign_id,
                batch_size=None,
            ),
            executor=seed_legacy,
        )

        replayed = await api_client.post(
            f"/api/suggestions/{campaign_id}/generate",
            headers={**auth_headers, "Idempotency-Key": idempotency_key},
        )

        assert replayed.status_code == 201, replayed.text
        body = replayed.json()
        assert body["idempotency_replay"] is True
        assert body["suggestions"]
        for suggestion in body["suggestions"]:
            assert suggestion["suggestion_id"] == suggestion["id"]
