"""OpenAPI documents runtime error and operation-failure contracts."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_authenticated_resource_routes_document_http_errors(api_client) -> None:
    response = await api_client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    responses = schema["paths"]["/api/v1/campaigns/{campaign_id}"]["get"]["responses"]

    for status_code in ("400", "401", "403", "404", "500"):
        assert status_code in responses

    assert (
        responses["401"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/HttpErrorResponse"
    )
    assert (
        responses["500"]["content"]["application/json"]["schema"]["$ref"]
        == "#/components/schemas/StructuredErrorEnvelope"
    )


@pytest.mark.asyncio
async def test_create_campaign_documents_operation_failure_and_idempotency(api_client) -> None:
    response = await api_client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    responses = schema["paths"]["/api/v1/campaigns"]["post"]["responses"]

    assert "201" in responses
    assert responses["200"]["description"].startswith("Operation-level campaign creation")
    assert responses["200"]["content"]["application/json"]["example"]["success"] is False
    assert responses["200"]["content"]["application/json"]["example"]["errors"]
    assert "409" in responses
    assert "Idempotency conflict" in responses["409"]["description"]


@pytest.mark.asyncio
async def test_submit_results_and_generate_suggestions_document_success_false(api_client) -> None:
    response = await api_client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()

    submit = schema["paths"]["/api/v1/results/{campaign_id}"]["post"]["responses"]
    assert submit["200"]["content"]["application/json"]["example"]["success"] is False
    assert "field_errors" in submit["200"]["content"]["application/json"]["example"]
    assert "409" in submit

    generate = schema["paths"]["/api/v1/suggestions/{campaign_id}/generate"]["post"]["responses"]
    assert generate["200"]["content"]["application/json"]["example"]["success"] is False
    assert generate["200"]["content"]["application/json"]["example"]["suggestions"] == []
