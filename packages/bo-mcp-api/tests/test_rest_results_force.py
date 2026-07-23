"""REST parity for ``force`` on result submission.

The MCP ``bo_submit_results`` tool exposes ``force=True`` to bypass the
exact-duplicate-coordinate check -- the documented path for submitting
an intentional replicate. The REST ``ResultBatchCreate`` schema
previously dropped the field, so a REST client could not submit a
replicate the optimizer had asked for; rejecting the suggestion instead
only retires that suggestion record without excluding the coordinates
from future generation, causing a repeated generate-and-reject loop.

Covered contracts:

- JSON submission rejects an exact duplicate by default (with the
  machine-readable ``DUPLICATE_RESULT`` code and duplicate diagnostics)
  and accepts it once ``force=True`` is set, matching the MCP tool.
- The recovery path recommended by the rejection ("Use force=True")
  interacts with idempotency: the rejection is cached under the
  submitted key and ``force`` is part of the request hash, so the
  forced retry needs a fresh ``Idempotency-Key``. This mirrors the
  key-reuse-with-different-payload semantics of the IETF
  Idempotency-Key draft
  (https://datatracker.ietf.org/doc/draft-ietf-httpapi-idempotency-key-header/).
- The CSV upload transport accepts the same override via a ``force``
  query parameter (FastAPI query-parameter pattern:
  https://fastapi.tiangolo.com/tutorial/query-params/).
- The OpenAPI schema documents ``force`` on both transports so
  spec-driven clients (e.g. the bo-agent OpenAPI inspector) can
  discover it.
"""

import pytest

from bo_mcp_server.client import ErrorCode
from bo_mcp_server.tools.create_campaign import create_campaign


async def _create_campaign(owner_id: str) -> str:
    intake = {
        "name": "REST Force Parity",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }
    created = await create_campaign(intake, owner_id)
    return created["campaign_id"]


def _result_row(x: float, y: float) -> dict:
    return {"parameter_values": {"x": x}, "objective_values": {"y": y}}


@pytest.mark.asyncio
async def test_exact_duplicate_rejected_without_force(
    api_client, auth_headers, persisted_user
) -> None:
    owner_id = str(persisted_user.id)
    campaign_id = await _create_campaign(owner_id)
    body = {"results": [_result_row(0.4, 1.0)], "source": "api"}

    first = await api_client.post(f"/api/results/{campaign_id}", json=body, headers=auth_headers)
    second = await api_client.post(f"/api/results/{campaign_id}", json=body, headers=auth_headers)

    assert first.status_code == 201, first.text
    assert second.status_code == 200, second.text
    envelope = second.json()
    assert envelope["success"] is False
    # Pin the rejection to the duplicate check specifically: any other
    # operation-level failure (campaign state, validation) must not
    # satisfy this test.
    assert envelope["error_code"] == ErrorCode.DUPLICATE_RESULT.value, envelope
    assert any("duplicate" in error.lower() for error in envelope["errors"]), envelope
    assert any(dup.get("is_exact") for dup in envelope["duplicates_detected"]), envelope


@pytest.mark.asyncio
async def test_exact_duplicate_accepted_with_force(
    api_client, auth_headers, persisted_user
) -> None:
    owner_id = str(persisted_user.id)
    campaign_id = await _create_campaign(owner_id)
    body = {"results": [_result_row(0.4, 1.0)], "source": "api"}
    forced_body = {"results": [_result_row(0.4, 1.0)], "source": "api", "force": True}

    first = await api_client.post(f"/api/results/{campaign_id}", json=body, headers=auth_headers)
    second = await api_client.post(
        f"/api/results/{campaign_id}", json=forced_body, headers=auth_headers
    )

    assert first.status_code == 201, first.text
    assert second.status_code == 201, second.text
    assert second.json()["success"] is True
    assert second.json()["error_code"] is None
    assert first.json()["result_ids"] != second.json()["result_ids"]


@pytest.mark.asyncio
async def test_forced_retry_of_rejected_key_conflicts_until_key_is_fresh(
    api_client, auth_headers, persisted_user
) -> None:
    """The documented recovery path needs a fresh Idempotency-Key.

    A duplicate rejection is terminal (non-retryable), so the
    idempotency cache stores it under the submitted key; ``force``
    participates in the canonical request hash. Following the
    rejection's "Use force=True" hint under the *same* key is
    therefore a key-reuse-with-different-payload conflict (409); the
    forced submission succeeds once the client mints a fresh key.
    """
    owner_id = str(persisted_user.id)
    campaign_id = await _create_campaign(owner_id)
    body = {"results": [_result_row(0.4, 1.0)], "source": "api"}
    forced_body = {"results": [_result_row(0.4, 1.0)], "source": "api", "force": True}
    rejected_key_headers = {**auth_headers, "Idempotency-Key": "force-retry-duplicate"}
    fresh_key_headers = {**auth_headers, "Idempotency-Key": "force-retry-duplicate-forced"}

    seeded = await api_client.post(f"/api/results/{campaign_id}", json=body, headers=auth_headers)
    assert seeded.status_code == 201, seeded.text

    rejected = await api_client.post(
        f"/api/results/{campaign_id}", json=body, headers=rejected_key_headers
    )
    assert rejected.status_code == 200, rejected.text
    assert rejected.json()["error_code"] == ErrorCode.DUPLICATE_RESULT.value

    conflicted = await api_client.post(
        f"/api/results/{campaign_id}", json=forced_body, headers=rejected_key_headers
    )
    assert conflicted.status_code == 409, conflicted.text
    assert conflicted.json()["detail"]["details"]["idempotency_conflict"] is True

    accepted = await api_client.post(
        f"/api/results/{campaign_id}", json=forced_body, headers=fresh_key_headers
    )
    assert accepted.status_code == 201, accepted.text
    assert accepted.json()["success"] is True


@pytest.mark.asyncio
async def test_upload_duplicate_rejected_then_accepted_with_force(
    api_client, auth_headers, persisted_user
) -> None:
    """The file-upload transport honours the same ``force`` override."""
    owner_id = str(persisted_user.id)
    campaign_id = await _create_campaign(owner_id)
    csv_file = {"file": ("replicate.csv", b"x,y\n0.4,1.0\n", "text/csv")}

    first = await api_client.post(
        f"/api/results/{campaign_id}/upload", files=csv_file, headers=auth_headers
    )
    assert first.status_code == 201, first.text

    rejected = await api_client.post(
        f"/api/results/{campaign_id}/upload", files=csv_file, headers=auth_headers
    )
    assert rejected.status_code == 200, rejected.text
    assert rejected.json()["success"] is False
    assert rejected.json()["error_code"] == ErrorCode.DUPLICATE_RESULT.value

    forced = await api_client.post(
        f"/api/results/{campaign_id}/upload",
        params={"force": "true"},
        files=csv_file,
        headers=auth_headers,
    )
    assert forced.status_code == 201, forced.text
    assert forced.json()["success"] is True
    assert forced.json()["result_ids"] != first.json()["result_ids"]


@pytest.mark.asyncio
async def test_forced_replicate_completes_pending_suggestion(
    api_client, auth_headers, persisted_user
) -> None:
    """End-to-end motivating scenario: replicate an already-observed point.

    A pending suggestion sits at coordinates that already carry a
    stored observation. Submitting its result is rejected as an exact
    duplicate without ``force``; with ``force=True`` the replicate
    persists AND the referenced suggestion transitions to
    ``completed`` — the outcome that rejecting the suggestion could
    never produce, since rejection neither records the measurement nor
    excludes the coordinates from future generation.
    """
    owner_id = str(persisted_user.id)
    campaign_id = await _create_campaign(owner_id)

    generated = await api_client.post(
        f"/api/suggestions/{campaign_id}/generate", headers=auth_headers
    )
    assert generated.status_code == 201, generated.text
    suggestion = generated.json()["suggestions"][0]

    observed = dict(suggestion["parameter_values"])
    free_row = {"parameter_values": observed, "objective_values": {"y": 1.0}}
    linked_row = {**free_row, "suggestion_id": suggestion["id"]}

    seeded = await api_client.post(
        f"/api/results/{campaign_id}", json={"results": [free_row]}, headers=auth_headers
    )
    assert seeded.status_code == 201, seeded.text

    rejected = await api_client.post(
        f"/api/results/{campaign_id}", json={"results": [linked_row]}, headers=auth_headers
    )
    assert rejected.status_code == 200, rejected.text
    assert rejected.json()["error_code"] == ErrorCode.DUPLICATE_RESULT.value

    forced = await api_client.post(
        f"/api/results/{campaign_id}",
        json={"results": [linked_row], "force": True},
        headers=auth_headers,
    )
    assert forced.status_code == 201, forced.text
    assert forced.json()["success"] is True

    listed = await api_client.get(f"/api/suggestions/{campaign_id}", headers=auth_headers)
    assert listed.status_code == 200, listed.text
    statuses = {s["id"]: s["status"] for s in listed.json()}
    assert statuses[suggestion["id"]] == "completed"


@pytest.mark.asyncio
async def test_openapi_documents_force_on_both_transports(api_client) -> None:
    """Spec-driven clients must be able to discover ``force``.

    The bo-agent drives REST calls from the live OpenAPI spec, so an
    undocumented boolean is effectively invisible: the JSON body field
    and the upload query parameter both need a non-empty description.
    """
    spec = (await api_client.get("/openapi.json")).json()

    body_field = spec["components"]["schemas"]["ResultBatchCreate"]["properties"]["force"]
    assert "duplicate" in body_field.get("description", "").lower(), body_field

    upload_params = spec["paths"]["/api/v1/results/{campaign_id}/upload"]["post"]["parameters"]
    force_param = next(p for p in upload_params if p["name"] == "force")
    assert "duplicate" in force_param.get("description", "").lower(), force_param
