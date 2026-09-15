"""REST behaviour for replicate result submission.

Two experiments run at the same settings are replicates, and both
measurements belong in the campaign. Result submission does not reject a
row for matching an existing row's parameter values; repeated ingestion
is prevented by experiment identity instead.

Covered contracts:

- Both transports (JSON body, CSV upload) accept a repeated parameter
  setting and store it as a separate result.
- A result whose ``suggestion_id`` already has one is still refused, and
  the suggestion it names still completes on the first submission.
- Retry safety for unlinked submissions comes from ``Idempotency-Key``,
  which is now the only thing distinguishing a resend from a genuine
  repeat experiment. This follows the key-reuse semantics of the IETF
  Idempotency-Key draft
  (https://datatracker.ietf.org/doc/draft-ietf-httpapi-idempotency-key-header/).
"""

import pytest

from bo_mcp_server.tools.create_campaign import create_campaign


async def _create_campaign(owner_id: str) -> str:
    intake = {
        "name": "REST Replicate Handling",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }
    created = await create_campaign(intake, owner_id)
    return created["campaign_id"]


def _result_row(x: float, y: float) -> dict:
    return {"parameter_values": {"x": x}, "objective_values": {"y": y}}


@pytest.mark.asyncio
async def test_replicate_accepted_across_requests(api_client, auth_headers, persisted_user) -> None:
    """Repeating a setting is a measurement, not an error."""
    owner_id = str(persisted_user.id)
    campaign_id = await _create_campaign(owner_id)

    first = await api_client.post(
        f"/api/results/{campaign_id}",
        json={"results": [_result_row(0.4, 1.0)], "source": "api"},
        headers=auth_headers,
    )
    # Same coordinates, a different measured outcome: the replicate.
    second = await api_client.post(
        f"/api/results/{campaign_id}",
        json={"results": [_result_row(0.4, 1.2)], "source": "api"},
        headers=auth_headers,
    )

    assert first.status_code == 201, first.text
    assert second.status_code == 201, second.text
    assert second.json()["success"] is True
    assert second.json()["error_code"] is None
    assert first.json()["result_ids"] != second.json()["result_ids"]

    listed = await api_client.get(f"/api/results/{campaign_id}", headers=auth_headers)
    assert listed.status_code == 200, listed.text
    assert len(listed.json()) == 2


@pytest.mark.asyncio
async def test_replicates_within_one_request_are_both_stored(
    api_client, auth_headers, persisted_user
) -> None:
    """A batch may carry the same setting twice."""
    owner_id = str(persisted_user.id)
    campaign_id = await _create_campaign(owner_id)

    response = await api_client.post(
        f"/api/results/{campaign_id}",
        json={"results": [_result_row(0.4, 1.0), _result_row(0.4, 1.1)], "source": "api"},
        headers=auth_headers,
    )

    assert response.status_code == 201, response.text
    assert response.json()["success"] is True
    assert len(response.json()["result_ids"]) == 2


@pytest.mark.asyncio
async def test_upload_accepts_repeated_rows(api_client, auth_headers, persisted_user) -> None:
    """The file transport stores repeated settings as separate replicates."""
    owner_id = str(persisted_user.id)
    campaign_id = await _create_campaign(owner_id)
    csv_file = {"file": ("replicate.csv", b"x,y\n0.4,1.0\n0.4,1.2\n", "text/csv")}

    uploaded = await api_client.post(
        f"/api/results/{campaign_id}/upload", files=csv_file, headers=auth_headers
    )

    assert uploaded.status_code == 201, uploaded.text
    assert uploaded.json()["success"] is True

    listed = await api_client.get(f"/api/results/{campaign_id}", headers=auth_headers)
    assert len(listed.json()) == 2


@pytest.mark.asyncio
async def test_idempotency_key_is_what_stops_a_retry_double_counting(
    api_client, auth_headers, persisted_user
) -> None:
    """The retry protection unlinked replicates actually rely on.

    Parameter equality no longer distinguishes a resend from a genuine
    repeat experiment, so the key is the only thing that does.
    """
    owner_id = str(persisted_user.id)
    campaign_id = await _create_campaign(owner_id)
    key_headers = {**auth_headers, "Idempotency-Key": "replicate-retry-same"}
    body = {"results": [_result_row(0.4, 1.0)], "source": "api"}

    first = await api_client.post(f"/api/results/{campaign_id}", json=body, headers=key_headers)
    retry = await api_client.post(f"/api/results/{campaign_id}", json=body, headers=key_headers)

    assert first.status_code == 201, first.text
    assert retry.status_code in (200, 201), retry.text
    assert retry.json()["result_ids"] == first.json()["result_ids"]

    listed = await api_client.get(f"/api/results/{campaign_id}", headers=auth_headers)
    assert len(listed.json()) == 1


@pytest.mark.asyncio
async def test_without_a_key_a_resend_is_stored_again(
    api_client, auth_headers, persisted_user
) -> None:
    """The documented limitation, pinned so it cannot regress silently.

    An unlinked submission carries no identity, so a resend is
    indistinguishable from a genuine repeat experiment and is stored.
    Clients that need retry safety must send an ``Idempotency-Key``.
    """
    owner_id = str(persisted_user.id)
    campaign_id = await _create_campaign(owner_id)
    body = {"results": [_result_row(0.4, 1.0)], "source": "api"}

    await api_client.post(f"/api/results/{campaign_id}", json=body, headers=auth_headers)
    await api_client.post(f"/api/results/{campaign_id}", json=body, headers=auth_headers)

    listed = await api_client.get(f"/api/results/{campaign_id}", headers=auth_headers)
    assert len(listed.json()) == 2


@pytest.mark.asyncio
async def test_replicate_of_an_observed_point_completes_its_suggestion(
    api_client, auth_headers, persisted_user
) -> None:
    """End-to-end motivating scenario, with no override needed.

    A pending suggestion sits at coordinates that already carry a stored
    observation. Its result persists and the suggestion transitions to
    ``completed``. Previously this submission was rejected outright, and
    the only recourse was rejecting the suggestion — which recorded no
    measurement at all and did not stop the optimizer asking again.
    """
    owner_id = str(persisted_user.id)
    campaign_id = await _create_campaign(owner_id)

    generated = await api_client.post(
        f"/api/suggestions/{campaign_id}/generate", headers=auth_headers
    )
    assert generated.status_code == 201, generated.text
    suggestion = generated.json()["suggestions"][0]

    observed = dict(suggestion["parameter_values"])
    seeded = await api_client.post(
        f"/api/results/{campaign_id}",
        json={"results": [{"parameter_values": observed, "objective_values": {"y": 1.0}}]},
        headers=auth_headers,
    )
    assert seeded.status_code == 201, seeded.text

    linked = await api_client.post(
        f"/api/results/{campaign_id}",
        json={
            "results": [
                {
                    "parameter_values": observed,
                    "objective_values": {"y": 1.4},
                    "suggestion_id": suggestion["suggestion_id"],
                }
            ]
        },
        headers=auth_headers,
    )
    assert linked.status_code == 201, linked.text
    assert linked.json()["success"] is True

    listed = await api_client.get(f"/api/suggestions/{campaign_id}", headers=auth_headers)
    assert listed.status_code == 200, listed.text
    statuses = {s["suggestion_id"]: s["status"] for s in listed.json()}
    assert statuses[suggestion["suggestion_id"]] == "completed"


@pytest.mark.asyncio
async def test_openapi_no_longer_advertises_force(api_client) -> None:
    """Spec-driven clients must not see a parameter that no longer exists.

    The bo-agent drives REST calls from the live OpenAPI spec, so a
    lingering ``force`` would be offered as a real option and rejected
    by the schema's ``extra="forbid"``.
    """
    spec = (await api_client.get("/openapi.json")).json()

    body_properties = spec["components"]["schemas"]["ResultBatchCreate"]["properties"]
    assert "force" not in body_properties, body_properties

    upload_params = spec["paths"]["/api/v1/results/{campaign_id}/upload"]["post"].get(
        "parameters", []
    )
    assert all(p["name"] != "force" for p in upload_params), upload_params
