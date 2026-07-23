"""REST ↔ MCP parity for mutation options and outcome fields.

The operating manual documents `dry_run`, `force`, `atomic`,
`continue_on_error`, and the structured ``error`` object as
transport-neutral. The shared operation layer has always supported
them; these tests pin that the REST transport actually exposes them
with the same semantics the MCP tools have:

- ``dry_run`` on every state-mutating endpoint validates, returns a
  preview, and persists nothing;
- ``force`` overrides duplicate detection on result submission;
- ``atomic=false`` + ``continue_on_error=true`` yields a per-row
  ``partial_results`` mapping;
- operation-level rejections carry the structured ``error`` object
  (``code`` / ``retryable`` / ``recovery_action``) in the 2xx envelope.

Reference: the dry-run / force / batch semantics under test are the
operation-layer contracts documented in
``bo_mcp_server.operations.submit_results`` and mirrored by the MCP
tools; the idempotency-header interplay follows the IETF draft
``draft-ietf-httpapi-idempotency-key-header``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytestmark = pytest.mark.usefixtures("persisted_user")

_INTAKE = {
    "name": "Parity Test Campaign",
    "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
    "objectives": [{"name": "y", "direction": "minimize"}],
    "backend": "botorch",
}


def _result_row(x: float, y: float, suggestion_id: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"parameter_values": {"x": x}, "objective_values": {"y": y}}
    if suggestion_id is not None:
        row["suggestion_id"] = suggestion_id
    return row


async def _create_campaign(api_client, auth_headers) -> str:
    response = await api_client.post(
        "/api/v1/campaigns", json={"intake": _INTAKE}, headers=auth_headers
    )
    assert response.status_code == 201, response.text
    data = response.json()
    assert data["success"] is True, data
    return data["campaign_id"]


class TestDryRunParity:
    @pytest.mark.asyncio
    async def test_create_dry_run_validates_without_persisting(
        self, api_client, auth_headers
    ) -> None:
        response = await api_client.post(
            "/api/v1/campaigns",
            json={"intake": _INTAKE, "dry_run": True},
            headers=auth_headers,
        )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["success"] is True
        assert data["dry_run"] is True
        assert data["campaign_id"] is None

        listing = await api_client.get("/api/v1/campaigns", headers=auth_headers)
        assert listing.json()["campaigns"] == []

    @pytest.mark.asyncio
    async def test_generate_dry_run_previews_without_persisting(
        self, api_client, auth_headers
    ) -> None:
        campaign_id = await _create_campaign(api_client, auth_headers)

        response = await api_client.post(
            f"/api/v1/suggestions/{campaign_id}/generate",
            params={"dry_run": "true"},
            headers=auth_headers,
        )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["success"] is True
        assert data["dry_run"] is True
        assert data["preview"], "dry-run generation must report a preview"
        assert data["suggestions"] == []

        listing = await api_client.get(f"/api/v1/suggestions/{campaign_id}", headers=auth_headers)
        assert listing.json() == []

    @pytest.mark.asyncio
    async def test_submit_dry_run_previews_without_persisting(
        self, api_client, auth_headers
    ) -> None:
        campaign_id = await _create_campaign(api_client, auth_headers)

        response = await api_client.post(
            f"/api/v1/results/{campaign_id}",
            json={"results": [_result_row(0.5, 1.5)], "dry_run": True},
            headers=auth_headers,
        )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["success"] is True
        assert data["dry_run"] is True
        assert data["preview"]["rows_would_persist"] == 1
        assert data["result_ids"] == []

        listing = await api_client.get(f"/api/v1/results/{campaign_id}", headers=auth_headers)
        assert listing.json() == []

    @pytest.mark.asyncio
    async def test_lifecycle_dry_run_previews_without_committing(
        self, api_client, auth_headers
    ) -> None:
        campaign_id = await _create_campaign(api_client, auth_headers)
        generated = await api_client.post(
            f"/api/v1/suggestions/{campaign_id}/generate", headers=auth_headers
        )
        assert generated.status_code == 201, generated.text

        response = await api_client.post(
            f"/api/v1/campaigns/{campaign_id}/lifecycle",
            json={"action": "pause", "dry_run": True},
            headers=auth_headers,
        )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["success"] is True
        assert data["dry_run"] is True
        assert data["preview"]["from_status"] == "running"
        assert data["preview"]["to_status"] == "paused"

        campaign = await api_client.get(f"/api/v1/campaigns/{campaign_id}", headers=auth_headers)
        assert campaign.json()["status"] == "running", "dry run must not transition"

    @pytest.mark.asyncio
    async def test_suggestion_status_dry_run_previews_without_committing(
        self, api_client, auth_headers
    ) -> None:
        campaign_id = await _create_campaign(api_client, auth_headers)
        generated = await api_client.post(
            f"/api/v1/suggestions/{campaign_id}/generate", headers=auth_headers
        )
        suggestion_id = generated.json()["suggestions"][0]["id"]

        response = await api_client.post(
            f"/api/v1/suggestions/{suggestion_id}/status",
            json={"status": "rejected", "dry_run": True},
            headers=auth_headers,
        )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["success"] is True
        assert data["dry_run"] is True
        assert data["preview"]

        listing = await api_client.get(f"/api/v1/suggestions/{campaign_id}", headers=auth_headers)
        assert listing.json()[0]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_upload_dry_run_previews_without_persisting(
        self, api_client, auth_headers
    ) -> None:
        campaign_id = await _create_campaign(api_client, auth_headers)
        csv_content = b"x,y\n0.25,2.5\n"

        response = await api_client.post(
            f"/api/v1/results/{campaign_id}/upload",
            params={"dry_run": "true"},
            files={"file": ("rows.csv", csv_content, "text/csv")},
            headers=auth_headers,
        )

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["success"] is True
        assert data["dry_run"] is True
        assert data["result_ids"] == []

        listing = await api_client.get(f"/api/v1/results/{campaign_id}", headers=auth_headers)
        assert listing.json() == []


class TestForceAndBatchModes:
    @pytest.mark.asyncio
    async def test_duplicate_rejection_carries_structured_error(
        self, api_client, auth_headers
    ) -> None:
        """The 2xx rejection envelope exposes code/retryable/recovery_action."""
        campaign_id = await _create_campaign(api_client, auth_headers)
        row = _result_row(0.5, 1.5)
        first = await api_client.post(
            f"/api/v1/results/{campaign_id}", json={"results": [row]}, headers=auth_headers
        )
        assert first.json()["success"] is True

        duplicate = await api_client.post(
            f"/api/v1/results/{campaign_id}", json={"results": [row]}, headers=auth_headers
        )

        assert duplicate.status_code == 200, duplicate.text
        data = duplicate.json()
        assert data["success"] is False
        assert data["error"] is not None, "rejection must carry the structured error"
        assert data["error"]["code"] == "E004"
        assert data["error"]["retryable"] is False
        assert data["error"]["recovery_action"]

    @pytest.mark.asyncio
    async def test_force_overrides_duplicate_detection(self, api_client, auth_headers) -> None:
        campaign_id = await _create_campaign(api_client, auth_headers)
        row = _result_row(0.5, 1.5)
        first = await api_client.post(
            f"/api/v1/results/{campaign_id}", json={"results": [row]}, headers=auth_headers
        )
        assert first.json()["success"] is True

        forced = await api_client.post(
            f"/api/v1/results/{campaign_id}",
            json={"results": [row], "force": True},
            headers=auth_headers,
        )

        assert forced.status_code == 201, forced.text
        assert forced.json()["success"] is True

        listing = await api_client.get(f"/api/v1/results/{campaign_id}", headers=auth_headers)
        assert len(listing.json()) == 2

    @pytest.mark.asyncio
    async def test_continue_on_error_reports_partial_results(
        self, api_client, auth_headers
    ) -> None:
        """Non-atomic submission persists valid rows and maps per-row outcomes."""
        campaign_id = await _create_campaign(api_client, auth_headers)
        seeded = await api_client.post(
            f"/api/v1/results/{campaign_id}",
            json={"results": [_result_row(0.5, 1.5)]},
            headers=auth_headers,
        )
        assert seeded.json()["success"] is True

        mixed = await api_client.post(
            f"/api/v1/results/{campaign_id}",
            json={
                # Row 0 duplicates the seeded row; row 1 is fresh.
                "results": [_result_row(0.5, 1.5), _result_row(0.75, 1.0)],
                "atomic": False,
                "continue_on_error": True,
            },
            headers=auth_headers,
        )

        data = mixed.json()
        assert data.get("partial_results") is not None, json.dumps(data)[:800]
        assert len(data["partial_results"]) == 2

        listing = await api_client.get(f"/api/v1/results/{campaign_id}", headers=auth_headers)
        assert len(listing.json()) == 2, "only the fresh row may persist"
