"""REST parity for ``measurement_uncertainty`` on result submission.

The MCP / domain / storage path supports per-objective
``measurement_uncertainty`` values that the bo-engine wires into
``train_Yvar`` to drive a ``FixedNoiseGaussianLikelihood``. The REST
``ResultCreate`` schema previously dropped the field, so REST callers
could not benefit from known measurement noise even though MCP callers
could.

These tests assert that REST submission accepts the field, that it
round-trips into storage, and that the same validation rules
(unknown-objective key warnings, finite/non-negative declared values)
apply to both transports.

Reference: BoTorch ``FixedNoiseGaussianLikelihood`` documentation
(https://botorch.org/docs/likelihoods) — the per-observation noise
input shape is ``train_Yvar`` (variance), which the engine derives by
squaring the per-objective stddev passed in via this field.
"""

from uuid import uuid4

import pytest

from bo_mcp_server.tools.create_campaign import create_campaign
from bo_mcp_server.tools.generate_suggestions import generate_suggestions


async def _create_campaign(owner_id: str) -> str:
    intake = {
        "name": "REST Uncertainty Parity",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }
    created = await create_campaign(intake, owner_id)
    return created["campaign_id"]


class TestMeasurementUncertaintyRoundTrip:
    """REST submit + REST query / list round-trip the field."""

    @pytest.mark.asyncio
    async def test_submit_accepts_uncertainty_field(
        self, api_client, auth_headers, persisted_user
    ) -> None:
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign(owner_id)
        gen = await generate_suggestions(campaign_id, batch_size=1)
        suggestion = gen["suggestions"][0]

        response = await api_client.post(
            f"/api/results/{campaign_id}",
            json={
                "results": [
                    {
                        "parameter_values": suggestion["parameter_values"],
                        "objective_values": {"y": 0.4},
                        "suggestion_id": suggestion["suggestion_id"],
                        "measurement_uncertainty": {"y": 0.05},
                    }
                ],
                "source": "api",
            },
            headers=auth_headers,
        )
        # 8.19: successful batch submit returns 201.
        assert response.status_code == 201, response.text
        data = response.json()
        assert data["success"] is True
        assert len(data["result_ids"]) == 1

    @pytest.mark.asyncio
    async def test_get_results_returns_uncertainty(
        self, api_client, auth_headers, persisted_user
    ) -> None:
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign(owner_id)
        gen = await generate_suggestions(campaign_id, batch_size=1)
        suggestion = gen["suggestions"][0]

        await api_client.post(
            f"/api/results/{campaign_id}",
            json={
                "results": [
                    {
                        "parameter_values": suggestion["parameter_values"],
                        "objective_values": {"y": 0.4},
                        "suggestion_id": suggestion["suggestion_id"],
                        "measurement_uncertainty": {"y": 0.07},
                    }
                ],
                "source": "api",
            },
            headers=auth_headers,
        )

        # GET /api/results/{campaign_id} returns ResultResponse[]; the
        # field must echo the persisted value.
        response = await api_client.get(f"/api/results/{campaign_id}", headers=auth_headers)
        assert response.status_code == 200
        results = response.json()
        assert len(results) == 1
        assert results[0]["measurement_uncertainty"] == {"y": 0.07}

    @pytest.mark.asyncio
    async def test_detailed_query_returns_uncertainty(
        self, api_client, auth_headers, persisted_user
    ) -> None:
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign(owner_id)
        gen = await generate_suggestions(campaign_id, batch_size=1)
        suggestion = gen["suggestions"][0]

        await api_client.post(
            f"/api/results/{campaign_id}",
            json={
                "results": [
                    {
                        "parameter_values": suggestion["parameter_values"],
                        "objective_values": {"y": 0.4},
                        "suggestion_id": suggestion["suggestion_id"],
                        "measurement_uncertainty": {"y": 0.03},
                    }
                ],
                "source": "api",
            },
            headers=auth_headers,
        )

        # Detailed verbosity exposes ``measurement_uncertainty`` on each
        # row -- the operation already returns it; this guards the REST
        # query envelope against silently dropping it.
        response = await api_client.post(
            f"/api/results/{campaign_id}/query",
            json={"verbosity": "detailed"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        envelope = response.json()
        assert envelope["success"] is True
        assert envelope["results"][0]["measurement_uncertainty"] == {"y": 0.03}


class TestMeasurementUncertaintyValidation:
    """REST validation matches the MCP envelope for invalid values."""

    @pytest.mark.asyncio
    async def test_negative_uncertainty_returns_field_error(
        self, api_client, auth_headers, persisted_user
    ) -> None:
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign(owner_id)
        gen = await generate_suggestions(campaign_id, batch_size=1)
        suggestion = gen["suggestions"][0]

        response = await api_client.post(
            f"/api/results/{campaign_id}",
            json={
                "results": [
                    {
                        "parameter_values": suggestion["parameter_values"],
                        "objective_values": {"y": 0.4},
                        "suggestion_id": suggestion["suggestion_id"],
                        "measurement_uncertainty": {"y": -0.1},
                    }
                ],
                "source": "api",
            },
            headers=auth_headers,
        )
        # The route returns 200 with success=false because the operation
        # owns the structured-error envelope; the field_errors map must
        # pin the offending objective key.
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["success"] is False
        assert "results[0].measurement_uncertainty['y']" in data["field_errors"]

    @pytest.mark.asyncio
    async def test_unknown_objective_uncertainty_key_is_warning_only(
        self, api_client, auth_headers, persisted_user
    ) -> None:
        """Unknown keys are dropped before bo-engine; row still commits.

        Mirrors ``test_invalid_measurement_uncertainty.py``'s MCP path
        so the two transports keep the same warning-vs-error policy.
        """
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign(owner_id)
        gen = await generate_suggestions(campaign_id, batch_size=1)
        suggestion = gen["suggestions"][0]

        response = await api_client.post(
            f"/api/results/{campaign_id}",
            json={
                "results": [
                    {
                        "parameter_values": suggestion["parameter_values"],
                        "objective_values": {"y": 0.4},
                        "suggestion_id": suggestion["suggestion_id"],
                        "measurement_uncertainty": {"y": 0.05, "stray": 0.0},
                    }
                ],
                "source": "api",
            },
            headers=auth_headers,
        )
        # 8.19: row still commits with a warning, so the batch is a
        # successful create — 201.
        assert response.status_code == 201, response.text
        data = response.json()
        assert data["success"] is True
        assert any("unknown" in w.lower() for w in data["warnings"])


class TestResultMetadataValidation:
    """REST result metadata uses the same strict schema as MCP intake."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("persisted_user")
    async def test_unknown_metadata_key_returns_422_not_500(self, api_client, auth_headers) -> None:
        response = await api_client.post(
            f"/api/results/{uuid4()}",
            json={
                "results": [
                    {
                        "parameter_values": {"x": 0.25},
                        "objective_values": {"y": 0.4},
                        "metadata": {"donor_name": "bad-rest-key"},
                    }
                ],
                "source": "api",
            },
            headers=auth_headers,
        )

        assert response.status_code == 422, response.text
        body = response.json()
        paths = [tuple(err.get("loc", ())) for err in body["detail"]]
        assert ("body", "results", 0, "metadata", "donor_name") in paths
