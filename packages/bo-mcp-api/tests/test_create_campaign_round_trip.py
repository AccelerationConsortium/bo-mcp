"""REST create-campaign route preserves advanced fields end-to-end.

Drives the FastAPI route the same way a real client would
(``POST /api/campaigns``) and reads the persisted ``CampaignSpec`` back
from storage. Fails if any field on the request body silently drops
between REST schema validation, ``CampaignIntakeInput`` conversion,
``create_campaign_operation``, and the ``CampaignSpecRepository`` save
+ load cycle.
"""

from __future__ import annotations

from uuid import UUID

import pytest

from bo_mcp_server.storage import CampaignRepository, CampaignSpecRepository, get_session

pytestmark = pytest.mark.usefixtures("persisted_user")


def _intake_payload() -> dict:
    return {
        "name": "Advanced REST Round Trip",
        "parameters": [
            {
                "name": "x",
                "type": "continuous",
                "bounds": [0.0, 1.0],
                "parameter_options": {"baybe": {"encoding": "ohe"}},
            }
        ],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "backend": "botorch",
        "backend_options": {"botorch": {"acquisition_optimizer": "lbfgsb"}},
        "use_input_warping": True,
        "use_cost_aware": True,
        "turbo_config": {"initial_length": 0.5},
        "outcome_constraints": [{"objective_name": "y", "threshold": 0.5, "greater_than": True}],
        "max_observations": 12,
        "random_seed": 42,
    }


@pytest.mark.asyncio
async def test_rest_create_campaign_persists_advanced_fields(api_client, auth_headers):
    """Advanced fields survive REST request → DB persistence → reload."""
    response = await api_client.post(
        "/api/campaigns",
        json={"intake": _intake_payload()},
        headers=auth_headers,
    )

    # 8.19: successful create returns 201 + Location pointing at the
    # new resource's canonical GET.
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["success"] is True, body
    campaign_id = body["campaign_id"]
    assert response.headers["Location"] == f"/api/v1/campaigns/{campaign_id}"

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        spec_repo = CampaignSpecRepository(session)
        campaign = await campaign_repo.get(UUID(campaign_id))
        assert campaign is not None
        spec = await spec_repo.get(campaign.spec_id)

    assert spec is not None
    assert spec.requested_backend == "botorch"
    assert spec.backend == "botorch"
    assert spec.backend_options == {"botorch": {"acquisition_optimizer": "lbfgsb"}}
    assert spec.parameters[0].parameter_options == {"baybe": {"encoding": "ohe"}}
    assert spec.use_input_warping is True
    assert spec.use_cost_aware is True
    assert spec.turbo_config is not None
    assert spec.turbo_config.initial_length == pytest.approx(0.5)
    assert len(spec.outcome_constraints) == 1
    assert spec.outcome_constraints[0].objective_name == "y"
    assert spec.max_observations == 12
    assert spec.random_seed == 42

    config_response = await api_client.get(
        f"/api/campaigns/{campaign_id}/config",
        headers=auth_headers,
    )
    assert config_response.status_code == 200, config_response.text
    config = config_response.json()
    assert "schema_version" not in config
    assert config["campaign_id"] == campaign_id
    assert config["spec_id"] == str(campaign.spec_id)
    assert config["backend_requested"] == "botorch"
    assert config["backend_resolved"] == "botorch"
    assert config["batch_size"] == 1
    assert config["max_observations"] == 12
    assert config["initial_design_size_requested"] is None
    assert config["initial_design_size"] == 2
    assert config["initial_design_size_source"] == "botorch_default"
    assert config["random_seed"] == 42
    assert config["parameters"][0]["parameter_options"] == {"baybe": {"encoding": "ohe"}}
    assert config["objectives"] == [
        {
            "name": "y",
            "direction": "minimize",
            "unit": "",
            "target": None,
            "log_transform": False,
            # Extended target surface (match modes, desirability inputs,
            # typed transforms) — unset for this plain minimize objective.
            "target_mode": None,
            "match_shape": None,
            "match_scale": None,
            "weight": None,
            "normalization_bounds": None,
            "transform": None,
        }
    ]
    assert config["backend_options"] == {"botorch": {"acquisition_optimizer": "lbfgsb"}}
    assert config["use_input_warping"] is True
    assert config["use_cost_aware"] is True
    assert config["turbo_config"]["initial_length"] == pytest.approx(0.5)
    assert config["outcome_constraints"][0]["objective_name"] == "y"


@pytest.mark.asyncio
async def test_campaign_config_preserves_requested_initial_design_size(api_client, auth_headers):
    payload = _intake_payload()
    payload["initial_design_size"] = 7

    create_response = await api_client.post(
        "/api/campaigns",
        json={"intake": payload},
        headers=auth_headers,
    )

    assert create_response.status_code == 201, create_response.text
    campaign_id = create_response.json()["campaign_id"]

    config_response = await api_client.get(
        f"/api/campaigns/{campaign_id}/config",
        headers=auth_headers,
    )

    assert config_response.status_code == 200, config_response.text
    config = config_response.json()
    assert config["initial_design_size_requested"] == 7
    assert config["initial_design_size"] == 7
    assert config["initial_design_size_source"] == "requested"


@pytest.mark.asyncio
async def test_campaign_config_raises_low_requested_initial_design_size_to_floor(
    api_client, auth_headers
):
    """A request below n_parameters + 1 is raised to the floor, not echoed verbatim.

    ``generate_next_batch`` never fits a GP on fewer than n_parameters + 1
    observations, so a 3-parameter campaign that requests
    ``initial_design_size=1`` still waits for 4. The config snapshot must
    reflect what will actually run rather than the raw request.
    """
    payload = _intake_payload()
    payload["parameters"] = [
        {"name": "x0", "type": "continuous", "bounds": [0.0, 1.0]},
        {"name": "x1", "type": "continuous", "bounds": [0.0, 1.0]},
        {"name": "x2", "type": "continuous", "bounds": [0.0, 1.0]},
    ]
    payload["initial_design_size"] = 1

    create_response = await api_client.post(
        "/api/campaigns",
        json={"intake": payload},
        headers=auth_headers,
    )
    assert create_response.status_code == 201, create_response.text
    campaign_id = create_response.json()["campaign_id"]

    config_response = await api_client.get(
        f"/api/campaigns/{campaign_id}/config",
        headers=auth_headers,
    )

    assert config_response.status_code == 200, config_response.text
    config = config_response.json()
    assert config["initial_design_size_requested"] == 1
    assert config["initial_design_size"] == 4
    assert config["initial_design_size_source"] == "botorch_default"


@pytest.mark.asyncio
async def test_rest_create_campaign_rejects_saasbo_on_botorch(api_client, auth_headers):
    """A pinned ``backend="botorch"`` must reject ``saasbo_config`` at intake.

    The BoTorch suggestion pipeline does not route SAASBO
    (``generate_next_batch`` raises a typed ``SAASBONotSupportedError``),
    so the backend reports the option ``UNSUPPORTED`` and campaign
    creation fails fast instead of silently fitting a dense-ARD GP under
    a sparse-prior advertisement. Operation-level rejections keep the
    route's historical ``200 OK`` + ``success=false`` envelope contract
    (see ``create_new_campaign``).
    """
    payload = _intake_payload()
    payload["saasbo_config"] = {"warmup_steps": 8, "num_samples": 8, "thinning": 2}

    response = await api_client.post(
        "/api/campaigns",
        json={"intake": payload},
        headers=auth_headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is False, body
    assert body["campaign_id"] is None
    assert any("saasbo" in error.lower() for error in body["errors"]), body["errors"]


def _baybe_acknowledged_intake() -> dict:
    """Minimal BayBE intake with a degradable knob and matching acknowledgement."""
    return {
        "name": "BayBE acknowledged degradation",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "backend": "baybe",
        "use_input_warping": True,
        "acknowledge_degradations": ["use_input_warping"],
    }


@pytest.mark.asyncio
async def test_rest_create_campaign_accepts_acknowledged_baybe_degradation(
    api_client, auth_headers
):
    """REST ``acknowledge_degradations`` must reach the domain intake.

    Without the field threading through ``_coerce_intake``, the route
    accepts the JSON shape and then silently drops the acknowledgement;
    create-time capability enforcement on BayBE then rejects the spec
    even though the caller asked for the degraded run. This regression
    test pins both the round-trip and the resulting success.
    """
    response = await api_client.post(
        "/api/campaigns",
        json={"intake": _baybe_acknowledged_intake()},
        headers=auth_headers,
    )
    # 8.19: successful create returns 201 even with acknowledged
    # degradations (the campaign is still persisted).
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["success"] is True, body
    warnings = body.get("warnings") or []
    assert any("Input warping" in w or "input_warping" in w for w in warnings), warnings

    async with get_session() as session:
        spec_repo = CampaignSpecRepository(session)
        campaign_repo = CampaignRepository(session)
        campaign = await campaign_repo.get(UUID(body["campaign_id"]))
        assert campaign is not None
        spec = await spec_repo.get(campaign.spec_id)
    assert spec is not None
    assert spec.acknowledge_degradations == ("use_input_warping",)


@pytest.mark.asyncio
async def test_rest_create_campaign_rejects_unacknowledged_baybe_degradation(
    api_client, auth_headers
):
    """Without acknowledgement the same payload must be rejected at create-time.

    Operation-level rejections in this codebase ride on the
    ``CampaignCreateResponse`` envelope: the route returns HTTP 200
    with ``success=False`` and the structured error in ``errors``
    (see e.g. ``test_rest_measurement_uncertainty.py`` for the
    established convention). Only request-body Pydantic validation
    returns 4xx. Pin both halves so a future route change cannot
    silently downgrade the contract.
    """
    intake = _baybe_acknowledged_intake()
    intake.pop("acknowledge_degradations")
    response = await api_client.post(
        "/api/campaigns",
        json={"intake": intake},
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is False, body
    assert body.get("campaign_id") is None
    errors = body.get("errors") or []
    joined = " ".join(errors)
    assert "Input warping" in joined or "use_input_warping" in joined, errors
