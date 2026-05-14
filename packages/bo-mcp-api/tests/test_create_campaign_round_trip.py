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
        "saasbo_config": {"warmup_steps": 8, "num_samples": 8, "thinning": 2},
        "outcome_constraints": [{"objective_name": "y", "threshold": 0.5, "greater_than": True}],
        "max_observations": 12,
        "random_seed": 42,
    }


@pytest.mark.asyncio
async def test_rest_create_campaign_persists_advanced_fields(
    api_client, auth_headers, persisted_user
):
    """Advanced fields survive REST request → DB persistence → reload."""
    response = await api_client.post(
        "/api/campaigns",
        json={"intake": _intake_payload()},
        headers=auth_headers,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is True, body
    campaign_id = body["campaign_id"]

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        spec_repo = CampaignSpecRepository(session)
        campaign = await campaign_repo.get(UUID(campaign_id))
        assert campaign is not None
        spec = await spec_repo.get(campaign.spec_id)

    assert spec is not None
    assert spec.backend == "botorch"
    assert spec.backend_options == {"botorch": {"acquisition_optimizer": "lbfgsb"}}
    assert spec.parameters[0].parameter_options == {"baybe": {"encoding": "ohe"}}
    assert spec.use_input_warping is True
    assert spec.use_cost_aware is True
    assert spec.turbo_config is not None
    assert spec.turbo_config.initial_length == pytest.approx(0.5)
    assert spec.saasbo_config is not None
    assert spec.saasbo_config.warmup_steps == 8
    assert len(spec.outcome_constraints) == 1
    assert spec.outcome_constraints[0].objective_name == "y"
    assert spec.max_observations == 12
    assert spec.random_seed == 42
