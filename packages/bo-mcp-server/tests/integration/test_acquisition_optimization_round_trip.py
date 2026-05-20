"""Integration test for the ``acquisition_optimization`` server pipeline.

Verifies that the per-campaign restart / raw-sample override flows through
intake -> create_campaign -> storage -> generate_suggestions all the way down
to the bo-engine acquisition optimizer. Closes the gap reviewers caught: the
field existed on ``CampaignSpec`` but intake forbade it, ``_build_spec_from_dict``
did not copy it, and storage did not persist or restore it.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch
from uuid import uuid4

import pytest

from bo_mcp_server.domain import ResultSubmissionInput

pytestmark = pytest.mark.usefixtures("setup_database")


def _to_result_inputs(results: list[dict[str, Any]]) -> list[ResultSubmissionInput]:
    return [ResultSubmissionInput.model_validate(r) for r in results]


@pytest.mark.asyncio
async def test_acquisition_optimization_overrides_round_trip() -> None:
    """Custom restart count survives MCP create -> DB -> generate.

    Strategy: create a campaign with explicit
    ``acquisition_optimization={num_restarts: 7, raw_samples: 33}``, then
    invoke ``generate_suggestions`` enough times to reach the BO phase
    (initial design exhausts before BO). Patch ``_resolve_restart_budget`` to
    capture the effective values passed into the BoTorch optimizer.
    """
    from bo_mcp_server.operations.submit_results import submit_results_operation
    from bo_mcp_server.tools.create_campaign import create_campaign
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    owner_id = str(uuid4())
    intake = {
        "name": "Acquisition Optimization Override",
        "parameters": [
            {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
        ],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "initial_design_size": 2,
        "batch_size": 1,
        "random_seed": 42,
        "acquisition_optimization": {"num_restarts": 7, "raw_samples": 33},
    }
    create_result = await create_campaign(intake, owner_id)
    assert create_result["success"] is True, create_result.get("errors")
    campaign_id = create_result["campaign_id"]

    # Burn through the initial design so the next call hits the BO path.
    init_gen = await generate_suggestions(campaign_id, batch_size=2)
    assert init_gen["success"] is True
    submissions = [
        {
            "suggestion_id": s["id"],
            "parameter_values": s["parameter_values"],
            "objective_values": {"y": float(s["parameter_values"]["x"])},
        }
        for s in init_gen["suggestions"]
    ]
    submit = await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(submissions),
        submitted_by=owner_id,
    )
    assert submit["success"] is True

    # Spy on the budget resolution helper to capture effective values.
    captured: dict[str, int] = {}
    from bo_engine import acquisition as bo_acq

    real_resolver = bo_acq._resolve_restart_budget

    def spy(spec, bounds, num_restarts, raw_samples):
        restarts, samples = real_resolver(spec, bounds, num_restarts, raw_samples)
        captured["num_restarts"] = restarts
        captured["raw_samples"] = samples
        return restarts, samples

    with patch.object(bo_acq, "_resolve_restart_budget", side_effect=spy):
        bo_gen = await generate_suggestions(campaign_id, batch_size=1)

    assert bo_gen["success"] is True
    assert captured["num_restarts"] == 7, (
        f"Override not applied; got num_restarts={captured.get('num_restarts')}"
    )
    assert captured["raw_samples"] == 33, (
        f"Override not applied; got raw_samples={captured.get('raw_samples')}"
    )


@pytest.mark.asyncio
async def test_intake_without_override_uses_dimension_adaptive_defaults() -> None:
    """Omitting the field falls back to the dimension-adaptive formula."""
    from bo_engine.constants import (
        NUM_RESTARTS_BASE,
        NUM_RESTARTS_PER_DIM,
        RAW_SAMPLES_MIN,
        RAW_SAMPLES_PER_DIM,
    )

    from bo_mcp_server.operations.submit_results import submit_results_operation
    from bo_mcp_server.tools.create_campaign import create_campaign
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    owner_id = str(uuid4())
    intake = {
        "name": "Default Acquisition Optimization",
        "parameters": [
            {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
        ],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "initial_design_size": 2,
        "batch_size": 1,
        "random_seed": 42,
    }
    create_result = await create_campaign(intake, owner_id)
    campaign_id = create_result["campaign_id"]

    init_gen = await generate_suggestions(campaign_id, batch_size=2)
    submissions = [
        {
            "suggestion_id": s["id"],
            "parameter_values": s["parameter_values"],
            "objective_values": {"y": float(s["parameter_values"]["x"])},
        }
        for s in init_gen["suggestions"]
    ]
    await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(submissions),
        submitted_by=owner_id,
    )

    captured: dict[str, int] = {}
    from bo_engine import acquisition as bo_acq

    real_resolver = bo_acq._resolve_restart_budget

    def spy(spec, bounds, num_restarts, raw_samples):
        restarts, samples = real_resolver(spec, bounds, num_restarts, raw_samples)
        captured["num_restarts"] = restarts
        captured["raw_samples"] = samples
        return restarts, samples

    with patch.object(bo_acq, "_resolve_restart_budget", side_effect=spy):
        bo_gen = await generate_suggestions(campaign_id, batch_size=1)
    assert bo_gen["success"] is True

    # One continuous parameter → dimension-adaptive formula at d=1.
    expected_restarts = NUM_RESTARTS_BASE + 1 * NUM_RESTARTS_PER_DIM
    expected_samples = max(RAW_SAMPLES_MIN, 1 * RAW_SAMPLES_PER_DIM)
    assert captured["num_restarts"] == expected_restarts
    assert captured["raw_samples"] == expected_samples
