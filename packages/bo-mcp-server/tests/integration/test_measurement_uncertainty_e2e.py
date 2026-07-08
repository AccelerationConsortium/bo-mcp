"""End-to-end test for measurement uncertainty flowing into the GP.

Submitting results with ``measurement_uncertainty`` through MCP must end up
as a ``train_yvar`` tensor inside the BO engine, which then routes the GP
onto a ``FixedNoiseGaussianLikelihood``. This closes the integration gap
between the model factories' new ``train_yvar`` parameter and the server's
existing measurement-uncertainty plumbing.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from bo_mcp_server.domain import ResultSubmissionInput
from tests.factories import seed_owner

pytestmark = pytest.mark.usefixtures("setup_database")


def _to_result_inputs(results: list[dict[str, Any]]) -> list[ResultSubmissionInput]:
    return [ResultSubmissionInput.model_validate(r) for r in results]


@pytest.mark.asyncio
async def test_measurement_uncertainty_drives_fixed_noise_likelihood() -> None:
    """Full-coverage uncertainty submissions -> FixedNoiseGaussianLikelihood."""
    # Patch where the GP factory is *resolved* at call time. The single-
    # objective dispatch lives in :mod:`bo_engine.suggestions_single_objective`
    # after the suggestions god-module split.
    from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

    from bo_engine import suggestions_single_objective as sugg_mod
    from bo_mcp_server.operations.submit_results import submit_results_operation
    from bo_mcp_server.tools.create_campaign import create_campaign
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    owner_id = await seed_owner()
    intake = {
        "name": "Measurement Uncertainty End-to-End",
        # train_yvar / FixedNoiseGaussianLikelihood is a BoTorch-only feature.
        "backend": "botorch",
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

    # Submit three results with measurement_uncertainty so the campaign
    # has enough observations to leave the initial-design path AND every
    # observation carries known noise.
    init_gen = await generate_suggestions(campaign_id, batch_size=2)
    submissions = [
        {
            "suggestion_id": s["id"],
            "parameter_values": s["parameter_values"],
            "objective_values": {"y": float(s["parameter_values"]["x"])},
            "measurement_uncertainty": {"y": 0.05},
        }
        for s in init_gen["suggestions"]
    ]
    await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(submissions),
        submitted_by=owner_id,
    )

    # One more observation, also with uncertainty, to push past the
    # ``min_data`` gate in ``generate_next_batch``.
    second_gen = await generate_suggestions(campaign_id, batch_size=1)
    s = second_gen["suggestions"][0]
    await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(
            [
                {
                    "suggestion_id": s["id"],
                    "parameter_values": s["parameter_values"],
                    "objective_values": {"y": float(s["parameter_values"]["x"])},
                    "measurement_uncertainty": {"y": 0.05},
                }
            ]
        ),
        submitted_by=owner_id,
    )

    captured: dict[str, object] = {}
    real_factory = sugg_mod.create_and_fit_single_task_model

    def spy(train_x, train_y, bounds, **kwargs):
        captured["train_yvar"] = kwargs.get("train_yvar")
        model = real_factory(train_x, train_y, bounds, **kwargs)
        captured["likelihood"] = model.likelihood
        return model

    with patch.object(sugg_mod, "create_and_fit_single_task_model", side_effect=spy):
        bo_gen = await generate_suggestions(campaign_id, batch_size=1)

    assert bo_gen["success"] is True
    assert captured.get("train_yvar") is not None, (
        "train_yvar was not propagated from measurement_uncertainty"
    )
    assert isinstance(captured["likelihood"], FixedNoiseGaussianLikelihood)


@pytest.mark.asyncio
async def test_partial_measurement_uncertainty_falls_back_to_trainable_noise() -> None:
    """If any observation is missing uncertainty, the GP keeps trainable noise."""
    # Patch where the GP factory is *resolved* at call time. The single-
    # objective dispatch lives in :mod:`bo_engine.suggestions_single_objective`
    # after the suggestions god-module split.
    from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood

    from bo_engine import suggestions_single_objective as sugg_mod
    from bo_mcp_server.operations.submit_results import submit_results_operation
    from bo_mcp_server.tools.create_campaign import create_campaign
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    owner_id = await seed_owner()
    intake = {
        "name": "Partial Measurement Uncertainty",
        # train_yvar / FixedNoiseGaussianLikelihood is a BoTorch-only feature.
        "backend": "botorch",
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
            "suggestion_id": init_gen["suggestions"][0]["id"],
            "parameter_values": init_gen["suggestions"][0]["parameter_values"],
            "objective_values": {"y": float(init_gen["suggestions"][0]["parameter_values"]["x"])},
            "measurement_uncertainty": {"y": 0.05},
        },
        # Second result has NO uncertainty -> full-coverage check fails.
        {
            "suggestion_id": init_gen["suggestions"][1]["id"],
            "parameter_values": init_gen["suggestions"][1]["parameter_values"],
            "objective_values": {"y": float(init_gen["suggestions"][1]["parameter_values"]["x"])},
        },
    ]
    await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(submissions),
        submitted_by=owner_id,
    )

    second_gen = await generate_suggestions(campaign_id, batch_size=1)
    s = second_gen["suggestions"][0]
    await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(
            [
                {
                    "suggestion_id": s["id"],
                    "parameter_values": s["parameter_values"],
                    "objective_values": {"y": float(s["parameter_values"]["x"])},
                }
            ]
        ),
        submitted_by=owner_id,
    )

    captured: dict[str, object] = {}
    real_factory = sugg_mod.create_and_fit_single_task_model

    def spy(train_x, train_y, bounds, **kwargs):
        captured["train_yvar"] = kwargs.get("train_yvar")
        model = real_factory(train_x, train_y, bounds, **kwargs)
        captured["likelihood"] = model.likelihood
        return model

    with patch.object(sugg_mod, "create_and_fit_single_task_model", side_effect=spy):
        await generate_suggestions(campaign_id, batch_size=1)

    assert captured["train_yvar"] is None
    assert isinstance(captured["likelihood"], GaussianLikelihood)
    assert not isinstance(captured["likelihood"], FixedNoiseGaussianLikelihood)
