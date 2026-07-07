"""Objective-scale resolution for the exploration/exploitation diagnostics.

Suggestion-provenance ``model_uncertainty`` is a raw-scale posterior std,
so before the engine compares it against dimensionless balance thresholds
it must be divided by an observed objective scale. ``None`` (no results,
degenerate spreads) tells the engine to fall back to its neutral
exploration ratio rather than treating raw units as dimensionless.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from bo_mcp_server.domain import CampaignSpec, Result, ResultSource
from bo_mcp_server.domain.campaign_spec import InputParameter, Objective, ParameterType
from bo_mcp_server.operations.diagnostics.suggestions_analysis import (
    _observed_objective_scale,
)


def _spec(objectives: tuple[Objective, ...]) -> CampaignSpec:
    return CampaignSpec(
        name="scale-test",
        parameters=(
            InputParameter(
                name="x",
                type=ParameterType.CONTINUOUS,
                bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
            ),
        ),
        objectives=objectives,
    )


def _result(values: dict[str, float]) -> Result:
    return Result(
        campaign_id=uuid4(),
        parameter_values={"x": 0.5},
        objective_values=values,
        source=ResultSource.API,
        submitted_by=uuid4(),
    )


class TestObservedObjectiveScale:
    def test_single_objective_spread(self) -> None:
        spec = _spec((Objective(name="y", direction="minimize"),))
        results = [_result({"y": v}) for v in (10.0, 30.0, 20.0)]
        assert _observed_objective_scale(results, spec) == pytest.approx(20.0)

    def test_multi_objective_mean_of_spreads(self) -> None:
        spec = _spec(
            (
                Objective(name="a", direction="minimize"),
                Objective(name="b", direction="maximize"),
            )
        )
        results = [_result({"a": 0.0, "b": 100.0}), _result({"a": 1.0, "b": 300.0})]
        # Mean of per-objective spreads: (1.0 + 200.0) / 2.
        assert _observed_objective_scale(results, spec) == pytest.approx(100.5)

    def test_no_results_is_unknown(self) -> None:
        spec = _spec((Objective(name="y", direction="minimize"),))
        assert _observed_objective_scale([], spec) is None

    def test_degenerate_spread_is_unknown(self) -> None:
        spec = _spec((Objective(name="y", direction="minimize"),))
        results = [_result({"y": 5.0}), _result({"y": 5.0})]
        assert _observed_objective_scale(results, spec) is None
