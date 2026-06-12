"""BayBE pending-experiment support tests.

Verifies that pending suggestions are forwarded to BayBE's native
``Campaign.recommend(pending_experiments=...)`` /
``Campaign.acquisition_values(pending_experiments=...)`` and that the
backend does not silently drop them. Validation failures downgrade to
warnings instead of raising, matching the protocol contract that
``pending_points`` is an optimization hint and not a correctness gate.

Reference: BayBE asynchronous-workflow docs (pending experiments).
"""

from __future__ import annotations

import pytest

from bo_engine.types import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)
from bo_engine_baybe.backend import BayBEBackend


@pytest.fixture
def finite_discrete_spec() -> OptimizationSpec:
    """Tiny finite discrete spec where pending-aware behavior matters."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x", type=ParameterType.DISCRETE, values=[0.0, 0.5, 1.0]),
            ParameterSpec(name="y", type=ParameterType.DISCRETE, values=[0.0, 0.5, 1.0]),
        ],
        objectives=[ObjectiveSpec(name="z", minimize=True)],
        batch_size=1,
    )


class TestPendingExperimentsAreHonored:
    def test_pending_point_not_reissued_on_discrete_space(
        self,
        finite_discrete_spec: OptimizationSpec,
    ) -> None:
        """The pending point is excluded from the next recommendation.

        With ``allow_recommending_already_measured=False`` and the new
        ``allow_recommending_pending_experiments=False`` default, BayBE's
        own discrete filter drops the pending row from the candidate set
        — so a 9-cell grid with one measurement and one pending point has
        7 remaining cells, and the recommender must never return the
        pending one.
        """
        backend = BayBEBackend()
        observations = [
            ObservationData(parameter_values={"x": 0.0, "y": 0.0}, objective_values={"z": 1.0}),
        ]
        pending = [{"x": 0.5, "y": 0.5}]
        batch = backend.generate_suggestions(
            spec=finite_discrete_spec,
            observations=observations,
            batch_size=1,
            iteration=1,
            pending_points=pending,
        )
        for sugg in batch.suggestions:
            assert sugg["parameter_values"] != pending[0]

    def test_pending_point_invalid_downgrades_to_warning(
        self,
        finite_discrete_spec: OptimizationSpec,
    ) -> None:
        """Pending entries missing parameters become a warning, not a crash."""
        backend = BayBEBackend()
        observations = [
            ObservationData(parameter_values={"x": 0.0, "y": 0.0}, objective_values={"z": 1.0}),
        ]
        batch = backend.generate_suggestions(
            spec=finite_discrete_spec,
            observations=observations,
            batch_size=1,
            iteration=1,
            pending_points=[{"x": 0.5}],  # missing "y"
        )
        assert any("Pending points dropped" in w for w in batch.warnings)


class TestPendingExperimentsInContinuousSpace:
    def test_continuous_pending_accepted(self) -> None:
        """A continuous campaign accepts the pending dataframe without dtype errors."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        backend = BayBEBackend()
        observations = [
            ObservationData(parameter_values={"x1": 0.1, "x2": 0.1}, objective_values={"y": 1.0}),
            ObservationData(parameter_values={"x1": 0.9, "x2": 0.1}, objective_values={"y": 0.5}),
            ObservationData(parameter_values={"x1": 0.5, "x2": 0.5}, objective_values={"y": 0.7}),
        ]
        batch = backend.generate_suggestions(
            spec=spec,
            observations=observations,
            batch_size=1,
            iteration=1,
            pending_points=[{"x1": 0.6, "x2": 0.6, "y": 0.5}],
        )
        # Should produce a valid suggestion and no "Pending points dropped" warning.
        assert len(batch.suggestions) == 1
        assert not any("Pending points dropped" in w for w in batch.warnings)
