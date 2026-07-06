"""Unit tests for the BayBE → engine-neutral exhaustion translation.

BayBE raises ``NotEnoughPointsLeftError`` when the *filtered* candidate set
— the discrete grid minus measured, previously-recommended, and pending
rows — is smaller than the requested batch (see the ``allow_recommending_*``
campaign flags: https://emdgroup.github.io/baybe/stable/userguide/campaigns.html).
The translated :class:`SearchSpaceExhaustedError` counts must mirror that
same filter; deriving ``n_available`` from measurements alone made the
server's SEARCH_SPACE_EXHAUSTED envelope contradict itself ("3 unseen
combinations remain" next to ``next_action_recommendation:
terminate_campaign``).
"""

import pytest

from bo_engine.initial_design import SearchSpaceExhaustedError
from bo_engine.types import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)
from bo_engine_baybe.backend import BayBEBackend


def _grid_spec(n_points: int) -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[
            ParameterSpec(
                name="x",
                type=ParameterType.DISCRETE,
                values=[float(i) for i in range(n_points)],
            ),
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
    )


class TestExhaustionCounts:
    def test_pending_points_counted_as_unavailable(self) -> None:
        """Grid 4, measured 1, pending 3, batch 2 ⇒ zero candidates remain.

        The pre-fix derivation (grid minus measured) reported
        ``n_available=3 >= n_requested=2`` for this exact shape while the
        recommender saw zero candidates.
        """
        backend = BayBEBackend()
        spec = _grid_spec(4)
        observations = [
            ObservationData(parameter_values={"x": 0.0}, objective_values={"y": 1.0}),
        ]
        pending = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}]
        with pytest.raises(SearchSpaceExhaustedError) as excinfo:
            backend.generate_suggestions(
                spec=spec,
                observations=observations,
                batch_size=2,
                iteration=1,
                pending_points=pending,
            )
        error = excinfo.value
        assert error.n_requested == 2
        assert error.n_total_combinations == 4
        assert error.n_available == 0

    def test_previously_recommended_points_counted_as_unavailable(self) -> None:
        """A restored campaign's recommended-row exclusions show up in the counts.

        ``allow_recommending_already_recommended=False`` drops rows the
        campaign already recommended, so after a batch of 2 on a 3-point
        grid only 1 candidate remains — with no pending points involved.
        """
        backend = BayBEBackend()
        spec = _grid_spec(3)
        first = backend.generate_suggestions(spec=spec, observations=[], batch_size=2, iteration=1)
        with pytest.raises(SearchSpaceExhaustedError) as excinfo:
            backend.generate_suggestions(
                spec=spec,
                observations=[],
                batch_size=2,
                iteration=2,
                backend_state=first.backend_state,
            )
        error = excinfo.value
        assert error.n_total_combinations == 3
        assert error.n_available == 1
        assert error.n_available < error.n_requested

    def test_counts_are_internally_consistent(self) -> None:
        """The envelope may never claim enough candidates while raising exhaustion."""
        backend = BayBEBackend()
        spec = _grid_spec(2)
        observations = [
            ObservationData(parameter_values={"x": 0.0}, objective_values={"y": 1.0}),
            ObservationData(parameter_values={"x": 1.0}, objective_values={"y": 2.0}),
        ]
        with pytest.raises(SearchSpaceExhaustedError) as excinfo:
            backend.generate_suggestions(
                spec=spec, observations=observations, batch_size=1, iteration=1
            )
        error = excinfo.value
        assert error.n_available < error.n_requested
        assert error.n_available <= (error.n_total_combinations or 0)
