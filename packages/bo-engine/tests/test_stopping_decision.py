"""Tests for budget / convergence-based automatic stopping.

The historical behaviour of ``OptimizationSpec`` quietly dropped any
``max_iterations`` field set at the server domain layer, so unattended
agent-driven campaigns could blow through their experimental budget without
warning. ``evaluate_stopping_decision`` now centralises three stopping
signals — iteration cap, observation cap, and convergence tolerance — into
a single :class:`StoppingDecision` consumed by the server's
``generate_suggestions`` entry point.

The convergence path delegates to the existing
``detect_single_objective_convergence`` implementation; see references in
:mod:`bo_engine.convergence` for the underlying methodology.
"""

from __future__ import annotations

import pytest

from bo_engine.convergence import (
    StoppingDecision,
    StoppingReason,
    evaluate_stopping_decision,
)
from bo_engine.types import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)


def _spec(
    *,
    max_iterations: int | None = None,
    max_observations: int | None = None,
    convergence_tolerance: float | None = None,
    minimize: bool = True,
) -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
        objectives=[ObjectiveSpec(name="y", minimize=minimize)],
        max_iterations=max_iterations,
        max_observations=max_observations,
        convergence_tolerance=convergence_tolerance,
    )


def _obs(values: list[float]) -> list[ObservationData]:
    return [ObservationData(parameter_values={"x": 0.5}, objective_values={"y": v}) for v in values]


class TestIterationBudget:
    """``max_iterations`` short-circuits suggestion generation."""

    def test_no_max_iterations_is_no_op(self) -> None:
        """Without a configured cap the decision is always to proceed."""
        decision = evaluate_stopping_decision(
            _spec(), observations=_obs([1.0, 2.0]), next_iteration=42
        )
        assert decision == StoppingDecision(False, None, "", {})

    def test_iteration_below_cap_continues(self) -> None:
        """At ``next_iteration == max_iterations`` the campaign still runs."""
        decision = evaluate_stopping_decision(
            _spec(max_iterations=5), observations=_obs([1.0]), next_iteration=5
        )
        assert decision.should_stop is False

    def test_iteration_above_cap_stops_with_recommendation(self) -> None:
        """At ``next_iteration > max_iterations`` the campaign is halted."""
        decision = evaluate_stopping_decision(
            _spec(max_iterations=5), observations=_obs([1.0]), next_iteration=6
        )
        assert decision.should_stop is True
        assert decision.reason == StoppingReason.BUDGET_EXCEEDED_ITERATIONS
        assert decision.details["next_action_recommendation"] == "terminate_campaign"
        assert decision.details["next_iteration"] == 6
        assert decision.details["max_iterations"] == 5


class TestObservationBudget:
    """``max_observations`` short-circuits independently of iteration grouping."""

    def test_below_cap_continues(self) -> None:
        """``len(observations) < max_observations`` keeps the campaign running."""
        decision = evaluate_stopping_decision(
            _spec(max_observations=3), observations=_obs([1.0, 2.0]), next_iteration=1
        )
        assert decision.should_stop is False

    def test_at_or_above_cap_stops(self) -> None:
        """Equal-or-greater triggers ``BUDGET_EXCEEDED_OBSERVATIONS``."""
        decision = evaluate_stopping_decision(
            _spec(max_observations=3),
            observations=_obs([1.0, 2.0, 3.0]),
            next_iteration=1,
        )
        assert decision.should_stop is True
        assert decision.reason == StoppingReason.BUDGET_EXCEEDED_OBSERVATIONS
        assert decision.details["n_observations"] == 3


class TestConvergenceStopping:
    """``convergence_tolerance`` defers to ``detect_single_objective_convergence``."""

    def test_steady_improvement_does_not_stop(self) -> None:
        """A still-improving trajectory must not trigger a stop."""
        # Strictly decreasing best-so-far on a minimization objective.
        values = [10.0 - i for i in range(20)]
        decision = evaluate_stopping_decision(
            _spec(convergence_tolerance=0.05),
            observations=_obs(values),
            next_iteration=1,
        )
        assert decision.should_stop is False

    def test_plateau_triggers_convergence_stop(self) -> None:
        """A long flat best-so-far trajectory triggers ``CONVERGED``."""
        # Drop once at the start, then a long plateau.
        values = [5.0, 1.0] + [1.0] * 18
        decision = evaluate_stopping_decision(
            _spec(convergence_tolerance=0.05),
            observations=_obs(values),
            next_iteration=1,
        )
        assert decision.should_stop is True
        assert decision.reason == StoppingReason.CONVERGED
        assert decision.details["convergence_tolerance"] == pytest.approx(0.05)
        assert decision.details["next_action_recommendation"] == "terminate_campaign"

    def test_maximize_direction_uses_running_max(self) -> None:
        """For maximize objectives the helper tracks the running maximum."""
        # Increasing best-so-far on a maximize objective then plateau.
        values = [0.0, 5.0] + [5.0] * 18
        decision = evaluate_stopping_decision(
            _spec(convergence_tolerance=0.05, minimize=False),
            observations=_obs(values),
            next_iteration=1,
        )
        assert decision.should_stop is True
        assert decision.reason == StoppingReason.CONVERGED


class TestMultiObjectiveConvergence:
    """``convergence_tolerance`` on multi-objective specs is treated as a no-op.

    The server-side ``CampaignSpec`` validator rejects this combination at
    create time, but callers using the engine directly may construct a
    multi-objective ``OptimizationSpec`` with ``convergence_tolerance`` set.
    Such specs must not silently use objective 0 — they short-circuit to
    ``should_stop=False`` instead.
    """

    def test_multi_objective_convergence_tolerance_is_no_op(self) -> None:
        """No stop reason fires when more than one objective is configured."""
        spec = OptimizationSpec(
            parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
            objectives=[
                ObjectiveSpec(name="y1", minimize=True),
                ObjectiveSpec(name="y2", minimize=True),
            ],
            convergence_tolerance=0.05,
        )
        plateau = [
            ObservationData(
                parameter_values={"x": 0.5},
                objective_values={"y1": 1.0, "y2": 2.0},
            )
            for _ in range(20)
        ]
        decision = evaluate_stopping_decision(spec, observations=plateau, next_iteration=1)
        assert decision.should_stop is False


class TestOrderIndependence:
    """Stopping decisions must depend on the *content* of observations only.

    ``ResultRepository.list_by_campaign`` now orders by ``(created_at, id)``
    so list order is stable, but the engine layer is also order-invariant
    for budget caps (which depend on count) and order-invariant for the
    convergence-stop trajectory **provided** the caller has already sorted
    the observations. The two cases below pin both invariants.
    """

    def test_budget_cap_is_order_independent(self) -> None:
        """Reordering observations cannot flip a ``BUDGET_EXCEEDED`` decision."""
        spec = _spec(max_observations=3)
        obs = _obs([1.0, 2.0, 3.0])
        baseline = evaluate_stopping_decision(spec, obs, next_iteration=1)
        shuffled = evaluate_stopping_decision(spec, list(reversed(obs)), next_iteration=1)
        assert baseline.reason == shuffled.reason == StoppingReason.BUDGET_EXCEEDED_OBSERVATIONS


class TestPriority:
    """Iteration cap beats observation cap beats convergence (deterministic)."""

    def test_iteration_cap_beats_observation_cap(self) -> None:
        """Both caps tripped → iteration reason wins for a predictable envelope."""
        decision = evaluate_stopping_decision(
            _spec(max_iterations=2, max_observations=2),
            observations=_obs([1.0, 2.0]),
            next_iteration=3,
        )
        assert decision.reason == StoppingReason.BUDGET_EXCEEDED_ITERATIONS

    def test_observation_cap_beats_convergence(self) -> None:
        """If observations are also exhausted, prefer the explicit budget cap."""
        values = [1.0] * 20
        decision = evaluate_stopping_decision(
            _spec(max_observations=20, convergence_tolerance=0.05),
            observations=_obs(values),
            next_iteration=1,
        )
        assert decision.reason == StoppingReason.BUDGET_EXCEEDED_OBSERVATIONS


class TestMatchModeConvergence:
    """MATCH objectives converge on the distance-to-target trajectory.

    A boolean direction cannot express "best = closest to target": the
    raw-value trajectory of an overshoot-then-approach run reads as
    stagnation (or as maximization progress), so the stopping metric is
    ``|value - target_value|`` with minimize semantics.
    """

    @staticmethod
    def _match_spec(convergence_tolerance: float) -> OptimizationSpec:
        from bo_engine.types import TargetMode

        return OptimizationSpec(
            parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
            objectives=[ObjectiveSpec(name="y", target_mode=TargetMode.MATCH, target_value=7.4)],
            convergence_tolerance=convergence_tolerance,
        )

    def test_steadily_approaching_target_does_not_stop(self) -> None:
        """Overshoot-then-approach keeps improving on the distance metric.

        The raw values oscillate around the 7.4 target while the distance
        shrinks every step — the pre-fix boolean read this as a maximize
        trajectory and could stop the campaign on a wrong signal.
        """
        values = [10.0, 5.4, 9.0, 6.4, 8.2, 6.9, 7.8, 7.2, 7.55, 7.32]
        decision = evaluate_stopping_decision(
            self._match_spec(convergence_tolerance=0.01), _obs(values), next_iteration=3
        )
        assert not decision.should_stop

    def test_distance_plateau_triggers_convergence_stop(self) -> None:
        values = [10.0, 7.5] + [7.41] * 10
        decision = evaluate_stopping_decision(
            self._match_spec(convergence_tolerance=0.01), _obs(values), next_iteration=3
        )
        assert decision.should_stop
        assert decision.reason == StoppingReason.CONVERGED

    def test_target_mode_direction_override_is_honored(self) -> None:
        """target_mode='maximize' with the boolean left at default resolves
        through effective_mode: a steadily increasing trajectory keeps
        improving instead of reading as a minimize plateau."""
        from bo_engine.types import TargetMode

        spec = OptimizationSpec(
            parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
            objectives=[ObjectiveSpec(name="y", target_mode=TargetMode.MAXIMIZE)],
            convergence_tolerance=0.01,
        )
        rising = [float(v) for v in range(1, 12)]
        decision = evaluate_stopping_decision(spec, _obs(rising), next_iteration=3)
        assert not decision.should_stop
