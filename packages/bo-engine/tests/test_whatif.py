"""What-if analysis: real-spec simulation and dimensionless value-of-information.

Two correctness gaps are pinned here:

* M21 — ``_compute_suggestion_impact`` used to rebuild a synthetic
  ``minimize=True`` all-continuous spec and run both ``generate_next_batch``
  calls unseeded, so the reported ``suggestion_shift`` confounded the
  hypothetical's effect with acquisition-optimizer RNG and silently optimized a
  different problem than a maximization / constrained campaign. The fix threads
  the campaign's real ``OptimizationSpec`` and shares one derived seed across
  both calls.
* M22 — ``value_of_information`` summed quantities in incompatible units
  (y², parameter units, objective^m) with magic weights, so the recommendation
  flipped when the objective was rescaled. The fix normalizes each component to
  a dimensionless scale, so the verdict is invariant under ``y -> 1000*y``
  (cf. ``tests/test_convergence_scale_invariance.py``).
"""

from __future__ import annotations

from typing import cast

import pytest
import torch

from bo_engine.types import (
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)
from bo_engine.whatif import (
    HypotheticalResult,
    _compute_pareto_impact,
    _estimate_value_of_information,
    simulate_result,
)


def _bowl_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """A small 2-D quadratic bowl dataset, minimized near (0.3, 0.6)."""
    torch.manual_seed(7)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    train_x = torch.rand(8, 2, dtype=torch.double)
    train_y = ((train_x[:, 0] - 0.3) ** 2 + (train_x[:, 1] - 0.6) ** 2).unsqueeze(-1)
    return train_x, train_y, bounds, ["x1", "x2"]


def _spec(minimize: bool) -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="y", minimize=minimize)],
    )


class TestSuggestionImpactSeeding:
    """Seeding both runs isolates the hypothetical's effect from RNG jitter (M21)."""

    def test_redundant_hypothetical_shifts_less_than_informative(self) -> None:
        """The shift reflects the hypothetical's information, not RNG jitter.

        Both ``generate_next_batch`` calls share one derived seed, so the
        acquisition-optimizer randomness is held fixed across the before/after
        runs. A hypothetical that merely restates an existing observation then
        moves the suggestions far less than one placed in an unexplored region
        with a surprising value. Without the shared seed the shift was dominated
        by RNG and this ordering would not be reliable.
        """
        train_x, train_y, bounds, names = _bowl_data()
        spec = _spec(minimize=True)
        redundant = HypotheticalResult(
            parameters={"x1": float(train_x[0, 0]), "x2": float(train_x[0, 1])},
            objective_values={"y": float(train_y[0, 0])},
        )
        informative = HypotheticalResult(
            parameters={"x1": 0.9, "x2": 0.05}, objective_values={"y": -5.0}
        )
        redundant_shift = simulate_result(
            train_x, train_y, bounds, redundant, names, spec=spec, seed=123
        ).suggestion_impact
        informative_shift = simulate_result(
            train_x, train_y, bounds, informative, names, spec=spec, seed=123
        ).suggestion_impact
        assert redundant_shift is not None
        assert informative_shift is not None
        assert redundant_shift.suggestion_shift < informative_shift.suggestion_shift, (
            "A redundant hypothetical should move suggestions less than an "
            f"informative one; got redundant={redundant_shift.suggestion_shift:.4f} "
            f"vs informative={informative_shift.suggestion_shift:.4f}."
        )

    def test_same_seed_is_reproducible(self) -> None:
        train_x, train_y, bounds, names = _bowl_data()
        spec = _spec(minimize=True)
        hyp = HypotheticalResult(parameters={"x1": 0.5, "x2": 0.5}, objective_values={"y": 0.05})
        first = simulate_result(train_x, train_y, bounds, hyp, names, spec=spec, seed=99)
        second = simulate_result(train_x, train_y, bounds, hyp, names, spec=spec, seed=99)
        assert first.suggestion_impact is not None
        assert second.suggestion_impact is not None
        assert first.suggestion_impact.new_suggestions == second.suggestion_impact.new_suggestions


class TestRealSpecThreading:
    """The simulation optimizes the campaign's real problem, not a synthetic one (M21)."""

    def test_maximization_spec_round_trips(self) -> None:
        """A maximization spec runs end-to-end and respects the bounds."""
        train_x, train_y, bounds, names = _bowl_data()
        hyp = HypotheticalResult(parameters={"x1": 0.5, "x2": 0.5}, objective_values={"y": 0.05})
        report = simulate_result(
            train_x, train_y, bounds, hyp, names, spec=_spec(minimize=False), seed=5
        )
        assert report.suggestion_impact is not None
        for suggestion in report.suggestion_impact.new_suggestions:
            assert 0.0 <= suggestion["x1"] <= 1.0
            assert 0.0 <= suggestion["x2"] <= 1.0

    def test_direction_changes_suggestions(self) -> None:
        """Minimize vs maximize specs produce different suggestions on the same data.

        Under the old synthetic ``minimize=True`` spec both directions optimized
        the *same* (minimization) problem, so this assertion would fail.
        """
        train_x, train_y, bounds, names = _bowl_data()
        hyp = HypotheticalResult(parameters={"x1": 0.5, "x2": 0.5}, objective_values={"y": 0.05})
        minimized = simulate_result(
            train_x, train_y, bounds, hyp, names, spec=_spec(minimize=True), seed=5
        )
        maximized = simulate_result(
            train_x, train_y, bounds, hyp, names, spec=_spec(minimize=False), seed=5
        )
        assert minimized.suggestion_impact is not None
        assert maximized.suggestion_impact is not None
        assert (
            minimized.suggestion_impact.new_suggestions
            != maximized.suggestion_impact.new_suggestions
        ), "Minimize and maximize specs must not optimize the same problem."


class TestValueOfInformationScaleInvariance:
    """The VOI score and recommendation are invariant to objective rescaling (M22)."""

    def test_recommendation_unchanged_under_y_rescaling(self) -> None:
        train_x, train_y, bounds, names = _bowl_data()
        spec = _spec(minimize=True)
        hyp = HypotheticalResult(parameters={"x1": 0.3, "x2": 0.6}, objective_values={"y": 0.0})

        baseline = simulate_result(train_x, train_y, bounds, hyp, names, spec=spec, seed=11)

        scale = 1000.0
        hyp_scaled = HypotheticalResult(
            parameters=dict(hyp.parameters),
            objective_values={k: v * scale for k, v in hyp.objective_values.items()},
        )
        rescaled = simulate_result(
            train_x, train_y * scale, bounds, hyp_scaled, names, spec=spec, seed=11
        )

        assert baseline.recommendation == rescaled.recommendation
        assert baseline.value_of_information == pytest.approx(
            rescaled.value_of_information, rel=1e-3, abs=1e-6
        )


class TestCategoricalSpecRejected:
    """The numeric-tensor what-if can't represent categorical params (review #1)."""

    def test_categorical_spec_raises_clear_error(self) -> None:
        """A categorical spec is rejected up front, not via an opaque TypeError.

        ``hyp_x`` is a float tensor, so assigning a category string previously
        raised ``TypeError: can't assign a str to a torch.DoubleTensor`` deep in
        ``simulate_result``. The guard surfaces a clear, actionable message.
        """
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                ParameterSpec(name="c", type=ParameterType.CATEGORICAL, categories=["a", "b"]),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        train_x = torch.tensor([[0.2, 0.0], [0.8, 1.0]], dtype=torch.double)
        train_y = torch.tensor([[0.5], [0.3]], dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
        # A categorical (string) value deliberately violates the numeric
        # ``parameters`` type to exercise the runtime guard.
        hyp = HypotheticalResult(
            parameters=cast("dict[str, float]", {"x1": 0.5, "c": "a"}),
            objective_values={"y": 0.1},
        )
        with pytest.raises(ValueError, match="does not support categorical"):
            simulate_result(train_x, train_y, bounds, hyp, ["x1", "c"], spec=spec, seed=1)


class TestParetoImpactDirection:
    """Pareto impact must honor objective direction, not assume minimization (review #2)."""

    def test_maximize_objectives_pareto_membership(self) -> None:
        """For maximize objectives, the dominating point is Pareto-optimal.

        ``[5, 5]`` dominates everything when both objectives are maximized, so
        appended as the hypothetical it must read ``hypothetical_is_pareto``.
        The minimization-hardcoded code marks it dominated instead.
        """
        train_y = torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.double)
        aug_y = torch.tensor([[1.0, 1.0], [2.0, 2.0], [5.0, 5.0]], dtype=torch.double)
        maximize_mask = torch.tensor([False, False])

        impact = _compute_pareto_impact(train_y, aug_y, minimize_mask=maximize_mask)
        assert impact.hypothetical_is_pareto is True
        assert impact.hypervolume_change > 0.0

        # Contrast: the legacy all-minimize default mislabels it.
        legacy = _compute_pareto_impact(train_y, aug_y)
        assert legacy.hypothetical_is_pareto is False


class TestValueOfInformationZeroHypervolume:
    """VOI stays dimensionless even when the baseline hypervolume is 0 (review #3)."""

    def test_scale_invariant_with_zero_baseline_hv(self) -> None:
        """A zero baseline HV must not reintroduce unit dependence.

        Dividing the HV gain by ``original_hypervolume + EPS`` made VOI scale
        with the objective units when the baseline HV was 0 (common early on);
        normalizing by the observed-objective box volume restores invariance.
        """
        vois = []
        for scale in (1.0, 1000.0):
            train_y = torch.tensor([[1.0, 1.0]], dtype=torch.double) * scale
            aug_y = torch.tensor([[1.0, 1.0], [0.0, 0.0]], dtype=torch.double) * scale
            impact = _compute_pareto_impact(train_y, aug_y)
            assert impact.original_hypervolume == pytest.approx(0.0)  # baseline HV is genuinely 0
            vois.append(_estimate_value_of_information(None, None, impact))

        assert vois[0] == pytest.approx(vois[1], rel=1e-6, abs=1e-9)
