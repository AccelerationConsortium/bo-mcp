"""Tests for cost-aware Bayesian Optimization (EIpu).

References:
- EIpu (Expected Improvement per Unit cost) concept:
  https://arxiv.org/abs/1406.2541 (Snoek et al., "Input Warping for Bayesian Optimization")
- BoTorch acquisition function patterns:
  https://botorch.org/docs/acquisition
"""

import numpy as np
import torch
from botorch.models import SingleTaskGP

from bo_engine import (
    AcquisitionMethod,
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
    generate_next_batch,
)
from bo_engine.acquisition import EIpuAcquisition
from bo_engine.models import create_and_fit_single_task_model


def make_cost_aware_spec() -> OptimizationSpec:
    """Create cost-aware optimization spec."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="f", minimize=True)],
        batch_size=2,
        use_cost_aware=True,
        acquisition_method=AcquisitionMethod.COST_WEIGHTED_EI,
    )


def generate_observations_with_cost() -> list[ObservationData]:
    """Generate observations with cost data."""
    return [
        ObservationData(
            parameter_values={"x1": 0.1, "x2": 0.2},
            objective_values={"f": 0.5},
            cost=10.0,
        ),
        ObservationData(
            parameter_values={"x1": 0.3, "x2": 0.4},
            objective_values={"f": 0.3},
            cost=50.0,  # Higher cost
        ),
        ObservationData(
            parameter_values={"x1": 0.5, "x2": 0.6},
            objective_values={"f": 0.2},
            cost=100.0,  # Very high cost
        ),
        ObservationData(
            parameter_values={"x1": 0.7, "x2": 0.8},
            objective_values={"f": 0.4},
            cost=5.0,  # Low cost
        ),
        ObservationData(
            parameter_values={"x1": 0.9, "x2": 0.1},
            objective_values={"f": 0.35},
            cost=20.0,
        ),
    ]


class TestCostAwareSpec:
    """Test cost-aware optimization specification."""

    def test_eipu_acquisition_method(self) -> None:
        """EIpu acquisition method is available."""
        assert AcquisitionMethod.COST_WEIGHTED_EI.value == "cost_weighted_ei"

    def test_use_cost_aware_flag(self) -> None:
        """use_cost_aware flag is set correctly."""
        spec = make_cost_aware_spec()
        assert spec.use_cost_aware is True
        assert spec.acquisition_method == AcquisitionMethod.COST_WEIGHTED_EI


class TestCostAwareOptimization:
    """Test cost-aware BO integration."""

    def test_generates_suggestions_with_cost(self, rng: np.random.Generator) -> None:
        """generate_next_batch works with cost-aware optimization."""
        spec = make_cost_aware_spec()
        observations = generate_observations_with_cost()

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 2
        for sugg in suggestions:
            assert "x1" in sugg.parameter_values
            assert "x2" in sugg.parameter_values
            assert 0.0 <= sugg.parameter_values["x1"] <= 1.0
            assert 0.0 <= sugg.parameter_values["x2"] <= 1.0

    def test_acquisition_function_is_eipu(self, rng: np.random.Generator) -> None:
        """Suggestions show EIpu as acquisition function."""
        spec = make_cost_aware_spec()
        observations = generate_observations_with_cost()

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        for sugg in suggestions:
            assert sugg.acquisition_function == "cost_weighted_ei"

    def test_cost_aware_with_missing_cost(self, rng: np.random.Generator) -> None:
        """Falls back gracefully when some observations lack cost."""
        spec = make_cost_aware_spec()
        observations = [
            ObservationData(
                parameter_values={"x1": 0.1, "x2": 0.2},
                objective_values={"f": 0.5},
                cost=10.0,
            ),
            ObservationData(
                parameter_values={"x1": 0.3, "x2": 0.4},
                objective_values={"f": 0.3},
                cost=None,  # Missing cost
            ),
            ObservationData(
                parameter_values={"x1": 0.5, "x2": 0.6},
                objective_values={"f": 0.2},
                cost=50.0,
            ),
            ObservationData(
                parameter_values={"x1": 0.7, "x2": 0.8},
                objective_values={"f": 0.4},
                cost=15.0,
            ),
            ObservationData(
                parameter_values={"x1": 0.9, "x2": 0.1},
                objective_values={"f": 0.35},
                cost=25.0,
            ),
        ]

        # Should still work (falls back to standard acquisition)
        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 2

    def test_cost_aware_disabled(self, rng: np.random.Generator) -> None:
        """Works normally when cost-aware is disabled."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="f", minimize=True)],
            batch_size=2,
            use_cost_aware=False,
        )

        observations = [
            ObservationData(
                parameter_values={"x1": 0.3},
                objective_values={"f": 0.5},
                cost=10.0,  # Cost is ignored
            ),
            ObservationData(
                parameter_values={"x1": 0.7},
                objective_values={"f": 0.3},
                cost=50.0,
            ),
            ObservationData(
                parameter_values={"x1": 0.5},
                objective_values={"f": 0.4},
                cost=30.0,
            ),
        ]

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 2
        # Should use standard acquisition, not EIpu
        for sugg in suggestions:
            assert sugg.acquisition_function in [
                "noisy_expected_improvement",
                "expected_improvement",
            ]


class TestObservationDataWithCost:
    """Test ObservationData with cost field."""

    def test_observation_with_cost(self) -> None:
        """ObservationData can store cost."""
        obs = ObservationData(
            parameter_values={"x": 0.5},
            objective_values={"f": 0.3},
            cost=25.0,
        )
        assert obs.cost == 25.0

    def test_observation_without_cost(self) -> None:
        """ObservationData cost defaults to None."""
        obs = ObservationData(
            parameter_values={"x": 0.5},
            objective_values={"f": 0.3},
        )
        assert obs.cost is None


class TestCostAwareWithConstraints:
    """Test cost-aware BO combined with other features."""

    def test_cost_aware_with_outcome_constraint(self, rng: np.random.Generator) -> None:
        """Cost-aware works with outcome constraints."""
        from bo_engine import OutcomeConstraintSpec

        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="f", minimize=True)],
            batch_size=2,
            use_cost_aware=True,
            acquisition_method=AcquisitionMethod.COST_WEIGHTED_EI,
            outcome_constraints=[
                OutcomeConstraintSpec(
                    objective_name="f",
                    threshold=0.5,
                    greater_than=False,  # f <= 0.5
                )
            ],
        )

        observations = [
            ObservationData(
                parameter_values={"x1": 0.2},
                objective_values={"f": 0.3},  # Feasible
                cost=10.0,
            ),
            ObservationData(
                parameter_values={"x1": 0.5},
                objective_values={"f": 0.4},  # Feasible
                cost=30.0,
            ),
            ObservationData(
                parameter_values={"x1": 0.8},
                objective_values={"f": 0.7},  # Infeasible
                cost=5.0,
            ),
        ]

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 2

    def test_cost_aware_not_for_multi_objective(self, rng: np.random.Generator) -> None:
        """Cost-aware (EIpu) is single-objective only."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[
                ObjectiveSpec(name="f1", minimize=True),
                ObjectiveSpec(name="f2", minimize=True),
            ],
            batch_size=2,
            use_cost_aware=True,  # Ignored for multi-objective
        )

        observations = [
            ObservationData(
                parameter_values={"x1": 0.3},
                objective_values={"f1": 0.5, "f2": 0.4},
                cost=10.0,
            ),
            ObservationData(
                parameter_values={"x1": 0.7},
                objective_values={"f1": 0.3, "f2": 0.6},
                cost=20.0,
            ),
            ObservationData(
                parameter_values={"x1": 0.5},
                objective_values={"f1": 0.4, "f2": 0.5},
                cost=15.0,
            ),
        ]

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
            rng=rng,
        )

        assert len(suggestions) == 2
        # Should use multi-objective acquisition, not EIpu
        for sugg in suggestions:
            assert sugg.acquisition_function in [
                "hypervolume_improvement",
                "scalarized_multi_objective",
            ]


def _toy_objective_and_cost_models() -> tuple[
    SingleTaskGP, SingleTaskGP, torch.Tensor, torch.Tensor
]:
    """A 1-D objective GP and a cost GP whose cost rises steeply with x.

    A steep cost gradient makes the quotient-rule term (``-EI * cost'`` / cost^2)
    materially change the EIpu gradient, so a detached-cost implementation is
    detectable.
    """
    torch.manual_seed(0)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    train_x = torch.linspace(0.0, 1.0, 8, dtype=torch.double).unsqueeze(-1)
    # Objective in maximization form (higher = better).
    train_y = torch.sin(3.0 * train_x)
    # Cost increases sharply with x (and stays well above the positivity floor).
    train_cost = 1.0 + 5.0 * train_x
    obj_model = create_and_fit_single_task_model(train_x, train_y, bounds)
    cost_model = create_and_fit_single_task_model(train_x, train_cost, bounds)
    return obj_model, cost_model, bounds, train_y


class TestEIpuCostGradient:
    """EIpu must differentiate through the cost (the quotient-rule term, M3)."""

    def test_eipu_gradient_includes_cost_term(self) -> None:
        """The EIpu gradient matches the full EI/cost quotient, not a detached one.

        The previous implementation evaluated the cost under ``torch.no_grad``,
        so L-BFGS-B ascended ``EI/c`` with ``c`` treated as locally constant —
        the gradient reduced to ``EI'/c`` and dropped the ``-EI * c' / c^2``
        term. Delegating to ``InverseCostWeightedUtility`` restores it.
        """
        obj_model, cost_model, _bounds, _train_y = _toy_objective_and_cost_models()
        # A deliberately low ``best_f`` makes EI large everywhere, so the
        # ``-EI * cost'`` term is substantial and a detached cost is detectable.
        acqf = EIpuAcquisition(model=obj_model, cost_model=cost_model, best_f=-5.0)

        # A point between training inputs, where EI is strictly positive (pure
        # inverse-cost weighting, no negative-delta scaling branch).
        x = torch.tensor([[[0.45]]], dtype=torch.double, requires_grad=True)
        eipu = acqf(x)
        assert float(eipu.item()) > 0.0
        (grad_actual,) = torch.autograd.grad(eipu.sum(), x)

        # Reference 1: EI/cost with the cost graph KEPT (the correct quotient).
        x_full = x.detach().clone().requires_grad_(True)
        ei_full = acqf.ei(x_full)
        cost_full = cost_model.posterior(x_full).mean.reshape(())
        full = (ei_full.reshape(()) / cost_full).reshape(1)
        (grad_full,) = torch.autograd.grad(full.sum(), x_full)

        # Reference 2: the old buggy gradient with the cost DETACHED.
        x_detached = x.detach().clone().requires_grad_(True)
        ei_detached = acqf.ei(x_detached)
        cost_detached = cost_model.posterior(x_detached).mean.reshape(()).detach()
        buggy = (ei_detached.reshape(()) / cost_detached).reshape(1)
        (grad_buggy,) = torch.autograd.grad(buggy.sum(), x_detached)

        assert torch.isfinite(grad_actual).all()
        # The real EIpu gradient equals the full quotient...
        assert torch.allclose(grad_actual, grad_full, rtol=1e-5, atol=1e-9)
        # ...and is meaningfully different from the detached-cost gradient.
        assert not torch.allclose(grad_actual, grad_buggy, rtol=1e-3, atol=1e-6)
