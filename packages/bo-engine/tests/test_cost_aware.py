"""Tests for cost-aware Bayesian Optimization (EIpu).

References:
- EI per unit cost ("EI per second"): Snoek, Larochelle & Adams (2012),
  "Practical Bayesian Optimization of Machine Learning Algorithms",
  https://arxiv.org/abs/1206.2944
- BoTorch acquisition function patterns:
  https://botorch.org/docs/acquisition
- qLogNEI (the improvement core): Ament et al. (2023), "Unexpected
  Improvements to Expected Improvement for Bayesian Optimization",
  https://arxiv.org/abs/2310.20708
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
from bo_engine.acquisition import create_cost_aware_acquisition, optimize_acquisition
from bo_engine.models import create_and_fit_single_task_model

# Number of candidates requested when exercising sequential-greedy batching.
EIPU_TEST_BATCH_SIZE = 3

# Minimum pairwise distance between sequential-greedy batch members (and
# between a pending-free and a pending-conditioned candidate). Calibrated on
# the toy models below: genuinely conditioned candidates separate by
# >= 2.5e-3, while an acquisition that ignores pending points collapses the
# batch to duplicates within ~1e-8.
EIPU_MIN_CANDIDATE_DISTANCE = 1e-4

# Gradient agreement tolerances: the acquisition gradient must match the
# reference computed with the cost graph intact...
EIPU_GRADIENT_MATCH_RTOL = 1e-5
EIPU_GRADIENT_MATCH_ATOL = 1e-9
# ...and must differ from the detached-cost reference by more than fit noise.
EIPU_GRADIENT_DIFF_RTOL = 1e-3
EIPU_GRADIENT_DIFF_ATOL = 1e-6

# Pointwise agreement tolerance for the log-space decomposition identity
# ``EIpu(x) = log qNEI(x) - log E[cost(x)]`` (identical tensor ops on both
# sides, so only floating-point roundoff separates them).
EIPU_DECOMPOSITION_RTOL = 1e-10

# Non-unit search-space width used to confirm the batch behavior is not an
# artifact of the unit cube.
NON_UNIT_UPPER_BOUND = 2.0


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


def _toy_objective_and_cost_models(
    upper_bound: float = 1.0,
) -> tuple[SingleTaskGP, SingleTaskGP, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A 1-D objective GP and a cost GP whose cost rises steeply with x.

    A steep cost gradient makes the log-cost term (``-cost'/cost``)
    materially change the EIpu gradient, so a detached-cost implementation is
    detectable. ``upper_bound`` widens the search space beyond the unit cube.
    """
    torch.manual_seed(0)
    bounds = torch.tensor([[0.0], [upper_bound]], dtype=torch.double)
    train_x = torch.linspace(0.0, upper_bound, 8, dtype=torch.double).unsqueeze(-1)
    # Objective in maximization form (higher = better).
    train_y = torch.sin(3.0 * train_x)
    # Cost increases sharply with x (and stays well above the positivity floor).
    train_cost = 1.0 + 5.0 * train_x
    obj_model = create_and_fit_single_task_model(train_x, train_y, bounds)
    cost_model = create_and_fit_single_task_model(train_x, train_cost, bounds)
    return obj_model, cost_model, bounds, train_x, train_y


def _toy_eipu_acquisition(upper_bound: float = 1.0):
    """Build the EIpu acquisition on the toy models, returning it with bounds."""
    obj_model, cost_model, bounds, train_x, train_y = _toy_objective_and_cost_models(upper_bound)
    acqf = create_cost_aware_acquisition(
        model=obj_model,
        train_x=train_x,
        train_y=train_y,
        cost_model=cost_model,
        maximize=True,
    )
    return acqf, cost_model, bounds, train_x


class TestEIpuCostGradient:
    """EIpu must differentiate through the cost (the quotient-rule term)."""

    def test_eipu_gradient_includes_cost_term(self) -> None:
        """The EIpu gradient matches log-improvement minus log-cost, graph intact.

        In log space ``d/dx [log EI(x) - log c(x)] = d log EI - c'(x)/c(x)``
        — the log-domain form of the EI-per-unit-cost quotient rule (Snoek
        et al. 2012, https://arxiv.org/abs/1206.2944). An implementation
        that evaluates the cost under ``torch.no_grad`` drops the ``c'/c``
        term, so L-BFGS-B would ascend a surface that treats the cost as
        locally constant. ``InverseCostWeightedUtility`` (log mode) keeps
        the cost posterior in the autograd graph.
        """
        acqf, cost_model, _bounds, _train_x = _toy_eipu_acquisition()

        # A point between training inputs, on the steep part of the cost.
        x = torch.tensor([[[0.45]]], dtype=torch.double, requires_grad=True)
        eipu = acqf(x)
        assert torch.isfinite(eipu).all()
        (grad_actual,) = torch.autograd.grad(eipu.sum(), x)

        # Reference 1: log improvement - log cost with the cost graph KEPT.
        x_full = x.detach().clone().requires_grad_(True)
        log_imp_full = acqf.improvement(x_full)
        cost_full = cost_model.posterior(x_full).mean.reshape(())
        full = (log_imp_full.reshape(()) - torch.log(cost_full)).reshape(1)
        (grad_full,) = torch.autograd.grad(full.sum(), x_full)

        # Reference 2: the cost DETACHED — its slope vanishes from the gradient.
        x_detached = x.detach().clone().requires_grad_(True)
        log_imp_detached = acqf.improvement(x_detached)
        cost_detached = cost_model.posterior(x_detached).mean.reshape(()).detach()
        buggy = (log_imp_detached.reshape(()) - torch.log(cost_detached)).reshape(1)
        (grad_buggy,) = torch.autograd.grad(buggy.sum(), x_detached)

        assert torch.isfinite(grad_actual).all()
        # The real EIpu gradient equals the full log-quotient...
        assert torch.allclose(
            grad_actual, grad_full, rtol=EIPU_GRADIENT_MATCH_RTOL, atol=EIPU_GRADIENT_MATCH_ATOL
        )
        # ...and is meaningfully different from the detached-cost gradient.
        assert not torch.allclose(
            grad_actual, grad_buggy, rtol=EIPU_GRADIENT_DIFF_RTOL, atol=EIPU_GRADIENT_DIFF_ATOL
        )

    def test_eipu_value_is_log_improvement_minus_log_cost(self) -> None:
        """EIpu decomposes as ``log qNEI(x) - log E[cost(x)]`` pointwise.

        This is the log-space form of EI per unit cost (Snoek et al. 2012,
        https://arxiv.org/abs/1206.2944): subtracting the log expected cost
        is what steers the maximizer toward cheaper regions when
        improvements are comparable. BoTorch's
        ``InverseCostWeightedUtility(log=True)`` implements exactly this
        subtraction (see ``botorch.acquisition.cost_aware``).
        """
        acqf, cost_model, _bounds, _train_x = _toy_eipu_acquisition()

        x = torch.tensor([[[0.25]], [[0.45]], [[0.85]]], dtype=torch.double)
        with torch.no_grad():
            actual = acqf(x)
            expected = acqf.improvement(x) - torch.log(cost_model.posterior(x).mean.reshape(-1))
        assert torch.allclose(actual, expected, rtol=EIPU_DECOMPOSITION_RTOL)


class TestEIpuBatchConditioning:
    """Batch and pending-point conditioning of the cost-aware acquisition.

    BoTorch's sequential-greedy batch loop (``optimize_acqf(...,
    sequential=True)``) conditions candidate *i* on candidates ``0..i-1``
    via ``set_X_pending``; qLogNEI consumes pending points by folding them
    into its incumbent baseline (Ament et al. 2023,
    https://arxiv.org/abs/2310.20708; see also BoTorch's joint-vs-sequential
    candidate generation discussion at https://botorch.org/docs/optimization/).
    An acquisition whose pending hook is a no-op maximizes the identical
    surface at every greedy step and returns a batch of duplicates.
    """

    def test_batch_members_are_distinct(self) -> None:
        """A cost-aware batch of 3 contains three distinct experiments."""
        acqf, _cost_model, bounds, _train_x = _toy_eipu_acquisition()

        candidates, _ = optimize_acquisition(acqf, bounds, batch_size=EIPU_TEST_BATCH_SIZE)

        assert candidates.shape[0] == EIPU_TEST_BATCH_SIZE
        distances = torch.cdist(candidates, candidates)
        off_diagonal = distances[~torch.eye(EIPU_TEST_BATCH_SIZE, dtype=torch.bool)]
        assert (off_diagonal > EIPU_MIN_CANDIDATE_DISTANCE).all(), (
            f"Sequential-greedy batch members collapsed to duplicates: "
            f"pairwise distances {off_diagonal.tolist()}"
        )

    def test_pending_points_shift_the_candidate(self) -> None:
        """Declaring the incumbent maximizer as pending relocates the argmax.

        Uses a non-unit search space. A fresh acquisition is built for the
        pending run so the comparison starts from an identical surface.
        """
        acqf_free, _cost_model, bounds, _train_x = _toy_eipu_acquisition(
            upper_bound=NON_UNIT_UPPER_BOUND
        )
        unconditioned, _ = optimize_acquisition(acqf_free, bounds, batch_size=1)

        acqf_pending, _cost_model, bounds, _train_x = _toy_eipu_acquisition(
            upper_bound=NON_UNIT_UPPER_BOUND
        )
        conditioned, _ = optimize_acquisition(
            acqf_pending,
            bounds,
            batch_size=1,
            X_pending=unconditioned.reshape(1, -1),
        )

        shift = (conditioned - unconditioned).norm().item()
        assert shift > EIPU_MIN_CANDIDATE_DISTANCE, (
            f"Pending point did not move the candidate (shift {shift:.3e}); "
            "X_pending is being ignored by the acquisition"
        )
