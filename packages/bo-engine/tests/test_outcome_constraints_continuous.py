"""Tests for the continuous outcome-constraint modeling path.

The legacy binary path collapses the constrained objective to {0, 1} feasibility
labels before fitting a GP — this throws away the distance-to-boundary signal
that constrained EI relies on near the feasibility frontier. The continuous
path fits the raw objective values directly so the acquisition's
Gaussian-CDF feasibility weighting reads boundary distance rather than a
classifier's confidence.

References:
    - Gardner et al., "Bayesian Optimization with Inequality Constraints",
      ICML 2014 — original constrained-EI formulation that assumes
      regression on the constrained outcome.
    - Letham et al., "Constrained Bayesian Optimization with Noisy
      Experiments", ICML 2019 — explicit recommendation against binarizing
      the constraint outcome at fit time.
"""

from __future__ import annotations

import dataclasses

import torch

from bo_engine.suggestions import _build_outcome_constraint_models
from bo_engine.types import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    OutcomeConstraintSpec,
    ParameterSpec,
    ParameterType,
)

N_OBS = 8
N_DIMS = 2


def _make_spec(method: str, *, greater_than: bool) -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name=f"x{i}", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))
            for i in range(N_DIMS)
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
        outcome_constraints=[
            OutcomeConstraintSpec(
                objective_name="y",
                threshold=0.5,
                greater_than=greater_than,
                feasibility_threshold=0.5,
            )
        ],
        outcome_constraint_method=method,
    )


def _make_observations(
    values: list[float],
) -> tuple[list[ObservationData], torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    train_x = torch.rand(len(values), N_DIMS, dtype=torch.float64)
    observations = [
        ObservationData(
            parameter_values={f"x{i}": float(train_x[k, i].item()) for i in range(N_DIMS)},
            objective_values={"y": values[k]},
        )
        for k in range(len(values))
    ]
    bounds = torch.stack(
        [torch.zeros(N_DIMS, dtype=torch.float64), torch.ones(N_DIMS, dtype=torch.float64)]
    )
    return observations, train_x, bounds


class TestContinuousIsDefault:
    """Spec default must be continuous so unspecified callers get the better path."""

    def test_default_method_is_continuous(self) -> None:
        spec = OptimizationSpec(
            parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
            objectives=[ObjectiveSpec(name="y")],
        )
        assert spec.outcome_constraint_method == "continuous"


class TestContinuousModelFit:
    """Continuous path fits the raw objective values, not binary labels."""

    def test_continuous_model_recovers_signed_distance(self) -> None:
        values = [0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9, 1.0]
        spec = _make_spec("continuous", greater_than=True)
        observations, train_x, bounds = _make_observations(values)

        models = _build_outcome_constraint_models(spec, observations, train_x, bounds)

        assert models is not None
        assert len(models) == 1
        model, threshold = models[0]
        # ``greater_than=True`` keeps the raw scale; threshold is the spec value.
        assert threshold == 0.5
        # The trained model should predict near the raw objective values at the
        # training points (sanity check that we fit raw values, not 0/1 labels).
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(train_x)
            preds = posterior.mean.squeeze(-1).cpu().numpy()
        # The predictions span the raw value range, not [0, 1] feasibility-only.
        assert preds.min() < 0.4
        assert preds.max() > 0.6

    def test_continuous_less_than_flips_sign(self) -> None:
        """``<=`` constraints flip the GP targets so the callable contract is preserved.

        The continuous path encodes a ``<=`` constraint by training on
        ``-obj_tensor`` with ``threshold = -oc.threshold``. The downstream
        callable ``samples - threshold`` then evaluates ``-obj > -bound``
        i.e. ``obj < bound``; positive remains "feasible".
        """
        values = [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]
        spec = _make_spec("continuous", greater_than=False)
        observations, train_x, bounds = _make_observations(values)

        models = _build_outcome_constraint_models(spec, observations, train_x, bounds)

        assert models is not None
        model, threshold = models[0]
        # ``<=`` is encoded by negating both targets and threshold so the
        # downstream callable ``samples - threshold`` evaluates ``-obj > -bound``
        # i.e. ``obj < bound``.
        assert threshold == -0.5
        model.eval()
        with torch.no_grad():
            preds = model.posterior(train_x).mean.squeeze(-1).cpu().numpy()
        assert preds.max() < 0  # negated raw scale


class TestBinaryFallbackPreserved:
    """The binary path is still reachable for genuinely-binary outcomes."""

    def test_binary_path_returns_feasibility_threshold(self) -> None:
        values = [0.1, 0.2, 0.6, 0.7]
        spec = _make_spec("binary", greater_than=True)
        observations, train_x, bounds = _make_observations(values)

        models = _build_outcome_constraint_models(spec, observations, train_x, bounds)

        assert models is not None
        _model, threshold = models[0]
        # Binary path keeps the spec's feasibility_threshold (0.5), not the
        # objective bound.
        assert threshold == 0.5


class TestInvalidMethod:
    """Unknown methods raise the typed configuration error."""

    def test_invalid_method_rejected(self) -> None:
        import pytest

        from bo_engine.suggestions import OutcomeConstraintConfigurationError

        spec = _make_spec("continuous", greater_than=True)
        bad = dataclasses.replace(spec, outcome_constraint_method="nonsense")
        observations, train_x, bounds = _make_observations([0.1, 0.5, 0.9])

        with pytest.raises(OutcomeConstraintConfigurationError, match="outcome_constraint_method"):
            _build_outcome_constraint_models(bad, observations, train_x, bounds)
