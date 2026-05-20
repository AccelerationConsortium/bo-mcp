"""Tests for end-to-end measurement-uncertainty routing into the GP.

The fix added a ``train_yvar`` parameter to
``create_single_task_model`` / ``create_model``, but that signature alone is
not enough -- callers must also build the variance tensor from each
``ObservationData``'s per-objective standard-deviation map and pass it into
the model factory. The helper under test
(:func:`bo_engine.suggestions._prepare_train_yvar`) is the bridge that
converts user-supplied stddev to BoTorch's variance representation, and the
``generate_next_batch`` integration smoke-test confirms the fitted GP
actually uses a fixed-noise likelihood when full coverage is provided.

Reference:
    BoTorch handles known measurement variance via
    ``SingleTaskGP(..., train_Yvar=...)`` which constructs a
    ``FixedNoiseGaussianLikelihood`` internally (BoTorch tutorials,
    https://botorch.org/tutorials/). The variance must be in the same
    objective-column order as ``train_Y``.
"""

from __future__ import annotations

import math

import torch
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood

from bo_engine.suggestions import _prepare_train_yvar, generate_next_batch
from bo_engine.types import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)


def _single_obj_spec() -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
        random_seed=0,
        initial_design_size=2,
    )


def _multi_obj_spec() -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[
            ObjectiveSpec(name="y1", minimize=True),
            ObjectiveSpec(name="y2", minimize=False),
        ],
        random_seed=0,
        initial_design_size=2,
    )


class TestPrepareTrainYvar:
    """``_prepare_train_yvar`` translates per-result stddev to a variance tensor."""

    def test_full_coverage_returns_variance_tensor(self) -> None:
        """All-objectives-known builds an ``(n_obs, n_obj)`` variance tensor."""
        spec = _multi_obj_spec()
        observations = [
            ObservationData(
                parameter_values={"x": 0.2},
                objective_values={"y1": 1.0, "y2": 5.0},
                measurement_uncertainty={"y1": 0.1, "y2": 0.5},
            ),
            ObservationData(
                parameter_values={"x": 0.7},
                objective_values={"y1": 2.0, "y2": 6.0},
                measurement_uncertainty={"y1": 0.2, "y2": 0.4},
            ),
        ]
        yvar = _prepare_train_yvar(observations, spec)
        assert yvar is not None
        assert yvar.shape == (2, 2)
        # Stored as variance == stddev ** 2, in objective order (y1, y2).
        assert math.isclose(float(yvar[0, 0]), 0.1**2, rel_tol=1e-9)
        assert math.isclose(float(yvar[0, 1]), 0.5**2, rel_tol=1e-9)
        assert math.isclose(float(yvar[1, 0]), 0.2**2, rel_tol=1e-9)
        assert math.isclose(float(yvar[1, 1]), 0.4**2, rel_tol=1e-9)

    def test_objective_column_order_matches_spec(self) -> None:
        """The columns follow ``spec.objectives`` order, not dict insertion."""
        spec = _multi_obj_spec()  # objectives ordered as (y1, y2)
        observations = [
            ObservationData(
                parameter_values={"x": 0.1},
                objective_values={"y2": 9.0, "y1": 1.0},  # reversed insertion
                measurement_uncertainty={"y2": 0.7, "y1": 0.3},
            ),
        ]
        yvar = _prepare_train_yvar(observations, spec)
        assert yvar is not None
        # Column 0 must be y1's variance, not y2's.
        assert math.isclose(float(yvar[0, 0]), 0.3**2, rel_tol=1e-9)
        assert math.isclose(float(yvar[0, 1]), 0.7**2, rel_tol=1e-9)

    def test_partial_coverage_returns_none(self) -> None:
        """A single missing observation reverts the campaign to trainable noise."""
        spec = _single_obj_spec()
        observations = [
            ObservationData(
                parameter_values={"x": 0.2},
                objective_values={"y": 1.0},
                measurement_uncertainty={"y": 0.1},
            ),
            ObservationData(
                parameter_values={"x": 0.7},
                objective_values={"y": 2.0},
                measurement_uncertainty=None,  # missing
            ),
        ]
        assert _prepare_train_yvar(observations, spec) is None

    def test_missing_objective_key_returns_none(self) -> None:
        """An uncertainty dict without one of the objective keys -> ``None``."""
        spec = _multi_obj_spec()
        observations = [
            ObservationData(
                parameter_values={"x": 0.2},
                objective_values={"y1": 1.0, "y2": 5.0},
                measurement_uncertainty={"y1": 0.1},  # y2 missing
            ),
        ]
        assert _prepare_train_yvar(observations, spec) is None

    def test_empty_observations_returns_none(self) -> None:
        """No observations -> ``None`` (no model is built)."""
        assert _prepare_train_yvar([], _single_obj_spec()) is None


class TestGenerateNextBatchUsesFixedNoise:
    """``generate_next_batch`` must propagate ``train_yvar`` to the GP factory.

    Smoke-test the full pipeline by intercepting the GP factory and checking
    the constructed model's likelihood. This catches the regression where
    every layer of plumbing exists but a call site silently drops the
    keyword.
    """

    def _force_bo_path(self) -> tuple[OptimizationSpec, list[ObservationData]]:
        """Build a campaign past the initial-design threshold."""
        spec = _single_obj_spec()
        # Single parameter -> ``min_data = max(2, n_params+1) = 2``; supply 3
        # observations so ``generate_next_batch`` enters the BO branch.
        observations = [
            ObservationData(
                parameter_values={"x": 0.1},
                objective_values={"y": 0.5},
                measurement_uncertainty={"y": 0.1},
            ),
            ObservationData(
                parameter_values={"x": 0.5},
                objective_values={"y": 0.3},
                measurement_uncertainty={"y": 0.05},
            ),
            ObservationData(
                parameter_values={"x": 0.9},
                objective_values={"y": 0.7},
                measurement_uncertainty={"y": 0.2},
            ),
        ]
        return spec, observations

    def test_uncertainty_routes_to_fixed_noise_likelihood(self, monkeypatch) -> None:
        """Full coverage flips the constructed GP to ``FixedNoiseGaussianLikelihood``."""
        # The single-objective dispatch lives in
        # :mod:`bo_engine.suggestions_single_objective` after the
        # suggestions god-module split; patch where the factory is
        # looked up at call time.
        from bo_engine import suggestions_single_objective as so_mod

        spec, observations = self._force_bo_path()
        captured: dict[str, object] = {}
        real_factory = so_mod.create_and_fit_single_task_model

        def spy(train_x, train_y, bounds, **kwargs):
            captured["train_yvar"] = kwargs.get("train_yvar")
            model = real_factory(train_x, train_y, bounds, **kwargs)
            captured["likelihood"] = model.likelihood
            return model

        monkeypatch.setattr(so_mod, "create_and_fit_single_task_model", spy)

        suggestions, _ = generate_next_batch(spec, observations, batch_size=1, iteration=1)
        assert suggestions, "should have produced at least one suggestion"
        train_yvar = captured["train_yvar"]
        assert torch.is_tensor(train_yvar)
        assert isinstance(captured["likelihood"], FixedNoiseGaussianLikelihood)

    def test_partial_coverage_keeps_trainable_noise(self, monkeypatch) -> None:
        """Missing uncertainty on any observation -> trainable noise path."""
        from bo_engine import suggestions_single_objective as so_mod

        spec, observations = self._force_bo_path()
        # Drop uncertainty from the middle observation.
        observations[1] = ObservationData(
            parameter_values=observations[1].parameter_values,
            objective_values=observations[1].objective_values,
            measurement_uncertainty=None,
        )

        captured: dict[str, object] = {}
        real_factory = so_mod.create_and_fit_single_task_model

        def spy(train_x, train_y, bounds, **kwargs):
            captured["train_yvar"] = kwargs.get("train_yvar")
            model = real_factory(train_x, train_y, bounds, **kwargs)
            captured["likelihood"] = model.likelihood
            return model

        monkeypatch.setattr(so_mod, "create_and_fit_single_task_model", spy)

        generate_next_batch(spec, observations, batch_size=1, iteration=1)
        assert captured["train_yvar"] is None
        assert isinstance(captured["likelihood"], GaussianLikelihood)
        assert not isinstance(captured["likelihood"], FixedNoiseGaussianLikelihood)
