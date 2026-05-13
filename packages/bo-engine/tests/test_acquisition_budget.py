"""Tests for dimension-adaptive acquisition-optimization budgets (TODO 1.43).

The L-BFGS-B multi-start used by ``optimize_acqf`` has a known failure mode in
high-dimensional and multimodal acquisition landscapes: a fixed restart count
(historical default 20) traps the optimizer in shallow local optima, so the
returned candidate looks plausible but is sub-optimal. Eriksson et al.
(SAASBO, UAI 2021; https://arxiv.org/abs/2103.00349) recommend scaling restart
count with problem dimensionality, and the BoTorch tutorial set
(https://botorch.org/tutorials/) follows the same growth pattern.

These tests pin the formula:

* ``num_restarts = NUM_RESTARTS_BASE + NUM_RESTARTS_PER_DIM * d`` (capped at
  ``NUM_RESTARTS_MAX``)
* ``raw_samples  = max(RAW_SAMPLES_MIN, RAW_SAMPLES_PER_DIM * d)`` (capped at
  ``RAW_SAMPLES_MAX``)

and verify that explicit overrides on the config and an explicit
``OptimizationSpec`` field both flow through the dispatcher.
"""

from __future__ import annotations

import torch

from bo_engine.acquisition import _resolve_restart_budget
from bo_engine.constants import (
    NUM_RESTARTS_BASE,
    NUM_RESTARTS_MAX,
    NUM_RESTARTS_PER_DIM,
    RAW_SAMPLES_MAX,
    RAW_SAMPLES_MIN,
    RAW_SAMPLES_PER_DIM,
)
from bo_engine.types import (
    AcquisitionOptimizationConfig,
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)


def _bounds(n_dims: int) -> torch.Tensor:
    return torch.stack(
        [
            torch.zeros(n_dims, dtype=torch.float64),
            torch.ones(n_dims, dtype=torch.float64),
        ]
    )


def _spec(n_dims: int, config: AcquisitionOptimizationConfig | None = None) -> OptimizationSpec:
    params = [
        ParameterSpec(name=f"x{i}", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))
        for i in range(n_dims)
    ]
    return OptimizationSpec(
        parameters=params,
        objectives=[ObjectiveSpec(name="y", minimize=True)],
        acquisition_optimization=config or AcquisitionOptimizationConfig(),
    )


class TestDimensionAdaptiveDefaults:
    """The default formula scales linearly with dimension up to the caps."""

    def test_low_dim_uses_base_restarts(self) -> None:
        """For ``d=2`` the formula gives ``base + 2 * per_dim`` restarts."""
        restarts, samples = _resolve_restart_budget(
            spec=_spec(2), bounds=_bounds(2), num_restarts=None, raw_samples=None
        )
        assert restarts == NUM_RESTARTS_BASE + 2 * NUM_RESTARTS_PER_DIM
        assert samples == RAW_SAMPLES_MIN  # floor at low d

    def test_high_dim_grows_raw_samples(self) -> None:
        """At d=30 the raw-sample budget grows past the floor."""
        d = 30
        restarts, samples = _resolve_restart_budget(
            spec=_spec(d), bounds=_bounds(d), num_restarts=None, raw_samples=None
        )
        assert restarts == NUM_RESTARTS_BASE + d * NUM_RESTARTS_PER_DIM
        assert samples == max(RAW_SAMPLES_MIN, d * RAW_SAMPLES_PER_DIM)

    def test_caps_prevent_blowup_in_very_high_dim(self) -> None:
        """The ``*_MAX`` constants clamp the formula for very large d."""
        d = 10_000
        restarts, samples = _resolve_restart_budget(
            spec=_spec(d), bounds=_bounds(d), num_restarts=None, raw_samples=None
        )
        assert restarts == NUM_RESTARTS_MAX
        assert samples == RAW_SAMPLES_MAX


class TestExplicitOverrides:
    """Spec-level and call-level overrides bypass the formula."""

    def test_spec_config_overrides_defaults(self) -> None:
        """Values on ``AcquisitionOptimizationConfig`` flow through."""
        config = AcquisitionOptimizationConfig(num_restarts=7, raw_samples=33)
        restarts, samples = _resolve_restart_budget(
            spec=_spec(5, config), bounds=_bounds(5), num_restarts=None, raw_samples=None
        )
        assert restarts == 7
        assert samples == 33

    def test_explicit_call_args_take_precedence(self) -> None:
        """Direct ``num_restarts`` / ``raw_samples`` win over the spec config."""
        config = AcquisitionOptimizationConfig(num_restarts=7, raw_samples=33)
        restarts, samples = _resolve_restart_budget(
            spec=_spec(5, config), bounds=_bounds(5), num_restarts=11, raw_samples=99
        )
        assert restarts == 11
        assert samples == 99

    def test_missing_spec_uses_dimension_default(self) -> None:
        """When ``spec`` is None the formula still applies based on bounds."""
        restarts, samples = _resolve_restart_budget(
            spec=None, bounds=_bounds(4), num_restarts=None, raw_samples=None
        )
        assert restarts == NUM_RESTARTS_BASE + 4 * NUM_RESTARTS_PER_DIM
        assert samples == max(RAW_SAMPLES_MIN, 4 * RAW_SAMPLES_PER_DIM)


class TestConfigResolveMonotonicity:
    """The formula must be monotone non-decreasing in dimension."""

    def test_restarts_non_decreasing(self) -> None:
        """``restarts(d+1) >= restarts(d)`` for every d up to the cap."""
        config = AcquisitionOptimizationConfig()
        last = -1
        for d in range(1, 1000):
            restarts, _ = config.resolve(d)
            assert restarts >= last
            last = restarts

    def test_samples_non_decreasing(self) -> None:
        """``raw_samples(d+1) >= raw_samples(d)`` for every d up to the cap."""
        config = AcquisitionOptimizationConfig()
        last = -1
        for d in range(1, 1000):
            _, samples = config.resolve(d)
            assert samples >= last
            last = samples
