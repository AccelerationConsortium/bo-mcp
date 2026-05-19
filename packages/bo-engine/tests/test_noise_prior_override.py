"""Tests for the configurable noise prior via ``OptimizationSpec`` (8.39c).

The default ``GammaPrior(NOISE_PRIOR_GAMMA_CONCENTRATION,
NOISE_PRIOR_GAMMA_RATE)`` is calibrated for unit-standardized targets. In
data-starved regimes (e.g. n=5 in 10D on a high-noise problem) the
inferred noise hyperparameter is miscalibrated and follows the prior
rather than the data. We expose ``OptimizationSpec.noise_prior_params``
so callers can pre-shape the prior toward an empirically known noise
floor instead of waiting for MLL to recover from the default.

References:
    - Rasmussen & Williams, "Gaussian Processes for Machine Learning",
      §5.4.1 — prior calibration in data-starved regimes.
"""

from __future__ import annotations

import pytest
import torch
from gpytorch.priors import GammaPrior

from bo_engine.suggestions import _resolve_noise_prior
from bo_engine.types import (
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)


def _base_spec(noise_prior_params: tuple[float, float] | None = None) -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
        noise_prior_params=noise_prior_params,
    )


class TestNoiseOverrideResolution:
    """``_resolve_noise_prior`` materializes the GammaPrior when params are set."""

    def test_default_returns_none(self) -> None:
        assert _resolve_noise_prior(_base_spec()) is None

    def test_override_returns_gamma_prior_with_matching_params(self) -> None:
        spec = _base_spec((2.5, 0.3))
        prior = _resolve_noise_prior(spec)
        assert isinstance(prior, GammaPrior)
        # GammaPrior records the concentration/rate; pull them via the
        # registered parameter values.
        assert float(prior.concentration.item()) == pytest.approx(2.5)
        assert float(prior.rate.item()) == pytest.approx(0.3)

    def test_non_positive_params_rejected(self) -> None:
        with pytest.raises(ValueError, match="noise_prior_params"):
            _resolve_noise_prior(_base_spec((0.0, 1.0)))
        with pytest.raises(ValueError, match="noise_prior_params"):
            _resolve_noise_prior(_base_spec((1.0, -0.5)))


class TestNoiseOverrideAppliedToModel:
    """The override flows through to the fitted GP likelihood."""

    def test_override_replaces_default_prior(self) -> None:
        from bo_engine.models import create_single_task_model

        torch.manual_seed(0)
        train_x = torch.rand(10, 1, dtype=torch.float64)
        train_y = torch.randn(10, 1, dtype=torch.float64)
        bounds = torch.stack(
            [torch.zeros(1, dtype=torch.float64), torch.ones(1, dtype=torch.float64)]
        )

        custom = GammaPrior(3.0, 0.1)
        model = create_single_task_model(train_x, train_y, bounds, noise_prior=custom)

        # The likelihood's prior is registered on the inner ``HomoskedasticNoise``
        # module as ``noise_prior``; the factory accepts a user prior and
        # installs it there.
        prior_module = model.likelihood.noise_covar.noise_prior  # ty: ignore[unresolved-attribute]
        assert isinstance(prior_module, GammaPrior)
        assert float(prior_module.concentration.item()) == pytest.approx(3.0)
        assert float(prior_module.rate.item()) == pytest.approx(0.1)
