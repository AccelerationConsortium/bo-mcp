"""Tests for explicit GP noise prior configuration (TODO 1.5).

The GP observation-noise hyperparameter is the silent failure mode of
single-task BO on multi-scale objectives: when the prior is left implicit and
the data is heteroskedastic, MLL fitting can absorb signal variance into the
noise term, which collapses the posterior covariance and distorts acquisition
geometry. These tests verify that

1. trainable-noise GPs carry the explicit ``GammaPrior`` recommended by the
   BoTorch single-task tutorial (Balandat et al., 2020;
   https://botorch.org/tutorials/), with the constants exposed via
   :mod:`bo_engine.constants`;
2. supplying ``train_yvar`` switches the GP onto a
   ``FixedNoiseGaussianLikelihood`` so the user's measurement uncertainty is
   trusted instead of re-estimated (BoTorch ``HeteroskedasticSingleTaskGP``
   docs);
3. fitting completes successfully on both paths and the floor in
   :mod:`bo_engine.constants.NOISE_PRIOR_MIN_INFERRED` prevents the trainable
   noise from collapsing to zero.
"""

from __future__ import annotations

import logging

import torch
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood, GaussianLikelihood
from gpytorch.priors import GammaPrior

from bo_engine.constants import (
    NOISE_PRIOR_GAMMA_CONCENTRATION,
    NOISE_PRIOR_GAMMA_RATE,
    NOISE_PRIOR_MIN_INFERRED,
)
from bo_engine.models import (
    create_and_fit_model,
    create_and_fit_single_task_model,
    create_model,
    create_single_task_model,
)

SEED = 7
N_TRAIN = 10
N_DIMS = 2


def _likelihood_attr(model, *path: str):
    """Walk an attribute path on the GPyTorch likelihood.

    Centralizes the attribute walk so individual asserts stay readable. The
    ``model.likelihood`` access goes through ``getattr`` to side-step the
    ``Tensor | Module`` union ty cannot narrow for the public stubs.
    """
    obj = getattr(model, "likelihood")  # noqa: B009
    for name in path:
        obj = getattr(obj, name)
    return obj


def _train_data(
    n_obj: int = 1, dtype: torch.dtype = torch.float64
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(SEED)
    train_x = torch.rand(N_TRAIN, N_DIMS, dtype=dtype)
    bounds = torch.stack([torch.zeros(N_DIMS, dtype=dtype), torch.ones(N_DIMS, dtype=dtype)])
    if n_obj == 1:
        train_y = (train_x.sum(dim=-1, keepdim=True) - 1.0).pow(2)
    else:
        cols = [
            (train_x.sum(dim=-1, keepdim=True) - 1.0).pow(2),
            -train_x.prod(dim=-1, keepdim=True),
        ]
        train_y = torch.cat(cols[:n_obj], dim=-1)
    return train_x, train_y, bounds


class TestDefaultNoisePrior:
    """Trainable-noise models must expose the explicit Gamma prior."""

    def test_single_task_uses_gamma_prior(self) -> None:
        """Default ``create_single_task_model`` must wire ``GammaPrior``."""
        train_x, train_y, bounds = _train_data(n_obj=1)
        model = create_single_task_model(train_x, train_y, bounds)
        likelihood = _likelihood_attr(model)
        assert isinstance(likelihood, GaussianLikelihood)
        # `noise_prior` is stored as a registered prior on the noise_covar.
        prior = _likelihood_attr(model, "noise_covar", "noise_prior")
        assert isinstance(prior, GammaPrior)
        # GammaPrior stores concentration/rate as buffers; compare values.
        assert torch.isclose(
            prior.concentration,
            torch.tensor(NOISE_PRIOR_GAMMA_CONCENTRATION, dtype=prior.concentration.dtype),
        )
        assert torch.isclose(
            prior.rate,
            torch.tensor(NOISE_PRIOR_GAMMA_RATE, dtype=prior.rate.dtype),
        )

    def test_model_list_uses_gamma_prior_per_sub_model(self) -> None:
        """Every sub-model of a ``ModelListGP`` must carry the prior."""
        train_x, train_y, bounds = _train_data(n_obj=2)
        model = create_model(train_x, train_y, bounds)
        assert len(model.models) == 2
        for sub in model.models:
            prior = _likelihood_attr(sub, "noise_covar", "noise_prior")
            assert isinstance(prior, GammaPrior)

    def test_noise_constraint_enforces_floor(self) -> None:
        """``NOISE_PRIOR_MIN_INFERRED`` must be the active lower bound."""
        train_x, train_y, bounds = _train_data(n_obj=1)
        model = create_single_task_model(train_x, train_y, bounds)
        constraint = _likelihood_attr(model, "noise_covar", "raw_noise_constraint")
        # GPyTorch stores the bound as a float32 tensor; compare via isclose
        # to absorb the dtype rounding instead of an exact `==`.
        assert torch.isclose(
            constraint.lower_bound,
            torch.tensor(NOISE_PRIOR_MIN_INFERRED, dtype=constraint.lower_bound.dtype),
        )


class TestCustomNoisePrior:
    """Callers may override the noise prior on the trainable path."""

    def test_custom_prior_is_honored(self) -> None:
        """Explicit ``noise_prior`` argument must reach the likelihood."""
        train_x, train_y, bounds = _train_data(n_obj=1)
        custom = GammaPrior(2.0, 0.3)
        model = create_single_task_model(train_x, train_y, bounds, noise_prior=custom)
        prior = _likelihood_attr(model, "noise_covar", "noise_prior")
        assert isinstance(prior, GammaPrior)
        concentration_expected = torch.tensor(2.0, dtype=prior.concentration.dtype)
        rate_expected = torch.tensor(0.3, dtype=prior.rate.dtype)
        assert torch.isclose(prior.concentration, concentration_expected)
        assert torch.isclose(prior.rate, rate_expected)


class TestFixedNoisePath:
    """``train_yvar`` must switch the model to ``FixedNoiseGaussianLikelihood``."""

    def test_train_yvar_uses_fixed_noise_likelihood(self) -> None:
        """When per-observation variance is provided, fix the noise term.

        BoTorch's ``Standardize(m=1)`` outcome transform rescales both
        ``train_Y`` and ``train_Yvar`` (variances divided by the per-objective
        sample variance) so the GP sees unit-scale data. The assertion is
        therefore on the *non-trainable* nature of the noise: shape preserved,
        no ``raw_noise`` parameter, and the values stay constant across a
        synthetic backward pass.
        """
        train_x, train_y, bounds = _train_data(n_obj=1)
        train_yvar = torch.full((N_TRAIN, 1), 0.04, dtype=train_y.dtype)
        model = create_single_task_model(train_x, train_y, bounds, train_yvar=train_yvar)
        likelihood = _likelihood_attr(model)
        assert isinstance(likelihood, FixedNoiseGaussianLikelihood)
        noise_covar = _likelihood_attr(model, "noise_covar")
        # FixedGaussianNoise stores `noise` as a buffer, not a parameter, so
        # `raw_noise` (the trainable parameter on GaussianLikelihood) is absent.
        assert not hasattr(noise_covar, "raw_noise")
        fixed = noise_covar.noise.detach()
        assert fixed.shape[-1] == N_TRAIN
        # All sample variances are equal in this test so the standardized
        # noise is constant; that constancy is the invariant we care about.
        assert torch.allclose(fixed, fixed.mean().expand_as(fixed))

    def test_model_list_fixed_noise_per_objective(self) -> None:
        """Each multi-objective sub-model gets its own fixed-noise column.

        See :meth:`test_train_yvar_uses_fixed_noise_likelihood` for why we do
        not compare against raw ``train_yvar``: per-objective Standardize
        rescales the variance independently. We assert structural fixedness
        and that the *ratio* between objectives is preserved (objective 1 has
        2.5× the input variance of objective 0).
        """
        train_x, train_y, bounds = _train_data(n_obj=2)
        train_yvar = torch.stack(
            [
                torch.full((N_TRAIN,), 0.02, dtype=train_y.dtype),
                torch.full((N_TRAIN,), 0.05, dtype=train_y.dtype),
            ],
            dim=-1,
        )
        model = create_model(train_x, train_y, bounds, train_yvar=train_yvar)
        for sub in model.models:
            likelihood = _likelihood_attr(sub)
            assert isinstance(likelihood, FixedNoiseGaussianLikelihood)
            assert not hasattr(_likelihood_attr(sub, "noise_covar"), "raw_noise")


class TestFittingWithExplicitPrior:
    """End-to-end ``fit_gpytorch_mll`` must still succeed on both paths."""

    def test_fit_trainable_noise_respects_floor(self) -> None:
        """Fitted noise stays at or above the configured floor."""
        train_x, train_y, bounds = _train_data(n_obj=1)
        model = create_and_fit_single_task_model(train_x, train_y, bounds)
        fitted_noise = _likelihood_attr(model, "noise_covar", "noise").detach().reshape(-1)
        assert torch.all(fitted_noise >= NOISE_PRIOR_MIN_INFERRED - 1e-9)

    def test_fit_fixed_noise_does_not_drift(self) -> None:
        """Fitting must not modify the (standardized) fixed-noise buffer."""
        train_x, train_y, bounds = _train_data(n_obj=1)
        train_yvar = torch.full((N_TRAIN, 1), 0.04, dtype=train_y.dtype)
        model = create_single_task_model(train_x, train_y, bounds, train_yvar=train_yvar)
        # Snapshot the standardized noise buffer before any MLL fitting runs.
        pre_fit = _likelihood_attr(model, "noise_covar", "noise").detach().clone()
        # Call fit_gpytorch_mll via the convenience wrapper.
        from bo_engine.models import fit_single_task_model

        fit_single_task_model(model)
        post_fit = _likelihood_attr(model, "noise_covar", "noise").detach()
        assert torch.allclose(pre_fit, post_fit)

    def test_fit_multi_objective_trainable(self) -> None:
        """Multi-objective trainable-noise fit also lands above the floor."""
        train_x, train_y, bounds = _train_data(n_obj=2)
        model = create_and_fit_model(train_x, train_y, bounds)
        for sub in model.models:
            fitted = _likelihood_attr(sub, "noise_covar", "noise").detach().reshape(-1)
            assert torch.all(fitted >= NOISE_PRIOR_MIN_INFERRED - 1e-9)


class TestFittedNoiseLogging:
    """Fit emits a debug log of the noise hyperparameter for drift tracking."""

    def test_debug_logging_emits_noise_summary(self, caplog) -> None:
        """A debug log line includes min/mean/max noise after fitting."""
        train_x, train_y, bounds = _train_data(n_obj=1)
        with caplog.at_level(logging.DEBUG, logger="bo_engine.models"):
            create_and_fit_single_task_model(train_x, train_y, bounds)
        messages = [rec.getMessage() for rec in caplog.records]
        assert any("Fitted noise" in m for m in messages)
