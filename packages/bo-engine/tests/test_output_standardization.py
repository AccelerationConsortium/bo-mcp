"""Tests for output standardization.

These tests guard the convention documented in
``bo_engine.models``: every GP we build attaches ``Standardize(m=1)`` as an
outcome transform, and BoTorch is trusted to

1. standardize ``train_Y`` during ``SingleTaskGP.__init__`` so
   ``fit_gpytorch_mll`` fits on unit-scale targets (verified by
   :func:`bo_engine.models.verify_standardization`);
2. untransform the posterior on inference so callers see the original scale.

References:
    - BoTorch ``Standardize`` tutorial and API notes:
      https://botorch.org/docs/models#outcome-transforms
    - Eriksson et al. 2019, "Scalable Global Optimization via Local Bayesian
      Optimization" (TuRBO §3) -- relies on unit-scale targets so the
      ``success_tolerance`` / ``length_min`` heuristics are meaningful.
    - Balandat et al. 2020, "BoTorch: A Framework for Efficient Monte-Carlo
      Bayesian Optimization", §4 -- outcome transforms make acquisition
      functions invariant to affine rescaling of objectives.
"""

from __future__ import annotations

import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize

from bo_engine.models import (
    _build_likelihood,
    create_and_fit_model,
    create_and_fit_single_task_model,
    fit_single_task_model,
    verify_standardization,
)

SEED = 12345
N_TRAIN = 12
N_DIMS = 3


def _make_training_data(
    scale: float = 1.0,
    offset: float = 0.0,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic quadratic-ish training data for GP fitting."""
    torch.manual_seed(SEED)
    train_x = torch.rand(N_TRAIN, N_DIMS, dtype=dtype)
    # Smooth non-trivial function so the GP has signal to fit.
    base_y = torch.sin(3.0 * train_x[:, 0]) + 0.5 * train_x[:, 1] ** 2 - 0.3 * train_x[:, 2]
    train_y = (scale * base_y + offset).unsqueeze(-1)
    bounds = torch.stack([torch.zeros(N_DIMS, dtype=dtype), torch.ones(N_DIMS, dtype=dtype)])
    return train_x, train_y, bounds


class TestVerifyStandardization:
    """BoTorch must internalize ``Standardize(m=1)`` during model construction."""

    def test_fitted_single_task_gp_has_unit_scale_targets(self) -> None:
        """``fit_gpytorch_mll`` sees zero-mean, unit-variance targets."""
        train_x, train_y, bounds = _make_training_data(scale=100.0, offset=500.0)
        model = create_and_fit_single_task_model(train_x, train_y, bounds)

        reports = verify_standardization(model)

        assert len(reports) == 1
        report = reports[0]
        assert report["standardized"] == pytest.approx(1.0)
        assert report["mean"] == pytest.approx(0.0, abs=1e-5)
        assert report["var"] == pytest.approx(1.0, abs=1e-4)

    def test_model_list_gp_standardizes_each_objective_independently(self) -> None:
        """Objectives with wildly different magnitudes are each unit-scaled."""
        torch.manual_seed(SEED)
        train_x = torch.rand(N_TRAIN, N_DIMS, dtype=torch.float64)
        # Two objectives on very different scales (yield ~0.5, throughput ~500).
        y0 = 0.5 * torch.sin(2.0 * train_x[:, 0]) + 0.2
        y1 = 500.0 * torch.sin(2.0 * train_x[:, 0]) + 200.0
        train_y = torch.stack([y0, y1], dim=-1)
        bounds = torch.stack(
            [torch.zeros(N_DIMS, dtype=torch.float64), torch.ones(N_DIMS, dtype=torch.float64)]
        )

        model = create_and_fit_model(train_x, train_y, bounds)
        reports = verify_standardization(model)

        assert len(reports) == 2
        for report in reports:
            assert report["standardized"] == pytest.approx(1.0)

    def test_verify_standardization_flags_missing_transform(self) -> None:
        """A GP fit without the transform must fail the verification."""
        from botorch.exceptions.warnings import InputDataWarning

        train_x, train_y, bounds = _make_training_data(scale=10.0, offset=50.0)
        # Raw SingleTaskGP with outcome_transform=None -> stored targets are raw.
        with pytest.warns(InputDataWarning):
            raw_model = SingleTaskGP(
                train_X=train_x,
                train_Y=train_y,
                input_transform=Normalize(d=N_DIMS, bounds=bounds),
                outcome_transform=None,
            )
        fit_single_task_model(raw_model)

        reports = verify_standardization(raw_model)

        assert reports[0]["standardized"] == pytest.approx(0.0)


class TestPosteriorMatchesManualStandardization:
    """Posterior must match a manually-standardized baseline in original scale."""

    def test_posterior_agrees_with_manual_standardization_baseline(self) -> None:
        """``outcome_transform=Standardize`` == manual-standardize + no transform.

        We fit two GPs on the same data: (a) our production model with
        ``Standardize(m=1)``, (b) a baseline where we standardize ``train_Y``
        manually and pass ``outcome_transform=None``. Both models are fit
        from the same torch seed, so hyperparameter optimization sees the
        same initialization. The posterior in original scale must match.
        """
        train_x, train_y, bounds = _make_training_data(scale=10.0, offset=3.0)

        # Production path: Standardize outcome transform.
        torch.manual_seed(SEED)
        prod_model = create_and_fit_single_task_model(train_x, train_y, bounds)

        # Manual baseline: standardize train_Y ourselves, no outcome_transform.
        # Use sample stdv (unbiased=True) to match BoTorch's ``nanstd`` --
        # otherwise the baseline targets have std sqrt(n/(n-1)) and the GP
        # would learn a different outputscale.
        mean = train_y.mean(dim=-2, keepdim=True)
        std = train_y.std(dim=-2, unbiased=True, keepdim=True)
        train_y_std = (train_y - mean) / std

        torch.manual_seed(SEED)
        baseline_model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y_std,
            input_transform=Normalize(d=N_DIMS, bounds=bounds),
            outcome_transform=None,
            # Match the production likelihood (explicit GammaPrior + GreaterThan
            # constraint) so this self-consistency check compares the
            # outcome-transform behaviour only, not differences in noise priors.
            likelihood=_build_likelihood(None),
        )
        fit_single_task_model(baseline_model)

        # Evaluate both at a shared test grid.
        test_x = torch.rand(20, N_DIMS, dtype=torch.float64)
        prod_model.eval()
        baseline_model.eval()
        with torch.no_grad():
            prod_post = prod_model.posterior(test_x)
            prod_mean = prod_post.mean.squeeze(-1)
            prod_std = prod_post.variance.sqrt().squeeze(-1)

            base_post = baseline_model.posterior(test_x)
            # Untransform the baseline's (standardized-space) posterior back to
            # the user scale -- this is exactly what Standardize does inside
            # ``prod_model.posterior``.
            base_mean_raw = base_post.mean.squeeze(-1) * std.squeeze() + mean.squeeze()
            base_std_raw = base_post.variance.sqrt().squeeze(-1) * std.squeeze()

        # Tight tolerance: this is a self-consistency check, not a
        # statistical claim about stochastic fitting.
        torch.testing.assert_close(prod_mean, base_mean_raw, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(prod_std, base_std_raw, rtol=1e-3, atol=1e-3)


class TestAcquisitionScaleInvariance:
    """Affine rescaling of train_y must not change the optimal candidate.

    This is the scale-invariance property that motivates output standardization:
    the user can measure throughput in items/hour or items/second, and BO must
    produce the same experiment queue.
    """

    def _fit_and_best_acquisition_point(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        bounds: torch.Tensor,
    ) -> torch.Tensor:
        """Fit GP + evaluate qLogEI on a fixed grid; return argmax location."""
        from botorch.acquisition.logei import qLogExpectedImprovement

        torch.manual_seed(SEED)
        model = create_and_fit_single_task_model(train_x, train_y, bounds)
        # best_f in *raw* scale because posterior is untransformed back.
        best_f = float(train_y.max().item())
        acqf = qLogExpectedImprovement(model=model, best_f=best_f)

        # Deterministic evaluation grid.
        torch.manual_seed(SEED + 1)
        grid = torch.rand(64, 1, N_DIMS, dtype=torch.float64)
        with torch.no_grad():
            values = acqf(grid)
        best_idx = int(torch.argmax(values).item())
        return grid[best_idx, 0]

    def test_scale_invariance_under_affine_target_rescaling(self) -> None:
        """Same argmax when we rescale y by (a, b) with a > 0."""
        train_x, train_y, bounds = _make_training_data(scale=1.0, offset=0.0)

        # Reference problem in unit scale.
        x_best_ref = self._fit_and_best_acquisition_point(train_x, train_y, bounds)

        # Affine rescaling -- positive slope preserves argmax of
        # expected improvement under the Standardize convention.
        scale = 137.0
        offset = -42.5
        train_y_rescaled = scale * train_y + offset
        x_best_scaled = self._fit_and_best_acquisition_point(train_x, train_y_rescaled, bounds)

        # Standardize collapses both problems to the same unit-scale GP, so
        # the argmax on a shared grid must be identical up to float noise.
        torch.testing.assert_close(x_best_ref, x_best_scaled, rtol=1e-6, atol=1e-6)
