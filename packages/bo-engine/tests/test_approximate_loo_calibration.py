"""Validate the single-fit LOO path against the exact ``batch_cross_validation``.

``compute_loo_cv_optimized`` routes to ``_compute_approximate_loo`` once
the dataset crosses ``CVConfig.approximate_threshold`` (default 100). That
path fits one GP and computes the exact LOO predictive moments from the
inverse train covariance (Rasmussen & Williams "GPML" §5.4.2,
Eqs. 5.10-5.12) instead of the ``n`` model refits the exact path requires
— O(N³) instead of O(N⁴). The downdate is exact at fixed hyperparameters,
so the only remaining discrepancy against ``batch_cross_validation`` is
the per-fold re-estimation of hyperparameters, which is small once each
fold retains n-1 ≈ n points. We pin that agreement near the routing
threshold, and pin the y-scale invariance that the standardized-space
computation guarantees.

References:
    - Sundararajan & Keerthi, "Predictive Approaches for Choosing
      Hyperparameters in Gaussian Processes", Neural Computation 2001 —
      LOO-CV with GP hyperparameters held fixed across folds.
    - Rasmussen & Williams "Gaussian Processes for Machine Learning",
      §5.4.2 — exact LOO posterior via partitioned-matrix inversion.
"""

from __future__ import annotations

import math

import pytest
import torch

from bo_engine.cross_validation import (
    _compute_approximate_loo,
    _compute_batch_loo_cv,
)

pytestmark = pytest.mark.slow

# Hyperparameters are re-estimated per fold on the exact path but reused
# from the full fit on the downdate path; with n-1 ≈ n points per fold the
# measured relative RMSE gap on this synthetic is 0.2-0.6% across all
# sizes, so 5% detects any real regression with an order-of-magnitude
# margin while staying immune to fit jitter.
RELATIVE_RMSE_TOLERANCE = 0.05
COVERAGE_TOLERANCE = 0.05


def _synthetic_dataset(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    d = 2
    train_x = torch.rand(n, d, dtype=torch.float64)
    base_y = torch.sin(2 * train_x[:, 0]) + 0.5 * train_x[:, 1] ** 2
    noise = 0.05 * torch.randn(n, dtype=torch.float64)
    train_y = (base_y + noise).unsqueeze(-1)
    bounds = torch.stack([torch.zeros(d, dtype=torch.float64), torch.ones(d, dtype=torch.float64)])
    return train_x, train_y, bounds


@pytest.mark.parametrize("n_samples", [80, 100, 120, 200])
def test_approximate_loo_matches_exact_within_tolerance(n_samples: int) -> None:
    """RMSE and coverage of the downdate LOO must agree with refit LOO.

    Both paths predict held-out *observations* (predictive moments include
    observation noise), so their coverage estimates must agree as well —
    a units mismatch between the two paths would show up here first.
    """
    train_x, train_y, bounds = _synthetic_dataset(n_samples, seed=11)

    exact = _compute_batch_loo_cv(train_x, train_y, bounds)
    approx = _compute_approximate_loo(train_x, train_y, bounds)

    # Both methods must return numeric (non-NaN) RMSE on the smoothly-noisy
    # synthetic — a NaN here would indicate the GP fit collapsed and the
    # underlying assumption (Cholesky factor well-conditioned) failed.
    assert not math.isnan(exact.rmse)
    assert not math.isnan(approx.rmse)

    rel_err = abs(approx.rmse - exact.rmse) / max(exact.rmse, 1e-6)
    assert rel_err <= RELATIVE_RMSE_TOLERANCE, (
        f"Downdate LOO RMSE diverged from refit LOO at n={n_samples}: "
        f"approx={approx.rmse:.4f}, exact={exact.rmse:.4f}, "
        f"rel_err={rel_err:.4f}. The exact downdate should track the "
        "refit path to well under a percent on this synthetic."
    )

    assert abs(approx.coverage_95 - exact.coverage_95) <= COVERAGE_TOLERANCE, (
        f"95% coverage disagrees between LOO paths at n={n_samples}: "
        f"approx={approx.coverage_95:.3f}, exact={exact.coverage_95:.3f}. "
        "Both must score predictive intervals for held-out observations."
    )


def test_approximate_loo_is_scale_invariant() -> None:
    """Rescaling y by 1000 must not change R², coverage or z-scores.

    The downdate operates on standardized targets, so all scale-free
    metrics are invariant and RMSE/MAE scale exactly with y. The historic
    failure mode was mixing original-scale posterior variance with
    standardized-scale likelihood noise, which made the reported metrics
    depend qualitatively on the units of y.
    """
    n_samples = 120  # above the routing threshold, the regime that matters
    train_x, train_y, bounds = _synthetic_dataset(n_samples, seed=11)

    base = _compute_approximate_loo(train_x, train_y, bounds)
    scaled = _compute_approximate_loo(train_x, 1000.0 * train_y, bounds)

    assert scaled.r_squared == pytest.approx(base.r_squared, abs=1e-6)
    assert scaled.rmse == pytest.approx(1000.0 * base.rmse, rel=1e-6)
    assert scaled.mae == pytest.approx(1000.0 * base.mae, rel=1e-6)
    assert scaled.mean_standardized_error == pytest.approx(base.mean_standardized_error, rel=1e-6)
    assert scaled.coverage_95 == pytest.approx(base.coverage_95, abs=1e-9)
