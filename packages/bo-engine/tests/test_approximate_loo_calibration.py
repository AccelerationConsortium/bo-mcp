"""Validate the approximate LOO path against the exact ``batch_cross_validation``.

``compute_loo_cv_optimized`` routes to ``_compute_approximate_loo`` once
the dataset crosses ``CVConfig.approximate_threshold`` (default 100). The
approximate path uses the PRESS-residual identity to avoid the ``n`` model
refits that the exact path requires — it is O(N³) instead of O(N⁴) but
relies on the assumption that the GP's leverage approximation is
accurate. We never validated that assumption near the routing threshold;
this test pins it.

The test compares the approximate LOO RMSE against the exact
``batch_cross_validation`` LOO RMSE at sizes ``80, 100, 120, 200`` on the
default ``SingleTaskGP`` with ``Normalize`` / ``Standardize`` (the same
construction ``_compute_approximate_loo`` builds internally) and asserts
agreement within a documented tolerance.

References:
    - Sundararajan & Keerthi, "Predictive Approaches for Choosing
      Hyperparameters in Gaussian Processes", Neural Computation 2001 —
      derives the PRESS-residual identity used by the approximate path.
    - Rasmussen & Williams "Gaussian Processes for Machine Learning",
      §5.4.2 — exact LOO posterior via partitioned-matrix inversion.
"""

from __future__ import annotations

import pytest
import torch

from bo_engine.cross_validation import (
    CVConfig,
    _compute_approximate_loo,
    _compute_batch_loo_cv,
)

pytestmark = pytest.mark.slow


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
    """RMSE of the approximate LOO must agree with the exact LOO within tolerance.

    The PRESS residual approximation can drift from the exact LOO when
    individual training points are highly influential (small effective
    leverage). We bound the agreement at 25 % relative RMSE which is the
    documented tolerance for the calibration test; tighter bounds would
    spuriously fail on the smoothly-noisy synthetic. The point of the test
    is to detect *large* divergences that would indicate a bug in the
    PRESS path or a torch-version change that broke the leverage
    extraction.
    """
    train_x, train_y, bounds = _synthetic_dataset(n_samples, seed=11)

    exact = _compute_batch_loo_cv(train_x, train_y)
    approx = _compute_approximate_loo(train_x, train_y, bounds)

    # Both methods must return numeric (non-NaN) RMSE on the smoothly-noisy
    # synthetic — a NaN here would indicate the GP fit collapsed and the
    # underlying assumption (Cholesky factor well-conditioned) failed.
    assert exact.rmse == exact.rmse  # not NaN
    assert approx.rmse == approx.rmse  # not NaN

    # The approximate RMSE should be within 25% of the exact RMSE.
    rel_err = abs(approx.rmse - exact.rmse) / max(exact.rmse, 1e-6)
    assert rel_err <= 0.25, (
        f"Approximate LOO RMSE diverged from exact LOO at n={n_samples}: "
        f"approx={approx.rmse:.4f}, exact={exact.rmse:.4f}, "
        f"rel_err={rel_err:.4f}. Either the PRESS approximation broke "
        "under a torch update or the calibration threshold "
        f"({CVConfig().approximate_threshold}) needs revisiting."
    )
