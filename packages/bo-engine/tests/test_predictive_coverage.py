"""Predictive-interval coverage tests for the trust layer.

Calibration scores, prediction intervals and LOO diagnostics all compare
model intervals against *noisy observations*, so the intervals must come
from the posterior predictive (latent function plus observation noise) —
R&W GPML §2.2, Eq. 2.24. Latent-only intervals systematically under-cover
noisy data: on the σ=0.3 dataset used below, latent 95% intervals cover
only ~60% of observations, while the predictive intervals land at the
nominal rate. The mechanism pins assert the predictive construction
directly; the nightly test asserts the resulting statistical guarantee
(Kuleshov et al., ICML 2018: observed coverage must match expected
coverage for trustworthy uncertainty).

References:
    - Rasmussen & Williams "GPML" §2.2, Eq. 2.24
    - Kuleshov et al., "Accurate Uncertainties for Deep Learning Using
      Calibrated Regression", ICML 2018
"""

from __future__ import annotations

import pytest
import torch

from bo_engine.calibration import compute_calibration_score, compute_loo_calibration
from bo_engine.constants import CI_95_Z_SCORE
from bo_engine.models import create_and_fit_single_task_model
from bo_engine.prediction_intervals import compute_prediction_intervals

BOUNDS_1D = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
NOISE_STD = 0.3


def _noisy_dataset(n: int = 40, seed: int = 11) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    train_x = torch.rand(n, 1, dtype=torch.float64)
    train_y = torch.sin(4 * train_x) + NOISE_STD * torch.randn(n, 1, dtype=torch.float64)
    return train_x, train_y


class TestPredictiveIntervalMechanism:
    """Intervals must be built from the observation-noise-inclusive posterior."""

    def test_prediction_interval_std_matches_predictive_posterior(self) -> None:
        """Pin: the interval std equals the posterior-predictive std and is
        strictly wider than the latent std wherever noise was fitted.
        """
        train_x, train_y = _noisy_dataset()
        model = create_and_fit_single_task_model(train_x, train_y, BOUNDS_1D)

        test_x = torch.linspace(0.1, 0.9, 5, dtype=torch.float64).unsqueeze(-1)
        intervals = compute_prediction_intervals(model, test_x)

        with torch.no_grad():
            predictive_std = (
                model.posterior(test_x, observation_noise=True).variance.sqrt().reshape(-1)
            )
            latent_std = model.posterior(test_x).variance.sqrt().reshape(-1)

        for i, point_intervals in enumerate(intervals):
            interval_std = point_intervals["obj_0"][0].std
            assert interval_std == pytest.approx(predictive_std[i].item(), rel=1e-9)
            assert interval_std > latent_std[i].item(), (
                "Interval std does not include observation noise — it answers "
                "the wrong question for 'what outcome should I expect'."
            )

    def test_calibration_observed_coverage_uses_predictive_posterior(self) -> None:
        """Pin: reported observed coverage equals coverage computed manually
        from the predictive posterior, and exceeds latent-interval coverage
        on this noisy dataset.
        """
        train_x, train_y = _noisy_dataset()
        model = create_and_fit_single_task_model(train_x, train_y, BOUNDS_1D)

        report = compute_calibration_score(model, train_x, train_y)
        observed_95 = next(
            c.observed_coverage
            for c in report.coverage_results
            if abs(c.confidence_level - 0.95) < 1e-9
        )

        actual = train_y.squeeze(-1)
        with torch.no_grad():
            predictive = model.posterior(train_x, observation_noise=True)
            latent = model.posterior(train_x)

        def coverage(posterior: object) -> float:
            mean = posterior.mean.squeeze(-1)  # ty: ignore[unresolved-attribute]
            std = posterior.variance.sqrt().squeeze(-1)  # ty: ignore[unresolved-attribute]
            inside = (actual >= mean - CI_95_Z_SCORE * std) & (actual <= mean + CI_95_Z_SCORE * std)
            return inside.to(torch.float64).mean().item()

        assert observed_95 == pytest.approx(coverage(predictive), abs=1e-9)
        assert observed_95 > coverage(latent), (
            "Predictive coverage should exceed latent coverage when noise "
            "is non-negligible; equality means observation noise was dropped."
        )

    def test_loo_calibration_returns_populated_report(self) -> None:
        """Structural smoke for the (expensive) LOO calibration path."""
        train_x, train_y = _noisy_dataset(n=10, seed=0)

        report = compute_loo_calibration(train_x, train_y, BOUNDS_1D)

        assert len(report.coverage_results) == 3
        for coverage_result in report.coverage_results:
            assert 0.0 <= coverage_result.observed_coverage <= 1.0
        assert 0.0 <= report.calibration_score <= 1.0


class TestCoverageStatistical:
    @pytest.mark.nightly
    def test_95_coverage_matches_nominal_for_well_specified_model(self) -> None:
        """y = f(x) + N(0, σ²) with known σ on a smooth f: observed 95%
        coverage must sit at the nominal rate (within a calibrated band).

        Measured across these seeds the predictive coverage averages ~0.975
        (slightly conservative, as the fitted noise upper-bounds σ on small
        data); latent-only intervals average ~0.60 and fall far outside the
        band, so this test pins the predictive construction statistically.
        """
        n_seeds = 5
        coverages: list[float] = []
        for seed in range(n_seeds):
            train_x, train_y = _noisy_dataset(seed=seed)
            model = create_and_fit_single_task_model(train_x, train_y, BOUNDS_1D)
            report = compute_calibration_score(model, train_x, train_y)
            coverages.append(
                next(
                    c.observed_coverage
                    for c in report.coverage_results
                    if abs(c.confidence_level - 0.95) < 1e-9
                )
            )

        mean_coverage = sum(coverages) / len(coverages)
        assert abs(mean_coverage - 0.95) <= 0.05, (
            f"Mean observed 95% coverage {mean_coverage:.3f} (per seed: "
            f"{[round(c, 3) for c in coverages]}) is outside the calibrated "
            "band [0.90, 1.00] for a well-specified model."
        )
