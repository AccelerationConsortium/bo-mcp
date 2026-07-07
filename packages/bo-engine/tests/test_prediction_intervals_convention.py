"""EI/PoI posterior convention for suggestion explanations.

The explanation surface must describe the quantity the acquisition
function actually optimized: BoTorch's (q)EI family evaluates the
**latent** (noise-free) posterior, so ``expected_improvement`` /
``probability_of_improvement`` are computed from the latent moments.
The prediction *intervals* deliberately stay predictive
(observation-noise-inflated) — they answer the different question
"what range of measured outcomes should I expect?".

References:
    - Rasmussen & Williams, GPML (2006), Eq. 2.24 vs 2.26 — predictive vs
      latent test distributions differ exactly by the observation noise.
    - BoTorch ``ExpectedImprovement`` — computed from ``model.posterior``
      without ``observation_noise=True``.
"""

from __future__ import annotations

import pytest
import torch
from scipy import stats as scipy_stats

from bo_engine.models import create_and_fit_single_task_model
from bo_engine.prediction_intervals import compute_suggestion_predictions


def _noisy_model() -> tuple:
    """A deliberately noisy 1-D fit so latent and predictive stds diverge."""
    generator = torch.Generator().manual_seed(17)
    train_x = torch.rand(12, 1, dtype=torch.double, generator=generator)
    noise = 0.3 * torch.randn(12, dtype=torch.double, generator=generator)
    train_y = (train_x.squeeze(-1) - 0.5) ** 2 + noise
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    model = create_and_fit_single_task_model(train_x, train_y, bounds)
    return model, float(train_y.min())


class TestLatentImprovementConvention:
    def test_poi_matches_latent_posterior_moments(self) -> None:
        model, best_value = _noisy_model()
        suggestion = {"x": 0.5}

        batch = compute_suggestion_predictions(
            model=model,
            suggestions=[suggestion],
            parameter_names=["x"],
            best_value=best_value,
            minimize=True,
        )
        prediction = batch.suggestions[0]

        with torch.no_grad():
            latent = model.posterior(torch.tensor([[0.5]], dtype=torch.double))
            latent_mean = latent.mean.item()
            latent_std = latent.variance.sqrt().item()
            predictive = model.posterior(
                torch.tensor([[0.5]], dtype=torch.double), observation_noise=True
            )
            predictive_std = predictive.variance.sqrt().item()

        # Sanity: the two conventions genuinely differ on this fixture.
        assert predictive_std > latent_std * 1.05

        expected_poi = float(scipy_stats.norm.cdf((best_value - latent_mean) / latent_std))
        assert prediction.probability_of_improvement == pytest.approx(expected_poi, abs=1e-9)

        # The reported interval std stays predictive.
        interval_std = prediction.objectives["obj_0"][0].std
        assert interval_std == pytest.approx(predictive_std, rel=1e-6)

    def test_latent_ei_not_inflated_by_observation_noise(self) -> None:
        """EI from the noise-inflated std would be strictly larger."""
        model, best_value = _noisy_model()

        batch = compute_suggestion_predictions(
            model=model,
            suggestions=[{"x": 0.5}],
            parameter_names=["x"],
            best_value=best_value,
            minimize=True,
        )
        prediction = batch.suggestions[0]

        with torch.no_grad():
            latent = model.posterior(torch.tensor([[0.5]], dtype=torch.double))
            mean = latent.mean.item()
            std = latent.variance.sqrt().item()

        z = (best_value - mean) / std
        expected_ei = max(
            0.0,
            (best_value - mean) * float(scipy_stats.norm.cdf(z))
            + std * float(scipy_stats.norm.pdf(z)),
        )
        assert prediction.expected_improvement == pytest.approx(expected_ei, abs=1e-9)
