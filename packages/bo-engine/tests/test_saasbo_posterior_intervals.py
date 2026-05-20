"""Posterior-interval reporting for SAASBO importance (8.36).

Previously :func:`bo_engine.saasbo.compute_saasbo_importance_report`
returned only the median lengthscale per dimension. Early in a campaign
the half-Cauchy posterior on each lengthscale is wide and the median
alone is an unreliable basis for pruning a dimension. We now report
the 25 % and 75 % posterior quantiles, the log10 width of the interval,
and a ``confident`` flag that is False when the interval spans more than
one order of magnitude.

References:
    - Eriksson & Jankowiak, "High-Dimensional Bayesian Optimization with
      Sparse Axis-Aligned Subspaces", UAI 2021, §3.3 & Appendix B.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch

from bo_engine.constants import SAASBO_WIDE_INTERVAL_LOG10_THRESHOLD
from bo_engine.saasbo import (
    SAASBOImportance,
    compute_saasbo_importance_report,
)


@dataclass
class _Sample:
    """Just enough surface to mimic a fitted SAASBO model for these tests."""

    lengthscale_samples: torch.Tensor


def _patched_model(samples: torch.Tensor):
    """Construct a stub model so we can drive the quantile reduction directly."""
    model = MagicMock()
    inner = MagicMock()
    inner.base_kernel = MagicMock()
    inner.base_kernel.lengthscale = samples
    model.covar_module = inner
    return model


class TestPosteriorIntervalsPopulated:
    """The new fields must reflect the per-dimension quantile spread."""

    def test_wide_interval_flagged_not_confident(self) -> None:
        # 8 posterior samples per dimension, two dimensions:
        # dim 0: tight (all near 1.0) → confident True
        # dim 1: wide (spans 0.01 to 100) → confident False
        tight = torch.linspace(0.9, 1.1, 8)
        wide = torch.tensor([0.01, 0.05, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0])
        samples = torch.stack([tight, wide], dim=-1).unsqueeze(0)
        # SAASBO stores lengthscale as ``(num_samples, 1, d)``; flatten so the
        # quantile helper sees ``(num_samples, d)``.
        samples = samples.reshape(8, 1, 2)

        with (
            patch(
                "bo_engine.saasbo.get_saasbo_lengthscale_quantiles",
                return_value=torch.tensor(
                    [
                        [0.95, 0.05],  # 25 %
                        [1.0, 1.0],  # 50 %
                        [1.05, 90.0],  # 75 %
                    ]
                ),
            ),
            patch(
                "bo_engine.saasbo.get_saasbo_lengthscales",
                return_value=torch.tensor([1.0, 1.0]),
            ),
        ):
            model = _patched_model(samples)
            report = compute_saasbo_importance_report(model, parameter_names=["a", "b"])

        assert len(report) == 2
        assert isinstance(report[0], SAASBOImportance)
        # Tight posterior on dim 0 → small log10 width, confident=True
        assert report[0].confident is True
        assert report[0].log10_interval_width < SAASBO_WIDE_INTERVAL_LOG10_THRESHOLD
        # Wide posterior on dim 1 → log10(90/0.05) > 3 → confident=False
        assert report[1].confident is False
        assert report[1].log10_interval_width > SAASBO_WIDE_INTERVAL_LOG10_THRESHOLD


class TestNarrowPosteriorAllConfident:
    """When every dimension's posterior is tight, every entry should be confident."""

    def test_all_dimensions_marked_confident(self) -> None:
        with (
            patch(
                "bo_engine.saasbo.get_saasbo_lengthscale_quantiles",
                return_value=torch.tensor([[0.9, 0.95], [1.0, 1.0], [1.1, 1.05]]),
            ),
            patch(
                "bo_engine.saasbo.get_saasbo_lengthscales",
                return_value=torch.tensor([1.0, 1.0]),
            ),
        ):
            model = _patched_model(torch.zeros(1, 1, 2))
            report = compute_saasbo_importance_report(model)

        assert all(entry.confident for entry in report)


class TestQuantileFields:
    """The raw 25 % / 75 % bounds must be populated for downstream UI display."""

    def test_quantile_bounds_match_input(self) -> None:
        with (
            patch(
                "bo_engine.saasbo.get_saasbo_lengthscale_quantiles",
                return_value=torch.tensor([[0.3], [1.0], [3.0]]),
            ),
            patch(
                "bo_engine.saasbo.get_saasbo_lengthscales",
                return_value=torch.tensor([1.0]),
            ),
        ):
            model = _patched_model(torch.zeros(1, 1, 1))
            report = compute_saasbo_importance_report(model)

        assert report[0].lengthscale_q25 == pytest.approx(0.3)
        assert report[0].lengthscale == pytest.approx(1.0)
        assert report[0].lengthscale_q75 == pytest.approx(3.0)
