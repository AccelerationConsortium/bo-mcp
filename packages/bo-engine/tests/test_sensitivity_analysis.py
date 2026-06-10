"""Tests for sensitivity analysis normalization and labeling.

The user-facing sensitivity labels ("high"/"medium"/"low") and the
robustness score are compared against fixed dimensionless thresholds, so
the underlying sensitivity must be dimensionless: gradients are
normalized by the parameter range (input side) AND the observed
objective spread (output side). Without the output-side normalization, a
yield expressed in percent reads "high sensitivity everywhere" while the
identical problem expressed as a fraction reads "low" — the labels would
describe measurement units, not the function.

The scale-invariance pattern follows
``tests/test_convergence_scale_invariance.py`` (same campaign expressed
in different units must produce identical verdicts).

References:
    - Kucherenko et al. "Monte Carlo evaluation of derivative-based
      global sensitivity measures" (2009) — DGSM normalize derivatives
      to dimensionless form before comparison.
    - Saltelli et al. "Global Sensitivity Analysis: The Primer" (2008),
      ch. 2: sensitivity measures must not depend on output units.
"""

from __future__ import annotations

import math

import pytest
import torch

from bo_engine.models import create_and_fit_single_task_model
from bo_engine.sensitivity_analysis import (
    compute_sensitivity,
    rank_parameters_by_sensitivity,
)

Y_SCALE = 1000.0


def _make_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """2-D data where x0 drives the objective and x1 barely matters."""
    torch.manual_seed(7)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    train_x = torch.rand(25, 2, dtype=torch.double)
    train_y = 3.0 * (train_x[:, 0:1] - 0.5) ** 2 + 0.05 * train_x[:, 1:2]
    return train_x, train_y, bounds


class TestScaleInvariance:
    """y and 1000*y must produce identical labels and robustness scores."""

    def test_labels_and_robustness_invariant_to_y_rescaling(self) -> None:
        train_x, train_y, bounds = _make_data()
        x_best = train_x[train_y.argmin()].unsqueeze(0)

        model = create_and_fit_single_task_model(train_x, train_y, bounds)
        model_scaled = create_and_fit_single_task_model(train_x, Y_SCALE * train_y, bounds)

        report = compute_sensitivity(model, x_best, bounds)
        report_scaled = compute_sensitivity(model_scaled, x_best, bounds)

        labels = [p.sensitivity_level for p in report.parameters]
        labels_scaled = [p.sensitivity_level for p in report_scaled.parameters]
        assert labels == labels_scaled, (
            f"Sensitivity labels changed under y-rescaling: {labels} vs "
            f"{labels_scaled} — the normalization is unit-dependent."
        )
        assert report.robustness_score == pytest.approx(report_scaled.robustness_score, abs=1e-3)
        assert report.overall_sensitivity == pytest.approx(
            report_scaled.overall_sensitivity, abs=1e-3
        )

    def test_raw_gradient_and_effect_size_stay_in_objective_units(self) -> None:
        """The raw gradient/effect size deliberately keep objective units."""
        train_x, train_y, bounds = _make_data()
        x_best = train_x[train_y.argmin()].unsqueeze(0)

        model = create_and_fit_single_task_model(train_x, train_y, bounds)
        model_scaled = create_and_fit_single_task_model(train_x, Y_SCALE * train_y, bounds)

        grad = compute_sensitivity(model, x_best, bounds).parameters[0].gradient
        grad_scaled = compute_sensitivity(model_scaled, x_best, bounds).parameters[0].gradient
        assert grad_scaled == pytest.approx(Y_SCALE * grad, rel=0.05)

    def test_ranking_invariant_to_y_rescaling(self) -> None:
        train_x, train_y, bounds = _make_data()
        model = create_and_fit_single_task_model(train_x, train_y, bounds)
        model_scaled = create_and_fit_single_task_model(train_x, Y_SCALE * train_y, bounds)

        ranking = rank_parameters_by_sensitivity(model, train_x, bounds)
        ranking_scaled = rank_parameters_by_sensitivity(model_scaled, train_x, bounds)
        assert [name for name, _ in ranking] == [name for name, _ in ranking_scaled]
        for (_, value), (_, value_scaled) in zip(ranking, ranking_scaled, strict=True):
            assert value == pytest.approx(value_scaled, abs=1e-3)


class TestEdgeCaseShapes:
    """1-D campaigns and degenerate observation sets must not crash.

    An unconstrained ``squeeze()`` on the per-parameter vectors collapses
    a ``(1, 1)`` tensor to 0-d for single-parameter campaigns (IndexError
    on indexing), and the default Bessel-corrected ``std()`` is ``nan``
    for one observation — ``max(nan, floor)`` stays ``nan``.
    """

    def test_one_parameter_campaign_report(self) -> None:
        torch.manual_seed(3)
        bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
        train_x = torch.linspace(0, 1, 10, dtype=torch.double).unsqueeze(-1)
        train_y = (train_x - 0.3) ** 2
        model = create_and_fit_single_task_model(train_x, train_y, bounds)

        report = compute_sensitivity(model, train_x[:1], bounds, parameter_names=["only"])
        assert len(report.parameters) == 1
        assert report.most_sensitive == "only"
        assert math.isfinite(report.parameters[0].sensitivity)
        assert math.isfinite(report.robustness_score)

    def test_one_parameter_ranking(self) -> None:
        torch.manual_seed(3)
        bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
        train_x = torch.linspace(0, 1, 10, dtype=torch.double).unsqueeze(-1)
        train_y = (train_x - 0.3) ** 2
        model = create_and_fit_single_task_model(train_x, train_y, bounds)

        ranking = rank_parameters_by_sensitivity(model, train_x, bounds)
        assert len(ranking) == 1
        assert math.isfinite(ranking[0][1])

    def test_single_observation_report_is_finite(self) -> None:
        torch.manual_seed(3)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
        train_x = torch.tensor([[0.4, 0.6]], dtype=torch.double)
        train_y = torch.tensor([[1.0]], dtype=torch.double)
        model = create_and_fit_single_task_model(train_x, train_y, bounds)

        report = compute_sensitivity(model, train_x, bounds)
        assert math.isfinite(report.overall_sensitivity)
        assert math.isfinite(report.robustness_score)
        assert all(math.isfinite(p.sensitivity) for p in report.parameters)

    def test_constant_observations_report_is_finite(self) -> None:
        torch.manual_seed(3)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
        train_x = torch.rand(8, 2, dtype=torch.double)
        train_y = torch.full((8, 1), 2.5, dtype=torch.double)
        model = create_and_fit_single_task_model(train_x, train_y, bounds)

        report = compute_sensitivity(model, train_x[:1], bounds)
        assert math.isfinite(report.overall_sensitivity)
        assert math.isfinite(report.robustness_score)


class TestSensitivityOrdering:
    """The influential parameter must outrank the inert one."""

    def test_driving_parameter_ranks_first(self) -> None:
        train_x, train_y, bounds = _make_data()
        model = create_and_fit_single_task_model(train_x, train_y, bounds)

        ranking = rank_parameters_by_sensitivity(
            model, train_x, bounds, parameter_names=["driver", "inert"]
        )
        assert ranking[0][0] == "driver"
        assert ranking[0][1] > ranking[1][1]

    def test_report_identifies_most_sensitive(self) -> None:
        train_x, train_y, bounds = _make_data()
        model = create_and_fit_single_task_model(train_x, train_y, bounds)
        # Off-minimum point so the driver's gradient is clearly nonzero.
        x_probe = torch.tensor([[0.9, 0.5]], dtype=torch.double)
        report = compute_sensitivity(model, x_probe, bounds, parameter_names=["driver", "inert"])
        assert report.most_sensitive == "driver"
        assert report.least_sensitive == "inert"
