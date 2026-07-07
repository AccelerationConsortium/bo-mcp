"""Tests for posterior predictive checks (residual-based GP validation).

The module's purpose is assumption checking, so its own statistical
contract must hold: for a *well-specified* model (data truly generated as
a smooth function plus Gaussian noise) the standardized residuals must be
approximately N(0, 1) and ``run_posterior_checks`` must not raise false
alarms. That guarantee only exists for leave-one-out residuals
standardized by the *predictive* (observation-noise-inclusive) std —
GPML §5.4.2. In-sample residuals standardized by the latent posterior std
have no such property: on the well-specified dataset used below they have
a standard deviation of ~2.7, which would (wrongly) fail every check.

References:
    - Rasmussen & Williams "GPML" §5.4.2 (LOO predictive moments)
    - Gelman et al., "Bayesian Data Analysis", Ch. 6 (model checking via
      residual diagnostics and posterior predictive checks)
"""

from __future__ import annotations

import math

import pytest
import torch

from bo_engine.cross_validation import compute_exact_loo_moments
from bo_engine.models import create_and_fit_model, create_and_fit_single_task_model
from bo_engine.posterior_checks import (
    analyze_residuals,
    check_multi_objective_posteriors,
    compute_standardized_residuals,
    run_posterior_checks,
)

BOUNDS_1D = torch.tensor([[0.0], [1.0]], dtype=torch.float64)

# Lab-typical, non-unit bounds. ``Normalize`` is the identity on the unit
# cube, so only non-unit bounds exercise the transformed-vs-raw distinction
# in the model's stored training inputs.
NON_UNIT_BOUNDS_2D = torch.tensor([[0.0, 0.0], [1000.0, 500.0]], dtype=torch.float64)


def _well_specified_dataset(
    n: int = 40, seed: int = 11, noise_std: float = 0.3
) -> tuple[torch.Tensor, torch.Tensor]:
    """Smooth sine plus Gaussian noise — a model the GP can match exactly."""
    torch.manual_seed(seed)
    train_x = torch.rand(n, 1, dtype=torch.float64)
    train_y = torch.sin(4 * train_x) + noise_std * torch.randn(n, 1, dtype=torch.float64)
    return train_x, train_y


def _well_specified_non_unit_dataset(
    n: int = 40, seed: int = 11, noise_std: float = 0.3
) -> tuple[torch.Tensor, torch.Tensor]:
    """Smooth signal plus Gaussian noise on lab-scale (non-unit) bounds."""
    torch.manual_seed(seed)
    span = NON_UNIT_BOUNDS_2D[1] - NON_UNIT_BOUNDS_2D[0]
    train_x = NON_UNIT_BOUNDS_2D[0] + span * torch.rand(n, 2, dtype=torch.float64)
    signal = torch.sin(train_x[:, 0:1] / 250.0) + train_x[:, 1:2] / 500.0
    return train_x, signal + noise_std * torch.randn(n, 1, dtype=torch.float64)


class TestStandardizedResiduals:
    """Residuals must be LOO predictive z-scores in the model's target space."""

    def test_residuals_pin_against_exact_loo_moments(self) -> None:
        """Mechanism pin: residuals equal (t_i - μ_{-i}) / σ_{-i} from the
        exact LOO downdate on the model's transformed targets.
        """
        train_x, train_y = _well_specified_dataset()
        model = create_and_fit_single_task_model(train_x, train_y, BOUNDS_1D)

        residuals = compute_standardized_residuals(model, train_x, train_y)

        loo_mean, loo_var = compute_exact_loo_moments(model)
        expected = (model.train_targets - loo_mean) / loo_var.sqrt()

        assert torch.allclose(residuals, expected, atol=1e-8)

    def test_residual_std_near_one_for_well_specified_model(self) -> None:
        """For matched model + Gaussian noise, z-scores must be ~N(0, 1).

        In-sample residuals standardized by the latent posterior std give a
        standard deviation of ~2.7 on this dataset (the denominator omits
        the noise that dominates the numerator), so this bound separates
        the correct construction from the broken one by a wide margin.
        """
        train_x, train_y = _well_specified_dataset()
        model = create_and_fit_single_task_model(train_x, train_y, BOUNDS_1D)

        residuals = compute_standardized_residuals(model, train_x, train_y)

        assert 0.7 <= residuals.std().item() <= 1.3
        assert abs(residuals.mean().item()) <= 0.2

    def test_model_list_gp_routes_objectives(self) -> None:
        torch.manual_seed(2)
        n = 20
        train_x = torch.rand(n, 2, dtype=torch.float64)
        train_y = torch.stack(
            [
                train_x[:, 0] + 0.05 * torch.randn(n, dtype=torch.float64),
                torch.sin(5 * train_x[:, 1]) + 0.05 * torch.randn(n, dtype=torch.float64),
            ],
            dim=-1,
        )
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        model = create_and_fit_model(train_x, train_y, bounds)

        residuals_0 = compute_standardized_residuals(
            model, train_x, train_y[:, 0], objective_index=0
        )
        residuals_1 = compute_standardized_residuals(
            model, train_x, train_y[:, 1], objective_index=1
        )

        assert residuals_0.shape == (n,)
        assert residuals_1.shape == (n,)
        assert torch.isfinite(residuals_0).all()
        assert torch.isfinite(residuals_1).all()
        assert not torch.allclose(residuals_0, residuals_1)

    def test_fallback_on_training_data_mismatch(self) -> None:
        """When the passed data is not the model's training set, the LOO
        downdate does not apply; the fallback must still return finite
        residuals of the requested length.
        """
        train_x, train_y = _well_specified_dataset()
        model = create_and_fit_single_task_model(train_x, train_y, BOUNDS_1D)

        subset = compute_standardized_residuals(model, train_x[:10], train_y[:10])

        assert subset.shape == (10,)
        assert torch.isfinite(subset).all()

    def test_same_length_mismatch_uses_passed_data_not_stored_set(self) -> None:
        """A same-length dataset that is not the model's training data must
        not be silently answered with the stored training set's LOO
        residuals — the result must be computed at the *passed* points.
        """
        train_x, train_y = _well_specified_dataset()
        model = create_and_fit_single_task_model(train_x, train_y, BOUNDS_1D)

        stored_set_residuals = compute_standardized_residuals(model, train_x, train_y)

        torch.manual_seed(0)
        permutation = torch.randperm(train_x.shape[0])
        shuffled = compute_standardized_residuals(model, train_x[permutation], train_y[permutation])

        with torch.no_grad():
            posterior = model.posterior(train_x[permutation], observation_noise=True)
            expected_fallback = (
                train_y[permutation].squeeze(-1) - posterior.mean.squeeze(-1)
            ) / posterior.variance.sqrt().squeeze(-1)

        assert not torch.allclose(shuffled, stored_set_residuals), (
            "Shuffled data was answered with the stored training set's "
            "residuals — the training-data guard did not detect the mismatch."
        )
        assert torch.allclose(shuffled, expected_fallback, atol=1e-8)

    def test_same_length_perturbed_targets_use_fallback(self) -> None:
        """Same x, shifted y is a mismatch: residuals must reflect the
        passed targets, not the stored ones.
        """
        train_x, train_y = _well_specified_dataset()
        model = create_and_fit_single_task_model(train_x, train_y, BOUNDS_1D)

        shifted_y = train_y + 1.0
        residuals = compute_standardized_residuals(model, train_x, shifted_y)

        with torch.no_grad():
            posterior = model.posterior(train_x, observation_noise=True)
            expected_fallback = (
                shifted_y.squeeze(-1) - posterior.mean.squeeze(-1)
            ) / posterior.variance.sqrt().squeeze(-1)

        assert torch.allclose(residuals, expected_fallback, atol=1e-8)


class TestLOOResidualsOnNonUnitBounds:
    """The LOO path must engage for production models on lab-scale bounds.

    BoTorch swaps ``train_inputs`` to the transformed (normalized)
    representation when a model enters eval mode (``Model.eval`` →
    ``_set_transformed_inputs``; ``botorch/models/model.py``), so a
    training-data check comparing them against raw caller inputs silently
    demotes every ``Normalize``-equipped model to in-sample residuals. In
    that regime the residual std collapses well below 1 (the posterior
    mean interpolates its own training data), corrupting the normality and
    outlier checks downstream — GPML §5.4.2's LOO z-scores are the
    construction with the N(0, 1) guarantee. Unit-cube fixtures cannot
    detect the demotion because ``Normalize`` is the identity there.
    """

    @pytest.mark.parametrize("mode", ["train", "eval"])
    def test_residuals_equal_downdate_z_scores(self, mode: str) -> None:
        train_x, train_y = _well_specified_non_unit_dataset()
        model = create_and_fit_single_task_model(train_x, train_y, NON_UNIT_BOUNDS_2D)
        if mode == "train":
            model.train()
        else:
            model.eval()

        residuals = compute_standardized_residuals(model, train_x, train_y)

        loo_mean, loo_var = compute_exact_loo_moments(model)
        expected = (model.train_targets - loo_mean) / loo_var.sqrt()
        assert torch.allclose(residuals, expected, atol=1e-8)

    def test_residual_std_near_one_on_non_unit_bounds(self) -> None:
        """The N(0, 1) contract must hold off the unit cube as well; the
        in-sample fallback would report a visibly shrunken std here.
        """
        train_x, train_y = _well_specified_non_unit_dataset()
        model = create_and_fit_single_task_model(train_x, train_y, NON_UNIT_BOUNDS_2D)

        residuals = compute_standardized_residuals(model, train_x, train_y)

        assert 0.7 <= residuals.std().item() <= 1.3
        assert abs(residuals.mean().item()) <= 0.2


class TestResidualAnalysis:
    def test_planted_outlier_is_flagged(self) -> None:
        """A grossly corrupted observation must appear in outlier_indices.

        LOO residuals are the right tool here: the corrupted point cannot
        explain itself away, because its own value is excluded from the
        prediction that scores it.
        """
        torch.manual_seed(5)
        train_x = torch.linspace(0, 1, 15, dtype=torch.float64).unsqueeze(-1)
        train_y = 10 * train_x.clone()
        outlier_index = 7
        train_y[outlier_index] = -30.0

        model = create_and_fit_single_task_model(train_x, train_y, BOUNDS_1D)
        analysis = analyze_residuals(model, train_x, train_y)

        assert outlier_index in analysis.outlier_indices
        assert analysis.max_abs_residual > 3.0

    def test_analysis_statistics_are_populated(self) -> None:
        train_x, train_y = _well_specified_dataset()
        model = create_and_fit_single_task_model(train_x, train_y, BOUNDS_1D)

        analysis = analyze_residuals(model, train_x, train_y)

        assert analysis.standardized_residuals.shape == (train_x.shape[0],)
        for value in (analysis.mean, analysis.std, analysis.skewness, analysis.kurtosis):
            assert not math.isnan(value)


class TestRunPosteriorChecks:
    def test_well_specified_model_passes_checks(self) -> None:
        """No false alarm on a single seeded well-specified dataset."""
        train_x, train_y = _well_specified_dataset()
        model = create_and_fit_single_task_model(train_x, train_y, BOUNDS_1D)

        report = run_posterior_checks(model, train_x, train_y)

        assert report.assumptions_met is True
        assert 0.7 <= report.residual_analysis.std <= 1.3
        assert report.normality_tests, "expected at least one normality test at n=40"
        assert len(report.qq_plot_data.sample_quantiles) == train_x.shape[0]

    def test_multi_objective_reports_per_objective(self) -> None:
        torch.manual_seed(2)
        n = 20
        train_x = torch.rand(n, 2, dtype=torch.float64)
        train_y = torch.stack(
            [
                train_x[:, 0] + 0.05 * torch.randn(n, dtype=torch.float64),
                train_x[:, 1] + 0.05 * torch.randn(n, dtype=torch.float64),
            ],
            dim=-1,
        )
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        model = create_and_fit_model(train_x, train_y, bounds)

        reports = check_multi_objective_posteriors(
            model, train_x, train_y, objective_names=["a", "b"]
        )

        assert set(reports) == {"a", "b"}

    @pytest.mark.nightly
    def test_no_false_alarms_across_seeds(self) -> None:
        """Statistical form (TESTING.md): across seeds, residual std stays in
        [0.7, 1.3] and the checks pass on all but at most one seed (the
        normality tests run at α=0.05, so occasional rejections are
        expected even under H0).
        """
        n_seeds = 5
        passes = 0
        for seed in range(n_seeds):
            train_x, train_y = _well_specified_dataset(seed=seed)
            model = create_and_fit_single_task_model(train_x, train_y, BOUNDS_1D)
            report = run_posterior_checks(model, train_x, train_y)
            assert 0.7 <= report.residual_analysis.std <= 1.3, (
                f"seed {seed}: residual std {report.residual_analysis.std:.3f} "
                "outside [0.7, 1.3] for a well-specified model"
            )
            passes += report.assumptions_met

        assert passes >= n_seeds - 1

    @pytest.mark.nightly
    def test_heavy_tailed_noise_is_detected(self) -> None:
        """Misspecification case: Student-t(df=2) noise has infinite
        kurtosis, so the kurtosis/normality checks must flag it on (almost)
        every seed.
        """
        n_seeds = 5
        detections = 0
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            n = 60
            train_x = torch.rand(n, 1, dtype=torch.float64)
            t_noise = torch.distributions.StudentT(df=2.0).sample((n, 1)).to(torch.float64)
            train_y = torch.sin(4 * train_x) + 0.3 * t_noise
            model = create_and_fit_single_task_model(train_x, train_y, BOUNDS_1D)
            report = run_posterior_checks(model, train_x, train_y)
            detections += not report.assumptions_met

        assert detections >= n_seeds - 1, (
            f"Heavy-tailed noise detected on only {detections}/{n_seeds} seeds."
        )
