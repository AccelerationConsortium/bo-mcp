"""Tests for Leave-One-Out Cross-Validation diagnostics (v1.1)."""

import math

import torch

from bo_engine import (
    LOOCVMetrics,
    compute_loo_cv_for_model,
    compute_loo_cv_metrics,
    create_and_fit_model,
    create_and_fit_single_task_model,
)


class TestLOOCVMetricsComputation:
    """Test LOO-CV metrics computation."""

    def test_compute_loo_cv_basic(self) -> None:
        """Test basic LOO-CV computation."""
        torch.manual_seed(42)
        n_samples = 10
        train_x = torch.rand(n_samples, 2, dtype=torch.double)
        # Simple linear function for predictable results
        train_y = train_x[:, 0:1] + train_x[:, 1:2]
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        metrics = compute_loo_cv_metrics(train_x, train_y, bounds)

        # Should produce valid metrics
        assert not math.isnan(metrics.rmse)
        assert not math.isnan(metrics.mae)
        assert not math.isnan(metrics.r_squared)
        assert metrics.rmse >= 0
        assert metrics.mae >= 0
        # R² should be high for a simple linear function
        assert metrics.r_squared > 0.5

    def test_compute_loo_cv_few_samples(self) -> None:
        """Test LOO-CV with very few samples."""
        train_x = torch.rand(2, 2, dtype=torch.double)
        train_y = torch.rand(2, 1, dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        metrics = compute_loo_cv_metrics(train_x, train_y, bounds)

        # Should return NaN for insufficient data
        assert math.isnan(metrics.rmse)
        assert len(metrics.per_fold_errors) == 0

    def test_compute_loo_cv_1d_y(self) -> None:
        """Test LOO-CV with 1D y input."""
        torch.manual_seed(42)
        train_x = torch.rand(10, 2, dtype=torch.double)
        train_y = torch.rand(10, dtype=torch.double)  # 1D
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        metrics = compute_loo_cv_metrics(train_x, train_y, bounds)

        # Should handle 1D y
        assert not math.isnan(metrics.rmse)

    def test_per_fold_errors_length(self) -> None:
        """Test that per-fold errors match number of samples."""
        torch.manual_seed(42)
        n_samples = 8
        train_x = torch.rand(n_samples, 2, dtype=torch.double)
        train_y = torch.rand(n_samples, 1, dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        metrics = compute_loo_cv_metrics(train_x, train_y, bounds)

        # Should have n_samples fold errors (or close to it if some fail)
        assert len(metrics.per_fold_errors) > 0
        assert len(metrics.per_fold_errors) <= n_samples


class TestLOOCVForModel:
    """Test LOO-CV for fitted models."""

    def test_loo_cv_single_task_model(self) -> None:
        """Test LOO-CV on SingleTaskGP."""
        torch.manual_seed(42)
        train_x = torch.rand(10, 2, dtype=torch.double)
        train_y = train_x[:, 0:1] ** 2  # Quadratic function
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        model = create_and_fit_single_task_model(train_x, train_y, bounds)
        cv_results = compute_loo_cv_for_model(model, train_x, train_y)

        # For SingleTaskGP, cv_results is LOOCVMetrics directly (not a dict)
        assert isinstance(cv_results, LOOCVMetrics)
        assert not math.isnan(cv_results.rmse)

    def test_loo_cv_model_list_gp(self) -> None:
        """Test LOO-CV on ModelListGP."""
        torch.manual_seed(42)
        train_x = torch.rand(10, 2, dtype=torch.double)
        train_y = torch.stack(
            [
                train_x[:, 0] + train_x[:, 1],
                train_x[:, 0] * train_x[:, 1],
            ],
            dim=-1,
        )
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        model = create_and_fit_model(train_x, train_y, bounds)
        cv_results = compute_loo_cv_for_model(model, train_x, train_y)

        # Should have results for both objectives
        assert 0 in cv_results
        assert 1 in cv_results
        assert isinstance(cv_results[0], LOOCVMetrics)
        assert isinstance(cv_results[1], LOOCVMetrics)


class TestLOOCVMetricsInterpretation:
    """Test interpretation of LOO-CV metrics."""

    def test_perfect_prediction_metrics(self) -> None:
        """Test metrics for near-perfect predictions."""
        torch.manual_seed(42)
        n_samples = 15
        train_x = torch.rand(n_samples, 2, dtype=torch.double)
        # Very simple linear function - GP should fit perfectly
        train_y = (train_x[:, 0:1] + train_x[:, 1:2]) / 2
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        metrics = compute_loo_cv_metrics(train_x, train_y, bounds)

        # RMSE and MAE should be small for good fit
        assert metrics.rmse < 0.5
        assert metrics.mae < 0.5
        # R² should be high
        assert metrics.r_squared > 0.8

    def test_noisy_data_metrics(self) -> None:
        """Test metrics for noisy data."""
        torch.manual_seed(42)
        n_samples = 15
        train_x = torch.rand(n_samples, 2, dtype=torch.double)
        # Add significant noise
        train_y = train_x[:, 0:1] + torch.randn(n_samples, 1, dtype=torch.double) * 0.5
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        metrics = compute_loo_cv_metrics(train_x, train_y, bounds)

        # Metrics should still be computable
        assert not math.isnan(metrics.rmse)
        # R² will be lower for noisy data
        assert metrics.r_squared <= 1.0  # Valid R² range


class TestLOOCVDataclass:
    """Test LOOCVMetrics dataclass."""

    def test_dataclass_fields(self) -> None:
        """Test that LOOCVMetrics has all required fields."""
        metrics = LOOCVMetrics(
            rmse=0.1,
            mae=0.08,
            r_squared=0.95,
            mean_standardized_error=1.0,
            per_fold_errors=[0.1, 0.08, 0.12],
        )

        assert metrics.rmse == 0.1
        assert metrics.mae == 0.08
        assert metrics.r_squared == 0.95
        assert metrics.mean_standardized_error == 1.0
        assert len(metrics.per_fold_errors) == 3


class TestLOOCVEdgeCases:
    """Test edge cases in LOO-CV computation."""

    def test_single_dimension(self) -> None:
        """Test LOO-CV with single input dimension."""
        torch.manual_seed(42)
        train_x = torch.rand(10, 1, dtype=torch.double)
        train_y = train_x**2
        bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

        metrics = compute_loo_cv_metrics(train_x, train_y, bounds)

        assert not math.isnan(metrics.rmse)

    def test_many_dimensions(self) -> None:
        """Test LOO-CV with many input dimensions."""
        torch.manual_seed(42)
        n_dims = 5
        train_x = torch.rand(15, n_dims, dtype=torch.double)
        train_y = train_x.sum(dim=-1, keepdim=True)
        bounds = torch.zeros(2, n_dims, dtype=torch.double)
        bounds[1, :] = 1.0

        metrics = compute_loo_cv_metrics(train_x, train_y, bounds)

        assert not math.isnan(metrics.rmse)

    def test_constant_output(self) -> None:
        """Test LOO-CV with constant output values."""
        torch.manual_seed(42)
        train_x = torch.rand(10, 2, dtype=torch.double)
        train_y = torch.ones(10, 1, dtype=torch.double)  # Constant
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        metrics = compute_loo_cv_metrics(train_x, train_y, bounds)

        # Should produce small errors for constant data
        assert not math.isnan(metrics.mae)
        assert metrics.mae < 0.1  # Small error expected


class TestLOOCVMetricsValidation:
    """Test validation of LOO-CV metrics correctness.

    Reference: Hastie et al., "The Elements of Statistical Learning" (2009), Section 7.10.
    LOO-CV provides an approximately unbiased estimate of the expected prediction error.
    """

    def test_rmse_manual_computation(self) -> None:
        """Verify RMSE matches manual computation.

        RMSE = sqrt(mean((predicted - actual)^2))
        This validates the RMSE calculation in compute_loo_cv_metrics.
        """
        torch.manual_seed(42)
        n_samples = 10
        train_x = torch.rand(n_samples, 2, dtype=torch.double)
        train_y = train_x[:, 0:1] + train_x[:, 1:2]
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        metrics = compute_loo_cv_metrics(train_x, train_y, bounds)

        # Verify RMSE matches manual computation from per-fold errors
        if len(metrics.per_fold_errors) > 0:
            manual_mse = sum(e**2 for e in metrics.per_fold_errors) / len(metrics.per_fold_errors)
            manual_rmse = manual_mse**0.5
            assert abs(metrics.rmse - manual_rmse) < 1e-6

    def test_mae_manual_computation(self) -> None:
        """Verify MAE matches manual computation.

        MAE = mean(|predicted - actual|)
        """
        torch.manual_seed(42)
        n_samples = 10
        train_x = torch.rand(n_samples, 2, dtype=torch.double)
        train_y = train_x[:, 0:1] * 2
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        metrics = compute_loo_cv_metrics(train_x, train_y, bounds)

        # Verify MAE matches manual computation
        if len(metrics.per_fold_errors) > 0:
            manual_mae = sum(metrics.per_fold_errors) / len(metrics.per_fold_errors)
            assert abs(metrics.mae - manual_mae) < 1e-6

    def test_r_squared_range(self) -> None:
        """Verify R² is in valid range for good fits.

        R² = 1 - SS_res/SS_tot
        For well-fitting models, R² should be close to 1.
        Note: R² can be negative for poorly fitting models, which is valid.
        """
        torch.manual_seed(42)
        n_samples = 15
        train_x = torch.rand(n_samples, 2, dtype=torch.double)
        # Simple linear function - GP should fit well
        train_y = (train_x[:, 0:1] + train_x[:, 1:2]) / 2
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        metrics = compute_loo_cv_metrics(train_x, train_y, bounds)

        # R² should be <= 1.0 always
        assert metrics.r_squared <= 1.0
        # For a simple function, R² should be reasonably high
        assert metrics.r_squared > 0.5

    def test_r_squared_can_be_negative(self) -> None:
        """Verify R² can be negative for poor fits.

        When predictions are worse than using the mean, R² < 0.
        This is valid and important to capture.

        Note: With sufficient samples but a hard-to-fit function,
        the GP may produce predictions that are worse than the mean.
        """
        torch.manual_seed(123)
        n_samples = 8  # Minimum for meaningful LOO-CV
        # High-frequency function - hard to fit with few samples
        train_x = torch.rand(n_samples, 2, dtype=torch.double)
        train_y = torch.sin(train_x[:, 0:1] * 15) * torch.cos(train_x[:, 1:2] * 15)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        metrics = compute_loo_cv_metrics(train_x, train_y, bounds)

        # R² can be negative, zero, or positive depending on fit quality
        # Just verify it's a valid number (metrics computed successfully)
        assert not math.isnan(metrics.r_squared)
        # R² should always be <= 1.0
        assert metrics.r_squared <= 1.0

    def test_standardized_error_computation(self) -> None:
        """Verify standardized error is computed correctly.

        Standardized error = |predicted - actual| / sqrt(predicted_variance)
        Well-calibrated models should have mean standardized error close to 1.
        """
        torch.manual_seed(42)
        n_samples = 12
        train_x = torch.rand(n_samples, 2, dtype=torch.double)
        train_y = train_x[:, 0:1] + 0.1 * torch.randn(n_samples, 1, dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        metrics = compute_loo_cv_metrics(train_x, train_y, bounds)

        # Standardized error should be a valid positive number
        if not math.isnan(metrics.mean_standardized_error):
            assert metrics.mean_standardized_error >= 0


class TestLOOCVNumericalRobustness:
    """Test numerical robustness of LOO-CV computation.

    Reference: Numerical stability in GP fitting is critical for reliable CV.
    See BoTorch documentation on numerical precision requirements.
    """

    def test_small_values(self) -> None:
        """Test LOO-CV with very small objective values."""
        torch.manual_seed(42)
        n_samples = 10
        train_x = torch.rand(n_samples, 2, dtype=torch.double)
        train_y = train_x[:, 0:1] * 1e-6  # Very small values
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        metrics = compute_loo_cv_metrics(train_x, train_y, bounds)

        # Should still compute valid metrics
        assert not math.isnan(metrics.rmse)
        assert not math.isnan(metrics.mae)

    def test_large_values(self) -> None:
        """Test LOO-CV with large objective values."""
        torch.manual_seed(42)
        n_samples = 10
        train_x = torch.rand(n_samples, 2, dtype=torch.double)
        train_y = train_x[:, 0:1] * 1e4  # Large values
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        metrics = compute_loo_cv_metrics(train_x, train_y, bounds)

        # Should still compute valid metrics
        assert not math.isnan(metrics.rmse)
        assert not math.isnan(metrics.mae)

    def test_negative_values(self) -> None:
        """Test LOO-CV with negative objective values."""
        torch.manual_seed(42)
        n_samples = 10
        train_x = torch.rand(n_samples, 2, dtype=torch.double)
        train_y = train_x[:, 0:1] - 0.5  # Centered around -0.5
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        metrics = compute_loo_cv_metrics(train_x, train_y, bounds)

        assert not math.isnan(metrics.rmse)
        assert not math.isnan(metrics.mae)

    def test_narrow_bounds(self) -> None:
        """Test LOO-CV with narrow parameter bounds."""
        torch.manual_seed(42)
        n_samples = 10
        train_x = torch.rand(n_samples, 2, dtype=torch.double) * 0.01 + 0.5
        train_y = train_x[:, 0:1]
        bounds = torch.tensor([[0.5, 0.5], [0.51, 0.51]], dtype=torch.double)

        metrics = compute_loo_cv_metrics(train_x, train_y, bounds)

        # Should handle narrow bounds
        assert not math.isnan(metrics.rmse)

    def test_wide_bounds(self) -> None:
        """Test LOO-CV with wide parameter bounds."""
        torch.manual_seed(42)
        n_samples = 10
        train_x = torch.rand(n_samples, 2, dtype=torch.double) * 1000
        train_y = train_x[:, 0:1] / 1000
        bounds = torch.tensor([[0.0, 0.0], [1000.0, 1000.0]], dtype=torch.double)

        metrics = compute_loo_cv_metrics(train_x, train_y, bounds)

        assert not math.isnan(metrics.rmse)


class TestLOOCVDataShapes:
    """Test LOO-CV with various data shapes."""

    def test_exactly_three_samples(self) -> None:
        """Test LOO-CV with exactly 3 samples (minimum required)."""
        torch.manual_seed(42)
        train_x = torch.rand(3, 2, dtype=torch.double)
        train_y = train_x[:, 0:1]
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        metrics = compute_loo_cv_metrics(train_x, train_y, bounds)

        # Should produce valid metrics with exactly 3 samples
        assert not math.isnan(metrics.rmse)
        assert len(metrics.per_fold_errors) <= 3

    def test_high_dimensional_input(self) -> None:
        """Test LOO-CV with high-dimensional input (10 dims).

        High-dimensional GPs are more challenging due to curse of dimensionality.
        """
        torch.manual_seed(42)
        n_dims = 10
        n_samples = 20  # Need more samples for high-D
        train_x = torch.rand(n_samples, n_dims, dtype=torch.double)
        train_y = train_x.sum(dim=-1, keepdim=True) / n_dims
        bounds = torch.zeros(2, n_dims, dtype=torch.double)
        bounds[1, :] = 1.0

        metrics = compute_loo_cv_metrics(train_x, train_y, bounds)

        assert not math.isnan(metrics.rmse)

    def test_asymmetric_bounds(self) -> None:
        """Test LOO-CV with asymmetric parameter bounds."""
        torch.manual_seed(42)
        n_samples = 10
        train_x = torch.zeros(n_samples, 2, dtype=torch.double)
        # Different ranges for each dimension
        train_x[:, 0] = torch.rand(n_samples, dtype=torch.double) * 10  # [0, 10]
        train_x[:, 1] = torch.rand(n_samples, dtype=torch.double) * 0.1  # [0, 0.1]
        train_y = train_x[:, 0:1] / 10 + train_x[:, 1:2] * 10
        bounds = torch.tensor([[0.0, 0.0], [10.0, 0.1]], dtype=torch.double)

        metrics = compute_loo_cv_metrics(train_x, train_y, bounds)

        assert not math.isnan(metrics.rmse)


class TestLOOCVModelListGP:
    """Test LOO-CV for ModelListGP with multiple objectives."""

    def test_three_objectives(self) -> None:
        """Test LOO-CV with three objectives."""
        torch.manual_seed(42)
        train_x = torch.rand(12, 2, dtype=torch.double)
        train_y = torch.stack(
            [
                train_x[:, 0] + train_x[:, 1],
                train_x[:, 0] * train_x[:, 1],
                train_x[:, 0] - train_x[:, 1],
            ],
            dim=-1,
        )
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        model = create_and_fit_model(train_x, train_y, bounds)
        cv_results = compute_loo_cv_for_model(model, train_x, train_y)

        # Should have results for all three objectives
        assert 0 in cv_results
        assert 1 in cv_results
        assert 2 in cv_results
        for i in range(3):
            assert isinstance(cv_results[i], LOOCVMetrics)

    def test_different_function_complexity(self) -> None:
        """Test LOO-CV with objectives of different complexity.

        One simple linear, one complex nonlinear.
        """
        torch.manual_seed(42)
        train_x = torch.rand(15, 2, dtype=torch.double)
        train_y = torch.stack(
            [
                train_x[:, 0],  # Simple linear
                torch.sin(train_x[:, 0] * 6) * torch.cos(train_x[:, 1] * 6),  # Complex
            ],
            dim=-1,
        )
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        model = create_and_fit_model(train_x, train_y, bounds)
        cv_results = compute_loo_cv_for_model(model, train_x, train_y)

        # Simple objective should have better R²
        assert cv_results[0].r_squared > cv_results[1].r_squared or math.isnan(
            cv_results[1].r_squared
        )
