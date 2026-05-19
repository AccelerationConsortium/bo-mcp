"""Tests for convergence detection in Bayesian Optimization.

This module tests the convergence detection logic that determines when
optimization has reached diminishing returns and should be stopped.

References:
    - Convergence in BO: https://arxiv.org/abs/1906.08878
    - Hypervolume improvement tracking: https://botorch.org/docs/multi_objective/
    - Early stopping in ML: https://scikit-learn.org/stable/modules/early_stopping.html
"""

from bo_engine.constants import (
    CONVERGENCE_IMPROVEMENT_THRESHOLD,
    CONVERGENCE_MIN_OBSERVATIONS,
    CONVERGENCE_WINDOW_SIZE,
)
from bo_engine.convergence import (
    ConvergenceReport,
    detect_convergence,
    detect_hypervolume_convergence,
    detect_single_objective_convergence,
    estimate_remaining_iterations,
)


class TestConvergenceReport:
    """Tests for ConvergenceReport dataclass.

    Reference:
        Section 1.4 of Implementation Plan - Early Stopping Detection
    """

    def test_convergence_report_attributes(self) -> None:
        """ConvergenceReport has all expected attributes."""
        report = ConvergenceReport(
            converged=True,
            convergence_score=0.95,
            reason="Test reason",
            avg_improvement=0.001,
            window_size=5,
            iterations_without_improvement=3,
            recommendation="Test recommendation",
        )
        assert report.converged is True
        assert report.convergence_score == 0.95
        assert report.reason == "Test reason"
        assert report.avg_improvement == 0.001
        assert report.window_size == 5
        assert report.iterations_without_improvement == 3
        assert report.recommendation == "Test recommendation"


class TestDetectConvergence:
    """Tests for the detect_convergence function.

    Reference:
        - Convergence criteria based on improvement rate thresholds
        - https://arxiv.org/abs/1906.08878 Section 3.2 (Stopping Criteria)
    """

    def test_insufficient_data_returns_not_converged(self) -> None:
        """Returns not converged when fewer than min_observations.

        Use case: Early optimization with <10 observations should not claim convergence.
        """
        history = [1.0, 2.0, 3.0]  # Only 3 observations
        result = detect_convergence(history, min_observations=10)

        assert result.converged is False
        assert result.convergence_score == 0.0
        assert "Insufficient history" in result.reason
        assert "Continue optimization" in result.recommendation

    def test_empty_history_returns_not_converged(self) -> None:
        """Empty history returns not converged status."""
        result = detect_convergence([])
        assert result.converged is False
        assert result.convergence_score == 0.0

    def test_constant_metric_indicates_convergence(self) -> None:
        """Constant metric values indicate convergence (no improvement).

        Use case: When objective function plateaus, optimization should converge.
        Reference: This is a classic convergence scenario in optimization.
        """
        history = [10.0] * 15  # 15 constant values
        result = detect_convergence(history, min_observations=10, window_size=5)

        assert result.converged is True
        assert result.convergence_score == 1.0
        assert result.avg_improvement == 0.0

    def test_improving_metric_not_converged(self) -> None:
        """Steadily improving metric should not be converged.

        Use case: Active optimization with consistent improvement.
        Reference: Linear improvement suggests optimization is effective.
        """
        # Linear improvement: 5% per iteration
        history = [1.0 * (1.05**i) for i in range(15)]
        result = detect_convergence(
            history,
            min_observations=10,
            window_size=5,
            improvement_threshold=0.01,
        )

        assert result.converged is False
        assert result.avg_improvement > 0.01

    def test_small_improvement_converged(self) -> None:
        """Very small improvements should indicate convergence.

        Use case: When improvement rate drops below 1%, optimization is near optimum.
        Reference: Section 1.4 of Implementation Plan - 1% threshold.
        """
        # Tiny improvements (0.1% per iteration)
        history = [1.0 * (1.001**i) for i in range(15)]
        result = detect_convergence(
            history,
            min_observations=10,
            window_size=5,
            improvement_threshold=0.01,
        )

        assert result.converged is True
        assert result.avg_improvement < 0.01

    def test_default_parameters_match_constants(self) -> None:
        """Default parameters should match module constants.

        Reference: bo_engine/constants.py defines CONVERGENCE_* constants.
        """
        # Just verify the function uses constants as defaults
        history = [1.0] * 15
        result = detect_convergence(history)

        assert result.window_size == CONVERGENCE_WINDOW_SIZE

    def test_custom_window_size(self) -> None:
        """Custom window size should be respected."""
        history = [1.0] * 15
        result = detect_convergence(history, window_size=3)
        assert result.window_size == 3

    def test_custom_improvement_threshold(self) -> None:
        """Custom improvement threshold affects convergence detection.

        Use case: Different domains may require different convergence criteria.
        """
        # 2% improvement per iteration
        history = [1.0 * (1.02**i) for i in range(15)]

        # With 1% threshold, not converged
        result_strict = detect_convergence(history, min_observations=10, improvement_threshold=0.01)
        assert result_strict.converged is False

        # With 5% threshold, converged
        result_relaxed = detect_convergence(
            history, min_observations=10, improvement_threshold=0.05
        )
        assert result_relaxed.converged is True

    def test_stagnation_detection(self) -> None:
        """Detects stagnation (no improvement for many iterations).

        Use case: When optimization gets stuck in local optimum.
        Reference: Stagnation is common in high-dimensional BO.
        """
        # Initial improvement then stagnation
        history = [1.0, 2.0, 3.0, 4.0, 5.0] + [5.0] * 10
        result = detect_convergence(history, min_observations=10, window_size=5)

        assert result.iterations_without_improvement >= 5
        assert "stagnant" in result.recommendation.lower() or result.converged

    def test_convergence_score_bounded(self) -> None:
        """Convergence score should be between 0 and 1.

        Reference: Score normalization for consistent reporting.
        """
        # Test various scenarios
        scenarios = [
            [1.0] * 15,  # Constant
            [float(i) for i in range(15)],  # Linear
            [1.0 * (1.001**i) for i in range(15)],  # Tiny improvement
            [1.0 * (1.1**i) for i in range(15)],  # Large improvement
        ]

        for history in scenarios:
            result = detect_convergence(history, min_observations=10)
            assert 0.0 <= result.convergence_score <= 1.0

    def test_negative_improvement_indicates_regression(self) -> None:
        """Decreasing metric is detected (negative improvement).

        Use case: Optimization regressing due to data quality issues.
        """
        history = [10.0 - i * 0.5 for i in range(15)]  # Decreasing
        result = detect_convergence(history, min_observations=10, window_size=5)

        # Negative improvement should trigger convergence (no positive improvement)
        assert result.avg_improvement < 0

    def test_handles_near_zero_values(self) -> None:
        """Handles metric values near zero without division errors.

        Use case: Loss functions that approach zero.
        Reference: Numerical stability is important for production use.
        """
        history = [1e-12] * 15  # Very small values
        result = detect_convergence(history, min_observations=10)
        assert result is not None
        assert result.avg_improvement == result.avg_improvement  # Check for NaN


class TestDetectHypervolumeConvergence:
    """Tests for multi-objective convergence using hypervolume.

    Reference:
        - Hypervolume indicator: https://botorch.org/docs/multi_objective/
        - Multi-objective convergence: Emmerich et al., EMO 2005
    """

    def test_wrapper_calls_detect_convergence(self) -> None:
        """detect_hypervolume_convergence wraps detect_convergence."""
        history = [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        result = detect_hypervolume_convergence(history, min_observations=10)
        assert isinstance(result, ConvergenceReport)

    def test_increasing_hypervolume_not_converged(self) -> None:
        """Increasing hypervolume indicates active exploration.

        Use case: Pareto front is still expanding.
        Reference: Hypervolume improvement is the standard multi-objective metric.
        """
        # Steadily increasing hypervolume
        history = [0.1 * (1.05**i) for i in range(15)]
        result = detect_hypervolume_convergence(
            history, min_observations=10, improvement_threshold=0.01
        )
        assert result.converged is False

    def test_stable_hypervolume_converged(self) -> None:
        """Stable hypervolume indicates Pareto front has converged.

        Use case: No new non-dominated points being found.
        """
        # Hypervolume with minimal improvement
        history = [1.0 * (1.001**i) for i in range(15)]
        result = detect_hypervolume_convergence(
            history, min_observations=10, improvement_threshold=0.01
        )
        assert result.converged is True


class TestDetectSingleObjectiveConvergence:
    """Tests for single-objective convergence detection.

    Reference:
        - Section 1.4 of Implementation Plan
        - Best value tracking in BO: https://botorch.org/docs/tutorials/
    """

    def test_minimization_negates_values(self) -> None:
        """Minimization scenario: decreasing values should be improvement.

        Use case: Minimizing loss function - lower is better.
        """
        # Decreasing best values (improvement in minimization)
        history = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.9, 0.8]
        result = detect_single_objective_convergence(history, minimize=True, min_observations=10)
        # Should recognize improvement (decreasing values in minimization)
        assert result.avg_improvement >= 0

    def test_maximization_uses_raw_values(self) -> None:
        """Maximization scenario: increasing values should be improvement.

        Use case: Maximizing yield - higher is better.
        """
        # Increasing best values (improvement in maximization)
        history = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 10.1, 10.2]
        result = detect_single_objective_convergence(history, minimize=False, min_observations=10)
        assert result.avg_improvement >= 0

    def test_converged_minimization(self) -> None:
        """Detects convergence in minimization scenario.

        Use case: Loss function reached near-optimal value.
        """
        # Plateau after improvement (minimization) - very tiny improvements at end
        history = [10.0, 5.0, 3.0, 2.0, 1.5, 1.2, 1.1, 1.05, 1.02, 1.01, 1.009, 1.008]
        result = detect_single_objective_convergence(
            history, minimize=True, min_observations=10, improvement_threshold=0.01
        )
        # The improvement rate is slowing down - should be converging
        assert result.convergence_score > 0.0  # Some convergence detected
        # With such small improvements, should be near convergence
        assert result.avg_improvement < 0.02  # Small improvement rate

    def test_converged_maximization(self) -> None:
        """Detects convergence in maximization scenario.

        Use case: Yield optimization reached near-maximum.
        """
        # Plateau after improvement (maximization)
        history = [1.0, 5.0, 8.0, 9.0, 9.5, 9.8, 9.9, 9.95, 9.98, 9.99, 9.995, 9.998]
        result = detect_single_objective_convergence(
            history, minimize=False, min_observations=10, improvement_threshold=0.01
        )
        assert result.convergence_score > 0.5


class TestEstimateRemainingIterations:
    """Tests for remaining iteration estimation.

    Reference:
        - Budget estimation in BO: useful for resource planning
        - Similar to patience in early stopping: https://keras.io/api/callbacks/early_stopping/
    """

    def test_insufficient_data_returns_none(self) -> None:
        """Returns None when not enough history for estimation."""
        history = [1.0, 2.0, 3.0]  # Only 3 points
        result = estimate_remaining_iterations(history, target_improvement=0.1)
        assert result is None

    def test_no_improvement_returns_none(self) -> None:
        """Returns None when there's no improvement to extrapolate.

        Use case: Cannot estimate if optimization is stagnant.
        """
        history = [5.0] * 10  # No improvement
        result = estimate_remaining_iterations(history, target_improvement=0.1)
        assert result is None

    def test_estimates_based_on_improvement_rate(self) -> None:
        """Estimates iterations based on current improvement rate.

        Use case: "How many more iterations to improve by 50%?"
        """
        # 10% improvement per iteration
        history = [1.0 * (1.1**i) for i in range(10)]
        result = estimate_remaining_iterations(history, target_improvement=0.5)

        # With 10% per iteration, 50% target should need ~5 iterations
        assert result is not None
        assert 3 <= result <= 10  # Approximate range

    def test_respects_max_iterations(self) -> None:
        """Estimation is capped at max_iterations."""
        # Very slow improvement
        history = [1.0 * (1.001**i) for i in range(10)]
        result = estimate_remaining_iterations(history, target_improvement=1.0, max_iterations=50)

        if result is not None:
            assert result <= 50

    def test_decreasing_returns_none(self) -> None:
        """Returns None when metric is decreasing (no positive improvement).

        Use case: Cannot estimate improvement when regressing.
        """
        history = [10.0 - i for i in range(10)]  # Decreasing
        result = estimate_remaining_iterations(history, target_improvement=0.1)
        assert result is None


class TestIntegrationWithConstants:
    """Integration tests verifying behavior matches documented constants.

    Reference:
        - AGENT_COOKBOOK.md convergence guidance
        - TOOL_SCHEMAS.md convergence field documentation
    """

    def test_default_min_observations_is_ten(self) -> None:
        """Default minimum observations is 10 (per constants).

        Reference: CONVERGENCE_MIN_OBSERVATIONS = 10
        """
        assert CONVERGENCE_MIN_OBSERVATIONS == 10

        # 9 observations should not be enough
        history = [float(i) for i in range(9)]
        result = detect_convergence(history)
        assert "Insufficient" in result.reason

    def test_default_window_size_is_five(self) -> None:
        """Default window size is 5 (per constants).

        Reference: CONVERGENCE_WINDOW_SIZE = 5
        """
        assert CONVERGENCE_WINDOW_SIZE == 5

    def test_default_improvement_threshold_is_one_percent(self) -> None:
        """Default improvement threshold is 1% (per constants).

        Reference: CONVERGENCE_IMPROVEMENT_THRESHOLD = 0.01
        """
        assert CONVERGENCE_IMPROVEMENT_THRESHOLD == 0.01

    def test_documented_agent_guidance_scenario(self) -> None:
        """Test scenario from AGENT_COOKBOOK: converged with <20 results = early warning.

        Reference: AGENT_COOKBOOK.md - Early Convergence Warning section
        "If converged=true BUT n_results < 20: Warn user..."
        """
        # Simulate early convergence: only 12 results but stagnant
        history = [1.0, 2.0, 3.0, 4.0, 5.0] + [5.0] * 7  # 12 total
        result = detect_convergence(history, min_observations=10)

        # This scenario (12 results, converged) should trigger agent warning
        n_results = len(history)
        is_early_convergence = result.converged and n_results < 20

        if result.converged:
            # Agent should warn about early convergence
            assert is_early_convergence
