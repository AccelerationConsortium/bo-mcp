"""Tests for end-to-end campaign-health computation.

Covers :func:`bo_engine.diagnostics.analyze_hypervolume_history`,
:func:`compute_single_objective_progress_status`, and
:func:`compute_campaign_health`. The server layer wraps the last one
verbatim; these tests pin the engine-level behavior so the server cannot
re-derive a divergent answer.

Reference: hypervolume-history-based stagnation detection is a standard
multi-objective BO diagnostic; see e.g. Daulton et al. (2020),
"Differentiable Expected Hypervolume Improvement for Parallel
Multi-Objective Bayesian Optimization."
"""

from __future__ import annotations

from bo_engine.constants import (
    DIAGNOSTICS_HYPERVOLUME_DECREASE_WARNING,
    FALLBACK_HYPERVOLUME_IMPROVEMENT,
    MIN_IMPROVEMENT_RATE,
)
from bo_engine.diagnostics import (
    analyze_hypervolume_history,
    compute_campaign_health,
    compute_single_objective_progress_status,
)


class TestAnalyzeHypervolumeHistory:
    def test_short_history_returns_fallback_when_hv_present(self) -> None:
        """Single-sample history with a positive HV uses the fallback delta."""
        improvement, stagnant = analyze_hypervolume_history(
            hypervolume_history=[2.0],
            n_results=3,
            current_hypervolume=2.0,
        )
        assert improvement == FALLBACK_HYPERVOLUME_IMPROVEMENT
        assert stagnant == 0

    def test_short_history_without_hv_signals_no_improvement(self) -> None:
        improvement, stagnant = analyze_hypervolume_history(
            hypervolume_history=[],
            n_results=3,
            current_hypervolume=0.0,
        )
        assert improvement == 0.0
        assert stagnant == 0

    def test_monotone_increase_zero_stagnation(self) -> None:
        improvement, stagnant = analyze_hypervolume_history(
            hypervolume_history=[1.0, 2.0, 3.5],
            n_results=3,
            current_hypervolume=3.5,
        )
        assert improvement == 1.5
        assert stagnant == 0

    def test_flat_tail_counts_stagnant_iterations(self) -> None:
        improvement, stagnant = analyze_hypervolume_history(
            hypervolume_history=[1.0, 2.0, 2.0, 2.0],
            n_results=4,
            current_hypervolume=2.0,
        )
        # Last delta is zero -> non-negative improvement.
        assert improvement == 0.0
        # Three identical tail samples means two stagnant transitions.
        assert stagnant == 2

    def test_regression_returns_signed_delta(self) -> None:
        """A genuine regression must surface as a negative delta.

        The downstream ``DIAGNOSTICS_HYPERVOLUME_DECREASE_WARNING``
        threshold is negative, so clamping to ``>= 0`` would silently
        suppress the warning — see :func:`determine_health_status`.
        """
        improvement, _ = analyze_hypervolume_history(
            hypervolume_history=[5.0, 4.0],
            n_results=2,
            current_hypervolume=4.0,
        )
        assert improvement == -1.0
        # Sanity: the delta is large enough to trip the decrease warning.
        assert improvement < DIAGNOSTICS_HYPERVOLUME_DECREASE_WARNING


class TestSingleObjectiveProgressStatus:
    def test_strictly_above_min_rate_is_improving(self) -> None:
        assert compute_single_objective_progress_status(MIN_IMPROVEMENT_RATE + 0.01) == "improving"

    def test_at_or_below_min_rate_is_stable(self) -> None:
        assert compute_single_objective_progress_status(MIN_IMPROVEMENT_RATE) == "stable"
        assert compute_single_objective_progress_status(0.0) == "stable"
        assert compute_single_objective_progress_status(-1.0) == "stable"


class TestComputeCampaignHealth:
    def test_single_objective_branch_routes_to_improvement_history(self) -> None:
        diagnostics = {
            "improvement_history": [0.0, 0.1, 0.2, 0.3, 0.45],
            "improvement_rate": 0.5,
        }
        status, warnings, progress = compute_campaign_health(
            is_single_objective=True,
            n_results=5,
            diagnostics=diagnostics,
            model_correlation=0.8,
            hypervolume_history=[],
        )
        assert status == "healthy"
        assert progress == "improving"
        assert isinstance(warnings, list)

    def test_multi_objective_branch_uses_history(self) -> None:
        status, warnings, progress = compute_campaign_health(
            is_single_objective=False,
            n_results=6,
            diagnostics={"hypervolume": 3.0},
            model_correlation=0.8,
            hypervolume_history=[1.0, 1.5, 2.5, 3.0],
        )
        assert status in {"healthy", "warning", "critical"}
        assert progress in {"improving", "stagnant", "regressing"}
        assert isinstance(warnings, list)

    def test_multi_objective_fallback_progress_when_history_empty(self) -> None:
        """Empty history but a non-zero current HV — progress uses the single-sample default."""
        status, _, progress = compute_campaign_health(
            is_single_objective=False,
            n_results=3,
            diagnostics={"hypervolume": 1.5},
            model_correlation=0.5,
            hypervolume_history=[],
        )
        # Single-sample progress defaults to "improving" per determine_progress_status.
        assert progress == "improving"
        assert status in {"healthy", "warning", "critical"}

    def test_hypervolume_decrease_warning_fires(self) -> None:
        """The decrease warning is reachable now that the delta is signed.

        Pre-fix: ``analyze_hypervolume_history`` clamped its delta to
        ``>= 0`` so the downstream ``< DIAGNOSTICS_HYPERVOLUME_DECREASE_WARNING``
        branch was dead code. This test pins the round-trip so the bug
        cannot regress silently.
        """
        # Big enough sample count to clear DIAGNOSTICS_MIN_RESULTS_FOR_CORRELATION_WARNING (5).
        _, warnings, _ = compute_campaign_health(
            is_single_objective=False,
            n_results=8,
            diagnostics={"hypervolume": 4.0},
            model_correlation=0.8,
            hypervolume_history=[5.0, 4.0, 3.0],
        )
        assert any("Hypervolume is decreasing" in w for w in warnings)

    def test_does_not_mutate_diagnostics(self) -> None:
        diagnostics = {"improvement_history": [], "improvement_rate": 0.0}
        snapshot = dict(diagnostics)
        compute_campaign_health(
            is_single_objective=True,
            n_results=0,
            diagnostics=diagnostics,
            model_correlation=0.0,
            hypervolume_history=[],
        )
        assert diagnostics == snapshot
