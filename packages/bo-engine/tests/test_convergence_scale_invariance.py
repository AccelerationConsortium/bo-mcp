"""Scale-invariant convergence detection (8.35).

The previous implementation divided the per-iteration delta by
``max(abs(prev_value), ABS_TOL)``. This makes the threshold scale with
the absolute magnitude of the metric, so rescaling the campaign by 1000
(e.g. cost in dollars vs cents) flipped the converged / still-improving
verdict for the same underlying optimization run. The fix replaces the
per-step denominator with a global robust scale (IQR → std → abs latest)
so the convergence verdict depends only on the *shape* of the
trajectory, not its units.

References:
    - Wand, "Data-Based Choice of Histogram Bin Width", American
      Statistician 1997 — IQR is the standard robust scale estimator.
    - Section 1.4 of the implementation plan (existing early-stopping
      detector).
"""

from __future__ import annotations

from bo_engine.convergence import detect_convergence, estimate_remaining_iterations


def _make_trajectory(scale: float) -> list[float]:
    """Build a 12-step trajectory with a tail that's stuck near-zero improvement."""
    return [
        scale * v
        for v in (
            1.0,
            1.5,
            2.2,
            2.6,
            2.95,
            3.0,
            3.001,
            3.0015,
            3.002,
            3.0025,
            3.003,
            3.0035,
        )
    ]


class TestRescalingInvariance:
    """Same trajectory at different absolute scales converges at the same iteration."""

    def test_baseline_vs_1000x_rescaled_agree(self) -> None:
        baseline = _make_trajectory(1.0)
        rescaled = _make_trajectory(1000.0)

        report_baseline = detect_convergence(baseline, window_size=5)
        report_rescaled = detect_convergence(rescaled, window_size=5)

        # The "converged" verdict must agree across scales.
        assert report_baseline.converged == report_rescaled.converged
        # And the count of stagnant iterations must match exactly because the
        # robust scale absorbs the rescaling.
        assert (
            report_baseline.iterations_without_improvement
            == report_rescaled.iterations_without_improvement
        )

    def test_tiny_scale_trajectory_still_detects_convergence(self) -> None:
        report = detect_convergence(_make_trajectory(1e-6), window_size=5)
        # A trajectory with the same shape — even at micro-scale — must still
        # land in the converged regime; previously the absolute floor
        # ``IMPROVEMENT_TOLERANCE_ABSOLUTE`` made tiny-scale deltas look
        # gigantic and forced the verdict to "still improving".
        assert report.converged


class TestEstimateRemainingScaleInvariance:
    """``estimate_remaining_iterations`` must share the scale-invariant rate.

    It previously divided each step by ``abs(prev)`` (with a ``1e-10`` skip),
    re-introducing the exact magnitude dependence that ``detect_convergence``
    was fixed to avoid — so the remaining-iteration estimate could disagree
    with the converged/not-converged verdict computed right beside it. It now
    reuses ``_step_denominator`` over ``_history_scale``.
    """

    def test_estimate_invariant_to_1000x_rescaling(self) -> None:
        # Steadily-improving trajectory; recent prev values sit well above the
        # absolute floor at both scales, so the estimate must be identical.
        history = [1.0, 1.4, 1.9, 2.3, 2.6, 2.8]
        rescaled = [1000.0 * v for v in history]

        assert estimate_remaining_iterations(
            history, target_improvement=0.5
        ) == estimate_remaining_iterations(rescaled, target_improvement=0.5)

    def test_near_zero_prev_does_not_explode_estimate(self) -> None:
        # The recent window opens at a near-zero value. The old
        # ``delta / abs(prev)`` divisor would divide by ~5e-7, producing an
        # astronomically large improvement rate and a degenerate (``None``)
        # estimate; the robust history scale keeps it finite and actionable.
        history = [5e-7, 0.5, 1.0, 1.5, 2.0]
        estimate = estimate_remaining_iterations(history, target_improvement=1.0)

        assert estimate is not None, (
            "A near-zero prev value exploded the improvement rate — the "
            "estimate is still using the scale-dependent abs(prev) divisor."
        )
        assert 1 <= estimate <= 100

    def test_flat_trajectory_returns_none(self) -> None:
        # A flat trajectory has no positive improvement, so no finite horizon
        # can be estimated (returns None rather than the capped maximum).
        history = [5.0, 5.0, 5.0, 5.0, 5.0]
        assert estimate_remaining_iterations(history, target_improvement=0.5) is None
