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

from bo_engine.convergence import detect_convergence


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
