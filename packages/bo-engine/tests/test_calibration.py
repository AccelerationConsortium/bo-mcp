"""Calibration report consistency tests.

The user-facing calibration report carries two verdicts derived from the
same ``mean_error``: the boolean ``is_well_calibrated`` flag and the
recommendation text. Both must key off the same threshold
(``CALIBRATION_GOOD_THRESHOLD``) — a report claiming
``is_well_calibrated=True`` while its recommendation warns about
systematic over/under-confidence is self-contradictory messaging.

References:
    - Kuleshov et al., "Accurate Uncertainties for Deep Learning Using
      Calibrated Regression" (ICML 2018) — calibration error as the mean
      absolute deviation between nominal and observed coverage, the metric
      both verdicts summarize.
"""

from __future__ import annotations

from bo_engine.calibration import _generate_calibration_recommendation
from bo_engine.constants import CALIBRATION_GOOD_THRESHOLD


class TestRecommendationAgreesWithFlag:
    """Flag and recommendation text must share one threshold."""

    def test_error_between_old_and_current_threshold_reads_well_calibrated(self) -> None:
        """``mean_error`` just under the threshold must produce agreeing verdicts.

        0.07 sits between the previously hardcoded 0.05 literal and
        ``CALIBRATION_GOOD_THRESHOLD`` (0.1): the flag says well-calibrated,
        so the recommendation must too.
        """
        mean_error = 0.07
        assert mean_error < CALIBRATION_GOOD_THRESHOLD  # flag: is_well_calibrated=True

        recommendation = _generate_calibration_recommendation(mean_error, coverage_results=[])
        assert "well-calibrated" in recommendation

    def test_error_above_threshold_does_not_read_well_calibrated(self) -> None:
        mean_error = CALIBRATION_GOOD_THRESHOLD + 0.05
        recommendation = _generate_calibration_recommendation(mean_error, coverage_results=[])
        assert "well-calibrated" not in recommendation
