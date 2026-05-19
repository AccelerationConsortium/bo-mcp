"""Tests for the calibration drift gate (TODO 8.56).

The nightly CI job runs ``scripts/calibrate_test_tolerances.py`` with
``--check <baseline>`` so a tolerance regression caused by an
algorithm or library change fails loudly instead of letting the
hard-coded thresholds in ``packages/bo-engine/tests/conftest.py`` rot.

Reference: the drift-band approach (relative magnitude with an eps
floor on the denominator) is the same shape as the
``relativeUncertainty`` knob in pytest-benchmark, documented at
https://pytest-benchmark.readthedocs.io/en/latest/usage.html.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make the scripts dir importable so we can reach into the helpers.
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from calibrate_test_tolerances import (  # type: ignore[import-not-found]  # noqa: E402  # ty: ignore[unresolved-import]
    DEFAULT_DRIFT_ABSOLUTE_FLOOR_BY_METRIC,
    DEFAULT_DRIFT_CONTINUOUS_FLOOR,
    DEFAULT_DRIFT_THRESHOLD,
    ToleranceReport,
    _relative_drift,
    check_against_baseline,
)


def _report(name: str, ci_tolerance: float) -> ToleranceReport:
    return ToleranceReport(
        metric_name=name,
        mean=0.0,
        std=0.0,
        min_val=0.0,
        max_val=0.0,
        percentiles={},
        recommended_ci_tolerance=ci_tolerance,
        recommended_nightly_tolerance=ci_tolerance * 0.9,
    )


def _write_baseline(path: Path, **metrics: float) -> None:
    payload = {
        name: {
            "metric_name": name,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "percentiles": {},
            "recommended_ci_tolerance": value,
            "recommended_nightly_tolerance": value * 0.9,
        }
        for name, value in metrics.items()
    }
    path.write_text(json.dumps(payload))


class TestRelativeDrift:
    def test_zero_drift_returns_zero(self) -> None:
        assert _relative_drift(1.0, 1.0) == pytest.approx(0.0)

    def test_positive_drift_is_relative_to_baseline(self) -> None:
        assert _relative_drift(2.0, 2.5) == pytest.approx(0.25)

    def test_negative_drift_uses_absolute_difference(self) -> None:
        assert _relative_drift(2.0, 1.5) == pytest.approx(0.25)

    def test_eps_floor_protects_against_zero_baseline(self) -> None:
        # A baseline of 0 must not divide by zero — the eps floor keeps
        # the call safe and surfaces an "infinite" drift, which the
        # caller can act on.
        drift = _relative_drift(0.0, 0.5)
        assert drift > 1e6


class TestCheckAgainstBaseline:
    def test_drift_within_threshold_is_silent(self, tmp_path: Path) -> None:
        baseline = tmp_path / "baseline.json"
        _write_baseline(baseline, pareto_max=4.0)
        # A 10% drift sits well inside the 25% default threshold.
        reports = {"pareto_max": _report("pareto_max", 4.4)}
        drifts = check_against_baseline(reports, baseline)
        assert drifts == []

    def test_drift_beyond_threshold_is_reported(self, tmp_path: Path) -> None:
        baseline = tmp_path / "baseline.json"
        _write_baseline(baseline, pareto_max=4.0)
        # 50% drift breaches the 25% default AND the absolute delta of
        # 2.0 breaches the continuous floor, so both gates trip.
        reports = {"pareto_max": _report("pareto_max", 6.0)}
        drifts = check_against_baseline(reports, baseline)
        assert len(drifts) == 1
        assert "pareto_max" in drifts[0]
        assert "drifted by" in drifts[0]

    def test_pareto_size_one_unit_shift_suppressed(self, tmp_path: Path) -> None:
        """A 1-unit shift on ``pareto_size`` is silenced by the per-metric floor.

        Mirrors the production case: ``pareto_size`` lives on integer
        values in ``[2, 8]`` and the recommended-CI formula is
        ``min * 0.9``. A one-unit shift on ``min`` between runs is a
        50 % relative drift but only a 0.9-unit absolute shift, well
        under the 1.0 floor that ``DEFAULT_DRIFT_ABSOLUTE_FLOOR_BY_METRIC``
        carries for that metric.
        """
        baseline = tmp_path / "baseline.json"
        _write_baseline(baseline, pareto_size=1.8)
        reports = {"pareto_size": _report("pareto_size", 0.9)}
        drifts = check_against_baseline(reports, baseline)
        assert drifts == []

    def test_continuous_metric_drift_not_suppressed_by_pareto_size_floor(
        self, tmp_path: Path
    ) -> None:
        """A continuous-metric drift below the pareto_size floor still surfaces.

        Regression guard for the previous global-floor implementation:
        a 95 % relative drift on ``final_hypervolume`` (baseline 0.974
        → current 1.9) carries an absolute delta of ~0.93, which the
        old global 1.0 floor silenced. The per-metric split means the
        unlisted ``final_hypervolume`` falls back to the tight
        continuous floor and the drift is correctly reported.
        """
        baseline = tmp_path / "baseline.json"
        _write_baseline(baseline, final_hypervolume=0.974)
        reports = {"final_hypervolume": _report("final_hypervolume", 1.9)}
        drifts = check_against_baseline(reports, baseline)
        assert drifts, "continuous-metric drift should not be suppressed"
        assert "final_hypervolume" in drifts[0]

    def test_large_absolute_drift_still_reported(self, tmp_path: Path) -> None:
        """Genuine regressions clear both the relative and the absolute gates."""
        baseline = tmp_path / "baseline.json"
        _write_baseline(baseline, pareto_max=4.0)
        # 100% drift AND 4.0 absolute units — both gates trip.
        reports = {"pareto_max": _report("pareto_max", 8.0)}
        drifts = check_against_baseline(reports, baseline)
        assert drifts
        assert "drifted by" in drifts[0]

    def test_custom_floor_overrides_default_per_metric(self, tmp_path: Path) -> None:
        """Per-metric overrides win over the continuous default."""
        baseline = tmp_path / "baseline.json"
        _write_baseline(baseline, pareto_max=4.0)
        # 10% drift, abs delta 0.4. With a custom 1.0 floor on
        # pareto_max, the absolute gate suppresses what the relative
        # gate would otherwise flag.
        reports = {"pareto_max": _report("pareto_max", 4.4)}
        drifts = check_against_baseline(
            reports,
            baseline,
            threshold=0.05,
            absolute_floor_by_metric={"pareto_max": 1.0},
        )
        assert drifts == []

    def test_continuous_floor_kwarg_overrides_default(self, tmp_path: Path) -> None:
        """``continuous_floor`` can be tightened/loosened by the caller."""
        baseline = tmp_path / "baseline.json"
        _write_baseline(baseline, pareto_max=4.0)
        reports = {"pareto_max": _report("pareto_max", 4.4)}
        # 10% drift, abs 0.4. A very tight continuous floor (0.001)
        # plus a 5% relative threshold both trip.
        drifts = check_against_baseline(
            reports,
            baseline,
            threshold=0.05,
            continuous_floor=0.001,
        )
        assert drifts
        assert "pareto_max" in drifts[0]

    def test_missing_baseline_file_is_reported(self, tmp_path: Path) -> None:
        baseline = tmp_path / "nonexistent.json"
        reports = {"pareto_max": _report("pareto_max", 4.0)}
        drifts = check_against_baseline(reports, baseline)
        assert len(drifts) == 1
        assert "Baseline file not found" in drifts[0]

    def test_missing_metric_in_baseline_is_reported(self, tmp_path: Path) -> None:
        baseline = tmp_path / "baseline.json"
        _write_baseline(baseline, pareto_max=4.0)
        reports = {
            "pareto_max": _report("pareto_max", 4.0),
            "new_metric": _report("new_metric", 1.0),
        }
        drifts = check_against_baseline(reports, baseline)
        assert drifts == ["new_metric: missing from baseline"]

    def test_default_threshold_is_sensible(self) -> None:
        """The 25% default is the audit-prescribed band."""
        assert pytest.approx(0.25) == DEFAULT_DRIFT_THRESHOLD

    def test_default_pareto_size_floor_carries_one_unit(self) -> None:
        """The integer-valued metric gets a 1.0-unit floor by default."""
        assert DEFAULT_DRIFT_ABSOLUTE_FLOOR_BY_METRIC.get("pareto_size") == pytest.approx(1.0)

    def test_default_continuous_floor_is_tight(self) -> None:
        """The continuous-metric default is small enough to surface real drift.

        Anything ≥ 0.1 would hide a roughly-doubled ``final_hypervolume``;
        we want the absolute gate to be effectively a noise filter, not
        a silencer.
        """
        assert DEFAULT_DRIFT_CONTINUOUS_FLOOR < 0.1
