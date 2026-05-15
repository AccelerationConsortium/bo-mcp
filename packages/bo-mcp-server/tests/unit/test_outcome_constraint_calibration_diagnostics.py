"""Tests for the outcome-constraint calibration diagnostics helper.

The helper plumbs ``bo_engine.compute_outcome_constraint_calibration`` into
the server-side diagnostics payload (TODO 1.52). These tests pin both the
happy-path emission and the threshold-driven WARN behaviour so agents see
a single, actionable signal when the feasibility GP is overconfident.

Reference: BO with unknown constraints (Gelbart et al., UAI 2014) shows
that miscalibrated feasibility predictions push the acquisition into
regions that are infeasible in practice; the calibration metric is the
canonical early-warning signal for that failure.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from bo_engine.constants import CONSTRAINT_CALIBRATION_WARN_THRESHOLD

from bo_mcp_server.domain import (
    CampaignSpec,
    InputParameter,
    Objective,
    OutcomeConstraint,
    ParameterType,
)
from bo_mcp_server.domain.result import Result, ResultSource
from bo_mcp_server.operations.diagnostics.constraints import (
    compute_outcome_constraint_calibration_metrics,
)


def _make_spec(
    *,
    threshold: float = 0.5,
    greater_than: bool = True,
) -> CampaignSpec:
    """Build a single-parameter, single-objective spec with one outcome constraint."""
    return CampaignSpec(
        name="diagnostics-cal",
        parameters=(
            InputParameter(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),  # ty: ignore[invalid-argument-type]
        ),
        objectives=(Objective(name="y", direction="minimize"),),
        outcome_constraints=(
            OutcomeConstraint(
                objective_name="y",
                threshold=threshold,
                greater_than=greater_than,
            ),
        ),
    )


def _make_result(x: float, y: float, *, campaign_id, submitted_by) -> Result:
    """Construct a domain ``Result`` with a single parameter/objective pair."""
    return Result(
        campaign_id=campaign_id,
        parameter_values={"x": x},
        objective_values={"y": y},
        source=ResultSource.API,
        submitted_by=submitted_by,
    )


def _balanced_results(*, campaign_id, submitted_by, threshold: float = 0.5) -> list[Result]:
    """8 feasible / 8 infeasible 1-D rows that the constraint GP can separate."""
    feasible_xs = [round(0.05 * (i + 1), 4) for i in range(8)]
    infeasible_xs = [round(0.55 + 0.05 * i, 4) for i in range(8)]
    rows = []
    for x in feasible_xs:
        rows.append(
            _make_result(
                x=x,
                y=threshold - 0.25,
                campaign_id=campaign_id,
                submitted_by=submitted_by,
            )
        )
    for x in infeasible_xs:
        rows.append(
            _make_result(
                x=x,
                y=threshold + 0.25,
                campaign_id=campaign_id,
                submitted_by=submitted_by,
            )
        )
    return rows


class TestOutcomeConstraintCalibrationDiagnostics:
    """``compute_outcome_constraint_calibration_metrics`` populates diagnostics."""

    def test_no_outcome_constraints_short_circuits(self) -> None:
        """A spec without outcome constraints sets the key to None."""
        spec = CampaignSpec(
            name="no-oc",
            parameters=(
                InputParameter(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),  # ty: ignore[invalid-argument-type]
            ),
            objectives=(Objective(name="y", direction="minimize"),),
        )
        diagnostics: dict = {}
        compute_outcome_constraint_calibration_metrics([], spec, diagnostics)
        assert diagnostics["outcome_constraint_calibration"] is None

    def test_emits_one_report_per_outcome_constraint(self) -> None:
        """Per-constraint metrics are surfaced under ``outcome_constraint_calibration``."""
        spec = _make_spec()
        campaign_id = uuid4()
        submitted_by = uuid4()
        results = _balanced_results(campaign_id=campaign_id, submitted_by=submitted_by)

        diagnostics: dict = {}
        compute_outcome_constraint_calibration_metrics(results, spec, diagnostics)

        reports = diagnostics["outcome_constraint_calibration"]
        assert isinstance(reports, list)
        assert len(reports) == 1
        report = reports[0]
        assert report["constraint_name"] == "y"
        assert "brier_score" in report
        assert "expected_calibration_error" in report
        assert "calibration_error" in report

    def test_no_warning_when_calibrated(self) -> None:
        """A well-separated dataset must not raise the miscalibration warning."""
        spec = _make_spec()
        campaign_id = uuid4()
        submitted_by = uuid4()
        results = _balanced_results(campaign_id=campaign_id, submitted_by=submitted_by)

        diagnostics: dict = {}
        compute_outcome_constraint_calibration_metrics(results, spec, diagnostics)

        assert "warnings" not in diagnostics or all(
            "Outcome-constraint model" not in w for w in diagnostics["warnings"]
        )

    def test_warning_fires_above_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Force the calibration_error past the threshold and check WARN text.

        Patches the engine helper so the test is fast and deterministic — the
        underlying calibration metric is exercised by the bo-engine test
        suite, this test pins only the server-side WARN routing.
        """
        spec = _make_spec()
        campaign_id = uuid4()
        submitted_by = uuid4()
        results = _balanced_results(campaign_id=campaign_id, submitted_by=submitted_by)

        bad_error = CONSTRAINT_CALIBRATION_WARN_THRESHOLD + 0.1

        def fake_compute(*, constraint_specs, **_: object) -> list[dict]:
            return [
                {
                    "constraint_name": cs.objective_name,
                    "threshold": cs.threshold,
                    "greater_than": cs.greater_than,
                    "calibration_error": bad_error,
                    "brier_score": 0.4,
                    "expected_calibration_error": 0.3,
                    "is_calibrated": False,
                    "auc": 0.5,
                }
                for cs in constraint_specs
            ]

        monkeypatch.setattr(
            "bo_mcp_server.operations.diagnostics.constraints"
            ".compute_outcome_constraint_calibration",
            fake_compute,
        )

        diagnostics: dict = {}
        compute_outcome_constraint_calibration_metrics(results, spec, diagnostics)

        warnings = diagnostics.get("warnings", [])
        assert any("Outcome-constraint model" in w for w in warnings)
        # Threshold is referenced in the message so agents know which bar to clear.
        assert any(str(CONSTRAINT_CALIBRATION_WARN_THRESHOLD) in w for w in warnings)

    def test_empty_results_short_circuits(self) -> None:
        """No results → no calibration to assess."""
        spec = _make_spec()
        diagnostics: dict = {}
        compute_outcome_constraint_calibration_metrics([], spec, diagnostics)
        assert diagnostics["outcome_constraint_calibration"] is None

    def test_partial_missing_objective_does_not_drop_section(self) -> None:
        """Rows without the constrained objective are skipped per-constraint.

        Regression: before the per-constraint filter, ``train_x`` was built
        from every result while objective values were only collected when
        present. With 8 of 10 rows carrying the objective, the constraint
        GP saw an 8-row label tensor against a 10-row input tensor and
        crashed the entire diagnostics section.
        """
        spec = _make_spec()
        campaign_id = uuid4()
        submitted_by = uuid4()
        results = _balanced_results(campaign_id=campaign_id, submitted_by=submitted_by)
        # Two extra rows that lack the constrained objective ("y") entirely —
        # simulates historical data with a different objective column set.
        for x in (0.91, 0.95):
            results.append(
                Result(
                    campaign_id=campaign_id,
                    parameter_values={"x": x},
                    objective_values={"other": 0.7},
                    source=ResultSource.API,
                    submitted_by=submitted_by,
                )
            )

        diagnostics: dict = {}
        compute_outcome_constraint_calibration_metrics(results, spec, diagnostics)

        reports = diagnostics["outcome_constraint_calibration"]
        assert isinstance(reports, list)
        assert len(reports) == 1
        report = reports[0]
        # Section is no longer dropped — the calibration metric is present.
        assert report.get("error") is None
        assert "brier_score" in report

    def test_constraint_with_no_observations_emits_marker(self) -> None:
        """An unobserved-objective constraint yields a structured marker.

        Pins that a constraint pointing at an objective that does not
        appear in any result row collapses to a ``missing_objective``
        entry rather than dropping the whole diagnostics section.
        """
        spec = _make_spec()
        campaign_id = uuid4()
        submitted_by = uuid4()
        # Submit results that only carry an unrelated objective so the
        # constraint on "y" has zero usable rows.
        results = [
            Result(
                campaign_id=campaign_id,
                parameter_values={"x": x / 10.0},
                objective_values={"other": 0.5},
                source=ResultSource.API,
                submitted_by=submitted_by,
            )
            for x in range(5)
        ]

        diagnostics: dict = {}
        compute_outcome_constraint_calibration_metrics(results, spec, diagnostics)

        reports = diagnostics["outcome_constraint_calibration"]
        assert reports == [
            {
                "constraint_name": "y",
                "threshold": 0.5,
                "greater_than": True,
                "error": "missing_objective",
            }
        ]

    def test_constraint_with_one_row_emits_insufficient_data_marker(self) -> None:
        """A constraint with a single usable row gets an ``insufficient_data`` marker."""
        spec = _make_spec()
        campaign_id = uuid4()
        submitted_by = uuid4()
        results = [
            Result(
                campaign_id=campaign_id,
                parameter_values={"x": 0.4},
                objective_values={"y": 0.25},
                source=ResultSource.API,
                submitted_by=submitted_by,
            ),
            # A second row without the objective — should not bump the count.
            Result(
                campaign_id=campaign_id,
                parameter_values={"x": 0.6},
                objective_values={"other": 0.9},
                source=ResultSource.API,
                submitted_by=submitted_by,
            ),
        ]

        diagnostics: dict = {}
        compute_outcome_constraint_calibration_metrics(results, spec, diagnostics)

        reports = diagnostics["outcome_constraint_calibration"]
        assert reports == [
            {
                "constraint_name": "y",
                "threshold": 0.5,
                "greater_than": True,
                "error": "insufficient_data",
                "n_rows": 1,
            }
        ]
