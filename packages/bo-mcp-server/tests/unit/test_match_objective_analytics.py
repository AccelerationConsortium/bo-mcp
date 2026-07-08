"""MATCH-aware directional analytics across the server surface.

The three-valued objective goal (minimize / maximize / match) cannot be
collapsed into the ``is_minimize`` boolean: for a MATCH objective "best"
is the observation closest to the target, and improvement is measured on
the distance-to-target trajectory. These tests pin the shared analysis
helpers plus every directional consumer (campaign comparison, the
best-point anchor, convergence diagnostics) and the goal-identity
surfaces (comparison signatures, transfer similarity, MCP resource
rendering) for both the legacy ``direction`` and the ``target_mode``
spellings. The M78-era direction tests remain untouched — these cases
are additive MATCH coverage.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from bo_engine.types import ObjectiveTransformKind, TargetMode
from bo_mcp_server.domain import CampaignSpec, Result
from bo_mcp_server.domain.campaign_spec import Bounds, InputParameter, Objective, ParameterType
from bo_mcp_server.domain.result import ResultSource
from bo_mcp_server.operations.helpers import (
    objective_analysis_is_minimize,
    objective_analysis_series,
    objective_identity,
)

_TARGET = 7.4


def _match_objective(name: str = "ph", target: float = _TARGET) -> Objective:
    return Objective(name=name, target_mode=TargetMode.MATCH, target=target)


def _spec(objectives: tuple[Objective, ...]) -> CampaignSpec:
    return CampaignSpec(
        name="Match Analytics",
        parameters=(
            InputParameter(
                name="x",
                type=ParameterType.CONTINUOUS,
                bounds=Bounds(lower=0.0, upper=1.0),
            ),
        ),
        objectives=objectives,
    )


def _results(spec: CampaignSpec, values: list[float]) -> list[Result]:
    campaign_id = uuid4()
    submitted_by = uuid4()
    name = spec.objectives[0].name
    return [
        Result(
            campaign_id=campaign_id,
            parameter_values={"x": 0.1 * i},
            objective_values={name: v},
            source=ResultSource.API,
            submitted_by=submitted_by,
        )
        for i, v in enumerate(values)
    ]


class TestAnalysisHelpers:
    def test_match_series_is_distance_to_target_minimizing(self) -> None:
        metric, minimize = objective_analysis_series(_match_objective(), [10.0, 7.0, 7.5])
        assert metric == pytest.approx([2.6, 0.4, 0.1])
        assert minimize is True
        assert objective_analysis_is_minimize(_match_objective()) is True

    def test_legacy_directions_pass_through_unchanged(self) -> None:
        maximize = Objective(name="y", direction="maximize")
        metric, minimize = objective_analysis_series(maximize, [1.0, 2.0])
        assert metric == [1.0, 2.0]
        assert minimize is False
        assert objective_analysis_is_minimize(maximize) is False

    def test_target_mode_directions_pass_through_unchanged(self) -> None:
        minimize = Objective(name="y", target_mode=TargetMode.MINIMIZE)
        metric, is_min = objective_analysis_series(minimize, [3.0, 1.0])
        assert metric == [3.0, 1.0]
        assert is_min is True


class TestGoalIdentity:
    def test_legacy_and_target_mode_spellings_are_identical(self) -> None:
        legacy = Objective(name="y", direction="minimize")
        modern = Objective(name="y", target_mode=TargetMode.MINIMIZE)
        assert objective_identity(legacy) == objective_identity(modern)

    def test_match_identity_includes_the_target_value(self) -> None:
        assert objective_identity(_match_objective(target=7.4)) != objective_identity(
            _match_objective(target=6.0)
        )

    def test_identity_never_renders_none(self) -> None:
        modern = Objective(name="y", target_mode=TargetMode.MAXIMIZE)
        assert "None" not in objective_identity(modern)

    def test_comparison_signature_groups_spellings_together(self) -> None:
        from bo_mcp_server.operations.compare_campaigns import _objective_signature

        legacy = _spec((Objective(name="y", direction="maximize"),))
        modern = _spec((Objective(name="y", target_mode=TargetMode.MAXIMIZE),))
        assert _objective_signature(legacy) == _objective_signature(modern)

    def test_transfer_similarity_matches_across_spellings(self) -> None:
        from bo_mcp_server.operations.transfer_candidates import (
            _compute_objective_similarity,
        )

        legacy = _spec((Objective(name="y", direction="maximize"),))
        modern = _spec((Objective(name="y", target_mode=TargetMode.MAXIMIZE),))
        assert _compute_objective_similarity(legacy, modern) == pytest.approx(1.0)

    def test_transfer_similarity_separates_different_match_targets(self) -> None:
        from bo_mcp_server.operations.transfer_candidates import (
            _compute_objective_similarity,
        )

        low = _spec((_match_objective(target=6.0),))
        high = _spec((_match_objective(target=7.4),))
        assert _compute_objective_similarity(low, high) == pytest.approx(0.0)


class TestCompareCampaignMetrics:
    @pytest.mark.asyncio
    async def test_match_best_value_is_closest_to_target(self) -> None:
        from bo_mcp_server.operations.compare_campaigns import _compute_campaign_metrics

        spec = _spec((_match_objective(),))
        # Raw max is 10.0 and raw min is 5.0; the closest-to-7.4 value is 7.5.
        results = _results(spec, [10.0, 5.0, 7.5, 6.0])
        metrics = await _compute_campaign_metrics(spec, results, iteration=4)
        assert metrics["best_value"] == pytest.approx(7.5)
        assert metrics["improvement_rate"] is not None

    @pytest.mark.asyncio
    async def test_legacy_direction_metrics_unchanged(self) -> None:
        from bo_mcp_server.operations.compare_campaigns import _compute_campaign_metrics

        spec = _spec((Objective(name="y", direction="minimize"),))
        results = _results(spec, [10.0, 5.0, 7.5])
        metrics = await _compute_campaign_metrics(spec, results, iteration=3)
        assert metrics["best_value"] == pytest.approx(5.0)


class TestConvergenceDiagnostics:
    def test_match_improvement_history_reads_as_minimizing(self) -> None:
        """A shrinking distance trajectory must not read as stagnation.

        The backend's MATCH ``improvement_history`` is a running-best
        distance (decreasing); the pre-fix boolean treated it as a
        maximize series, negating the convergence verdict.
        """
        from bo_mcp_server.operations.diagnostics.convergence import (
            _compute_single_obj_convergence,
        )

        spec = _spec((_match_objective(),))
        # Long enough for the detector's min-observation gate; steadily
        # shrinking distances. Read with the wrong (maximize) direction
        # this trajectory is monotonically "worsening" and converges
        # immediately — the MATCH-aware direction keeps it improving.
        history = [2.6, 2.0, 1.5, 1.1, 0.8, 0.6, 0.45, 0.33, 0.24, 0.17, 0.12]
        diagnostics: dict = {"improvement_history": history}
        _compute_single_obj_convergence(spec, diagnostics)
        assert diagnostics["convergence"] is not None
        assert diagnostics["convergence"]["converged"] is False
        assert "Insufficient" not in diagnostics["convergence"]["reason"]

    def test_match_distance_plateau_converges(self) -> None:
        from bo_mcp_server.operations.diagnostics.convergence import (
            _compute_single_obj_convergence,
        )

        spec = _spec((_match_objective(),))
        diagnostics: dict = {"improvement_history": [2.6, 0.05] + [0.05] * 10}
        _compute_single_obj_convergence(spec, diagnostics)
        assert diagnostics["convergence"]["converged"] is True


class TestResourceRendering:
    def test_target_mode_objective_never_renders_none(self) -> None:
        from bo_mcp_server.resources.campaign_resource import _format_objectives_section

        lines = _format_objectives_section(_spec((_match_objective(),)))
        assert lines == [f"- **ph**: match (target: {_TARGET})"]

    def test_legacy_direction_rendering_is_unchanged(self) -> None:
        from bo_mcp_server.resources.campaign_resource import _format_objectives_section

        lines = _format_objectives_section(_spec((Objective(name="y", direction="minimize"),)))
        assert lines == ["- **y**: minimize"]

    def test_objective_ranges_direction_is_resolved(self) -> None:
        from bo_mcp_server.operations.diagnostics.enrichment import compute_objective_ranges

        spec = _spec((Objective(name="y", target_mode=TargetMode.MAXIMIZE),))
        results = _results(spec, [1.0, 2.0])
        ranges = compute_objective_ranges(spec, results)
        assert ranges["y"]["direction"] == "maximize"


class TestObjectiveTransformIntakeValidation:
    """Per-kind require-and-forbid rules the domain docstring promises."""

    def test_missing_required_field_rejected(self) -> None:
        from bo_mcp_server.domain.campaign_spec import ObjectiveTransform

        with pytest.raises(ValueError, match="requires bounds"):
            ObjectiveTransform(kind=ObjectiveTransformKind.CLAMP)
        with pytest.raises(ValueError, match="requires exponent"):
            ObjectiveTransform(kind=ObjectiveTransformKind.POWER)
        with pytest.raises(ValueError, match="requires center"):
            ObjectiveTransform(kind=ObjectiveTransformKind.SIGMOID, steepness=1.0)

    def test_wrong_kind_extras_rejected(self) -> None:
        from bo_mcp_server.domain.campaign_spec import ObjectiveTransform

        with pytest.raises(ValueError, match="does not take bounds"):
            ObjectiveTransform(kind=ObjectiveTransformKind.LOG, bounds=(0.0, 1.0))
        with pytest.raises(ValueError, match="does not take exponent"):
            ObjectiveTransform(kind=ObjectiveTransformKind.CLAMP, bounds=(0.0, 1.0), exponent=2)

    def test_valid_shapes_accepted(self) -> None:
        from bo_mcp_server.domain.campaign_spec import ObjectiveTransform

        assert ObjectiveTransform(kind=ObjectiveTransformKind.LOG)
        assert ObjectiveTransform(kind=ObjectiveTransformKind.CLAMP, bounds=(0.0, 1.0))
        assert ObjectiveTransform(kind=ObjectiveTransformKind.POWER, exponent=2)
        assert ObjectiveTransform(kind=ObjectiveTransformKind.SIGMOID, center=0.5, steepness=2.0)
