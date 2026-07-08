"""Extended objective surface on the BayBE backend.

Covers the neutral match-a-target modes (``NumericalTarget.match_*``),
the typed target-transform union (log/clamp/power/sigmoid), and the
desirability scalarization (``DesirabilityObjective``), plus the
capability/converter symmetry that keeps intake validation and
construction in lockstep.

References: BayBE targets userguide
(https://emdgroup.github.io/baybe/stable/userguide/targets.html) and
objectives userguide
(https://emdgroup.github.io/baybe/stable/userguide/objectives.html).
"""

from __future__ import annotations

import math
import re
from typing import Any

import pytest
from baybe.objectives import DesirabilityObjective, SingleTargetObjective
from baybe.targets import NumericalTarget

from bo_engine.types import (
    MatchShape,
    ObjectiveSpec,
    ObjectiveTransformKind,
    ObjectiveTransformSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
    ScalarizationMode,
    ScalarizerKind,
    TargetMode,
)
from bo_engine_baybe.backend import BayBEBackend
from bo_engine_baybe.converters import (
    _build_baybe_target,
    objective_support_issues,
    spec_to_objective,
)


def _spec(objectives: list[ObjectiveSpec], **kwargs: Any) -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
        objectives=objectives,
        **kwargs,
    )


class TestMatchTargets:
    def test_match_absolute_default_shape(self) -> None:
        o = ObjectiveSpec(name="y", target_mode=TargetMode.MATCH, target_value=7.4)
        assert _build_baybe_target(o) == NumericalTarget.match_absolute("y", match_value=7.4)

    def test_match_quadratic(self) -> None:
        o = ObjectiveSpec(
            name="y",
            target_mode=TargetMode.MATCH,
            target_value=7.4,
            match_shape=MatchShape.QUADRATIC,
        )
        assert _build_baybe_target(o) == NumericalTarget.match_quadratic("y", match_value=7.4)

    def test_match_bell_uses_scale_as_sigma(self) -> None:
        o = ObjectiveSpec(
            name="y",
            target_mode=TargetMode.MATCH,
            target_value=7.4,
            match_shape=MatchShape.BELL,
            match_scale=0.5,
        )
        assert _build_baybe_target(o) == NumericalTarget.match_bell("y", match_value=7.4, sigma=0.5)

    def test_match_triangular_uses_scale_as_width(self) -> None:
        o = ObjectiveSpec(
            name="y",
            target_mode=TargetMode.MATCH,
            target_value=7.4,
            match_shape=MatchShape.TRIANGULAR,
            match_scale=2.0,
        )
        assert _build_baybe_target(o) == NumericalTarget.match_triangular(
            "y", match_value=7.4, width=2.0
        )


class TestTransformUnion:
    def test_log_transform_boolean_back_compat_is_identical(self) -> None:
        """The legacy boolean and the typed LOG transform build the same target."""
        legacy = _build_baybe_target(ObjectiveSpec(name="y", minimize=True, log_transform=True))
        typed = _build_baybe_target(
            ObjectiveSpec(
                name="y",
                minimize=True,
                transform=ObjectiveTransformSpec(kind=ObjectiveTransformKind.LOG),
            )
        )
        expected = NumericalTarget(name="y", minimize=True).log()
        assert legacy == expected
        assert typed == expected

    def test_clamp_transform(self) -> None:
        o = ObjectiveSpec(
            name="y",
            minimize=True,
            transform=ObjectiveTransformSpec(kind=ObjectiveTransformKind.CLAMP, bounds=(0.0, 10.0)),
        )
        expected = NumericalTarget(name="y", minimize=True).clamp(min=0.0, max=10.0)
        assert _build_baybe_target(o) == expected

    def test_power_transform(self) -> None:
        o = ObjectiveSpec(
            name="y",
            minimize=True,
            transform=ObjectiveTransformSpec(kind=ObjectiveTransformKind.POWER, exponent=2),
        )
        expected = NumericalTarget(name="y", minimize=True).power(2)
        assert _build_baybe_target(o) == expected

    def test_sigmoid_transform_reconstructs_center_and_steepness(self) -> None:
        o = ObjectiveSpec(
            name="y",
            minimize=False,
            transform=ObjectiveTransformSpec(
                kind=ObjectiveTransformKind.SIGMOID, center=5.0, steepness=2.0
            ),
        )
        lower_y = 1.0 / (1.0 + math.e)
        expected = NumericalTarget.normalized_sigmoid(
            "y",
            anchors=[(5.0 - 0.5, lower_y), (5.0 + 0.5, 1.0 - lower_y)],
            minimize=False,
        )
        assert _build_baybe_target(o) == expected

    def test_sigmoid_direction_resolves_via_effective_mode(self) -> None:
        """``target_mode`` overrides the raw ``minimize`` boolean for sigmoid targets.

        The engine dataclass leaves ``minimize=True`` by default, so a
        ``target_mode='maximize'`` sigmoid spec must build the identical
        ascending target as the legacy ``minimize=False`` spelling — the
        stale boolean must not flip the direction.
        """
        transform = ObjectiveTransformSpec(
            kind=ObjectiveTransformKind.SIGMOID, center=5.0, steepness=2.0
        )
        via_mode = _build_baybe_target(
            ObjectiveSpec(name="y", target_mode=TargetMode.MAXIMIZE, transform=transform)
        )
        legacy_max = _build_baybe_target(
            ObjectiveSpec(name="y", minimize=False, transform=transform)
        )
        assert via_mode == legacy_max
        assert via_mode.minimize is False

    def test_sigmoid_direction_override_to_minimize(self) -> None:
        """``target_mode='minimize'`` wins over an explicit ``minimize=False``."""
        transform = ObjectiveTransformSpec(
            kind=ObjectiveTransformKind.SIGMOID, center=5.0, steepness=2.0
        )
        target = _build_baybe_target(
            ObjectiveSpec(
                name="y",
                minimize=False,
                target_mode=TargetMode.MINIMIZE,
                transform=transform,
            )
        )
        assert target.minimize is True


class TestValidationSymmetry:
    """Intake-time issues and construction-time errors come from one source."""

    @pytest.mark.parametrize(
        ("objective", "fragment"),
        [
            (
                ObjectiveSpec(name="y", target_mode=TargetMode.MATCH),
                "requires target_value",
            ),
            (
                ObjectiveSpec(
                    name="y",
                    target_mode=TargetMode.MATCH,
                    target_value=1.0,
                    match_shape=MatchShape.BELL,
                ),
                "requires match_scale",
            ),
            (
                ObjectiveSpec(
                    name="y",
                    target_mode=TargetMode.MATCH,
                    target_value=1.0,
                    match_shape=MatchShape.ABSOLUTE,
                    match_scale=1.0,
                ),
                "takes no match_scale",
            ),
            (
                ObjectiveSpec(name="y", target_value=1.0),
                "require target_mode='match'",
            ),
            (
                ObjectiveSpec(
                    name="y",
                    log_transform=True,
                    transform=ObjectiveTransformSpec(kind=ObjectiveTransformKind.LOG),
                ),
                "both log_transform and transform",
            ),
            (
                ObjectiveSpec(
                    name="y",
                    transform=ObjectiveTransformSpec(kind=ObjectiveTransformKind.CLAMP),
                ),
                "requires bounds",
            ),
            (
                ObjectiveSpec(
                    name="y",
                    transform=ObjectiveTransformSpec(kind=ObjectiveTransformKind.SIGMOID),
                ),
                "requires center and steepness",
            ),
        ],
    )
    def test_invalid_configuration_reported_and_rejected(
        self, objective: ObjectiveSpec, fragment: str
    ) -> None:
        spec = _spec([objective])
        issues = objective_support_issues(spec)
        assert any(fragment in reason for _key, reason in issues), (
            f"expected issue containing {fragment!r}, got {issues}"
        )
        with pytest.raises(ValueError, match=re.escape(fragment)):
            spec_to_objective(spec)
        result = BayBEBackend().validate_capabilities(spec)
        assert not result.is_compatible

    def test_valid_match_spec_is_compatible(self) -> None:
        spec = _spec([ObjectiveSpec(name="y", target_mode=TargetMode.MATCH, target_value=0.5)])
        assert objective_support_issues(spec) == []
        objective = spec_to_objective(spec)
        assert isinstance(objective, SingleTargetObjective)
        assert BayBEBackend().validate_capabilities(spec).is_compatible


class TestDesirability:
    def _desirability_spec(self) -> OptimizationSpec:
        return _spec(
            [
                ObjectiveSpec(
                    name="yield_", minimize=False, weight=2.0, normalization_bounds=(0.0, 100.0)
                ),
                ObjectiveSpec(
                    name="cost", minimize=True, weight=1.0, normalization_bounds=(0.0, 50.0)
                ),
            ],
            scalarization=ScalarizationMode.DESIRABILITY,
            scalarizer=ScalarizerKind.MEAN,
        )

    def test_builds_desirability_objective(self) -> None:
        objective = spec_to_objective(self._desirability_spec())
        assert isinstance(objective, DesirabilityObjective)

    def test_missing_normalization_bounds_is_reported(self) -> None:
        spec = _spec(
            [
                ObjectiveSpec(name="a", minimize=True, normalization_bounds=(0.0, 1.0)),
                ObjectiveSpec(name="b", minimize=True),
            ],
            scalarization=ScalarizationMode.DESIRABILITY,
        )
        issues = objective_support_issues(spec)
        assert any("normalization_bounds" in reason for _key, reason in issues)
        assert not BayBEBackend().validate_capabilities(spec).is_compatible

    def test_scalarizer_without_desirability_is_reported(self) -> None:
        spec = _spec(
            [ObjectiveSpec(name="y", minimize=True)],
            scalarizer=ScalarizerKind.MEAN,
        )
        issues = objective_support_issues(spec)
        assert any("scalarizer" in key for key, _reason in issues)

    def test_bell_match_target_is_allowed_in_desirability(self) -> None:
        spec = _spec(
            [
                ObjectiveSpec(name="a", minimize=True, normalization_bounds=(0.0, 1.0)),
                ObjectiveSpec(
                    name="b",
                    target_mode=TargetMode.MATCH,
                    target_value=5.0,
                    match_shape=MatchShape.BELL,
                    match_scale=1.0,
                ),
            ],
            scalarization=ScalarizationMode.DESIRABILITY,
        )
        assert objective_support_issues(spec) == []
        assert isinstance(spec_to_objective(spec), DesirabilityObjective)


@pytest.mark.slow
class TestBehavioral:
    def test_match_objective_drives_toward_target(self) -> None:
        """A match objective recommends near the target region.

        With a clean linear signal y = x observed on a coarse grid and a
        match target of y = 0.5, the recommended x should land in the
        central region rather than at the extremes (invariant assertion,
        not an exact point — stochastic-test guidance in TESTING.md).
        Mirrors the "match a setpoint" pattern of the BayBE targets
        userguide (https://emdgroup.github.io/baybe/stable/userguide/targets.html).
        """
        spec = _spec(
            [ObjectiveSpec(name="y", target_mode=TargetMode.MATCH, target_value=0.5)],
            random_seed=5,
        )
        observations = [
            ObservationData(parameter_values={"x": v}, objective_values={"y": v})
            for v in (0.0, 0.1, 0.9, 1.0)
        ]
        batch = BayBEBackend().generate_suggestions(
            spec=spec, observations=observations, batch_size=1, iteration=1
        )
        x = float(batch.suggestions[0]["parameter_values"]["x"])
        assert abs(x - 0.5) <= 0.35

    def test_desirability_campaign_generates(self) -> None:
        """A desirability campaign produces a valid suggestion end to end."""
        spec = _spec(
            [
                ObjectiveSpec(
                    name="a", minimize=False, weight=2.0, normalization_bounds=(0.0, 1.0)
                ),
                ObjectiveSpec(name="b", minimize=True, weight=1.0, normalization_bounds=(0.0, 1.0)),
            ],
            scalarization=ScalarizationMode.DESIRABILITY,
            random_seed=7,
        )
        observations = [
            ObservationData(
                parameter_values={"x": v},
                objective_values={"a": v, "b": v**2},
            )
            for v in (0.1, 0.5, 0.9)
        ]
        batch = BayBEBackend().generate_suggestions(
            spec=spec, observations=observations, batch_size=1, iteration=1
        )
        assert len(batch.suggestions) == 1
        x = float(batch.suggestions[0]["parameter_values"]["x"])
        assert 0.0 <= x <= 1.0


class TestDirectionValidationSymmetry:
    """M-class symmetry: transform/log rules resolve via ``effective_mode``.

    Validation (:func:`objective_support_issues`) and construction
    (:func:`_build_baybe_target`) must agree case-by-case whenever the
    three-valued ``target_mode`` overrides the boolean ``minimize`` —
    the pre-fix validator read the raw boolean and produced deferred
    crashes, false rejections, and silent drops.
    """

    @staticmethod
    def _issues(o: ObjectiveSpec) -> list[str]:
        return [reason for _key, reason in objective_support_issues(_spec([o]))]

    def test_log_flag_with_target_mode_maximize_is_rejected_at_intake(self) -> None:
        o = ObjectiveSpec(name="y", log_transform=True, target_mode=TargetMode.MAXIMIZE)
        issues = self._issues(o)
        reports = BayBEBackend().validate_capabilities(_spec([o])).unsupported
        # Reported at intake (capability layer) even though the boolean
        # ``minimize`` default is True; construction agrees by raising.
        assert any(r.key == "objectives[0].log_transform" for r in reports)
        assert not issues  # the boolean path reports via _log_transform_reports
        with pytest.raises(ValueError, match="log_transform=True requires minimize=True"):
            _build_baybe_target(o)

    def test_typed_log_with_target_mode_maximize_is_rejected(self) -> None:
        o = ObjectiveSpec(
            name="y",
            target_mode=TargetMode.MAXIMIZE,
            transform=ObjectiveTransformSpec(kind=ObjectiveTransformKind.LOG),
        )
        assert any("requires minimize=True" in reason for reason in self._issues(o))

    def test_explicit_maximize_flag_with_target_mode_minimize_and_log_is_accepted(self) -> None:
        """The pre-fix validator falsely rejected this consistent spec."""
        o = ObjectiveSpec(
            name="y",
            minimize=False,
            target_mode=TargetMode.MINIMIZE,
            transform=ObjectiveTransformSpec(kind=ObjectiveTransformKind.LOG),
        )
        assert not self._issues(o)
        target = _build_baybe_target(o)
        assert isinstance(target, NumericalTarget)

    def test_boolean_log_flag_with_target_mode_minimize_builds(self) -> None:
        o = ObjectiveSpec(
            name="y", minimize=False, target_mode=TargetMode.MINIMIZE, log_transform=True
        )
        assert not self._issues(o)
        assert isinstance(_build_baybe_target(o), NumericalTarget)

    def test_log_flag_with_match_mode_is_rejected_not_dropped(self) -> None:
        """Silently dropping the flag was the fourth pre-fix sub-case."""
        o = ObjectiveSpec(
            name="y", log_transform=True, target_mode=TargetMode.MATCH, target_value=7.4
        )
        assert any(
            "log_transform cannot be combined with target_mode='match'" in reason
            for reason in self._issues(o)
        )
        with pytest.raises(ValueError, match="log_transform cannot be combined"):
            spec_to_objective(_spec([o]))


class TestDesirabilityTransformRejection:
    """M-class: desirability must reject declared transforms, not drop them."""

    @staticmethod
    def _desirability(objectives: list[ObjectiveSpec]) -> OptimizationSpec:
        return _spec(objectives, scalarization=ScalarizationMode.DESIRABILITY)

    def test_typed_transform_under_desirability_is_reported(self) -> None:
        spec = self._desirability(
            [
                ObjectiveSpec(
                    name="a",
                    minimize=True,
                    normalization_bounds=(0.0, 1.0),
                    transform=ObjectiveTransformSpec(kind=ObjectiveTransformKind.LOG),
                ),
                ObjectiveSpec(name="b", minimize=True, normalization_bounds=(0.0, 1.0)),
            ]
        )
        issues = [reason for _key, reason in objective_support_issues(spec)]
        assert any("cannot be combined with" in r and "desirability" in r for r in issues)
        with pytest.raises(ValueError, match="desirability"):
            spec_to_objective(spec)

    def test_boolean_log_flag_under_desirability_is_reported(self) -> None:
        spec = self._desirability(
            [
                ObjectiveSpec(
                    name="a", minimize=True, log_transform=True, normalization_bounds=(0.0, 1.0)
                ),
                ObjectiveSpec(name="b", minimize=True, normalization_bounds=(0.0, 1.0)),
            ]
        )
        issues = [reason for _key, reason in objective_support_issues(spec)]
        assert any("log_transform" in r and "desirability" in r for r in issues)

    def test_plain_desirability_stays_accepted(self) -> None:
        spec = self._desirability(
            [
                ObjectiveSpec(name="a", minimize=True, normalization_bounds=(0.0, 1.0)),
                ObjectiveSpec(name="b", minimize=False, normalization_bounds=(0.0, 1.0)),
            ]
        )
        assert isinstance(spec_to_objective(spec), DesirabilityObjective)


class TestBuilderMemberCompleteness:
    """Every enum member is explicitly handled; unknown members raise.

    Guards the builders' final-branch checks: a future ``MatchShape`` /
    ``ObjectiveTransformKind`` member is exercised here automatically and
    fails loudly if a converter branch is missing, instead of silently
    routing to the last constructor.
    """

    @pytest.mark.parametrize("shape", list(MatchShape))
    def test_every_match_shape_builds(self, shape: MatchShape) -> None:
        scale = 1.0 if shape in (MatchShape.BELL, MatchShape.TRIANGULAR) else None
        o = ObjectiveSpec(
            name="y",
            target_mode=TargetMode.MATCH,
            target_value=7.4,
            match_shape=shape,
            match_scale=scale,
        )
        assert isinstance(_build_baybe_target(o), NumericalTarget)

    @pytest.mark.parametrize("kind", list(ObjectiveTransformKind))
    def test_every_transform_kind_builds(self, kind: ObjectiveTransformKind) -> None:
        transform = {
            ObjectiveTransformKind.LOG: ObjectiveTransformSpec(kind=kind),
            ObjectiveTransformKind.CLAMP: ObjectiveTransformSpec(kind=kind, bounds=(0.0, 1.0)),
            ObjectiveTransformKind.POWER: ObjectiveTransformSpec(kind=kind, exponent=2),
            ObjectiveTransformKind.SIGMOID: ObjectiveTransformSpec(
                kind=kind, center=0.5, steepness=2.0
            ),
        }[kind]
        o = ObjectiveSpec(name="y", minimize=True, transform=transform)
        assert isinstance(_build_baybe_target(o), NumericalTarget)
