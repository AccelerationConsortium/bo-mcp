"""BoTorch capability reports for the extended objective surface.

Match-a-target objectives, the typed target-transform union, and
desirability scalarization are honored by BayBE only; the BoTorch backend
must report them ``UNSUPPORTED`` so ``backend="auto"`` routes to BayBE and
a pinned ``backend="botorch"`` fails loudly at intake — the
substance-guardrail routing-safety pattern applied to objectives.
"""

from __future__ import annotations

from typing import Any

from bo_engine.botorch_backend import BoTorchBackend
from bo_engine.types import (
    ObjectiveSpec,
    ObjectiveTransformKind,
    ObjectiveTransformSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
    ScalarizationMode,
    TargetMode,
)


def _spec(objectives: list[ObjectiveSpec], **kwargs: Any) -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
        objectives=objectives,
        **kwargs,
    )


def test_match_objective_is_unsupported_on_botorch() -> None:
    spec = _spec([ObjectiveSpec(name="y", target_mode=TargetMode.MATCH, target_value=0.5)])
    result = BoTorchBackend().validate_capabilities(spec)
    assert not result.is_compatible
    assert any("target_mode" in r.key for r in result.unsupported)


def test_transform_union_is_unsupported_on_botorch() -> None:
    spec = _spec(
        [
            ObjectiveSpec(
                name="y",
                minimize=True,
                transform=ObjectiveTransformSpec(
                    kind=ObjectiveTransformKind.CLAMP, bounds=(0.0, 1.0)
                ),
            )
        ]
    )
    result = BoTorchBackend().validate_capabilities(spec)
    assert not result.is_compatible
    assert any(".transform" in r.key for r in result.unsupported)


def test_desirability_is_unsupported_on_botorch() -> None:
    spec = _spec(
        [
            ObjectiveSpec(name="a", minimize=True, normalization_bounds=(0.0, 1.0)),
            ObjectiveSpec(name="b", minimize=True, normalization_bounds=(0.0, 1.0)),
        ],
        scalarization=ScalarizationMode.DESIRABILITY,
    )
    result = BoTorchBackend().validate_capabilities(spec)
    assert not result.is_compatible
    assert any(r.key == "scalarization" for r in result.unsupported)


def test_plain_objectives_remain_supported() -> None:
    spec = _spec(
        [ObjectiveSpec(name="y", minimize=True, log_transform=True)],
    )
    result = BoTorchBackend().validate_capabilities(spec)
    assert result.is_compatible


def test_target_mode_disagreeing_with_boolean_is_unsupported() -> None:
    """M-class: BoTorch reads the boolean only, so a disagreeing override is vetoed.

    ``target_mode='maximize'`` with the boolean ``minimize`` left at its
    default previously validated fully compatible — and BoTorch minimized
    what BayBE maximizes for the identical spec.
    """
    spec = _spec([ObjectiveSpec(name="y", target_mode=TargetMode.MAXIMIZE)])
    result = BoTorchBackend().validate_capabilities(spec)
    assert not result.is_compatible
    reports = [r for r in result.unsupported if r.key == "objectives[0].target_mode"]
    assert reports
    assert "opposite direction" in reports[0].reason


def test_target_mode_agreeing_with_boolean_stays_supported() -> None:
    spec = _spec(
        [
            ObjectiveSpec(name="a", minimize=True, target_mode=TargetMode.MINIMIZE),
            ObjectiveSpec(name="b", minimize=False, target_mode=TargetMode.MAXIMIZE),
        ]
    )
    assert BoTorchBackend().validate_capabilities(spec).is_compatible


def test_cross_backend_direction_agreement_on_consistent_spec() -> None:
    """Both backends accept the consistently-declared target_mode spelling."""
    from bo_engine_baybe.backend import BayBEBackend

    spec = _spec([ObjectiveSpec(name="y", minimize=False, target_mode=TargetMode.MAXIMIZE)])
    assert BoTorchBackend().validate_capabilities(spec).is_compatible
    assert BayBEBackend().validate_capabilities(spec).is_compatible
