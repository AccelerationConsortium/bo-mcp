"""BoTorch intake guardrail for mixed categorical spaces it cannot optimize."""

from bo_engine.backend_base import CapabilityStatus
from bo_engine.botorch_backend import BoTorchBackend
from bo_engine.constants import MIXED_CATEGORICAL_COMBO_THRESHOLD
from bo_engine.transforms import (
    mixed_space_combo_limit_message,
    mixed_space_combo_overflow,
)
from bo_engine.types import (
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)


def _mixed_spec(*, category_count: int) -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[
            ParameterSpec(
                name="temperature",
                type=ParameterType.CONTINUOUS,
                bounds=(20.0, 120.0),
            ),
            ParameterSpec(
                name="catalyst",
                type=ParameterType.CATEGORICAL,
                categories=[f"c{i}" for i in range(category_count)],
            ),
            ParameterSpec(
                name="solvent",
                type=ParameterType.CATEGORICAL,
                categories=["a", "b"],
            ),
        ],
        objectives=[ObjectiveSpec(name="yield", minimize=False)],
    )


def test_large_mixed_space_is_rejected_during_capability_validation() -> None:
    category_count = MIXED_CATEGORICAL_COMBO_THRESHOLD // 2 + 1

    result = BoTorchBackend().validate_capabilities(_mixed_spec(category_count=category_count))

    assert not result.is_compatible
    report = next(
        item
        for item in result.option_reports
        if item.key == "parameters" and item.status == CapabilityStatus.UNSUPPORTED
    )
    assert str(MIXED_CATEGORICAL_COMBO_THRESHOLD) in report.reason


def test_supported_mixed_space_remains_compatible() -> None:
    category_count = MIXED_CATEGORICAL_COMBO_THRESHOLD // 2

    result = BoTorchBackend().validate_capabilities(_mixed_spec(category_count=category_count))

    assert result.is_compatible


def test_capability_reason_matches_acquisition_error_verbatim() -> None:
    """Intake reports the exact message the acquisition guard would raise.

    Both surfaces consume ``mixed_space_combo_overflow`` /
    ``mixed_space_combo_limit_message``; this pins the coupling so a future
    change to the acquisition limit cannot silently drift away from what
    intake tells users.
    """
    spec = _mixed_spec(category_count=MIXED_CATEGORICAL_COMBO_THRESHOLD // 2 + 1)

    combo_overflow = mixed_space_combo_overflow(spec)
    assert combo_overflow is not None

    result = BoTorchBackend().validate_capabilities(spec)
    report = next(
        item
        for item in result.option_reports
        if item.key == "parameters" and item.status == CapabilityStatus.UNSUPPORTED
    )
    assert report.reason == mixed_space_combo_limit_message(combo_overflow)


def test_categorical_without_categories_does_not_raise() -> None:
    """validate_capabilities stays a reporting API for malformed specs.

    ``ParameterSpec`` is an unvalidated dataclass, so a categorical parameter
    with ``categories=None`` is representable; the capability check must not
    turn that into a ValueError — the canonical rejection is produced by the
    parameter builders / schema validation upstream.
    """
    spec = OptimizationSpec(
        parameters=[
            ParameterSpec(
                name="temperature",
                type=ParameterType.CONTINUOUS,
                bounds=(20.0, 120.0),
            ),
            ParameterSpec(name="broken", type=ParameterType.CATEGORICAL, categories=None),
        ],
        objectives=[ObjectiveSpec(name="yield", minimize=False)],
    )

    result = BoTorchBackend().validate_capabilities(spec)

    assert not any(
        item.key == "parameters" and item.status == CapabilityStatus.UNSUPPORTED
        for item in result.option_reports
    )
