"""BoTorch auto-routing guardrail for BayBE molecular (substance) parameters.

A ``role=substance`` parameter encodes a molecule as SMILES → cheminformatics
descriptors (BayBE's ``SubstanceParameter``; see the BayBE substance/solvent
screening example at
https://emdgroup.github.io/baybe/stable/examples/Basics/parameters.html).
BoTorch has no chemistry kernel, so optimizing such a spec on BoTorch would
treat the SMILES labels as opaque one-hot categories and silently drop the
chemistry — a correctness bug, not a crash. These tests pin the guardrail in
:meth:`BoTorchBackend.validate_capabilities` that classifies a substance
parameter as ``UNSUPPORTED`` so ``backend="auto"`` routes it to BayBE and a
pinned ``backend="botorch"`` is rejected at intake.
"""

from __future__ import annotations

from bo_engine.backend_base import CapabilityStatus
from bo_engine.botorch_backend import BoTorchBackend
from bo_engine.interop import (
    BAYBE_BACKEND_NAME,
    BAYBE_PARAMETER_ROLE_KEY,
    BAYBE_SUBSTANCE_ROLE,
)
from bo_engine.types import (
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)


def _substance_spec(acknowledge: tuple[str, ...] = ()) -> OptimizationSpec:
    """Single-parameter spec declaring a BayBE substance role via the marker."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(
                name="solvent",
                type=ParameterType.CATEGORICAL,
                categories=["water", "ethanol"],
                parameter_options={
                    BAYBE_BACKEND_NAME: {
                        BAYBE_PARAMETER_ROLE_KEY: BAYBE_SUBSTANCE_ROLE,
                        "substance_data": {"water": "O", "ethanol": "CCO"},
                    }
                },
            ),
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
        acknowledge_degradations=acknowledge,
    )


class TestSubstanceIsHardIncompatible:
    """Pinned ``backend="botorch"`` + substance must be hard-incompatible."""

    def test_substance_parameter_makes_botorch_incompatible(self) -> None:
        result = BoTorchBackend().validate_capabilities(_substance_spec())
        assert not result.is_compatible
        substance_reports = [
            r
            for r in result.option_reports
            if r.status == CapabilityStatus.UNSUPPORTED
            and r.key.endswith(f".{BAYBE_BACKEND_NAME}.{BAYBE_PARAMETER_ROLE_KEY}")
        ]
        assert substance_reports
        assert "solvent" in substance_reports[0].key

    def test_acknowledge_degradations_cannot_bypass_substance_gate(self) -> None:
        # Listing any/every field in acknowledge_degradations must NOT
        # downgrade the substance veto — it is a correctness gate, not an
        # option degradation.
        for ack in (
            ("role",),
            ("parameter_options",),
            ("solvent",),
            ("turbo_config", "saasbo_config", "outcome_constraints", "role"),
        ):
            result = BoTorchBackend().validate_capabilities(_substance_spec(acknowledge=ack))
            assert not result.is_compatible, f"acknowledge={ack} unexpectedly bypassed the gate"


class TestNonSubstanceRoutingUnchanged:
    """Ordinary specs and other BayBE roles must NOT be newly rejected."""

    def test_plain_continuous_spec_stays_compatible(self) -> None:
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        assert BoTorchBackend().validate_capabilities(spec).is_compatible

    def test_categorical_with_baybe_encoding_option_not_rejected(self) -> None:
        # A vanilla categorical carrying an unrelated BayBE option (encoding)
        # must keep routing to BoTorch exactly as before the guardrail.
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(
                    name="color",
                    type=ParameterType.CATEGORICAL,
                    categories=["red", "blue"],
                    parameter_options={BAYBE_BACKEND_NAME: {"encoding": "OHE"}},
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        result = BoTorchBackend().validate_capabilities(spec)
        assert result.is_compatible
        assert not any(
            r.key.endswith(f".{BAYBE_BACKEND_NAME}.{BAYBE_PARAMETER_ROLE_KEY}")
            for r in result.option_reports
        )

    def test_task_role_not_rejected(self) -> None:
        # ``role=task`` on BoTorch is a separate, pre-existing concern and is
        # explicitly out of scope for this guardrail — it must not flip.
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(
                    name="lab",
                    type=ParameterType.CATEGORICAL,
                    categories=["A", "B"],
                    parameter_options={BAYBE_BACKEND_NAME: {"role": "task"}},
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        result = BoTorchBackend().validate_capabilities(spec)
        assert result.is_compatible
