"""End-to-end coverage for BayBE custom-representation campaigns.

``role=custom`` maps each category label to a user-supplied numeric
representation (e.g. quantum-chemistry descriptors) which BayBE consumes via
:class:`baybe.parameters.CustomDiscreteParameter`. Unlike the substance role,
no cheminformatics stack runs — the descriptor table is provided directly — so
these tests stay in the fast per-PR lane. GP fitting is not byte-stable, so the
round-trip assertions check invariants (the suggested value is one of the
declared category labels) rather than exact suggestions.
"""

from __future__ import annotations

from baybe.parameters import CustomDiscreteParameter

from bo_engine.backend_base import CapabilityStatus
from bo_engine.types import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)
from bo_engine_baybe.backend import BayBEBackend
from bo_engine_baybe.converters import spec_to_parameters

# A small ligand panel: category label -> {descriptor name: value}.
_LIGANDS: dict[str, dict[str, float]] = {
    "L1": {"volume": 1.0, "charge": -0.2},
    "L2": {"volume": 2.5, "charge": 0.1},
    "L3": {"volume": 3.1, "charge": 0.4},
    "L4": {"volume": 4.8, "charge": -0.7},
}


def _custom_parameter(
    descriptors: dict[str, dict[str, float]] | None = None,
    categories: list[str] | None = None,
) -> ParameterSpec:
    options: dict[str, object] = {
        "role": "custom",
        "custom_descriptors": _LIGANDS if descriptors is None else descriptors,
    }
    return ParameterSpec(
        name="ligand",
        type=ParameterType.CATEGORICAL,
        categories=list(_LIGANDS) if categories is None else categories,
        parameter_options={"baybe": options},
    )


def _spec(
    descriptors: dict[str, dict[str, float]] | None = None,
    categories: list[str] | None = None,
) -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[_custom_parameter(descriptors, categories)],
        objectives=[ObjectiveSpec(name="yield", minimize=False)],
    )


def _seed_observations(spec: OptimizationSpec, n: int = 3) -> list[ObservationData]:
    labels = list(_LIGANDS)
    return [
        ObservationData(
            parameter_values={"ligand": labels[i]},
            objective_values={o.name: float(i + 1) for o in spec.objectives},
        )
        for i in range(n)
    ]


def test_converter_builds_real_custom_parameter() -> None:
    """``role=custom`` + descriptor table yields a BayBE ``CustomDiscreteParameter``."""
    params = spec_to_parameters(_spec())
    assert len(params) == 1
    ligand = params[0]
    assert isinstance(ligand, CustomDiscreteParameter)
    assert ligand.name == "ligand"
    # Labels come from the DataFrame index; descriptor columns are preserved.
    assert set(ligand.values) == set(_LIGANDS)
    assert set(ligand.data.columns) == {"volume", "charge"}


def test_suggestion_round_trip_returns_category_label() -> None:
    """create -> observe -> suggest returns a declared category LABEL, not a vector."""
    backend = BayBEBackend()
    spec = _spec()

    initial = backend.generate_initial_design(spec, n_points=2)
    assert initial
    for row in initial:
        assert row["ligand"] in _LIGANDS

    batch = backend.generate_suggestions(spec, _seed_observations(spec), batch_size=1, iteration=1)
    assert len(batch.suggestions) == 1
    assert batch.suggestions[0]["parameter_values"]["ligand"] in _LIGANDS


def test_missing_descriptors_rejected_at_intake() -> None:
    """A descriptor table that omits a declared category is UNSUPPORTED at intake."""
    # Declare four categories but only supply descriptors for three.
    partial = {k: _LIGANDS[k] for k in ("L1", "L2", "L3")}
    spec = _spec(descriptors=partial, categories=list(_LIGANDS))
    result = BayBEBackend().validate_capabilities(spec)
    assert not result.is_compatible
    bad = [
        r
        for r in result.option_reports
        if r.status == CapabilityStatus.UNSUPPORTED and r.key.endswith(".baybe.custom_descriptors")
    ]
    assert bad
    assert "L4" in bad[0].reason


def test_duplicate_descriptor_rows_rejected_and_named() -> None:
    """Two labels with identical vectors are rejected, and the report names them."""
    dup = {
        "L1": {"volume": 1.0, "charge": -0.2},
        "L2": {"volume": 5.0, "charge": 0.9},
        "L3": {"volume": 1.0, "charge": -0.2},  # identical to L1
    }
    spec = _spec(descriptors=dup, categories=["L1", "L2", "L3"])
    result = BayBEBackend().validate_capabilities(spec)
    assert not result.is_compatible
    bad = [
        r
        for r in result.option_reports
        if r.status == CapabilityStatus.UNSUPPORTED and r.key.endswith(".baybe.custom_descriptors")
    ]
    assert bad
    # The specific colliding labels are named so a caller can enrich just those.
    assert "L1" in bad[0].reason
    assert "L3" in bad[0].reason


def test_constant_descriptor_column_rejected_at_intake() -> None:
    """BayBE rejects a column with a single unique value; surface it at intake."""
    constant = {
        "L1": {"volume": 1.0, "charge": 0.5},
        "L2": {"volume": 2.0, "charge": 0.5},
    }
    spec = _spec(descriptors=constant, categories=["L1", "L2"])
    result = BayBEBackend().validate_capabilities(spec)
    assert not result.is_compatible
    assert any(
        r.status == CapabilityStatus.UNSUPPORTED and r.key.endswith(".baybe.custom_descriptors")
        for r in result.option_reports
    )
