"""End-to-end functional coverage for BayBE molecular (substance) campaigns.

Until now every substance test exercised only the *rejection* path. These
tests prove a substance campaign can actually be built, searched, suggested,
observed, and persisted. They mirror BayBE's solvent-screening
``SubstanceParameter`` example — a categorical solvent choice whose labels map
to SMILES that BayBE encodes into cheminformatics descriptors
(https://emdgroup.github.io/baybe/stable/examples/Basics/parameters.html).

All tests are ``@pytest.mark.slow``: each builds a descriptor table and/or
fits a GP on the substance features (the MORDRED default is the cost driver),
so they run on main rather than in the per-PR lane (see TESTING.md). GP
fitting is not byte-stable, so we assert invariants (the suggested value is one
of the declared category labels, batch size) rather than exact suggestions.
"""

from __future__ import annotations

import pytest
from baybe.parameters import SubstanceParameter
from baybe.parameters.enum import SubstanceEncoding
from bo_engine.types import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)

from bo_engine_baybe.backend import BayBEBackend
from bo_engine_baybe.converters import spec_to_parameters, spec_to_searchspace

# A small, valid solvent-screening panel: category label -> SMILES.
_SOLVENTS: dict[str, str] = {
    "water": "O",
    "ethanol": "CCO",
    "methanol": "CO",
    "acetone": "CC(=O)C",
    "toluene": "Cc1ccccc1",
}


def _substance_parameter(encoding: str | None = None) -> ParameterSpec:
    options: dict[str, object] = {"role": "substance", "substance_data": dict(_SOLVENTS)}
    if encoding is not None:
        options["substance_encoding"] = encoding
    return ParameterSpec(
        name="solvent",
        type=ParameterType.CATEGORICAL,
        categories=list(_SOLVENTS),
        parameter_options={"baybe": options},
    )


def _single_objective_spec(encoding: str | None = None) -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[_substance_parameter(encoding)],
        objectives=[ObjectiveSpec(name="yield", minimize=False)],
    )


def _multi_objective_spec() -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[_substance_parameter(encoding="ECFP")],
        objectives=[
            ObjectiveSpec(name="yield", minimize=False),
            ObjectiveSpec(name="cost", minimize=True),
        ],
    )


def _seed_observations(spec: OptimizationSpec, n: int = 3) -> list[ObservationData]:
    labels = list(_SOLVENTS)
    rows: list[ObservationData] = []
    for i in range(n):
        objective_values = {o.name: float(i + 1) for o in spec.objectives}
        rows.append(
            ObservationData(
                parameter_values={"solvent": labels[i]},
                objective_values=objective_values,
            )
        )
    return rows


@pytest.mark.slow
def test_converter_builds_real_substance_parameter() -> None:
    """``role=substance`` + valid SMILES yields a BayBE ``SubstanceParameter``."""
    params = spec_to_parameters(_single_objective_spec())
    assert len(params) == 1
    solvent = params[0]
    assert isinstance(solvent, SubstanceParameter)
    assert solvent.name == "solvent"
    # Unset encoding falls back to the documented MORDRED default.
    assert solvent.encoding == SubstanceEncoding.MORDRED
    assert set(solvent.data) == set(_SOLVENTS)


@pytest.mark.slow
def test_converter_honors_requested_encoding() -> None:
    params = spec_to_parameters(_single_objective_spec(encoding="ECFP"))
    solvent = params[0]
    assert isinstance(solvent, SubstanceParameter)
    assert solvent.encoding == SubstanceEncoding.ECFP


@pytest.mark.slow
def test_searchspace_construction_builds_descriptor_table() -> None:
    """``spec_to_searchspace`` builds the descriptor-backed discrete space."""
    searchspace = spec_to_searchspace(_single_objective_spec(encoding="ECFP"))
    assert any(p.name == "solvent" for p in searchspace.parameters)
    # The discrete subspace exposes one row per declared category label.
    assert len(searchspace.discrete.exp_rep) == len(_SOLVENTS)


@pytest.mark.slow
def test_suggestion_round_trip_returns_category_label() -> None:
    """create -> initial design -> observe -> suggest returns a category LABEL.

    Guards the ``dataframe_to_suggestions`` pass-through: BayBE returns the
    experimental representation (the solvent label), not a descriptor vector.
    """
    backend = BayBEBackend()
    spec = _single_objective_spec(encoding="ECFP")

    initial = backend.generate_initial_design(spec, n_points=2)
    assert initial
    for row in initial:
        assert row["solvent"] in _SOLVENTS

    batch = backend.generate_suggestions(spec, _seed_observations(spec), batch_size=1, iteration=1)
    assert len(batch.suggestions) == 1
    suggested = batch.suggestions[0]["parameter_values"]["solvent"]
    assert suggested in _SOLVENTS


@pytest.mark.slow
def test_state_persistence_round_trip() -> None:
    """Serialize the campaign state, restore it, and suggest again consistently."""
    backend = BayBEBackend()
    spec = _single_objective_spec(encoding="ECFP")
    observations = _seed_observations(spec)

    first = backend.generate_suggestions(spec, observations, batch_size=1, iteration=1)
    state = first.backend_state
    assert state is not None

    restored = backend.generate_suggestions(
        spec, observations, batch_size=1, iteration=2, backend_state=state
    )
    assert len(restored.suggestions) == 1
    assert restored.suggestions[0]["parameter_values"]["solvent"] in _SOLVENTS


@pytest.mark.slow
def test_multi_objective_substance_smoke() -> None:
    """A substance parameter works under a multi-objective (Pareto) spec."""
    backend = BayBEBackend()
    spec = _multi_objective_spec()
    batch = backend.generate_suggestions(spec, _seed_observations(spec), batch_size=1, iteration=1)
    assert len(batch.suggestions) == 1
    assert batch.suggestions[0]["parameter_values"]["solvent"] in _SOLVENTS
