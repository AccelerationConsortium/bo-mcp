"""Malformed SMILES are reported UNSUPPORTED at intake (fail-loud).

A SMILES string passes the ``dict[str, str]`` shape but can still be
unparseable; BayBE's ``SubstanceParameter`` only discovers that deep inside
RDKit at construction. The capability layer parse-checks SMILES at intake so a
bad molecule is rejected with a clear reason instead of an opaque RDKit
traceback at suggestion time. The check is gated on ``_CHEMISTRY_AVAILABLE``
with a lazy RDKit import, so a chem-missing run never attempts the parse — this
suite pins both halves of that contract. Mirrors a BayBE solvent-screening
``SubstanceParameter`` spec
(https://emdgroup.github.io/baybe/stable/examples/Basics/parameters.html).
"""

from __future__ import annotations

from bo_engine.backend_base import CapabilityStatus
from bo_engine.types import (
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)
from bo_engine_baybe import capabilities
from bo_engine_baybe.backend import BayBEBackend


def _spec(substance_data: dict[str, str]) -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[
            ParameterSpec(
                name="solvent",
                type=ParameterType.CATEGORICAL,
                categories=["water", "ethanol"],
                parameter_options={
                    "baybe": {"role": "substance", "substance_data": substance_data},
                },
            ),
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
    )


def _smiles_reports(result) -> list:
    return [
        r
        for r in result.option_reports
        if r.status == CapabilityStatus.UNSUPPORTED and "cannot parse" in (r.reason or "")
    ]


def _substance_data_reports(result) -> list:
    return [
        r
        for r in result.option_reports
        if r.status == CapabilityStatus.UNSUPPORTED
        and r.key == "parameter_options[solvent].baybe.substance_data"
    ]


def test_valid_smiles_passes() -> None:
    # Water + ethanol are both valid SMILES; no parse rejection.
    result = BayBEBackend().validate_capabilities(_spec({"water": "O", "ethanol": "CCO"}))
    assert result.is_compatible
    assert not _smiles_reports(result)


def test_malformed_smiles_is_unsupported() -> None:
    result = BayBEBackend().validate_capabilities(
        _spec({"water": "O", "ethanol": "this-is-not-a-smiles"})
    )
    assert not result.is_compatible
    bad = _smiles_reports(result)
    assert bad
    assert "ethanol" in bad[0].reason


def test_substance_data_extra_category_is_unsupported() -> None:
    result = BayBEBackend().validate_capabilities(
        _spec({"water": "O", "ethanol": "CCO", "ghost": "CC"})
    )
    assert not result.is_compatible
    bad = _substance_data_reports(result)
    assert bad
    assert "extra SMILES" in bad[0].reason
    assert "ghost" in bad[0].reason


def test_duplicate_substances_are_unsupported() -> None:
    result = BayBEBackend().validate_capabilities(_spec({"water": "O", "ethanol": "[OH2]"}))
    assert not result.is_compatible
    bad = _substance_data_reports(result)
    assert bad
    assert "same substance" in bad[0].reason
    assert "ethanol" in bad[0].reason
    assert "water" in bad[0].reason


def test_chem_missing_path_skips_rdkit_parse(monkeypatch) -> None:
    """When chem is unavailable, the SMILES parse must not run.

    The chem-missing report must fire instead, and (critically) RDKit must
    never be invoked — the parse helper is patched to a tripwire that fails
    the test if the chem-missing guard ever falls through to it.
    """

    def _tripwire(*_args, **_kwargs):
        msg = "RDKit parse attempted on the chem-missing path"
        raise AssertionError(msg)

    monkeypatch.setattr(capabilities, "_CHEMISTRY_AVAILABLE", False)
    monkeypatch.setattr(
        capabilities,
        "_CHEMISTRY_UNAVAILABLE_REASON",
        "simulated missing scikit-fingerprints",
    )
    monkeypatch.setattr(capabilities, "_invalid_smiles_reports", _tripwire)

    # Even a malformed SMILES must not reach the parse helper here.
    result = BayBEBackend().validate_capabilities(
        _spec({"water": "O", "ethanol": "this-is-not-a-smiles"})
    )
    assert not result.is_compatible
    assert any(
        r.status == CapabilityStatus.UNSUPPORTED and "baybe[chem]" in (r.reason or "")
        for r in result.option_reports
    )
