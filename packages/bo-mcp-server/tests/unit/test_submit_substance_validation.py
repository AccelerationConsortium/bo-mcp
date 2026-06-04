"""Submitted substance values validate against category LABELS, not SMILES.

A ``role=substance`` parameter is categorical at the experimental level: the
user submits the solvent *label* (``"water"``), not the descriptor key
(``"O"``). Result validation must therefore check the submitted value against
the declared ``categories``, not against the ``substance_data`` SMILES map —
otherwise a perfectly valid observation would be flagged. This pins that
behavior on the categorical path. Mirrors a BayBE solvent-screening
``SubstanceParameter`` spec
(https://emdgroup.github.io/baybe/stable/examples/Basics/parameters.html).
"""

from __future__ import annotations

from bo_mcp_server.domain import InputParameter, ParameterType
from bo_mcp_server.operations.submit_results_validation import _validate_parameter_value


def _substance_parameter() -> InputParameter:
    return InputParameter(
        name="solvent",
        type=ParameterType.CATEGORICAL,
        categories=("water", "ethanol"),
        parameter_options={
            "baybe": {
                "role": "substance",
                "substance_data": {"water": "O", "ethanol": "CCO"},
            }
        },
    )


def test_submitted_category_label_validates() -> None:
    warnings: list[str] = []
    _validate_parameter_value(_substance_parameter(), "water", 0, warnings)
    assert warnings == []


def test_submitted_raw_smiles_is_flagged_not_a_category() -> None:
    # "O" is ethanol/water's SMILES key, NOT a declared category label, so it
    # must be flagged — proving validation keys off categories, not SMILES.
    warnings: list[str] = []
    _validate_parameter_value(_substance_parameter(), "O", 0, warnings)
    assert warnings
    assert "not in allowed categories" in warnings[0]
