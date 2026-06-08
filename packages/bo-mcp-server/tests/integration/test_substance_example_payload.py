"""The shipped solvent-screening example payload validates cleanly.

``docs/examples/substance_solvent_screening.json`` is the runnable molecular
(substance) recipe referenced from the docs. It lives under tracked ``docs/``
(not the git-ignored ``data/``) so it survives a fresh checkout/release. This
test loads it through ``validate_intake_operation`` and asserts ``valid is
True`` so the documented example cannot silently rot when the intake schema
evolves — and, by requiring the file on disk, it fails loudly in CI if the
example is ever dropped or re-ignored. Mirrors a BayBE solvent-screening
``SubstanceParameter`` spec
(https://emdgroup.github.io/baybe/stable/examples/Basics/parameters.html).
"""

from __future__ import annotations

import json
from pathlib import Path

from bo_mcp_server.operations.validate_intake import validate_intake_operation

# tests/integration/<file> -> repo root is four levels up from this file
# (.../packages/bo-mcp-server/tests/integration/<file>).
_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[4] / "docs" / "examples" / "substance_solvent_screening.json"
)


def test_substance_example_payload_validates() -> None:
    assert _EXAMPLE_PATH.is_file(), f"example payload missing at {_EXAMPLE_PATH}"
    payload = json.loads(_EXAMPLE_PATH.read_text())

    result = validate_intake_operation(payload)

    assert result["valid"] is True, result.get("errors")
    # The substance role survives intake into the canonical spec.
    spec = result["spec"]
    solvent = next(p for p in spec["parameters"] if p["name"] == "solvent")
    assert solvent["parameter_options"]["baybe"]["role"] == "substance"
