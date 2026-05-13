"""REST intake exposes the full :class:`CampaignSpec` field surface.

Pre-1.66 the REST ``IntakeData`` model dropped the new typed-options
fields (``backend``, ``backend_options``, ``parameter_options``,
``use_input_warping``, ``turbo_config``, …) because ``model_dump()``
omits anything not declared on the schema. The MCP intake honored those
fields, but a REST request shaped the same way silently fell back to
``backend="auto"`` with default knobs.

These tests pin the new behavior:

* The fields survive the REST schema round trip and reach
  ``CampaignSpec`` unchanged.
* ``extra="forbid"`` rejects unknown keys at the REST boundary so
  misspellings fail loudly.
"""

from __future__ import annotations

from bo_mcp_server.domain import CampaignIntakeInput, CampaignSpec

from api.schemas.intake import IntakeData


def _minimal_intake() -> dict:
    return {
        "name": "Advanced",
        "parameters": [
            {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
        ],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }


def test_rest_intake_preserves_backend_and_options() -> None:
    payload = {
        **_minimal_intake(),
        "backend": "baybe",
        "backend_options": {"baybe": {"encoding": "ohe"}},
    }
    rest = IntakeData.model_validate(payload)
    dumped = rest.model_dump()
    assert dumped["backend"] == "baybe"
    assert dumped["backend_options"] == {"baybe": {"encoding": "ohe"}}
    mcp = CampaignIntakeInput.model_validate(dumped)
    assert mcp.backend == "baybe"
    assert mcp.backend_options == {"baybe": {"encoding": "ohe"}}


def test_rest_intake_preserves_parameter_options() -> None:
    payload = {
        **_minimal_intake(),
        "parameters": [
            {
                "name": "x",
                "type": "continuous",
                "bounds": [0.0, 1.0],
                "parameter_options": {"baybe": {"encoding": "ohe"}},
            }
        ],
    }
    rest = IntakeData.model_validate(payload)
    dumped = rest.model_dump()
    mcp = CampaignIntakeInput.model_validate(dumped)
    assert mcp.parameters[0].parameter_options == {"baybe": {"encoding": "ohe"}}


def test_rest_intake_preserves_advanced_knobs() -> None:
    payload = {
        **_minimal_intake(),
        "use_input_warping": True,
        "use_cost_aware": True,
        "turbo_config": {"initial_length": 0.5},
        "saasbo_config": {"warmup_steps": 8, "num_samples": 8, "thinning": 2},
        "outcome_constraints": [{"objective_name": "y", "threshold": 0.5, "greater_than": True}],
    }
    rest = IntakeData.model_validate(payload)
    mcp = CampaignIntakeInput.model_validate(rest.model_dump())
    spec = CampaignSpec.model_validate(mcp.model_dump())
    assert spec.use_input_warping is True
    assert spec.use_cost_aware is True
    assert spec.turbo_config is not None
    assert spec.saasbo_config is not None
    assert len(spec.outcome_constraints) == 1
    assert spec.outcome_constraints[0].objective_name == "y"


def test_rest_intake_rejects_unknown_extras() -> None:
    """Misspelled or unknown keys must fail at intake instead of silently disappearing."""
    import pytest
    from pydantic import ValidationError

    payload = {
        **_minimal_intake(),
        "backedn": "baybe",  # typo
    }
    with pytest.raises(ValidationError):
        IntakeData.model_validate(payload)


def test_rest_intake_rejects_unknown_parameter_extras() -> None:
    import pytest
    from pydantic import ValidationError

    payload = {
        **_minimal_intake(),
        "parameters": [
            {
                "name": "x",
                "type": "continuous",
                "bounds": [0.0, 1.0],
                "parmaeter_options": {"baybe": {}},  # typo
            }
        ],
    }
    with pytest.raises(ValidationError):
        IntakeData.model_validate(payload)
