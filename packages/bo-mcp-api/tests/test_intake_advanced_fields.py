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

import json

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


def test_openapi_advertises_typed_advanced_configs() -> None:
    """The advanced knobs are typed configs in OpenAPI, not opaque objects.

    Parity with the MCP tool schema: a REST/OpenAPI client must be able to
    discover the fields of ``turbo_config``/``saasbo_config``/etc. straight
    from the spec (e.g. ``TurboConfig.initial_length``,
    ``OutcomeConstraint.objective_name``) rather than seeing a bare
    ``object``.
    """
    from api.main import create_app

    schemas = create_app().openapi()["components"]["schemas"]
    intake = schemas["IntakeData"]["properties"]

    # field -> (referenced component schema, a representative property on it)
    expected = {
        "turbo_config": ("TurboConfig", "initial_length"),
        "saasbo_config": ("SaasboConfig", "warmup_steps"),
        "fidelity_parameter": ("FidelityParameter", "target"),
        "transfer_learning": ("TransferLearningConfig", "prior_campaign_ids"),
        "acquisition_optimization": ("AcquisitionOptimizationConfig", "num_restarts"),
        "outcome_constraints": ("OutcomeConstraint", "objective_name"),
    }

    for field, (component, sample_prop) in expected.items():
        # The field $refs its typed config (Optional -> anyOf[$ref, null];
        # tuple -> items.$ref); checking the serialized node is enough and
        # avoids brittle anyOf/items navigation.
        field_blob = json.dumps(intake[field])
        assert f"#/components/schemas/{component}" in field_blob, (
            f"{field} should reference {component}, got {field_blob}"
        )
        assert sample_prop in schemas[component]["properties"], (
            f"{component} should expose '{sample_prop}'"
        )


def test_openapi_advertises_acquisition_method_enum() -> None:
    """``acquisition_method`` is the AcquisitionMethod enum in OpenAPI, not a bare string.

    Parity with the MCP tool schema: a REST/OpenAPI client must discover the
    valid acquisition methods from the spec rather than guessing and only
    learning the value was wrong on a failed request.
    """
    from api.main import create_app

    schemas = create_app().openapi()["components"]["schemas"]
    acq_node = schemas["IntakeData"]["properties"]["acquisition_method"]
    assert "#/components/schemas/AcquisitionMethod" in json.dumps(acq_node)
    enum_values = set(schemas["AcquisitionMethod"]["enum"])
    assert {"auto", "expected_improvement", "hypervolume_improvement"} <= enum_values


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
