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

import annotated_types
import pytest
from pydantic import ValidationError

from api.schemas.intake import IntakeData
from bo_mcp_server.client import MAX_GENERATION_BATCH_SIZE, field_docs
from bo_mcp_server.domain import CampaignIntakeInput, CampaignSpec


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


def test_intake_data_field_set_matches_campaign_intake_input() -> None:
    """REST ``IntakeData`` must mirror the MCP ``CampaignIntakeInput`` field set.

    The two models are hand-maintained duplicates — the REST schema adds
    REST-specific ``Field`` limits but its docstring promises "Field set
    mirrors CampaignIntakeInput". Without a structural fitness function the
    sets drift silently: a field added to ``CampaignIntakeInput`` but not
    ``IntakeData`` makes the REST transport reject (``extra="forbid"``) a
    payload the MCP transport accepts, breaking the project's transport-
    parity goal without any test failing. This codifies the contract.
    """
    assert set(IntakeData.model_fields) == set(CampaignIntakeInput.model_fields)


def test_intake_data_annotations_match_campaign_intake_input() -> None:
    """Each shared field carries the same type annotation on both transports.

    Field-level metadata (the REST ``max_length`` limits) may differ by
    design, but the declared *type* of every field must be identical so the
    same JSON validates equivalently on either transport. Pins the
    ``list[str]`` vs ``tuple[str, ...]`` drift that previously existed on
    ``acknowledge_degradations`` (REST accepted a ``list`` the MCP model
    typed as a ``tuple``).
    """
    rest_fields = IntakeData.model_fields
    mcp_fields = CampaignIntakeInput.model_fields
    mismatches = {
        name: (rest_fields[name].annotation, mcp_fields[name].annotation)
        for name in rest_fields
        if name in mcp_fields and rest_fields[name].annotation != mcp_fields[name].annotation
    }
    assert mismatches == {}


def test_intake_data_descriptions_match_campaign_intake_input() -> None:
    """Each shared field carries the same description on both transports.

    The ``Field(description=...)`` strings feed the REST OpenAPI document
    and the MCP tool schema — the contracts agents read when constructing
    payloads. A one-sided edit makes the two transports document different
    behavior for the same field, so descriptions are single-sourced in
    :mod:`bo_mcp_server.domain.field_docs`; this guards against a model
    reverting to an inline literal. Same architecture-fitness-function
    rationale as the field-set test above (cf. ArchUnit / importlinter).
    """
    rest_fields = IntakeData.model_fields
    mcp_fields = CampaignIntakeInput.model_fields
    mismatches = {
        name: (rest_fields[name].description, mcp_fields[name].description)
        for name in rest_fields
        if rest_fields[name].description != mcp_fields[name].description
    }
    assert mismatches == {}


def test_intake_data_numeric_constraints_match_campaign_intake_input() -> None:
    """Numeric range constraints (ge / gt / le / lt) agree on both transports.

    Pins the asymmetry where the REST schema used to accept born-dead
    budgets (``max_iterations=0``, ``initial_design_size=0``) and
    uncapped ``batch_size`` values that the MCP model rejects: the
    published OpenAPI then advertises values the downstream coercion
    422s, breaking schema-driven client codegen (OpenAPI guidance is
    that the schema must describe exactly what the server accepts:
    https://spec.openapis.org/oas/v3.1.0#schema-object). Collection
    length limits (``min_length``/``max_length``) legitimately differ —
    REST adds the ``MAX_INTAKE_*`` payload caps — so only numeric-range
    metadata is compared. The comparison is an equality, not a subset:
    dropping a bound on either side fails.
    """
    range_types = (annotated_types.Ge, annotated_types.Gt, annotated_types.Le, annotated_types.Lt)
    rest_fields = IntakeData.model_fields
    mcp_fields = CampaignIntakeInput.model_fields
    mismatches = {}
    for name, rest_field in rest_fields.items():
        rest_bounds = {m for m in rest_field.metadata if isinstance(m, range_types)}
        mcp_bounds = {m for m in mcp_fields[name].metadata if isinstance(m, range_types)}
        if rest_bounds != mcp_bounds:
            mismatches[name] = (rest_bounds, mcp_bounds)
    assert mismatches == {}


def test_rest_intake_rejects_out_of_range_budgets_at_boundary() -> None:
    """Range violations fail in ``IntakeData`` itself, not downstream.

    Before the constraints were aligned, ``max_iterations=0`` passed the
    REST boundary and was only rejected by the internal coercion with an
    error location the published schema never declared.
    """
    for overrides in (
        {"max_iterations": 0},
        {"initial_design_size": 0},
        {"batch_size": MAX_GENERATION_BATCH_SIZE + 1},
    ):
        with pytest.raises(ValidationError):
            IntakeData.model_validate({**_minimal_intake(), **overrides})


def test_intake_examples_validate_on_both_transports() -> None:
    """Every advertised schema example is a valid payload on both models.

    OpenAPI guidance requires examples to validate against their schema
    (https://swagger.io/docs/specification/adding-examples/); both models
    use ``extra="forbid"``, so a field rename or shape change invalidates
    the example and fails here instead of shipping a copy-paste-broken
    example to agents.
    """
    assert field_docs.INTAKE_EXAMPLES
    for example in field_docs.INTAKE_EXAMPLES:
        IntakeData.model_validate(example)
        CampaignIntakeInput.model_validate(example)


def test_intake_examples_are_the_shared_constant_on_both_transports() -> None:
    """Both transports advertise exactly the shared example payloads.

    Ties the round-trip test above to what the schemas actually publish:
    if a model swaps in its own inline example dict, the validated
    constant and the advertised example diverge silently.
    """
    expected = {"examples": field_docs.INTAKE_EXAMPLES}
    assert IntakeData.model_config.get("json_schema_extra") == expected
    assert CampaignIntakeInput.model_config.get("json_schema_extra") == expected
