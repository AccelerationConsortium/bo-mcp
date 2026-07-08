"""Schema-surface gate: the extended neutral spec surface on the MCP transport.

Every new neutral enum member (constraint types, target modes, match
shapes, scalarization modes) and every new neutral field (objective
match/desirability/transform fields, constraint cardinality/interpoint
fields, scalarization) must be advertised by the enriched MCP intake
schema — the guardrail that prevents transport drift as the surface
grows. The REST half lives in
``packages/bo-mcp-api/tests/test_neutral_surface_schema_openapi.py``.
"""

from __future__ import annotations

from typing import Any

from bo_engine.types import ConstraintType, MatchShape, ScalarizationMode, TargetMode
from bo_mcp_server.schema_extension import intake_schema_with_backend_extensions


def _enum_values(node: dict[str, Any]) -> set[str]:
    if "enum" in node:
        return set(node["enum"])
    values: set[str] = set()
    for branch in node.get("anyOf", []):
        if isinstance(branch, dict) and "enum" in branch:
            values.update(branch["enum"])
    return values


def _item_properties(schema: dict[str, Any], field: str) -> dict[str, Any]:
    return schema["properties"][field]["items"]["properties"]


def test_every_constraint_type_is_advertised() -> None:
    schema = intake_schema_with_backend_extensions()
    advertised = _enum_values(_item_properties(schema, "constraints")["type"])
    missing = {m.value for m in ConstraintType} - advertised
    assert not missing, f"MCP intake schema is missing constraint types: {sorted(missing)}"


def test_constraint_fields_are_advertised() -> None:
    props = _item_properties(intake_schema_with_backend_extensions(), "constraints")
    assert {"min_cardinality", "max_cardinality", "is_interpoint"} <= set(props)


def test_objective_fields_and_enums_are_advertised() -> None:
    props = _item_properties(intake_schema_with_backend_extensions(), "objectives")
    expected_fields = {
        "target_mode",
        "match_shape",
        "match_scale",
        "weight",
        "normalization_bounds",
        "transform",
    }
    assert expected_fields <= set(props)
    assert {m.value for m in TargetMode} <= _enum_values(props["target_mode"])
    assert {m.value for m in MatchShape} <= _enum_values(props["match_shape"])


def test_scalarization_fields_are_advertised() -> None:
    schema = intake_schema_with_backend_extensions()
    assert "scalarization" in schema["properties"]
    assert "scalarizer" in schema["properties"]
    advertised = _enum_values(schema["properties"]["scalarization"])
    assert {m.value for m in ScalarizationMode} <= advertised


def test_scalarizer_and_transform_kind_enums_are_complete() -> None:
    """Member completeness for the two enums the earlier gate left unasserted."""
    from bo_engine.types import ObjectiveTransformKind, ScalarizerKind

    schema = intake_schema_with_backend_extensions()
    assert {m.value for m in ScalarizerKind} <= _enum_values(schema["properties"]["scalarizer"])
    props = _item_properties(schema, "objectives")
    kind_node = next(
        branch["properties"]["kind"]
        for branch in props["transform"]["anyOf"]
        if isinstance(branch, dict) and "properties" in branch
    )
    assert {m.value for m in ObjectiveTransformKind} <= _enum_values(kind_node)
