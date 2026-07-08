"""The REST OpenAPI advertises the extended neutral spec surface.

REST half of the neutral-surface schema gate (constraint types +
cardinality/interpoint fields, objective match/desirability/transform
fields, scalarization); the MCP half lives in
``packages/bo-mcp-server/tests/unit/test_neutral_surface_schema_gate.py``.
"""

from __future__ import annotations


def _schemas() -> dict:
    from api.main import app

    return app.openapi()["components"]["schemas"]


def _enum_union(schemas: dict, members: set[str]) -> set[str]:
    """Union of every component enum containing the given anchor members."""
    values: set[str] = set()
    for schema in schemas.values():
        if isinstance(schema, dict) and schema.get("enum") and members <= set(schema["enum"]):
            values.update(schema["enum"])
    return values


def test_openapi_advertises_full_constraint_type_enum() -> None:
    from bo_engine.types import ConstraintType

    advertised = _enum_union(_schemas(), {"sum_equals", "linear"})
    missing = {m.value for m in ConstraintType} - advertised
    assert not missing, f"OpenAPI is missing constraint types: {sorted(missing)}"


def test_openapi_advertises_objective_and_constraint_fields() -> None:
    schemas = _schemas()
    objective_carriers = [
        s
        for s in schemas.values()
        if isinstance(s, dict) and "target_mode" in s.get("properties", {})
    ]
    assert objective_carriers, "No OpenAPI schema advertises the objective target_mode field"
    for carrier in objective_carriers:
        assert {"match_shape", "match_scale", "weight", "normalization_bounds", "transform"} <= (
            set(carrier["properties"])
        )
    constraint_carriers = [
        s
        for s in schemas.values()
        if isinstance(s, dict) and "min_cardinality" in s.get("properties", {})
    ]
    assert constraint_carriers, "No OpenAPI schema advertises the cardinality fields"


def test_openapi_advertises_scalarization() -> None:
    schemas = _schemas()
    carriers = [
        s
        for s in schemas.values()
        if isinstance(s, dict) and "scalarization" in s.get("properties", {})
    ]
    assert carriers, "No OpenAPI schema advertises the scalarization field"


def test_openapi_goal_enums_are_member_complete() -> None:
    """Member completeness for every objective-goal enum on the REST transport.

    Mirrors the MCP gate's enum-union assertions so a new enum member can
    never appear on one transport only.
    """
    from bo_engine.types import (
        MatchShape,
        ObjectiveTransformKind,
        ScalarizationMode,
        ScalarizerKind,
        TargetMode,
    )

    schemas = _schemas()
    anchor_by_enum = {
        TargetMode: {"minimize", "match"},
        MatchShape: {"absolute", "bell"},
        ScalarizationMode: {"pareto", "desirability"},
        ScalarizerKind: {"mean", "geom_mean"},
        ObjectiveTransformKind: {"log", "clamp"},
    }
    for enum_cls, anchors in anchor_by_enum.items():
        advertised = _enum_union(schemas, anchors)
        missing = {m.value for m in enum_cls} - advertised
        assert not missing, f"OpenAPI is missing {enum_cls.__name__} members: {sorted(missing)}"
