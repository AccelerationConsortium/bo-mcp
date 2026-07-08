"""The REST OpenAPI advertises the typed per-campaign BayBE options.

REST parity with the MCP tool schemas: the intake request body's
``backend_options`` is enriched with each backend's typed campaign-level
shape in the generated OpenAPI, so REST/OpenAPI clients discover e.g.
BayBE's recommender configuration exactly like MCP clients. This is the
REST half of the positive schema-surface gate; the MCP half lives in
``packages/bo-mcp-server/tests/unit/test_backend_options_schema_surface.py``.
"""

from __future__ import annotations


def _intake_schemas(openapi: dict) -> list[dict]:
    """Return every generated intake component schema carrying backend_options."""
    schemas = openapi["components"]["schemas"]
    return [
        v
        for v in schemas.values()
        if isinstance(v, dict) and "backend_options" in v.get("properties", {})
    ]


def test_openapi_intake_advertises_baybe_backend_options() -> None:
    from api.main import app
    from bo_engine_baybe.options import BayBEBackendOptions

    openapi = app.openapi()
    intake_schemas = _intake_schemas(openapi)
    assert intake_schemas, "OpenAPI has no component schema with a backend_options field"

    for schema in intake_schemas:
        bo = schema["properties"]["backend_options"]
        object_branch = next(m for m in bo["anyOf"] if m.get("type") == "object")
        # Unknown/foreign backends remain accepted via additionalProperties.
        assert "additionalProperties" in object_branch
        baybe = object_branch["properties"]["baybe"]
        props = set(baybe["properties"])
        missing = set(BayBEBackendOptions.model_fields) - props
        assert not missing, f"OpenAPI backend_options.baybe is missing fields: {sorted(missing)}"
