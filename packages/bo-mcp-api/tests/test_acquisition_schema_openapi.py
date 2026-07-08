"""The REST OpenAPI advertises the full neutral acquisition surface.

REST parity gate for the acquisition-method family: every member of the
neutral :class:`bo_engine.types.AcquisitionMethod` enum and the
``acquisition_beta`` knob must appear in the generated OpenAPI document
(the MCP half lives in
``packages/bo-mcp-server/tests/unit/test_acquisition_schema_surface.py``).
"""

from __future__ import annotations


def test_openapi_advertises_full_acquisition_method_enum() -> None:
    from api.main import app
    from bo_engine.types import AcquisitionMethod

    openapi = app.openapi()
    schemas = openapi["components"]["schemas"]
    enum_values: set[str] = set()
    for schema in schemas.values():
        if (
            isinstance(schema, dict)
            and schema.get("enum")
            and {"auto", "noisy_expected_improvement"} <= set(schema["enum"])
        ):
            enum_values.update(schema["enum"])
    missing = {m.value for m in AcquisitionMethod} - enum_values
    assert not missing, f"OpenAPI is missing acquisition methods: {sorted(missing)}"


def test_openapi_advertises_acquisition_beta() -> None:
    from api.main import app

    openapi = app.openapi()
    schemas = openapi["components"]["schemas"]
    carriers = [
        name
        for name, schema in schemas.items()
        if isinstance(schema, dict) and "acquisition_beta" in schema.get("properties", {})
    ]
    assert carriers, "No OpenAPI component schema advertises acquisition_beta"
