"""The REST OpenAPI advertises the BayBE molecular (substance) recipe.

REST parity with the MCP tool schemas: the intake request body's
``parameters[].parameter_options`` is enriched with each backend's typed
shape in the generated OpenAPI, so REST/OpenAPI clients discover the BayBE
``role=substance`` recipe (``substance_data`` SMILES map + ``substance_encoding``)
exactly like MCP clients. Mirrors a BayBE solvent-screening
``SubstanceParameter`` spec
(https://emdgroup.github.io/baybe/stable/examples/Basics/parameters.html).
"""

from __future__ import annotations


def _input_parameter_schemas(openapi: dict) -> list[dict]:
    """Return every generated ``InputParameter`` component schema."""
    schemas = openapi["components"]["schemas"]
    return [v for k, v in schemas.items() if k.startswith("InputParameter")]


def test_openapi_input_parameter_advertises_substance_role() -> None:
    from api.main import app

    openapi = app.openapi()
    input_parameter_schemas = _input_parameter_schemas(openapi)
    assert input_parameter_schemas, "OpenAPI is missing the InputParameter component schema"

    for schema in input_parameter_schemas:
        po = schema["properties"]["parameter_options"]
        object_branch = next(m for m in po["anyOf"] if m.get("type") == "object")
        # Unknown/foreign backends remain accepted via additionalProperties.
        assert "additionalProperties" in object_branch
        baybe = object_branch["properties"]["baybe"]
        props = baybe["properties"]
        assert {"role", "substance_data", "substance_encoding"} <= set(props)
        assert "substance" in props["role"]["enum"]
        encoding_enum = next(
            branch["enum"] for branch in props["substance_encoding"]["anyOf"] if "enum" in branch
        )
        assert {"MORDRED", "ECFP", "RDKIT"} <= set(encoding_enum)
