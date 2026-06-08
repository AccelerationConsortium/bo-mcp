"""The MCP tool schemas advertise the BayBE molecular (substance) recipe.

``parameter_options`` used to be an opaque per-backend ``dict`` in the
generated tool schema, so an agent could not discover that a categorical
parameter can carry ``role=substance`` + ``substance_data`` (a SMILES map) +
``substance_encoding`` without reading source. The schema-extension layer now
splices each backend's typed ``parameter_options`` shape into the tool
schemas. These tests pin that both intake tools (``bo_create_campaign`` and
``bo_validate_intake``) expose the BayBE ``role`` / ``substance_data`` /
``substance_encoding`` surface, mirroring a BayBE solvent-screening
``SubstanceParameter`` spec
(https://emdgroup.github.io/baybe/stable/examples/Basics/parameters.html).
"""

from __future__ import annotations

import pytest

_INTAKE_TOOLS = ("bo_create_campaign", "bo_validate_intake")


def _baybe_parameter_options(tool_input_schema: dict) -> dict:
    """Drill into a tool inputSchema to the spliced ``parameter_options.baybe``."""
    intake = tool_input_schema["properties"]["intake_data"]
    po = intake["properties"]["parameters"]["items"]["properties"]["parameter_options"]
    object_branch = next(m for m in po["anyOf"] if m.get("type") == "object")
    # additionalProperties is preserved so unknown/foreign backends still
    # validate — the typed surface documents, it does not lock the field down.
    assert "additionalProperties" in object_branch
    return object_branch["properties"]["baybe"]


@pytest.mark.parametrize("tool_name", _INTAKE_TOOLS)
async def test_intake_tool_schema_advertises_substance_role(tool_name: str) -> None:
    from bo_mcp_server.server import create_mcp_server

    server = create_mcp_server()
    tools = await server.list_tools()
    tool = next(t for t in tools if t.name == tool_name)

    baybe = _baybe_parameter_options(tool.inputSchema)
    props = baybe["properties"]
    assert {"role", "substance_data", "substance_encoding"} <= set(props)

    # ``role`` advertises the substance option.
    assert "substance" in props["role"]["enum"]

    # ``substance_encoding`` advertises the curated encoding enum (Optional
    # renders as anyOf[enum, null] after inlining).
    encoding_enum = next(
        branch["enum"] for branch in props["substance_encoding"]["anyOf"] if "enum" in branch
    )
    assert {"MORDRED", "ECFP", "RDKIT2DDESCRIPTORS", "RDKITFINGERPRINT"} <= set(encoding_enum)
    assert "RDKIT" not in set(encoding_enum)

    # ``substance_data`` is the SMILES map (object of string values).
    substance_data = props["substance_data"]
    object_branch = next(m for m in substance_data["anyOf"] if m.get("type") == "object")
    assert object_branch["additionalProperties"]["type"] == "string"
