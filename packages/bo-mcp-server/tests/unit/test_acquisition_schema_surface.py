"""Schema-surface gate: the acquisition surface on the MCP transport.

Every member of the neutral :class:`bo_engine.types.AcquisitionMethod`
enum and the ``acquisition_beta`` knob must be advertised by the enriched
MCP intake schema — the guardrail that prevents a new neutral enum member
from shipping undiscoverable (the REST half lives in
``packages/bo-mcp-api/tests/test_acquisition_schema_openapi.py``).
"""

from __future__ import annotations

from typing import Any

from bo_engine.types import AcquisitionMethod
from bo_mcp_server.schema_extension import intake_schema_with_backend_extensions


def _enum_values(node: dict[str, Any]) -> set[str]:
    """Collect enum values from a field schema (inlined or anyOf-wrapped)."""
    if "enum" in node:
        return set(node["enum"])
    values: set[str] = set()
    for branch in node.get("anyOf", []):
        if isinstance(branch, dict) and "enum" in branch:
            values.update(branch["enum"])
    return values


def test_every_acquisition_method_is_advertised() -> None:
    schema = intake_schema_with_backend_extensions()
    advertised = _enum_values(schema["properties"]["acquisition_method"])
    missing = {m.value for m in AcquisitionMethod} - advertised
    assert not missing, f"MCP intake schema is missing acquisition methods: {sorted(missing)}"


def test_acquisition_beta_is_advertised() -> None:
    schema = intake_schema_with_backend_extensions()
    assert "acquisition_beta" in schema["properties"]
