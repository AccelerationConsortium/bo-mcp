"""Schema-surface gate: per-campaign ``backend_options`` on both transports.

Per-parameter BayBE options are already spliced into the MCP tool schemas
and the REST OpenAPI through the ``parameter_options_schema`` hook. This
suite is the equivalent **positive** gate for the per-campaign
``backend_options`` surface: every typed :class:`BayBEBackendOptions`
field must be advertised in the enriched intake schema, so a new
campaign-level option can never ship silently undiscoverable on both
transports (a naive MCP-vs-REST diff cannot catch that failure mode —
both sides would be equally blind).

The REST half of the gate lives in
``packages/bo-mcp-api/tests/test_backend_options_schema_openapi.py``,
which asserts the same field set on the generated OpenAPI document.
"""

from __future__ import annotations

from typing import Any

import pytest

from bo_engine_baybe.options import BayBEBackendOptions
from bo_mcp_server.schema_extension import (
    augment_backend_options,
    backend_options_fragments,
    intake_schema_with_backend_extensions,
)


def _object_branch(node: dict[str, Any]) -> dict[str, Any]:
    """Return the object-typed branch of an ``anyOf`` (nullable) field schema."""
    any_of = node.get("anyOf")
    if isinstance(any_of, list):
        return next(m for m in any_of if isinstance(m, dict) and m.get("type") == "object")
    assert node.get("type") == "object"
    return node


def _baybe_backend_options_properties(schema: dict[str, Any]) -> dict[str, Any]:
    """Extract the spliced ``backend_options.baybe`` properties from an intake schema."""
    field = schema["properties"]["backend_options"]
    branch = _object_branch(field)
    return branch["properties"]["baybe"]["properties"]


class TestBackendOptionsFragments:
    def test_baybe_contributes_a_fragment(self) -> None:
        fragments = backend_options_fragments()
        assert "baybe" in fragments
        assert fragments["baybe"]["type"] == "object"

    def test_botorch_contributes_no_fragment(self) -> None:
        # BoTorch has no typed campaign-level options; a None fragment
        # must keep it out of the spliced schema entirely.
        fragments = backend_options_fragments()
        assert "botorch" not in fragments


class TestMCPIntakeSchemaSurface:
    @pytest.mark.parametrize("field_name", sorted(BayBEBackendOptions.model_fields))
    def test_every_baybe_backend_option_is_advertised(self, field_name: str) -> None:
        """Every typed BayBE campaign-level option appears in the MCP intake schema."""
        schema = intake_schema_with_backend_extensions()
        props = _baybe_backend_options_properties(schema)
        assert field_name in props, (
            f"BayBEBackendOptions.{field_name} is missing from the enriched intake "
            "schema — the backend_options_schema hook or the splice is broken."
        )

    def test_recommender_config_fields_are_inlined(self) -> None:
        """Nested recommender fields survive the ``$defs`` inlining."""
        schema = intake_schema_with_backend_extensions()
        props = _baybe_backend_options_properties(schema)
        recommender = _object_branch(props["recommender"])
        assert "switch_after" in recommender["properties"]

    def test_parameter_options_still_advertised(self) -> None:
        """The combined enrichment keeps the per-parameter splice intact."""
        schema = intake_schema_with_backend_extensions()
        items = schema["properties"]["parameters"]["items"]
        po = _object_branch(items["properties"]["parameter_options"])
        assert "role" in po["properties"]["baybe"]["properties"]

    def test_unknown_backends_remain_accepted(self) -> None:
        """The splice must not close the unknown-backend passthrough."""
        schema = intake_schema_with_backend_extensions()
        branch = _object_branch(schema["properties"]["backend_options"])
        assert "additionalProperties" in branch


class TestOpenAPIStyleAugmentation:
    def test_augment_backend_options_is_idempotent(self) -> None:
        doc: dict[str, Any] = {
            "components": {
                "schemas": {
                    "IntakeData": {
                        "properties": {
                            "backend_options": {
                                "anyOf": [
                                    {
                                        "type": "object",
                                        "additionalProperties": {"type": "object"},
                                    },
                                    {"type": "null"},
                                ],
                            },
                        },
                    },
                },
            },
        }
        augment_backend_options(doc)
        first = doc["components"]["schemas"]["IntakeData"]["properties"]["backend_options"]
        augment_backend_options(doc)
        second = doc["components"]["schemas"]["IntakeData"]["properties"]["backend_options"]
        assert first == second
        props = _object_branch(first)["properties"]["baybe"]["properties"]
        assert set(BayBEBackendOptions.model_fields) <= set(props)
