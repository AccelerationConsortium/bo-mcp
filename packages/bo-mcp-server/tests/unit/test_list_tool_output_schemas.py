"""``bo_list_*`` tool output schemas must type their list items, not just the envelope.

Regression guard for the response-model gap left open when tools were
first annotated with real Pydantic models (see ``tools/response_models.py``):
the top-level envelope (``success``, ``campaigns``, ``total_count``, ...)
became discoverable, but each list entry stayed ``dict[str, Any]``, so
``outputSchema.properties.<list>.items`` still advertised nothing but
``additionalProperties: true``. This test pins the fix -- each item schema
must expose its real field names -- so a future refactor cannot silently
regress list items back to an opaque blob.

Reference: MCP tool output schemas
https://modelcontextprotocol.io/specification/2025-06-18/server/tools#output-schema
"""

from __future__ import annotations

from typing import Any

from bo_mcp_server.server import create_mcp_server


def _list_item_properties(tool_name: str, list_field: str) -> set[str]:
    """Resolve the ``items`` sub-schema of ``list_field`` on ``tool_name``'s outputSchema.

    Pydantic/FastMCP emit nested models via a top-level ``$ref`` into
    ``$defs`` rather than inlining the object schema, so this follows
    the ref when present.
    """
    mcp = create_mcp_server()
    tool = mcp._tool_manager.get_tool(tool_name)
    assert tool is not None, f"tool {tool_name!r} is not registered"
    output_schema: dict[str, Any] | None = tool.output_schema
    assert output_schema is not None, f"tool {tool_name!r} has no outputSchema"

    items = output_schema["properties"][list_field]["items"]
    ref = items.get("$ref")
    if ref is None:
        return set(items.get("properties", {}))
    def_name = ref.rsplit("/", maxsplit=1)[-1]
    return set(output_schema["$defs"][def_name]["properties"])


def test_list_campaigns_items_are_typed() -> None:
    properties = _list_item_properties("bo_list_campaigns", "campaigns")
    assert {"campaign_id", "name", "status", "iteration", "created_at"} <= properties


def test_list_suggestions_items_are_typed() -> None:
    properties = _list_item_properties("bo_list_suggestions", "suggestions")
    assert {"suggestion_id", "status", "parameter_values", "created_at"} <= properties


def test_list_results_items_are_typed() -> None:
    properties = _list_item_properties("bo_list_results", "results")
    assert {"result_id", "objective_values", "parameter_values", "created_at"} <= properties
