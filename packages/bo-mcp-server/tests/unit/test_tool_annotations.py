"""MCP tools carry the right safety-hint annotations.

Background: ``ToolAnnotations`` are advisory hints that let agents pick
a retry policy without trial-and-error. ``readOnlyHint=True`` marks
tools that are always safe to retry; ``destructiveHint=True`` marks
operations that may be irreversible; ``idempotentHint=True`` marks
operations whose repeats have no compounding effect. This suite pins
the mapping so the registration cannot silently regress.

The MCP tool registry on FastMCP stores annotations on each tool, so
the assertions read directly off the registered objects rather than
re-importing each decorator output.
"""

from __future__ import annotations

from typing import Any

import pytest

from bo_mcp_server.server import create_mcp_server


@pytest.fixture(scope="module")
def registered_tools() -> dict[str, Any]:
    """Build the MCP server once and expose tools keyed by name."""
    server = create_mcp_server()
    return {t.name: t for t in server._tool_manager.list_tools()}


READ_ONLY_TOOLS = {
    "bo_health_check",
    "bo_list_capabilities",
    "bo_list_campaigns",
    "bo_list_results",
    "bo_list_suggestions",
    "bo_batch_get_status",
    "bo_get_diagnostics",
    "bo_get_suggestion_explanation",
    "bo_compare_campaigns",
    "bo_discover_transfer_candidates",
    "bo_export_campaign",
    "bo_validate_intake",
}

DESTRUCTIVE_TOOLS = {"bo_terminate_campaign"}

IDEMPOTENT_MUTATION_TOOLS = {"bo_pause_campaign", "bo_resume_campaign"}

NON_IDEMPOTENT_TOOLS = {
    "bo_create_campaign",
    "bo_submit_results",
    "bo_upload_results_file",
    "bo_generate_suggestions",
    "bo_update_suggestion_status",
}


@pytest.mark.parametrize("name", sorted(READ_ONLY_TOOLS))
def test_read_only_tools(registered_tools: dict[str, Any], name: str) -> None:
    tool = registered_tools[name]
    annotations = tool.annotations
    assert annotations is not None, f"{name} missing annotations"
    assert annotations.readOnlyHint is True
    assert annotations.destructiveHint is False


@pytest.mark.parametrize("name", sorted(DESTRUCTIVE_TOOLS))
def test_destructive_tools(registered_tools: dict[str, Any], name: str) -> None:
    tool = registered_tools[name]
    annotations = tool.annotations
    assert annotations is not None
    assert annotations.readOnlyHint is False
    assert annotations.destructiveHint is True


@pytest.mark.parametrize("name", sorted(IDEMPOTENT_MUTATION_TOOLS))
def test_idempotent_mutating_tools(registered_tools: dict[str, Any], name: str) -> None:
    tool = registered_tools[name]
    annotations = tool.annotations
    assert annotations is not None
    assert annotations.readOnlyHint is False
    assert annotations.destructiveHint is False
    assert annotations.idempotentHint is True


@pytest.mark.parametrize("name", sorted(NON_IDEMPOTENT_TOOLS))
def test_non_idempotent_mutating_tools(registered_tools: dict[str, Any], name: str) -> None:
    tool = registered_tools[name]
    annotations = tool.annotations
    assert annotations is not None
    assert annotations.readOnlyHint is False
    assert annotations.destructiveHint is False
    assert annotations.idempotentHint is False
