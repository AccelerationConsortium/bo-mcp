"""Suggestion explanation tool wrapper for MCP."""

from typing import Any

from bo_mcp_server.operations.suggestion_explanation import (
    get_suggestion_explanation_operation,
)
from bo_mcp_server.server import mcp
from bo_mcp_server.tools.annotations import READ_ONLY


@mcp.tool(name="bo_get_suggestion_explanation", annotations=READ_ONLY)
async def get_suggestion_explanation(suggestion_id: str) -> dict[str, Any]:
    """Get detailed explanation for why a suggestion was generated.

    Workflow: Call after bo_generate_suggestions to understand the reasoning
    behind a specific suggestion (acquisition value, model confidence, etc.).

    Args:
        suggestion_id: UUID of the suggestion to explain.

    Returns:
        Dictionary with:
            - success: Boolean
            - explanation: Human-readable explanation
            - provenance: Full provenance data (model type, acquisition function, etc.)
            - errors: List of error messages
    """
    return await get_suggestion_explanation_operation(suggestion_id)
