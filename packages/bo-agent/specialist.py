"""Bayesian optimization specialist subagent configuration."""

from typing import Any

from bo_mcp.tools import build_bo_mcp_openapi_toolset
from prompts import BO_SPECIALIST_INSTRUCTIONS

BO_SPECIALIST_NAME = "bo-specialist"
BO_SPECIALIST_DESCRIPTION = (
    "Designs BO-MCP campaigns and integrates the appropriate function-evaluation packages."
)


def build_bo_specialist_subagent() -> dict[str, Any]:
    """Return the BO specialist's subagent configuration."""
    return {
        "name": BO_SPECIALIST_NAME,
        "description": BO_SPECIALIST_DESCRIPTION,
        "instructions": BO_SPECIALIST_INSTRUCTIONS,
        "toolsets": [build_bo_mcp_openapi_toolset()],
    }
