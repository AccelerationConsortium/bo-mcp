"""Bayesian optimization specialist subagent configuration."""

from typing import Any

from bo_mcp.tools import build_bo_mcp_openapi_toolset, build_optional_bo_mcp_toolset
from prompts import BO_SPECIALIST_INSTRUCTIONS
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_deep import DeepAgentDeps, create_deep_agent

BO_SPECIALIST_NAME = "bo-specialist"
BO_SPECIALIST_DESCRIPTION = (
    "Designs BO-MCP campaigns and integrates the appropriate function-evaluation packages."
)


def build_bo_specialist_agent(config: dict[str, Any]) -> Agent[DeepAgentDeps, str]:
    """Build the specialist without pydantic-deep's default base instructions."""
    return create_deep_agent(
        model=config["model"],
        instructions=config["instructions"],
        include_filesystem=True,
        include_execute=True,
        include_todo=True,
        web_search=False,
        web_fetch=False,
        thinking=False,
        include_subagents=False,
        include_skills=False,
        include_plan=False,
        include_teams=False,
        include_monitoring=False,
        include_builtin_subagents=False,
        context_manager=False,
        cost_tracking=False,
        include_memory=False,
        extra_toolsets=tuple(config.get("toolsets") or []),
    )


def build_bo_specialist_subagent(model: str | Model) -> dict[str, Any]:
    """Return the BO specialist's subagent configuration."""
    toolsets = [build_bo_mcp_openapi_toolset()]
    if mcp_toolset := build_optional_bo_mcp_toolset():
        toolsets.append(mcp_toolset)

    return {
        "name": BO_SPECIALIST_NAME,
        "description": BO_SPECIALIST_DESCRIPTION,
        "model": model,
        "instructions": BO_SPECIALIST_INSTRUCTIONS,
        "agent_factory": build_bo_specialist_agent,
        "toolsets": toolsets,
    }
