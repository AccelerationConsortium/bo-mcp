"""Toolset builders and registration helpers for BO-MCP."""

from __future__ import annotations

import os
from typing import Any

from pydantic_ai import Agent, FunctionToolset, RunContext, Tool
from pydantic_ai.mcp import MCPToolset, SSETransport

from bo_mcp.openapi import (
    inspect_bo_mcp_openapi_operation,
    inspect_bo_mcp_openapi_overview,
)

BO_MCP_TOOLSET_ID = "bo_mcp_toolset"
BO_MCP_OPENAPI_TOOLSET_ID = "bo_mcp_openapi_toolset"


def _bo_mcp_sse_url() -> str:
    sse_url = os.getenv("BO_MCP_SSE_URL", "").strip()
    if not sse_url:
        message = "BO_MCP_SSE_URL is not set; configure the BO-MCP SSE endpoint."
        raise ValueError(message)
    return sse_url


def build_bo_mcp_toolset() -> MCPToolset[Any]:
    """Build the BO MCP client used by the chat runtime."""
    return MCPToolset(SSETransport(_bo_mcp_sse_url()), id=BO_MCP_TOOLSET_ID)


def register_bo_mcp_tools(agent: Agent[Any, Any]) -> None:
    """Register the BO-MCP SSE toolset on an agent once."""
    if any(toolset.id == BO_MCP_TOOLSET_ID for toolset in agent.toolsets):
        return

    @agent.toolset(per_run_step=False, id=BO_MCP_TOOLSET_ID)
    def bo_mcp_toolset(_ctx: RunContext[Any]) -> MCPToolset[Any]:
        return build_bo_mcp_toolset()


def build_bo_mcp_openapi_toolset() -> FunctionToolset[object]:
    """Build tools that inspect the live BO-MCP OpenAPI schema."""
    return FunctionToolset(
        id=BO_MCP_OPENAPI_TOOLSET_ID,
        tools=[
            Tool(
                inspect_bo_mcp_openapi_overview,
                name="inspect_bo_mcp_openapi_overview",
                max_retries=2,
            ),
            Tool(
                inspect_bo_mcp_openapi_operation,
                name="inspect_bo_mcp_openapi_operation",
                max_retries=2,
            ),
        ],
    )
