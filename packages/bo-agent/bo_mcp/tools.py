from __future__ import annotations

import os
from typing import Any

from pydantic_ai import FunctionToolset, Tool
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.mcp import MCPServerSSE

from domains.bo_mcp.openapi import (
    inspect_bo_mcp_openapi_overview,
    inspect_bo_mcp_openapi_operation,
)
from grafico.agents.compat import unwrap_mutable_agent
from grafico.tools.toolset_registration import register_persistent_toolset

BO_MCP_TOOLSET_ID = "bo_mcp_toolset"
BO_MCP_OPENAPI_TOOLSET_ID = "bo_mcp_openapi_toolset"


def _bo_mcp_sse_url() -> str:
    return os.getenv("BO_MCP_SSE_URL", "http://mcp:8001/sse")


def build_bo_mcp_toolset() -> MCPServerSSE:
    """Build the BO MCP client used by the chat runtime."""
    return MCPServerSSE(_bo_mcp_sse_url(), id=BO_MCP_TOOLSET_ID)


def build_bo_mcp_openapi_toolset() -> FunctionToolset[object]:
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


def register_bo_mcp_tools(agent: AbstractAgent[Any, Any]) -> None:
    mutable_agent = unwrap_mutable_agent(agent)
    register_persistent_toolset(
        mutable_agent,
        toolset_id=BO_MCP_TOOLSET_ID,
        build_toolset=build_bo_mcp_toolset,
    )


def register_bo_mcp_openapi_tools(agent: AbstractAgent[Any, Any]) -> None:
    mutable_agent = unwrap_mutable_agent(agent)
    register_persistent_toolset(
        mutable_agent,
        toolset_id=BO_MCP_OPENAPI_TOOLSET_ID,
        build_toolset=build_bo_mcp_openapi_toolset,
    )
