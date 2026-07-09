"""Toolset builders and registration helpers for BO-MCP."""

from __future__ import annotations

import os
from typing import Any

from pydantic_ai import FunctionToolset, Tool
from pydantic_ai.mcp import MCPToolset, SSETransport

from bo_mcp.openapi import (
    inspect_bo_mcp_openapi_operation,
    inspect_bo_mcp_openapi_overview,
)

BO_MCP_TOOLSET_ID = "bo_mcp_toolset"
BO_MCP_OPENAPI_TOOLSET_ID = "bo_mcp_openapi_toolset"


def _bo_mcp_sse_url() -> str:
    return os.getenv("BO_MCP_SSE_URL", "http://mcp:8001/sse")


def build_bo_mcp_toolset() -> MCPToolset[Any]:
    """Build the BO MCP client used by the chat runtime."""
    return MCPToolset(SSETransport(_bo_mcp_sse_url()), id=BO_MCP_TOOLSET_ID)


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
