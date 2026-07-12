"""Toolset builders and registration helpers for BO-MCP."""

from __future__ import annotations

import logging
import os
from typing import Any, Self

from pydantic_ai import FunctionToolset, RunContext, Tool
from pydantic_ai.mcp import MCPToolset, SSETransport
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool

from bo_mcp.openapi import (
    inspect_bo_mcp_openapi_operation,
    inspect_bo_mcp_openapi_overview,
)

BO_MCP_TOOLSET_ID = "bo_mcp_toolset"
BO_MCP_OPENAPI_TOOLSET_ID = "bo_mcp_openapi_toolset"

logger = logging.getLogger(__name__)


def _bo_mcp_sse_url() -> str:
    sse_url = os.getenv("BO_MCP_SSE_URL", "").strip()
    if not sse_url:
        message = "BO_MCP_SSE_URL is not set; configure the BO-MCP SSE endpoint."
        raise ValueError(message)
    return sse_url


def build_bo_mcp_toolset() -> MCPToolset[Any]:
    """Build the BO MCP client used by the chat runtime."""
    return MCPToolset(SSETransport(_bo_mcp_sse_url()), id=BO_MCP_TOOLSET_ID)


class OptionalBoMcpToolset(AbstractToolset[Any]):
    """Expose BO-MCP tools when SSE is available without aborting the agent run."""

    def __init__(self, sse_url: str) -> None:
        """Configure a per-run connection to the optional SSE endpoint."""
        self._sse_url = sse_url
        self._mcp_toolset: MCPToolset[Any] | None = None

    @property
    def id(self) -> str:
        """Return the stable toolset identifier."""
        return BO_MCP_TOOLSET_ID

    async def for_run(self, ctx: RunContext[Any]) -> AbstractToolset[Any]:
        """Return isolated connection state for each specialist run."""
        del ctx
        return OptionalBoMcpToolset(self._sse_url)

    async def __aenter__(self) -> Self:
        """Connect when possible, otherwise continue with no MCP tools."""
        mcp_toolset = MCPToolset(SSETransport(self._sse_url), id=BO_MCP_TOOLSET_ID)
        try:
            await mcp_toolset.__aenter__()
        except Exception as exc:  # noqa: BLE001 — optional service failures must degrade safely
            logger.warning(
                "BO-MCP SSE unavailable; continuing without interactive MCP tools",
                extra={
                    "sse_url": self._sse_url,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            return self

        self._mcp_toolset = mcp_toolset
        return self

    async def __aexit__(self, *args: object) -> bool | None:
        """Close the MCP connection when one was established."""
        if self._mcp_toolset is None:
            return None
        try:
            return await self._mcp_toolset.__aexit__(*args)
        finally:
            self._mcp_toolset = None

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        """Return no tools while the optional service is unavailable."""
        if self._mcp_toolset is None:
            return {}
        return await self._mcp_toolset.get_tools(ctx)

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
    ) -> Any:  # noqa: ANN401 — tool return values are defined by the remote MCP server
        """Delegate calls to the connected MCP toolset."""
        if self._mcp_toolset is None:
            message = "BO-MCP SSE toolset is unavailable"
            raise RuntimeError(message)
        return await self._mcp_toolset.call_tool(name, tool_args, ctx, tool)


def build_optional_bo_mcp_toolset() -> OptionalBoMcpToolset | None:
    """Build a fail-soft SSE toolset when the endpoint is configured."""
    sse_url = os.getenv("BO_MCP_SSE_URL", "").strip()
    if not sse_url:
        logger.info("BO_MCP_SSE_URL is unset; interactive BO-MCP tools are disabled")
        return None

    return OptionalBoMcpToolset(sse_url)


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
