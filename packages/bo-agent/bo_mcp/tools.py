"""Toolset builders and registration helpers for BO-MCP."""

from __future__ import annotations

import logging
import os
from typing import Any, Self

from pydantic_ai import FunctionToolset, RunContext, Tool
from pydantic_ai.mcp import MCPToolset, StreamableHttpTransport
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool

from bo_mcp.openapi import (
    inspect_bo_mcp_openapi_operation,
    inspect_bo_mcp_openapi_overview,
)

BO_MCP_TOOLSET_ID = "bo_mcp_toolset"
BO_MCP_OPENAPI_TOOLSET_ID = "bo_mcp_openapi_toolset"

logger = logging.getLogger(__name__)


def _read_bo_mcp_url() -> str:
    """Return the configured MCP endpoint URL, or an empty string when unset.

    The server moved from the deprecated SSE transport to streamable-http;
    a leftover ``BO_MCP_SSE_URL`` cannot work against the new transport, so
    it is ignored with a loud migration hint instead of failing obscurely
    at connect time.
    """
    url = os.getenv("BO_MCP_URL", "").strip()
    if not url and os.getenv("BO_MCP_SSE_URL", "").strip():
        logger.warning(
            "BO_MCP_SSE_URL is set but no longer used: the BO-MCP server "
            "moved from the deprecated SSE transport to streamable-http. "
            "Set BO_MCP_URL to the /mcp endpoint "
            "(e.g. http://127.0.0.1:8001/mcp)."
        )
    return url


def _bo_mcp_url() -> str:
    url = _read_bo_mcp_url()
    if not url:
        message = (
            "BO_MCP_URL is not set; configure the BO-MCP streamable-http "
            "endpoint (e.g. http://127.0.0.1:8001/mcp)."
        )
        raise ValueError(message)
    return url


def build_bo_mcp_toolset() -> MCPToolset[Any]:
    """Build the BO MCP client used by the chat runtime."""
    return MCPToolset(StreamableHttpTransport(_bo_mcp_url()), id=BO_MCP_TOOLSET_ID)


class OptionalBoMcpToolset(AbstractToolset[Any]):
    """Expose BO-MCP tools when the endpoint is reachable without aborting the run."""

    def __init__(self, url: str) -> None:
        """Configure a per-run connection to the optional MCP endpoint."""
        self._url = url
        self._mcp_toolset: MCPToolset[Any] | None = None

    @property
    def id(self) -> str:
        """Return the stable toolset identifier."""
        return BO_MCP_TOOLSET_ID

    async def for_run(self, ctx: RunContext[Any]) -> AbstractToolset[Any]:
        """Return isolated connection state for each specialist run."""
        del ctx
        return OptionalBoMcpToolset(self._url)

    async def __aenter__(self) -> Self:
        """Connect when possible, otherwise continue with no MCP tools."""
        mcp_toolset = MCPToolset(StreamableHttpTransport(self._url), id=BO_MCP_TOOLSET_ID)
        try:
            await mcp_toolset.__aenter__()
        except Exception as exc:  # noqa: BLE001 — optional service failures must degrade safely
            logger.warning(
                "BO-MCP endpoint unavailable; continuing without interactive MCP tools",
                extra={
                    "url": self._url,
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
            message = "BO-MCP toolset is unavailable"
            raise RuntimeError(message)
        return await self._mcp_toolset.call_tool(name, tool_args, ctx, tool)


def build_optional_bo_mcp_toolset() -> OptionalBoMcpToolset | None:
    """Build a fail-soft MCP toolset when the endpoint is configured."""
    url = _read_bo_mcp_url()
    if not url:
        logger.info("BO_MCP_URL is unset; interactive BO-MCP tools are disabled")
        return None

    return OptionalBoMcpToolset(url)


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
