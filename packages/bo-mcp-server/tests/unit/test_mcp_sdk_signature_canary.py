"""Canary for the private FastMCP internals monkeypatched at the tool/resource boundary.

``tool_boundary.py`` and ``resource_boundary.py`` replace three private
SDK methods (``ToolManager.call_tool``, ``FastMCP.read_resource``,
``ResourceManager.get_resource``) to get structured error envelopes —
behavior the SDK does not expose a public hook for. Those methods are
not a public contract, so a routine ``mcp`` version bump could change
their signature and silently revert agents to opaque ``ToolError``
text / double-wrapped resource errors (the exact failure mode this
code exists to prevent).

This test pins the parameter names the wrappers rely on so a
``uv lock`` bump that changes them fails loudly here instead of
surfacing as a subtle behavior regression in production.
"""

from __future__ import annotations

import inspect

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.resources.resource_manager import ResourceManager
from mcp.server.fastmcp.tools import ToolManager


def test_tool_manager_call_tool_signature_unchanged() -> None:
    params = list(inspect.signature(ToolManager.call_tool).parameters)
    assert params == ["self", "name", "arguments", "context", "convert_result"]


def test_fastmcp_read_resource_signature_unchanged() -> None:
    params = list(inspect.signature(FastMCP.read_resource).parameters)
    assert params == ["self", "uri"]


def test_resource_manager_get_resource_signature_unchanged() -> None:
    params = list(inspect.signature(ResourceManager.get_resource).parameters)
    assert params == ["self", "uri", "context"]
