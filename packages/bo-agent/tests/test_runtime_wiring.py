"""Runtime import and assembled-agent wiring contract tests."""

import importlib
from collections.abc import Iterator
from types import ModuleType
from typing import Any, cast

import logfire
import pytest
from bo_mcp.tools import BO_MCP_TOOLSET_ID
from prompts import BO_SPECIALIST_INSTRUCTIONS


@pytest.fixture(scope="module")
def agent_module() -> Iterator[ModuleType]:
    """Import the web entrypoint with real instrumentation but no telemetry export."""
    monkeypatch = pytest.MonkeyPatch()
    configure = logfire.configure

    def configure_without_sending(*args: Any, **kwargs: Any) -> Any:
        kwargs["send_to_logfire"] = False
        return configure(*args, **kwargs)

    monkeypatch.setattr(logfire, "configure", configure_without_sending)
    try:
        yield importlib.import_module("agent")
    finally:
        monkeypatch.undo()


def _runtime_agent(agent_module: ModuleType) -> Any:
    """Return the assembled pydantic-ai agent from the imported entrypoint."""
    return agent_module.__dict__["agent"]


def _toolsets_by_id(agent_module: ModuleType) -> dict[str, Any]:
    """Index the assembled agent's named toolsets."""
    runtime_agent = _runtime_agent(agent_module)
    return {
        toolset_id: toolset
        for toolset in runtime_agent.toolsets
        if (toolset_id := getattr(toolset, "id", None)) is not None
    }


@pytest.mark.smoke
def test_runtime_modules_import_in_standalone_environment(agent_module: ModuleType) -> None:
    """All directly deployed modules import using bo-agent's declared dependencies."""
    assert importlib.import_module("agent") is agent_module
    for module_name in (
        "bo_mcp.client",
        "bo_mcp.openapi",
        "bo_mcp.tools",
        "prompts",
        "specialist",
    ):
        assert importlib.import_module(module_name).__name__ == module_name


def test_main_agent_uses_builtin_execute_and_filesystem_tools(agent_module: ModuleType) -> None:
    """The main agent exposes pydantic-deep's console instead of a custom bash tool."""
    toolsets = _toolsets_by_id(agent_module)
    console_tools = cast(dict[str, Any], toolsets["deep-console"].tools)

    assert {"execute", "read_file", "ls"} <= console_tools.keys()

    static_tool_names = {
        tool_name
        for toolset in _runtime_agent(agent_module).toolsets
        for tool_name in cast(dict[str, Any], getattr(toolset, "tools", {}))
    }
    assert "bash" not in static_tool_names


@pytest.mark.smoke
def test_prompted_client_module_is_importable() -> None:
    """The canonical client module named in the specialist prompt resolves at runtime."""
    module_name = "bo_mcp.client"

    assert module_name in BO_SPECIALIST_INSTRUCTIONS
    assert importlib.import_module(module_name).__name__ == module_name


@pytest.mark.smoke
def test_prompted_memory_tools_are_registered(agent_module: ModuleType) -> None:
    """Memory operations promised by the specialist prompt exist on the main agent."""
    memory_tools = cast(dict[str, Any], _toolsets_by_id(agent_module)["deep-memory"].tools)

    for tool_name in ("write_memory", "update_memory"):
        assert tool_name in BO_SPECIALIST_INSTRUCTIONS
        assert tool_name in memory_tools


@pytest.mark.smoke
def test_bo_mcp_toolset_is_registered(agent_module: ModuleType) -> None:
    """The assembled main agent includes the BO-MCP toolset promised by its wiring."""
    assert "BO-MCP MCP tools" in BO_SPECIALIST_INSTRUCTIONS
    assert BO_MCP_TOOLSET_ID in _toolsets_by_id(agent_module)
