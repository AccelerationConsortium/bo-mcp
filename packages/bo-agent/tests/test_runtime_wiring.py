"""Runtime import and assembled-agent wiring contract tests."""

import asyncio
import importlib
from collections.abc import Iterator
from types import ModuleType
from typing import Any, cast

import logfire
import pydantic_deep
import pytest
import specialist as specialist_module
import subagents_pydantic_ai.toolset as subagent_toolsets
from prompts import BO_SPECIALIST_INSTRUCTIONS
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel


@pytest.fixture(scope="module")
def compiled_subagents() -> dict[str, Any]:
    """Collect the real subagents compiled while assembling the main agent."""
    return {}


@pytest.fixture(scope="module")
def agent_module(compiled_subagents: dict[str, Any]) -> Iterator[ModuleType]:
    """Import the web entrypoint with a local model and no telemetry export."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.delenv("BO_MCP_SSE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_ADMIN_KEY", raising=False)
    configure = logfire.configure
    create_deep_agent = pydantic_deep.create_deep_agent
    compile_subagent = subagent_toolsets._compile_subagent
    to_web = Agent.to_web

    def configure_without_sending(*args: Any, **kwargs: Any) -> Any:
        kwargs["send_to_logfire"] = False
        return configure(*args, **kwargs)

    def create_deep_agent_with_test_model(
        model: Model | str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        del model
        test_model = TestModel(call_tools=[], custom_output_text="ok")
        return create_deep_agent(test_model, *args, **kwargs)

    def to_web_with_test_model(
        self: Agent[Any, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        kwargs["models"] = {"Test": TestModel(call_tools=[], custom_output_text="ok")}
        return to_web(self, *args, **kwargs)

    def capture_compiled_subagent(*args: Any, **kwargs: Any) -> Any:
        compiled = compile_subagent(*args, **kwargs)
        compiled_subagents[compiled.name] = compiled.agent
        return compiled

    monkeypatch.setattr(logfire, "configure", configure_without_sending)
    monkeypatch.setattr(pydantic_deep, "create_deep_agent", create_deep_agent_with_test_model)
    monkeypatch.setattr(specialist_module, "create_deep_agent", create_deep_agent_with_test_model)
    monkeypatch.setattr(subagent_toolsets, "_compile_subagent", capture_compiled_subagent)
    monkeypatch.setattr(Agent, "to_web", to_web_with_test_model)
    try:
        yield importlib.import_module("agent")
    finally:
        monkeypatch.undo()


def _runtime_agent(agent_module: ModuleType) -> Any:
    """Return the assembled pydantic-ai agent from the imported entrypoint."""
    return agent_module.__dict__["agent"]


def _toolsets_by_id(agent_module: ModuleType) -> dict[str, Any]:
    """Index the assembled agent's named toolsets."""
    return _agent_toolsets_by_id(_runtime_agent(agent_module))


def _agent_toolsets_by_id(runtime_agent: Any) -> dict[str, Any]:
    """Index an assembled agent's named toolsets."""
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
def test_assembled_main_agent_runs_without_mcp_configuration(agent_module: ModuleType) -> None:
    """A trivial main-agent run succeeds without configuring the optional MCP service."""
    deps = agent_module.__dict__["deps"]
    result = asyncio.run(_runtime_agent(agent_module).run("hi", deps=deps))

    assert result.output == "ok"


@pytest.mark.smoke
def test_prompted_client_module_is_importable() -> None:
    """The canonical client module named in the specialist prompt resolves at runtime."""
    module_name = "bo_mcp.client"

    assert module_name in BO_SPECIALIST_INSTRUCTIONS
    assert importlib.import_module(module_name).__name__ == module_name


@pytest.mark.smoke
def test_prompted_memory_tools_are_registered_on_specialist(
    agent_module: ModuleType,
    compiled_subagents: dict[str, Any],
) -> None:
    """Memory operations promised by the specialist prompt exist on that specialist."""
    del agent_module  # importing the entrypoint populates compiled_subagents
    assert "bo-specialist" in compiled_subagents
    specialist = compiled_subagents["bo-specialist"]
    memory_tools = cast(
        dict[str, Any],
        _agent_toolsets_by_id(specialist)["deep-memory"].tools,
    )

    for tool_name in ("write_memory", "update_memory"):
        assert tool_name in BO_SPECIALIST_INSTRUCTIONS
        assert tool_name in memory_tools
