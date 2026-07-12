"""Behavioral tests for the specialist's optional BO-MCP transport."""

import asyncio
import os
from unittest.mock import patch

from bo_mcp.tools import BO_MCP_TOOLSET_ID
from prompts import BO_SPECIALIST_INSTRUCTIONS
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from specialist import build_bo_specialist_subagent


def test_specialist_run_survives_unreachable_mcp_server() -> None:
    """An unavailable optional SSE endpoint must not abort specialist startup."""
    with patch.dict(
        os.environ,
        {"BO_MCP_SSE_URL": "http://127.0.0.1:1/sse"},
        clear=True,
    ):
        specialist_config = build_bo_specialist_subagent()

    specialist = Agent(
        TestModel(call_tools=[], custom_output_text="ok"),
        instructions=specialist_config["instructions"],
        toolsets=specialist_config["toolsets"],
    )

    result = asyncio.run(specialist.run("hi"))

    assert result.output == "ok"


def test_optional_mcp_toolset_is_registered_on_specialist() -> None:
    """Interactive BO-MCP guidance and its optional toolset target the specialist."""
    with patch.dict(
        os.environ,
        {"BO_MCP_SSE_URL": "http://127.0.0.1:8001/sse"},
        clear=True,
    ):
        specialist_config = build_bo_specialist_subagent()

    assert "BO-MCP MCP tools" in BO_SPECIALIST_INSTRUCTIONS
    assert BO_MCP_TOOLSET_ID in {toolset.id for toolset in specialist_config["toolsets"]}
