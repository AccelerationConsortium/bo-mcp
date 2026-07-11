"""Configuration contract tests for BO-MCP service endpoints."""

import os
from unittest.mock import patch

import pytest
from bo_mcp.openapi import _default_openapi_url
from bo_mcp.tools import BO_MCP_TOOLSET_ID, _bo_mcp_sse_url, register_bo_mcp_tools
from pydantic_ai import Agent


def test_explicit_openapi_url_takes_precedence() -> None:
    environment = {
        "BO_MCP_OPENAPI_URL": "https://schema.example/openapi.json",
        "BO_MCP_API_URL": "https://api.example",
    }
    with patch.dict(os.environ, environment, clear=True):
        assert _default_openapi_url() == "https://schema.example/openapi.json"


def test_openapi_url_is_derived_from_api_url() -> None:
    with patch.dict(os.environ, {"BO_MCP_API_URL": "http://127.0.0.1:8000/"}, clear=True):
        assert _default_openapi_url() == "http://127.0.0.1:8000/openapi.json"


def test_legacy_rest_url_does_not_mask_missing_configuration() -> None:
    with (
        patch.dict(os.environ, {"BO_REST_URL": "http://legacy.example"}, clear=True),
        pytest.raises(ValueError, match="BO_MCP_OPENAPI_URL or BO_MCP_API_URL"),
    ):
        _default_openapi_url()


def test_sse_url_comes_from_environment() -> None:
    with patch.dict(
        os.environ,
        {"BO_MCP_SSE_URL": "http://127.0.0.1:8001/sse"},
        clear=True,
    ):
        assert _bo_mcp_sse_url() == "http://127.0.0.1:8001/sse"


def test_missing_sse_url_fails_fast() -> None:
    with (
        patch.dict(os.environ, {}, clear=True),
        pytest.raises(ValueError, match="BO_MCP_SSE_URL is not set"),
    ):
        _bo_mcp_sse_url()


def test_register_bo_mcp_tools_is_lazy_and_idempotent() -> None:
    test_agent = Agent()

    register_bo_mcp_tools(test_agent)
    register_bo_mcp_tools(test_agent)

    toolset_ids = [toolset.id for toolset in test_agent.toolsets]
    assert toolset_ids.count(BO_MCP_TOOLSET_ID) == 1
