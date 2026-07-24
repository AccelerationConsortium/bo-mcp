"""Configuration contract tests for BO-MCP service endpoints."""

import logging
import os
from unittest.mock import patch

import pytest
from bo_mcp.openapi import _default_openapi_url
from bo_mcp.tools import BO_MCP_TOOLSET_ID, _bo_mcp_url, build_optional_bo_mcp_toolset


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


def test_mcp_url_comes_from_environment() -> None:
    with patch.dict(
        os.environ,
        {"BO_MCP_URL": "http://127.0.0.1:8001/mcp"},
        clear=True,
    ):
        assert _bo_mcp_url() == "http://127.0.0.1:8001/mcp"


def test_missing_mcp_url_fails_fast() -> None:
    with (
        patch.dict(os.environ, {}, clear=True),
        pytest.raises(ValueError, match="BO_MCP_URL is not set"),
    ):
        _bo_mcp_url()


def test_optional_mcp_toolset_is_disabled_without_mcp_url() -> None:
    with patch.dict(os.environ, {}, clear=True):
        assert build_optional_bo_mcp_toolset() is None


def test_optional_mcp_toolset_uses_configured_mcp_url() -> None:
    with patch.dict(os.environ, {"BO_MCP_URL": "http://127.0.0.1:8001/mcp"}, clear=True):
        toolset = build_optional_bo_mcp_toolset()

    assert toolset is not None
    assert toolset.id == BO_MCP_TOOLSET_ID


def test_legacy_sse_url_is_ignored_with_migration_hint(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A leftover ``BO_MCP_SSE_URL`` is ignored, with a migration warning.

    The HTTP+SSE transport was deprecated by MCP protocol revision
    2025-03-26 in favor of streamable-http, and the server no longer
    serves ``/sse`` — silently honoring the legacy URL would fail
    obscurely at connect time instead of pointing at the rename.
    """
    with (
        patch.dict(os.environ, {"BO_MCP_SSE_URL": "http://127.0.0.1:8001/sse"}, clear=True),
        caplog.at_level(logging.WARNING),
    ):
        assert build_optional_bo_mcp_toolset() is None

    assert any("BO_MCP_SSE_URL" in record.message for record in caplog.records)
