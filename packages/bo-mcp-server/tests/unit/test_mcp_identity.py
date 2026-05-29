"""MCP mutating tools resolve user identity internally."""

from __future__ import annotations

from uuid import uuid4

import pytest

from bo_mcp_server.client import (
    DEV_USER_EMAIL,
    AuthenticationConfigurationError,
    ensure_mcp_startup_user,
    get_campaign_with_spec,
    list_campaign_results,
)
from bo_mcp_server.server import create_mcp_server
from bo_mcp_server.tools.create_campaign import _create_campaign_tool
from bo_mcp_server.tools.submit_results import _submit_results_tool
from bo_mcp_server.tools.upload_results_file import _upload_results_file_tool

pytestmark = pytest.mark.usefixtures("setup_database")


def _toy_intake() -> dict:
    return {
        "name": "MCP identity smoke",
        "description": "Minimal campaign for MCP identity tests",
        "parameters": [
            {
                "name": "x",
                "type": "continuous",
                "bounds": {"lower": 0.0, "upper": 1.0},
            }
        ],
        "objectives": [{"name": "score", "direction": "minimize"}],
        "batch_size": 1,
        "max_observations": 1,
    }


def _tool_parameters(name: str) -> dict:
    server = create_mcp_server()
    tools = {tool.name: tool for tool in server._tool_manager.list_tools()}
    return tools[name].parameters


def test_create_campaign_schema_hides_owner_id() -> None:
    params = _tool_parameters("bo_create_campaign")

    assert "owner_id" not in params["properties"]
    assert "owner_id" not in params.get("required", [])


def test_submit_results_schema_hides_submitted_by() -> None:
    params = _tool_parameters("bo_submit_results")

    assert "submitted_by" not in params["properties"]
    assert "submitted_by" not in params.get("required", [])


def test_upload_results_file_schema_hides_submitted_by() -> None:
    params = _tool_parameters("bo_upload_results_file")

    assert "submitted_by" not in params["properties"]
    assert "submitted_by" not in params.get("required", [])


@pytest.mark.asyncio
async def test_mcp_dev_auth_creates_campaign_and_results_with_dev_user(monkeypatch) -> None:
    monkeypatch.setenv("DEV_AUTH", "1")
    monkeypatch.setenv("API_ENV", "development")

    dev_user = await ensure_mcp_startup_user()
    assert dev_user is not None
    assert dev_user.email == DEV_USER_EMAIL

    created = await _create_campaign_tool(_toy_intake(), verbosity="minimal")

    assert created["success"] is True
    campaign_id = created["campaign_id"]
    campaign, _spec = await get_campaign_with_spec(campaign_id, dev_user.id)
    assert campaign.owner_id == dev_user.id

    submitted = await _submit_results_tool(
        campaign_id=campaign_id,
        results=[
            {
                "parameter_values": {"x": 0.0},
                "objective_values": {"score": 1.0},
            }
        ],
        verbosity="minimal",
    )

    assert submitted["success"] is True
    results = await list_campaign_results(campaign_id, dev_user.id)
    assert len(results) == 1
    assert results[0].submitted_by == dev_user.id


@pytest.mark.asyncio
async def test_mcp_dev_auth_uploads_results_with_dev_user(monkeypatch) -> None:
    monkeypatch.setenv("DEV_AUTH", "1")
    monkeypatch.setenv("API_ENV", "development")

    dev_user = await ensure_mcp_startup_user()
    assert dev_user is not None

    created = await _create_campaign_tool(_toy_intake(), verbosity="minimal")
    assert created["success"] is True
    campaign_id = created["campaign_id"]

    uploaded = await _upload_results_file_tool(
        campaign_id=campaign_id,
        file_content="param_x,obj_score\n0.0,1.0\n",
    )

    assert uploaded["success"] is True
    assert uploaded["results_created"] == 1
    results = await list_campaign_results(campaign_id, dev_user.id)
    assert len(results) == 1
    assert results[0].submitted_by == dev_user.id


@pytest.mark.asyncio
async def test_create_campaign_without_dev_auth_returns_missing_identity(monkeypatch) -> None:
    """Without DEV_AUTH the MCP create tool returns a structured missing-identity error.

    This is the configuration the direct-MCP eval runs by default (no DEV_AUTH
    set). It must surface a clean E199 envelope flagged ``missing_identity``
    rather than crash or silently invent an owner — see ``resolve_mcp_user``.
    """
    monkeypatch.setenv("DEV_AUTH", "0")
    monkeypatch.setenv("API_ENV", "development")

    result = await _create_campaign_tool(_toy_intake(), verbosity="minimal")

    assert result["success"] is False
    assert result["error"]["code"] == "E199"
    assert result["error"]["details"]["missing_identity"] is True
    assert result["error"]["details"]["transport"] == "mcp"


@pytest.mark.asyncio
async def test_submit_results_without_dev_auth_returns_missing_identity(monkeypatch) -> None:
    """Without DEV_AUTH the MCP submit tool returns a structured missing-identity error."""
    monkeypatch.setenv("DEV_AUTH", "0")
    monkeypatch.setenv("API_ENV", "development")

    result = await _submit_results_tool(
        campaign_id=str(uuid4()),
        results=[{"parameter_values": {"x": 0.0}, "objective_values": {"score": 1.0}}],
        verbosity="minimal",
    )

    assert result["success"] is False
    assert result["error"]["code"] == "E199"
    assert result["error"]["details"]["missing_identity"] is True


@pytest.mark.asyncio
async def test_upload_results_file_without_dev_auth_returns_missing_identity(monkeypatch) -> None:
    """Without DEV_AUTH the MCP upload tool returns a structured missing-identity error."""
    monkeypatch.setenv("DEV_AUTH", "0")
    monkeypatch.setenv("API_ENV", "development")

    result = await _upload_results_file_tool(
        campaign_id=str(uuid4()),
        file_content="param_x,obj_score\n0.0,1.0\n",
    )

    assert result["success"] is False
    assert result["error"]["code"] == "E199"
    assert result["error"]["details"]["missing_identity"] is True


@pytest.mark.asyncio
async def test_upload_malformed_file_reports_validation_before_identity(monkeypatch) -> None:
    """A malformed upload yields the payload-validation error, not the identity error.

    The upload tool validates the file payload before resolving identity, so an
    agent gets an actionable field error even when identity is unconfigured —
    matching bo_create_campaign / bo_submit_results. Here DEV_AUTH is unset yet
    the unsupported-format envelope (E005) wins over the missing-identity E199.
    """
    monkeypatch.setenv("DEV_AUTH", "0")
    monkeypatch.setenv("API_ENV", "development")

    result = await _upload_results_file_tool(
        campaign_id=str(uuid4()),
        file_content="",
        file_format="xml",
    )

    assert result["success"] is False
    assert result["error"]["code"] == "E005"
    assert result["error"].get("details", {}).get("missing_identity") is None


@pytest.mark.asyncio
async def test_dev_auth_refused_in_production(monkeypatch) -> None:
    """DEV_AUTH=1 with API_ENV=production must refuse rather than bootstrap the shared user.

    The shared dev user's API key is checked into source, so allowing the
    bypass in production would publish a master credential. Startup calls
    ``ensure_mcp_startup_user`` (see ``cli.main_async``), which must raise — and
    thereby crash the server at boot — instead of seeding the user. Mirrors the
    API package's ``test_create_app_refuses_dev_auth_in_production``.
    """
    monkeypatch.setenv("DEV_AUTH", "1")
    monkeypatch.setenv("API_ENV", "production")

    with pytest.raises(AuthenticationConfigurationError, match="DEV_AUTH=1 is not allowed"):
        await ensure_mcp_startup_user()


@pytest.mark.asyncio
async def test_dev_auth_allowed_outside_production(monkeypatch) -> None:
    """The same DEV_AUTH=1 bootstrap is permitted when API_ENV is not production."""
    monkeypatch.setenv("DEV_AUTH", "1")
    monkeypatch.setenv("API_ENV", "staging")

    user = await ensure_mcp_startup_user()

    assert user is not None
    assert user.email == DEV_USER_EMAIL
