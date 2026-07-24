"""Every read operation must carry ``schema_version``, even on a not-found path.

``response_formatter.py`` documents "every MCP-tool / REST response
carries ``schema_version``", but seven read surfaces were missed when
``with_response_metadata``/``attach_response_metadata`` was applied
opportunistically elsewhere: agents/clients that dispatch on
``schema_version`` would hit a ``KeyError`` on exactly these calls.
This test calls each one against a campaign/suggestion id that does
not exist (or, for the two operations that need no id at all, with no
arguments) and asserts the envelope is present regardless of the
success/error outcome.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from bo_mcp_server.backend_context import (
    campaign_backend_var,
    get_campaign_backend,
    set_campaign_backend,
)
from bo_mcp_server.operations.export_campaign import export_campaign_operation
from bo_mcp_server.operations.list_campaigns import list_campaigns_operation
from bo_mcp_server.operations.list_capabilities import list_capabilities_operation
from bo_mcp_server.operations.list_results import list_results_operation
from bo_mcp_server.operations.list_suggestions import list_suggestions_operation
from bo_mcp_server.operations.suggestion_explanation import (
    get_suggestion_explanation_operation,
)
from bo_mcp_server.tools.health_check import health_check

pytestmark = pytest.mark.usefixtures("setup_database")


@pytest.mark.asyncio
async def test_list_campaigns_operation_carries_schema_version() -> None:
    response = await list_campaigns_operation(owner_id=uuid4())
    assert "schema_version" in response


@pytest.mark.asyncio
async def test_list_campaigns_metadata_survives_leaked_campaign_binding() -> None:
    """Cross-campaign list stamps the server default despite a same-task bind.

    In-process callers (client facade, scripts) can invoke operations
    without the transports' ``campaign_backend_scope``; a binding left by
    e.g. ``create_campaign`` must not mislabel ``bo_list_campaigns``'
    envelope as campaign-scoped (issue #82 follow-up review).
    """
    set_campaign_backend("botorch")
    try:
        response = await list_campaigns_operation(owner_id=uuid4())
        assert response["_metadata"]["backend_source"] == "server_default"
        # The operation must restore the caller's binding on exit.
        assert get_campaign_backend() == "botorch"
    finally:
        campaign_backend_var.set(None)


def test_list_capabilities_metadata_survives_leaked_campaign_binding() -> None:
    """Capability listing is campaign-agnostic even after a same-task bind."""
    set_campaign_backend("botorch")
    try:
        response = list_capabilities_operation()
        assert response["_metadata"]["backend_source"] == "server_default"
        assert get_campaign_backend() == "botorch"
    finally:
        campaign_backend_var.set(None)


@pytest.mark.asyncio
async def test_health_check_metadata_survives_leaked_campaign_binding() -> None:
    """Health check is campaign-agnostic even after a same-task bind."""
    set_campaign_backend("botorch")
    try:
        response = await health_check()
        assert response["_metadata"]["backend_source"] == "server_default"
        assert get_campaign_backend() == "botorch"
    finally:
        campaign_backend_var.set(None)


@pytest.mark.asyncio
async def test_list_suggestions_operation_carries_schema_version_on_not_found() -> None:
    response = await list_suggestions_operation(campaign_id=str(uuid4()))
    assert "schema_version" in response


@pytest.mark.asyncio
async def test_list_results_operation_carries_schema_version_on_not_found() -> None:
    response = await list_results_operation(campaign_id=str(uuid4()))
    assert "schema_version" in response


@pytest.mark.asyncio
async def test_export_campaign_operation_carries_schema_version_on_not_found() -> None:
    response = await export_campaign_operation(campaign_id=str(uuid4()))
    assert "schema_version" in response


@pytest.mark.asyncio
async def test_get_suggestion_explanation_operation_carries_schema_version_on_not_found() -> None:
    response = await get_suggestion_explanation_operation(suggestion_id=str(uuid4()))
    assert "schema_version" in response


def test_list_capabilities_operation_carries_schema_version() -> None:
    response = list_capabilities_operation()
    assert "schema_version" in response


@pytest.mark.asyncio
async def test_health_check_tool_carries_schema_version() -> None:
    response = await health_check()
    assert "schema_version" in response
