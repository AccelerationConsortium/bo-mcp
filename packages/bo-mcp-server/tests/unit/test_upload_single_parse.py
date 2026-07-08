"""The upload tool parses its CSV exactly once per call.

The MCP tool validates the payload *before* resolving identity (so a
malformed file gets an actionable error even when identity is
unconfigured) and previously re-parsed the same content inside the
inner pipeline — up to seconds of doubled event-loop blocking at the
10 MB cap. The pre-identity parse result is now threaded into the
pipeline, and the parse itself runs in a worker thread.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from bo_mcp_server.domain import User
from bo_mcp_server.result_upload_parser import parse_prefixed_result_rows
from bo_mcp_server.tools.upload_results_file import (
    _upload_results_file_tool,
    upload_results_file,
)

pytestmark = pytest.mark.usefixtures("setup_database")

CSV_CONTENT = "param_x,obj_score\n0.1,1.0\n0.2,2.0\n"


@pytest.fixture
def parse_counter(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count invocations of the row parser inside the upload module."""
    counter = [0]

    def _counting_parse(*args: Any, **kwargs: Any) -> Any:
        counter[0] += 1
        return parse_prefixed_result_rows(*args, **kwargs)

    monkeypatch.setattr(
        "bo_mcp_server.tools.upload_results_file.parse_prefixed_result_rows",
        _counting_parse,
    )
    return counter


@pytest.fixture
def fake_identity(monkeypatch: pytest.MonkeyPatch) -> User:
    user = User(name="Uploader", email="uploader@example.com", api_key_hash="x" * 64)

    async def _resolve() -> User:
        return user

    monkeypatch.setattr("bo_mcp_server.tools.upload_results_file.resolve_mcp_user", _resolve)
    return user


@pytest.mark.asyncio
@pytest.mark.usefixtures("fake_identity")
async def test_mcp_tool_parses_csv_once(parse_counter: list[int]) -> None:
    """One parse covers both the pre-identity check and the pipeline."""
    result = await _upload_results_file_tool(
        campaign_id=str(uuid4()),
        file_content=CSV_CONTENT,
    )

    # Unknown campaign: the submit stage fails, but the payload was
    # parsed exactly once on the way there.
    assert result["success"] is False
    assert parse_counter[0] == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures("fake_identity")
async def test_mcp_tool_dry_run_parses_csv_once(parse_counter: list[int]) -> None:
    result = await _upload_results_file_tool(
        campaign_id=str(uuid4()),
        file_content=CSV_CONTENT,
        dry_run=True,
    )

    assert result["success"] is False
    assert parse_counter[0] == 1


@pytest.mark.asyncio
async def test_compat_helper_parses_csv_once(parse_counter: list[int]) -> None:
    result = await upload_results_file(
        campaign_id=str(uuid4()),
        file_content=CSV_CONTENT,
        submitted_by=str(uuid4()),
    )

    assert result["success"] is False
    assert parse_counter[0] == 1
