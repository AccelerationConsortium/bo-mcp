"""End-to-end mapping of ``CorruptedJsonColumnError`` to the client envelope.

The unit tests in
``tests/unit/test_storage/test_strict_json_column_decode.py`` only
assert that the ORM-level property fails loud — they do not exercise
the transport boundary. The audit's promise was a structured envelope
on the response, so this module corrupts a column via raw SQL and then
asserts the envelope shape via both transport-relevant code paths the
boundary catches sit on:

1. **MCP tool boundary** (``tool_boundary.install_validation_envelope_wrapper``).
   The reviewer's finding called out that the corruption used to
   propagate as a raw ``RuntimeError`` because the wrapper only
   handled ``ToolError`` from Pydantic. We now exercise the wrapper
   directly with a tool whose body reads a corrupted spec and assert
   the envelope.

2. **REST exception handler** is covered by a focused unit test in
   ``packages/bo-mcp-api/tests/test_corrupted_json_envelope.py``; this
   module covers the server-side mapping so the contract is locked
   regardless of which transport is in play.

Both paths route through the shared
``bo_mcp_server.errors.make_corrupted_json_response`` so the envelope
keys (``error.code == DATA_INTEGRITY_ERROR``, ``retryable=False``,
sanitized ``details.column``) stay consistent. The code is deliberately
distinct from the broader ``DATABASE_ERROR`` because the failure is
deterministic — see the docstring on
``make_corrupted_json_response`` for the retryability rationale.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text

from bo_mcp_server.domain import (
    CampaignSpec,
    InputParameter,
    Objective,
    ParameterType,
    User,
)
from bo_mcp_server.domain.campaign import Campaign, CampaignStatus
from bo_mcp_server.errors import ErrorCode
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    CorruptedJsonColumnError,
    UserRepository,
    close_database,
    get_session,
    init_database,
)

pytestmark = pytest.mark.usefixtures("fresh_database")


@pytest_asyncio.fixture
async def fresh_database() -> AsyncGenerator[None]:
    """Reset the engine + session factory between tests."""
    await close_database()
    await init_database()
    try:
        yield
    finally:
        await close_database()


async def _seed_campaign() -> dict[str, str]:
    async with get_session() as session:
        unique = str(uuid4())
        user = User(
            name="Owner",
            email=f"owner-{unique}@example.com",
            api_key_hash=hashlib.sha256(unique.encode()).hexdigest(),
        )
        user = await UserRepository(session).save(user)
        spec = CampaignSpec(
            name="Boundary Test",
            parameters=(
                InputParameter(
                    name="x",
                    type=ParameterType.CONTINUOUS,
                    bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
                ),
            ),
            objectives=(Objective(name="y", direction="maximize"),),
        )
        spec_id = uuid4()
        await CampaignSpecRepository(session).save(spec, spec_id=spec_id)
        campaign = Campaign(spec_id=spec_id, owner_id=user.id, status=CampaignStatus.RUNNING)
        await CampaignRepository(session).save(campaign)
        await session.commit()
        return {"spec_id": str(spec_id), "campaign_id": str(campaign.id)}


async def _corrupt_spec_parameters(spec_id: str) -> None:
    """Overwrite ``campaign_specs.parameters_json`` with non-JSON via raw SQL."""
    stmt = text("UPDATE campaign_specs SET parameters_json = :p WHERE id = :i")
    async with get_session() as session:
        await session.execute(stmt, {"p": "not-json", "i": spec_id})
        await session.commit()


@pytest.mark.asyncio
async def test_mcp_tool_boundary_returns_database_error_envelope() -> None:
    """A tool whose body trips ``CorruptedJsonColumnError`` returns the envelope.

    The wrapper installed by
    ``install_validation_envelope_wrapper`` catches the typed
    exception and converts it to the structured envelope. Without the
    catch the exception would propagate to FastMCP and surface as a
    raw ``RuntimeError`` text instead of the canonical
    ``DATA_INTEGRITY_ERROR`` shape (a distinct code from
    ``DATABASE_ERROR`` because data corruption is deterministically
    non-retryable — see ``make_corrupted_json_response``).
    """
    from mcp.server.fastmcp import FastMCP

    from bo_mcp_server.tool_boundary import install_validation_envelope_wrapper

    ids = await _seed_campaign()
    await _corrupt_spec_parameters(ids["spec_id"])

    mcp = FastMCP(name="boundary-test")

    @mcp.tool()
    async def _read_spec_parameters(spec_id: str) -> dict[str, Any]:
        """Read ``parsed_parameters`` to trip the strict decoder."""
        async with get_session() as session:
            repo = CampaignSpecRepository(session)
            spec = await repo.get(uuid4().__class__(spec_id))
        if spec is None:
            return {"success": False, "error": "spec_missing"}
        return {"success": True, "n_parameters": len(spec.parameters)}

    install_validation_envelope_wrapper(mcp)

    response = await mcp._tool_manager.call_tool(
        "_read_spec_parameters",
        {"spec_id": ids["spec_id"]},
    )
    # ``call_tool`` returns the structured envelope our wrapper produced.
    envelope = response if isinstance(response, dict) else json.loads(response)
    assert envelope["success"] is False
    assert envelope["error"]["code"] == ErrorCode.DATA_INTEGRITY_ERROR.value
    # Reviewer's correctness finding: corrupted data is deterministic,
    # so the envelope must NOT advertise retryability — a client retry
    # loop would only re-trigger the same broken row.
    assert envelope["error"]["retryable"] is False
    assert envelope["error"]["retry_after"] is None
    # The sanitized envelope identifies the offending column but never
    # leaks the raw payload bytes.
    assert "parameters" in envelope["error"]["details"]["column"]
    assert "raw" not in envelope["error"]["details"]
    assert "not-json" not in json.dumps(envelope)


@pytest.mark.asyncio
async def test_make_corrupted_json_response_uses_data_integrity_code() -> None:
    """The shared mapper produces the canonical ``DATA_INTEGRITY_ERROR`` envelope.

    Locks the surface that both the MCP wrapper and the REST handler
    consume. A drift in code or in retryability propagates to every
    consumer; pinning it here avoids per-transport drift tests.

    Reviewer follow-up: an earlier version of this test asserted
    ``retryable is True`` because the response reused
    ``ErrorCode.DATABASE_ERROR``. That code lives in the retryable
    family (transient connection blips, deadlocks), but corrupted JSON
    is deterministic — retrying the same request just re-trips the
    same broken row. The mapper now uses the dedicated
    ``DATA_INTEGRITY_ERROR`` code whose retry hint is
    ``(retryable=False, retry_after=None)`` so clients (especially LLM
    agents) do not enter a retry storm against unrecoverable state.
    """
    from bo_mcp_server.errors import make_corrupted_json_response

    ids = await _seed_campaign()
    await _corrupt_spec_parameters(ids["spec_id"])

    # The repository's ``_to_entity`` walks ``parsed_parameters`` while
    # materialising the domain object, so the strict decoder trips
    # inside ``get()`` itself — not on a later attribute access.
    captured: CorruptedJsonColumnError | None = None
    async with get_session() as session:
        try:
            await CampaignSpecRepository(session).get(uuid4().__class__(ids["spec_id"]))
        except CorruptedJsonColumnError as exc:
            captured = exc

    assert captured is not None, "Expected the strict decoder to raise"
    envelope = make_corrupted_json_response(captured)
    assert envelope["success"] is False
    assert envelope["error"]["code"] == ErrorCode.DATA_INTEGRITY_ERROR.value
    assert envelope["error"]["details"]["column"] == captured.context
    # Retryability comes from the central ``ERROR_CODE_RETRY_HINTS``
    # table; ``DATA_INTEGRITY_ERROR`` is the non-retryable family.
    assert envelope["error"]["retryable"] is False
    assert envelope["error"]["retry_after"] is None
    # The error message itself steers the caller away from retrying.
    assert "Do not retry" in envelope["error"]["message"]
