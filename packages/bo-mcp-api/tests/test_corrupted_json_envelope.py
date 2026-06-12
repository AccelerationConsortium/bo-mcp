"""REST mapping of ``CorruptedJsonColumnError`` to the ``DATA_INTEGRITY_ERROR`` envelope.

The strict JSON decoder raises a typed exception
when a persisted JSON column cannot be decoded. The unit tests in
``packages/bo-mcp-server/tests/unit/test_storage/test_strict_json_column_decode.py``
pin the ORM-level behaviour; this module covers the **REST transport
boundary** so the audit's documented promise of a structured envelope
is verifiable end-to-end. Without the mapping the catch-all in
:mod:`api.error_handlers` used to convert the exception into a
generic ``INTERNAL_ERROR`` (``E199``) envelope, hiding the fact that
the failure is a *known* data-corruption signal operators can address.

The envelope uses the dedicated ``DATA_INTEGRITY_ERROR`` (``E108``)
code rather than the broader ``DATABASE_ERROR`` because corrupted
JSON is **deterministic** — retrying the same request re-trips the
same row and is wasted effort. See the docstring on
``make_corrupted_json_response`` for the retryability rationale.

The route fixture mounts a synthetic ``/raise-corrupted-json`` endpoint
that raises the typed exception directly — corrupting a real spec row
mid-request would require fixture infrastructure shared with the
MCP-side test, and a synthetic route exercises the exact same
exception-handling code path with less setup.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from fastapi import APIRouter
from httpx import ASGITransport, AsyncClient

from api.main import create_app
from bo_mcp_server.storage.models import CorruptedJsonColumnError


@pytest_asyncio.fixture
async def corruption_api_client(request: pytest.FixtureRequest) -> AsyncGenerator[AsyncClient]:
    """API client whose ``/raise-corrupted-json`` route trips the typed error.

    The endpoint raises a synthetic :class:`CorruptedJsonColumnError`
    so we exercise the boundary without depending on the storage
    layer to actually contain a malformed row. The real ORM-level
    decode path is covered in
    ``packages/bo-mcp-server/tests/unit/test_storage/test_strict_json_column_decode.py``.
    """
    request.getfixturevalue("setup_database")
    app = create_app()
    router = APIRouter()

    @router.get("/raise-corrupted-json")
    async def _raise() -> dict:
        msg = "CampaignSpec(test-spec-id).parameters"
        raise CorruptedJsonColumnError(
            msg,
            "not-json-bytes-with-secret-content",
            ValueError("Expecting value"),
        )

    app.include_router(router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client


@pytest.mark.asyncio
async def test_corrupted_json_returns_data_integrity_envelope(
    corruption_api_client,
) -> None:
    """REST route raising ``CorruptedJsonColumnError`` yields 500 + ``DATA_INTEGRITY_ERROR``.

    Reproducer for the reviewer's finding chain: the original audit
    asked for a structured envelope at the REST boundary; the
    follow-up review noted that reusing ``DATABASE_ERROR`` mis-tagged
    the failure as retryable. We now emit the dedicated
    ``DATA_INTEGRITY_ERROR`` (``E108``) code so clients can branch on
    a deterministic-failure signal without entering a retry storm.
    """
    response = await corruption_api_client.get(
        "/raise-corrupted-json",
        headers={"X-Request-ID": "corrupt-trace-1"},
    )
    assert response.status_code == 500, response.text
    body = response.json()
    assert body["success"] is False
    # The dedicated DATA_INTEGRITY_ERROR code (E108) — distinct from
    # both the transient DATABASE_ERROR (E103) and the catch-all
    # INTERNAL_ERROR (E199).
    assert body["error"]["code"] == "E108"
    # Deterministic failure: the envelope must NOT advertise retry.
    assert body["error"]["retryable"] is False
    assert body["error"]["retry_after"] is None
    assert body["error"]["details"]["column"] == "CampaignSpec(test-spec-id).parameters"
    # Header pivot back to the server-side log.
    assert response.headers["X-Request-ID"] == "corrupt-trace-1"
    # Request id is surfaced under details so operators correlate to logs.
    assert body["error"]["details"]["request_id"] == "corrupt-trace-1"


@pytest.mark.asyncio
async def test_corrupted_json_envelope_does_not_leak_raw_bytes(
    corruption_api_client,
) -> None:
    """Sanitization: the raw column contents never appear in the response body.

    Corrupted columns can carry user-submitted chemistry / formulation IP;
    surfacing the first 200 chars to a client would be the wrong default
    even for a 500 response. The full excerpt stays in the server log.
    """
    response = await corruption_api_client.get(
        "/raise-corrupted-json",
        headers={"X-Request-ID": "corrupt-trace-2"},
    )
    body_text = response.text
    assert "not-json-bytes-with-secret-content" not in body_text
    # The exception message reference is also kept off the wire.
    assert "Expecting value" not in body_text
