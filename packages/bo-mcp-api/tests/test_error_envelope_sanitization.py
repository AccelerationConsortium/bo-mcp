"""Global error sanitization for the REST transport (TODO 8.18).

The audit found two leak surfaces:

* The CSV/XLSX upload route returned ``Failed to parse file: {e}``,
  where ``{e}`` was a pandas/openpyxl exception string that included
  library name, file path, and dependency version.
* No global exception handler normalised unhandled exceptions, so a
  surprise traceback was the response body whenever a route raised
  outside the curated try-blocks.

These tests pin the fix end-to-end:

* The upload route emits a sanitized 400 when the body is not a
  parseable CSV/Excel file; the response body carries no library
  names or stack frames.
* The global ``Exception`` handler converts an unhandled error into
  the structured envelope :func:`make_error_response` produces, with
  the request id surfaced under ``error.details.request_id``.

Reference: FastAPI exception-handlers documentation
https://fastapi.tiangolo.com/tutorial/handling-errors/.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from bo_mcp_server.tools.create_campaign import create_campaign
from fastapi import APIRouter
from httpx import ASGITransport, AsyncClient

from api.main import create_app


async def _create_campaign_for_owner(owner_id: str) -> str:
    result = await create_campaign(
        {
            "name": "Error sanitization",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        },
        owner_id,
    )
    return result["campaign_id"]


def _contains_library_fingerprint(payload: str) -> bool:
    """Heuristic: does ``payload`` look like a leaked library/path/trace?"""
    suspicious = [
        "openpyxl",
        "pandas",
        "Traceback",
        "/site-packages/",
        '.py"',
        "line ",
        "tokenizer",
    ]
    lowered = payload.lower()
    return any(token.lower() in lowered for token in suspicious)


class TestUploadParseErrorSanitization:
    """A malformed file produces a sanitized 400 envelope."""

    @pytest.mark.asyncio
    async def test_corrupt_excel_returns_400_without_library_names(
        self, api_client, auth_headers, persisted_user
    ) -> None:
        campaign_id = await _create_campaign_for_owner(str(persisted_user.id))

        response = await api_client.post(
            f"/api/results/{campaign_id}/upload",
            files={"file": ("broken.xlsx", b"not actually an xlsx file", "application/x-xls")},
            headers=auth_headers,
        )

        assert response.status_code == 400, response.text
        body_text = response.text
        assert not _contains_library_fingerprint(body_text), body_text
        assert response.headers.get("X-Request-ID")


class _ExplodingError(RuntimeError):
    """Synthetic error used to force the global handler path."""


@pytest_asyncio.fixture
async def exploding_api_client(request: pytest.FixtureRequest) -> AsyncGenerator[AsyncClient]:
    """API client whose ``/explode`` route always raises.

    The fixture mounts a fresh app with an extra router so the
    global :class:`Exception` handler is exercised end-to-end without
    needing to corrupt a real route in production code.
    """
    request.getfixturevalue("setup_database")
    app = create_app()
    router = APIRouter()

    @router.get("/explode")
    async def explode() -> dict:
        msg = "leak-me: /Users/secret/path/to/file.py line 42 openpyxl=3.1.2"
        raise _ExplodingError(msg)

    app.include_router(router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client


class TestGlobalExceptionHandlerSanitization:
    """An unhandled exception is normalised to the structured envelope."""

    @pytest.mark.asyncio
    async def test_unhandled_exception_returns_sanitized_envelope(
        self, exploding_api_client, auth_headers, caplog
    ) -> None:
        with caplog.at_level(logging.ERROR):
            response = await exploding_api_client.get(
                "/explode",
                headers={**auth_headers, "X-Request-ID": "explode-trace-1"},
            )

        assert response.status_code == 500, response.text
        body = response.json()
        assert body["success"] is False
        assert body["error"]["code"] == "E199"
        # The response carries the request id under the structured
        # ``details`` envelope so operators can correlate it to logs.
        assert body["error"]["details"]["request_id"] == "explode-trace-1"
        # The response header echoes the request id too so retry
        # middleware can pivot without parsing the body.
        assert response.headers["X-Request-ID"] == "explode-trace-1"

        # Body must not contain the exception's message or any of the
        # library / path fingerprints we logged it with.
        body_text = response.text
        assert "leak-me" not in body_text
        assert not _contains_library_fingerprint(body_text), body_text

        # Server-side log must include the exception for triage.
        leaked_to_log = [r for r in caplog.records if "leak-me" in r.getMessage() or r.exc_info]
        assert leaked_to_log, "global handler must log the full exception"
