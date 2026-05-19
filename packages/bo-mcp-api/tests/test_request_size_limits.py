"""Request-size bounds enforced at the REST transport boundary.

Each test pins one of the three caps the audit called out:

* Schema-level ``Field(max_length=...)`` on intake and batch
  collections so an oversize payload is rejected by Pydantic before
  reaching a route handler.
* The global JSON-body middleware that rejects payloads whose
  advertised ``Content-Length`` exceeds the configured cap.
* The per-route streaming reader on the file-upload endpoint.

Reference: RFC 9110 §15.5.14 (413 Content Too Large).
"""

from __future__ import annotations

import io
from typing import Any

import pytest
from pydantic import ValidationError

from api.limits import (
    MAX_BATCH_CAMPAIGN_IDS,
    MAX_BATCH_RESULTS,
    MAX_COMPARE_CAMPAIGN_IDS,
    MAX_INTAKE_CONSTRAINTS,
    MAX_INTAKE_OBJECTIVES,
    MAX_INTAKE_PARAMETERS,
    MAX_JSON_REQUEST_BODY_BYTES,
    MAX_UPLOAD_FILE_SIZE_BYTES,
)
from api.schemas.campaign import BatchStatusRequest, CompareCampaignsRequest
from api.schemas.intake import IntakeData
from api.schemas.result import ResultBatchCreate


def _continuous_param(name: str) -> dict[str, Any]:
    return {"name": name, "type": "continuous", "bounds": [0.0, 1.0]}


def _objective(name: str) -> dict[str, Any]:
    return {"name": name, "direction": "minimize"}


def _intake_with(params: int, objectives: int, constraints: int) -> dict[str, Any]:
    return {
        "name": "Limits",
        "parameters": [_continuous_param(f"p{i}") for i in range(params)],
        "objectives": [_objective(f"o{i}") for i in range(objectives)],
        "constraints": [
            {
                "type": "linear",
                "parameters": [f"p{i}"],
                "coefficients": [1.0],
                "value": 1.0,
                "operator": "<=",
            }
            for i in range(constraints)
        ],
    }


class TestIntakeListCaps:
    """Pydantic-level bounds on the intake collections."""

    def test_parameters_at_limit_pass(self) -> None:
        IntakeData.model_validate(_intake_with(MAX_INTAKE_PARAMETERS, 1, 0))

    def test_parameters_over_limit_reject(self) -> None:
        with pytest.raises(ValidationError) as exc:
            IntakeData.model_validate(_intake_with(MAX_INTAKE_PARAMETERS + 1, 1, 0))
        assert any("parameters" in str(err["loc"]) for err in exc.value.errors())

    def test_objectives_over_limit_reject(self) -> None:
        with pytest.raises(ValidationError) as exc:
            IntakeData.model_validate(_intake_with(1, MAX_INTAKE_OBJECTIVES + 1, 0))
        assert any("objectives" in str(err["loc"]) for err in exc.value.errors())

    def test_constraints_over_limit_reject(self) -> None:
        with pytest.raises(ValidationError) as exc:
            IntakeData.model_validate(_intake_with(1, 1, MAX_INTAKE_CONSTRAINTS + 1))
        assert any("constraints" in str(err["loc"]) for err in exc.value.errors())


class TestBatchListCaps:
    """Pydantic-level bounds on the batch / compare route collections."""

    def test_batch_result_list_over_limit_rejects(self) -> None:
        payload = {
            "results": [
                {"parameter_values": {"x": 0.0}, "objective_values": {"y": 0.0}}
                for _ in range(MAX_BATCH_RESULTS + 1)
            ]
        }
        with pytest.raises(ValidationError):
            ResultBatchCreate.model_validate(payload)

    def test_batch_status_ids_over_limit_rejects(self) -> None:
        payload = {
            "campaign_ids": [f"campaign-{i}" for i in range(MAX_BATCH_CAMPAIGN_IDS + 1)],
        }
        with pytest.raises(ValidationError):
            BatchStatusRequest.model_validate(payload)

    def test_compare_ids_over_limit_rejects(self) -> None:
        payload = {
            "campaign_ids": [f"campaign-{i}" for i in range(MAX_COMPARE_CAMPAIGN_IDS + 1)],
        }
        with pytest.raises(ValidationError):
            CompareCampaignsRequest.model_validate(payload)


class TestJsonBodySizeMiddleware:
    """The global middleware rejects oversize JSON requests.

    The middleware applies two complementary checks:
    1. A ``Content-Length`` preflight that short-circuits the fast
       path without reading the body.
    2. A streaming receive wrapper that counts actual bytes, so a
       request that lies about ``Content-Length`` (or omits it
       entirely via chunked transfer encoding) is still rejected.
    """

    @pytest.mark.asyncio
    async def test_oversize_content_length_returns_413(self, api_client, auth_headers) -> None:
        oversize_payload = b"{" + b"a" * (MAX_JSON_REQUEST_BODY_BYTES + 1) + b"}"
        headers = {
            **auth_headers,
            "Content-Type": "application/json",
        }
        response = await api_client.post(
            "/api/campaigns",
            content=oversize_payload,
            headers=headers,
        )
        assert response.status_code == 413, response.text

    @pytest.mark.asyncio
    async def test_chunked_request_without_content_length_is_caught_by_stream_counter(
        self, api_client, auth_headers
    ) -> None:
        """A chunked request with no ``Content-Length`` still 413s.

        Pre-fix the middleware only inspected ``Content-Length``; a
        client using ``Transfer-Encoding: chunked`` (or otherwise
        omitting the header) bypassed the cap. The streaming
        receive wrapper sees the actual bytes as they arrive and
        raises once the running total crosses the cap, even when no
        length was advertised up front.
        """

        async def _chunked_oversize_body():
            # 65 KiB chunks; total exceeds MAX_JSON_REQUEST_BODY_BYTES.
            chunk = b"a" * (64 * 1024)
            sent = 0
            yield b"{"
            sent += 1
            while sent <= MAX_JSON_REQUEST_BODY_BYTES + 1:
                yield chunk
                sent += len(chunk)
            yield b"}"

        headers = {
            **auth_headers,
            "Content-Type": "application/json",
        }
        response = await api_client.post(
            "/api/campaigns",
            content=_chunked_oversize_body(),
            headers=headers,
        )
        assert response.status_code == 413, response.text

    @pytest.mark.asyncio
    async def test_multipart_on_non_upload_route_is_not_exempt(
        self, api_client, auth_headers
    ) -> None:
        """``multipart/...`` on a JSON route is bounded too.

        Pre-fix the middleware exempted any ``multipart/`` content
        type, so a multipart payload sent to ``POST /api/campaigns``
        would bypass the body-size cap entirely. The fix restricts
        the multipart exemption to the upload route's path.
        """
        oversize_payload = b"a" * (MAX_JSON_REQUEST_BODY_BYTES + 1)
        response = await api_client.post(
            "/api/campaigns",
            files={"file": ("payload.bin", oversize_payload, "application/octet-stream")},
            headers=auth_headers,
        )
        assert response.status_code == 413, response.text


class TestUploadFileSizeCap:
    """The per-route streaming reader caps uploads at the configured size."""

    @pytest.mark.asyncio
    async def test_oversize_upload_returns_413(
        self, api_client, auth_headers, persisted_user
    ) -> None:
        from bo_mcp_server.tools.create_campaign import create_campaign

        created = await create_campaign(
            {
                "name": "Upload size cap",
                "parameters": [_continuous_param("x")],
                "objectives": [_objective("y")],
            },
            str(persisted_user.id),
        )
        campaign_id = created["campaign_id"]

        oversize = io.BytesIO(b"x" * (MAX_UPLOAD_FILE_SIZE_BYTES + 1))
        response = await api_client.post(
            f"/api/results/{campaign_id}/upload",
            files={"file": ("big.csv", oversize, "text/csv")},
            headers=auth_headers,
        )
        assert response.status_code == 413, response.text


class TestUploadBatchRowCap:
    """The upload path applies the same per-batch row cap as JSON submit.

    Before the fix, ``MAX_BATCH_RESULTS`` was only enforced by the
    JSON ``ResultBatchCreate`` schema; a CSV / XLSX under the 25 MiB
    file-size cap could still submit far more rows than the JSON
    transport accepts. This test pins the upload row cap.
    """

    @pytest.mark.asyncio
    async def test_upload_with_too_many_rows_returns_413(
        self, api_client, auth_headers, persisted_user
    ) -> None:
        from bo_mcp_server.tools.create_campaign import create_campaign

        created = await create_campaign(
            {
                "name": "Upload row cap",
                "parameters": [_continuous_param("x")],
                "objectives": [_objective("y")],
            },
            str(persisted_user.id),
        )
        campaign_id = created["campaign_id"]

        header_line = b"x,y\n"
        row_count = MAX_BATCH_RESULTS + 1
        # Each row is ~10 bytes; well under the 25 MiB upload cap so
        # the rejection has to come from the row-count check.
        body = header_line + (b"0.1,0.2\n" * row_count)

        response = await api_client.post(
            f"/api/results/{campaign_id}/upload",
            files={"file": ("rows.csv", body, "text/csv")},
            headers=auth_headers,
        )
        assert response.status_code == 413, response.text
        detail = response.json().get("detail", "")
        assert str(MAX_BATCH_RESULTS) in detail, detail

    @pytest.mark.asyncio
    async def test_upload_with_only_headers_reports_empty_not_missing_columns(
        self, api_client, auth_headers, persisted_user
    ) -> None:
        """Header-only CSV is "no rows", not "missing columns".

        Pre-fix the route computed required columns from the union
        of *row* keys; a header-only CSV produced an empty union and
        falsely reported "Missing columns" for every declared
        column. Post-fix the column check runs against the parsed
        header row, and empty data gets its own explicit error.
        """
        from bo_mcp_server.tools.create_campaign import create_campaign

        created = await create_campaign(
            {
                "name": "Header-only upload",
                "parameters": [_continuous_param("x")],
                "objectives": [_objective("y")],
            },
            str(persisted_user.id),
        )
        campaign_id = created["campaign_id"]

        body = b"x,y\n"  # header only, no data rows

        response = await api_client.post(
            f"/api/results/{campaign_id}/upload",
            files={"file": ("empty.csv", body, "text/csv")},
            headers=auth_headers,
        )
        assert response.status_code == 400, response.text
        detail = response.json().get("detail", "")
        assert "Missing columns" not in detail, detail
        assert "no result rows" in detail.lower(), detail
