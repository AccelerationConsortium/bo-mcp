"""REST CSV/XLSX upload route persists ``source_file`` provenance.

Regression coverage for the audit finding that
``POST /api/results/{campaign_id}/upload`` was injecting
``metadata={"source_file": filename}`` while the
:class:`bo_mcp_server.domain.ResultMetadata` schema declared
``extra="forbid"`` and lacked a ``source_file`` field. Every REST file
upload therefore 500'd at intake; the MCP upload path used
``source_row`` and was unaffected, which is why this slipped past tests.

These tests assert that:
- the route responds with 200 and ``success=true`` for a well-formed CSV;
- the ``source_file`` (and ``source_row``) metadata round-trip into
  storage and surface through the detailed result query envelope;
- the same metadata semantics work for XLSX uploads via the pandas /
  openpyxl read path.

Reference: FastAPI multipart upload pattern documented at
https://fastapi.tiangolo.com/tutorial/request-files/.
"""

from __future__ import annotations

import io

import pandas as pd
import pytest
from bo_mcp_server.tools.create_campaign import create_campaign


async def _create_campaign(owner_id: str) -> str:
    intake = {
        "name": "REST Upload Source File Provenance",
        "parameters": [
            {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
            {"name": "z", "type": "continuous", "bounds": [-1.0, 1.0]},
        ],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }
    created = await create_campaign(intake, owner_id)
    return created["campaign_id"]


def _csv_bytes(rows: list[dict[str, float]]) -> bytes:
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


def _xlsx_bytes(rows: list[dict[str, float]]) -> bytes:
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


async def _fetch_detailed_results(
    api_client, headers: dict[str, str], campaign_id: str
) -> list[dict]:
    response = await api_client.post(
        f"/api/results/{campaign_id}/query",
        json={"verbosity": "detailed"},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    envelope = response.json()
    assert envelope["success"] is True, envelope
    return envelope["results"]


class TestCsvUploadPersistsSourceFile:
    """CSV upload writes ``source_file`` into result metadata."""

    @pytest.mark.asyncio
    async def test_csv_upload_returns_200_and_persists_filename(
        self, api_client, auth_headers, persisted_user
    ) -> None:
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign(owner_id)
        filename = "experiments_q1.csv"
        payload = _csv_bytes(
            [
                {"x": 0.1, "z": 0.2, "y": 0.42},
                {"x": 0.3, "z": -0.1, "y": 0.31},
            ]
        )

        response = await api_client.post(
            f"/api/results/{campaign_id}/upload",
            files={"file": (filename, payload, "text/csv")},
            headers=auth_headers,
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["success"] is True, body
        assert len(body["result_ids"]) == 2

        rows = await _fetch_detailed_results(api_client, auth_headers, campaign_id)
        assert len(rows) == 2
        for row in rows:
            assert row["metadata"]["source_file"] == filename
            # CSV header counts as row 1, so data rows start at 2.
            assert row["metadata"]["source_row"] in {2, 3}
        observed_rows = {row["metadata"]["source_row"] for row in rows}
        assert observed_rows == {2, 3}

    @pytest.mark.asyncio
    async def test_xlsx_upload_persists_source_file(
        self, api_client, auth_headers, persisted_user
    ) -> None:
        owner_id = str(persisted_user.id)
        campaign_id = await _create_campaign(owner_id)
        filename = "experiments_q1.xlsx"
        payload = _xlsx_bytes([{"x": 0.5, "z": 0.0, "y": 0.7}])

        response = await api_client.post(
            f"/api/results/{campaign_id}/upload",
            files={
                "file": (
                    filename,
                    payload,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
            headers=auth_headers,
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["success"] is True, body
        assert len(body["result_ids"]) == 1

        rows = await _fetch_detailed_results(api_client, auth_headers, campaign_id)
        assert len(rows) == 1
        assert rows[0]["metadata"]["source_file"] == filename
        assert rows[0]["metadata"]["source_row"] == 2
