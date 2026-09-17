"""Transport contract tests for :class:`bo_mcp.client.BoMcpClient`.

``BoMcpClient`` owns only the transport (paths, headers, error
mapping); payload shapes belong to the BO-MCP API. These tests pin the
request the client actually emits by recording ``requests.Session``
calls with a stub, the standard pattern for testing ``requests``-based
clients without a live server (see the requests developer interface,
https://requests.readthedocs.io/en/latest/api/#requests.Session.request,
and the test-double taxonomy in
https://martinfowler.com/articles/mocksArentStubs.html).

The submit payload is pinned by exact equality because the server
request schema uses ``extra="forbid"``: any stray top-level key fails
validation with a 422 rather than being ignored.
"""

from typing import Any

import pytest
from bo_mcp.client import BoMcpClient, BoMcpOperationError
from pydantic import BaseModel, ConfigDict

_HTTP_OK = 200


class _ResultBatchCreate(BaseModel):
    """Top-level request shape accepted by the API.

    Mirrors the API's ``ResultBatchCreate`` contract: ``extra="forbid"``
    means any unknown top-level key fails validation with a 422.
    """

    model_config = ConfigDict(extra="forbid")

    results: list[dict[str, Any]]
    source: str


class _RecordedResponse:
    """Minimal stand-in for :class:`requests.Response`."""

    def __init__(self, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
        self.status_code = _HTTP_OK
        self.text = ""
        self._payload = payload

    def json(self) -> dict[str, Any] | list[dict[str, Any]]:
        return self._payload


class _RecordingSession:
    """Records every request and answers with a canned JSON payload."""

    def __init__(self, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
        self.headers: dict[str, str] = {}
        self.calls: list[dict[str, Any]] = []
        self._payload = payload

    def request(self, method: str, url: str, **kwargs: Any) -> _RecordedResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        return _RecordedResponse(self._payload)


def _client_with_session(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any] | list[dict[str, Any]],
) -> tuple[BoMcpClient, _RecordingSession]:
    client = BoMcpClient(base_url="http://bo-mcp.test", api_key="test-key")
    session = _RecordingSession(payload)
    monkeypatch.setattr(client, "session", session)
    return client, session


def test_client_guidance_distinguishes_replicates_from_rejections() -> None:
    guidance = " ".join((BoMcpClient.__doc__ or "").split())

    assert "Do not reject a suggestion solely because" in guidance
    assert "may intentionally recommend a replicate" in guidance
    assert "Run it and submit the result" in guidance
    assert "does not exclude" in guidance


def test_generate_suggestions_uses_own_generous_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, session = _client_with_session(monkeypatch, {"success": True, "suggestions": []})

    client.generate_suggestions("campaign-1")

    assert session.calls[0]["timeout"] == 900.0


def test_generate_suggestions_honors_explicit_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, session = _client_with_session(monkeypatch, {"success": True, "suggestions": []})

    client.generate_suggestions("campaign-1", timeout_s=1800.0)

    assert session.calls[0]["timeout"] == 1800.0


def test_get_results_reads_campaign_scoped_result_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, session = _client_with_session(monkeypatch, [{"id": "result-1"}])

    assert client.get_results("campaign-1") == [{"id": "result-1"}]
    call = session.calls[0]
    assert call["method"] == "GET"
    assert call["url"] == "http://bo-mcp.test/api/v1/results/campaign-1"


def test_get_results_rejects_non_list_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _ = _client_with_session(monkeypatch, {"results": []})

    with pytest.raises(BoMcpOperationError, match="non-list result payload"):
        client.get_results("campaign-1")


def test_submit_results_payload_carries_no_extra_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, session = _client_with_session(monkeypatch, {"success": True, "result_ids": ["r1"]})
    results = [{"parameter_values": {"x": 0.4}, "objective_values": {"y": 1.0}}]

    client.submit_results("campaign-1", results=results, idempotency_key="submit-1")

    call = session.calls[0]
    assert call["url"] == "http://bo-mcp.test/api/v1/results/campaign-1"
    # Exact payload equality: an unknown top-level key would 422.
    assert call["json"] == {"results": results, "source": "api"}
    assert call["headers"]["Idempotency-Key"] == "submit-1"
    _ResultBatchCreate.model_validate(call["json"])


def test_submit_results_surfaces_operation_rejection_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rejection = {
        "success": False,
        "result_ids": [],
        "errors": ["A result already exists for this suggestion"],
    }
    client, _ = _client_with_session(monkeypatch, rejection)

    with pytest.raises(BoMcpOperationError) as excinfo:
        client.submit_results(
            "campaign-1",
            results=[{"parameter_values": {"x": 0.4}, "objective_values": {"y": 1.0}}],
            idempotency_key="submit-3",
        )

    assert excinfo.value.payload == rejection
