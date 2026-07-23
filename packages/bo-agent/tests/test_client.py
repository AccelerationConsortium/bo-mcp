"""Transport contract tests for :class:`bo_mcp.client.BoMcpClient`.

``BoMcpClient`` owns only the transport (paths, headers, error
mapping); payload shapes belong to the BO-MCP API. These tests pin the
request the client actually emits by recording ``requests.Session``
calls with a stub, the standard pattern for testing ``requests``-based
clients without a live server (see the requests developer interface,
https://requests.readthedocs.io/en/latest/api/#requests.Session.request,
and the test-double taxonomy in
https://martinfowler.com/articles/mocksArentStubs.html).

The ``force`` assertions guard the REST/MCP parity contract in both
directions: a client that silently dropped ``force`` from a forced
submission would regress REST users back to the generate-and-reject
loop the flag exists to break, while a client that always sent the key
would 422 every ordinary submission against an API server that
predates the field (the server request schema uses ``extra="forbid"``,
so unknown keys are rejected rather than ignored). ``force`` must
therefore appear exactly when requested and never otherwise.
"""

from typing import Any

import pytest
from bo_mcp.client import BoMcpClient, BoMcpOperationError
from pydantic import BaseModel, ConfigDict

_HTTP_OK = 200


class _PreForceResultBatchCreate(BaseModel):
    """Top-level request shape of servers that predate the ``force`` field.

    Mirrors the API's ``ResultBatchCreate`` contract before ``force``
    existed: ``extra="forbid"`` means any unknown top-level key fails
    validation with a 422. Default submissions must stay valid against
    this shape so a newer client works during client/server version
    skew.
    """

    model_config = ConfigDict(extra="forbid")

    results: list[dict[str, Any]]
    source: str


class _RecordedResponse:
    """Minimal stand-in for :class:`requests.Response`."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.status_code = _HTTP_OK
        self.text = ""
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class _RecordingSession:
    """Records every request and answers with a canned JSON payload."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.headers: dict[str, str] = {}
        self.calls: list[dict[str, Any]] = []
        self._payload = payload

    def request(self, method: str, url: str, **kwargs: Any) -> _RecordedResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        return _RecordedResponse(self._payload)


def _client_with_session(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]
) -> tuple[BoMcpClient, _RecordingSession]:
    client = BoMcpClient(base_url="http://bo-mcp.test", api_key="test-key")
    session = _RecordingSession(payload)
    monkeypatch.setattr(client, "session", session)
    return client, session


def test_submit_results_default_payload_keeps_legacy_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, session = _client_with_session(monkeypatch, {"success": True, "result_ids": ["r1"]})
    results = [{"parameter_values": {"x": 0.4}, "objective_values": {"y": 1.0}}]

    client.submit_results("campaign-1", results=results, idempotency_key="submit-1")

    call = session.calls[0]
    assert call["url"] == "http://bo-mcp.test/api/v1/results/campaign-1"
    # Exact payload equality: ``force`` must be absent (not merely falsy)
    # so pre-``force`` servers with ``extra="forbid"`` accept the request.
    assert call["json"] == {"results": results, "source": "api"}
    assert call["headers"]["Idempotency-Key"] == "submit-1"
    _PreForceResultBatchCreate.model_validate(call["json"])


def test_submit_results_sends_force_true_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    client, session = _client_with_session(monkeypatch, {"success": True, "result_ids": ["r2"]})

    client.submit_results(
        "campaign-1",
        results=[{"parameter_values": {"x": 0.4}, "objective_values": {"y": 1.0}}],
        idempotency_key="submit-2",
        force=True,
    )

    assert session.calls[0]["json"]["force"] is True


def test_submit_results_surfaces_operation_rejection_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rejection = {
        "success": False,
        "result_ids": [],
        "errors": ["Exact duplicate results detected. Use force=True to override."],
    }
    client, _ = _client_with_session(monkeypatch, rejection)

    with pytest.raises(BoMcpOperationError) as excinfo:
        client.submit_results(
            "campaign-1",
            results=[{"parameter_values": {"x": 0.4}, "objective_values": {"y": 1.0}}],
            idempotency_key="submit-3",
        )

    assert excinfo.value.payload == rejection
