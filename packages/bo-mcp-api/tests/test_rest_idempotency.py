"""REST endpoints honour the ``Idempotency-Key`` header.

The same idempotency contract that the MCP tools enforce must now
apply to the REST routes: a retry with the same ``Idempotency-Key``
replays the cached response instead of creating a duplicate
mutation. Pushing :func:`apply_idempotency` into the operations
layer (via :func:`bo_mcp_server.client.run_idempotent_operation`)
is what unblocks this — REST handlers re-use the MCP cache
namespace so a retry that hits the other transport sees the
original call's response.

Coverage:
- happy-path replay (same key + same payload returns the same
  ``campaign_id`` / ``result_ids`` without a duplicate write);
- omitting the header keeps the legacy "always execute" behaviour;
- key reuse with a different payload returns the typed
  :class:`~bo_mcp_server.errors.ErrorCode.IDEMPOTENCY_CONFLICT`
  envelope (409, not a 500 from the route adapter — F2);
- cross-transport replay: an MCP ``bo_create_campaign`` call
  followed by a REST ``POST /api/campaigns`` with the same key
  replays the MCP response (F3 — shared canonical payload hash).

References:
- IETF draft ``draft-ietf-httpapi-idempotency-key-header``
  https://datatracker.ietf.org/doc/draft-ietf-httpapi-idempotency-key/
- Stripe's idempotency contract:
  https://stripe.com/docs/api/idempotent_requests
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from bo_mcp_server.client import canonical_create_campaign_payload
from bo_mcp_server.idempotency import canonical_request_hash
from bo_mcp_server.storage import get_session
from bo_mcp_server.storage.models import IdempotencyCacheModel
from bo_mcp_server.tools.create_campaign import create_campaign


def _intake_payload(name: str) -> dict:
    """Minimal intake payload — single continuous parameter, single objective."""
    return {
        "name": name,
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "backend": "botorch",
    }


async def _create_campaign_for_owner(owner_id: str, name: str) -> str:
    """Create a campaign via the MCP path and return its id."""
    result = await create_campaign(_intake_payload(name), owner_id)
    assert result["success"] is True, result
    return result["campaign_id"]


def _result_row(x: float, y: float, suggestion_id: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "parameter_values": {"x": x},
        "objective_values": {"y": y},
    }
    if suggestion_id is not None:
        row["suggestion_id"] = suggestion_id
    return row


# ---------------------------------------------------------------------------
# Happy-path / no-key replay tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openapi_documents_idempotency_key_contract(api_client) -> None:
    """The schema must explain when clients should reuse ``Idempotency-Key``."""
    response = await api_client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    create_campaign = schema["paths"]["/api/v1/campaigns"]["post"]
    header = next(
        parameter
        for parameter in create_campaign["parameters"]
        if parameter["name"] == "Idempotency-Key"
    )

    description = header["description"]
    assert "reuse that same key only when retrying the exact same request" in description
    assert "Do not reuse a key for a different payload" in description
    assert "shared with the MCP tools" in description


@pytest.mark.asyncio
async def test_create_campaign_idempotency_key_replays_response(
    api_client, auth_headers, persisted_user
) -> None:
    """Same Idempotency-Key + same payload → replayed response (no duplicate write).

    The second call must return the same ``campaign_id``: the
    response is the cached envelope from the original mutation.
    Anything else means we double-wrote.
    """
    _ = persisted_user
    headers = {**auth_headers, "Idempotency-Key": "test-idem-create-1"}
    payload = {"intake": _intake_payload("idempotency-replay")}

    first = await api_client.post("/api/campaigns", json=payload, headers=headers)
    # 8.19: fresh create returns 201; the replay re-emits the
    # original status so clients see the same response shape.
    assert first.status_code == 201, first.text
    first_body = first.json()
    assert first_body["success"] is True

    second = await api_client.post("/api/campaigns", json=payload, headers=headers)
    assert second.status_code == 201, second.text
    second_body = second.json()

    # The campaign_id must match — same logical mutation, replayed.
    assert second_body["campaign_id"] == first_body["campaign_id"]


@pytest.mark.asyncio
async def test_create_campaign_without_idempotency_key_creates_fresh_each_call(
    api_client, auth_headers, persisted_user
) -> None:
    """Omitting the header preserves the legacy "always execute" behaviour.

    Without an Idempotency-Key, every retry is treated as a distinct
    request — no surprise caching for clients that have not opted
    in to the contract.
    """
    _ = persisted_user
    payload = {"intake": _intake_payload("idempotency-fresh-each")}

    first = await api_client.post("/api/campaigns", json=payload, headers=auth_headers)
    second = await api_client.post("/api/campaigns", json=payload, headers=auth_headers)

    assert first.status_code == 201, first.text
    assert second.status_code == 201, second.text
    # Two distinct campaigns are created when no key is supplied.
    assert first.json()["campaign_id"] != second.json()["campaign_id"]


@pytest.mark.asyncio
async def test_submit_results_idempotency_key_replays_response(
    api_client, auth_headers, persisted_user
) -> None:
    """Submit-results route inherits idempotency via the operations layer.

    Two POSTs with the same Idempotency-Key must replay the original
    response (same ``result_ids``) instead of persisting the batch
    twice. Without this contract a network retry that landed at a
    REST gateway would silently duplicate every result row.
    """
    owner_id = str(persisted_user.id)
    campaign_id = await _create_campaign_for_owner(owner_id, "submit-replay")
    headers = {**auth_headers, "Idempotency-Key": "test-idem-submit-1"}
    body = {"results": [_result_row(0.5, 1.0)], "source": "api"}

    first = await api_client.post(
        f"/api/results/{campaign_id}",
        json=body,
        headers=headers,
    )
    # 8.19: synchronous batch create returns 201; the replay re-emits
    # the same status the original mutation carried.
    assert first.status_code == 201, first.text
    first_body = first.json()
    assert first_body["success"] is True, first_body
    first_ids = first_body["result_ids"]
    assert len(first_ids) == 1

    second = await api_client.post(
        f"/api/results/{campaign_id}",
        json=body,
        headers=headers,
    )
    assert second.status_code == 201, second.text
    second_body = second.json()
    # The result IDs must match — replay, not a duplicate insert.
    assert second_body["result_ids"] == first_ids


@pytest.mark.asyncio
async def test_submit_results_without_idempotency_key_persists_each_call(
    api_client, auth_headers, persisted_user
) -> None:
    """Without the header, each call writes a fresh result row.

    Mirrors the create-campaign no-header test: idempotency is opt-in,
    so legacy clients keep the previous semantics.

    The two bodies carry distinct parameter values so neither call
    trips the operation's exact-duplicate detector — that detector
    is a separate concern from the idempotency-replay contract this
    test is pinning, and would otherwise mask the no-replay path
    with an operation-level rejection.
    """
    owner_id = str(persisted_user.id)
    campaign_id = await _create_campaign_for_owner(owner_id, "submit-no-key")
    first_body = {"results": [_result_row(0.4, 1.0)], "source": "api"}
    second_body = {"results": [_result_row(0.6, 2.0)], "source": "api"}

    first = await api_client.post(
        f"/api/results/{campaign_id}", json=first_body, headers=auth_headers
    )
    second = await api_client.post(
        f"/api/results/{campaign_id}", json=second_body, headers=auth_headers
    )

    assert first.status_code == 201, first.text
    assert second.status_code == 201, second.text
    assert first.json()["result_ids"] != second.json()["result_ids"]


# ---------------------------------------------------------------------------
# Negative paths (F2 regression): idempotency-layer envelopes must surface
# as a typed HTTP error, not crash the route adapter.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_campaign_idempotency_conflict_returns_409(
    api_client, auth_headers, persisted_user
) -> None:
    """Reusing a key with a *different* payload returns a 409 envelope.

    Pre-fix the REST adapter unpacked ``result["campaign_id"]`` from the
    conflict envelope (which carries only the structured error) and
    raised :class:`KeyError`, producing a 500. Post-fix the route
    detects the missing success-shape field and re-raises as
    :class:`HTTPException` with the status derived from
    :class:`~bo_mcp_server.errors.ErrorCode.IDEMPOTENCY_CONFLICT`.
    """
    _ = persisted_user
    headers = {**auth_headers, "Idempotency-Key": "conflict-key-create"}

    first = await api_client.post(
        "/api/campaigns",
        json={"intake": _intake_payload("conflict-original")},
        headers=headers,
    )
    assert first.status_code == 201, first.text

    # Same key, different payload → IDEMPOTENCY_CONFLICT (409).
    second = await api_client.post(
        "/api/campaigns",
        json={"intake": _intake_payload("conflict-different-name")},
        headers=headers,
    )
    assert second.status_code == 409, second.text
    detail = second.json()["detail"]
    assert detail["code"] == "E015", detail
    assert detail["retryable"] is False
    assert detail["details"]["idempotency_conflict"] is True


@pytest.mark.asyncio
async def test_submit_results_idempotency_conflict_returns_409(
    api_client, auth_headers, persisted_user
) -> None:
    """Same conflict semantics for submit-results — F2 covers both routes."""
    owner_id = str(persisted_user.id)
    campaign_id = await _create_campaign_for_owner(owner_id, "submit-conflict")
    headers = {**auth_headers, "Idempotency-Key": "conflict-key-submit"}

    first = await api_client.post(
        f"/api/results/{campaign_id}",
        json={"results": [_result_row(0.5, 1.0)], "source": "api"},
        headers=headers,
    )
    assert first.status_code == 201, first.text

    second = await api_client.post(
        f"/api/results/{campaign_id}",
        json={"results": [_result_row(0.5, 2.0)], "source": "api"},
        headers=headers,
    )
    assert second.status_code == 409, second.text
    detail = second.json()["detail"]
    assert detail["code"] == "E015", detail


# ---------------------------------------------------------------------------
# Cross-transport replay (F3 regression): the canonical request-payload
# builder makes REST and MCP hash identically, so a key registered on one
# transport replays on the other.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_then_rest_create_campaign_replays_across_transports(
    api_client, auth_headers, persisted_user
) -> None:
    """An MCP create call followed by a REST POST with the same key replays.

    Pre-F3 the two transports built their request_payload independently:
    REST hashed the validated model dump (defaults filled), MCP hashed
    the raw boundary dict (defaults omitted), so the hashes diverged
    even for semantically identical inputs. The shared
    ``canonical_create_campaign_payload`` builder eliminates that drift;
    this test pins the contract end-to-end.
    """
    owner_id = str(persisted_user.id)
    headers = {**auth_headers, "Idempotency-Key": "cross-transport-create"}
    intake = _intake_payload("cross-transport-create")

    # First call: MCP boundary (raw dict). The MCP wrapper internally
    # binds the idempotency key the same way the REST route does.
    mcp_result = await create_campaign(
        intake_data=intake,
        owner_id=owner_id,
        idempotency_key="cross-transport-create",
    )
    assert mcp_result["success"] is True
    mcp_campaign_id = mcp_result["campaign_id"]

    # Second call: REST with the same key + semantically identical
    # payload. The hash must match so this short-circuits as a replay
    # of the MCP response, not a fresh write.
    rest_response = await api_client.post(
        "/api/campaigns",
        json={"intake": intake},
        headers=headers,
    )
    # Cross-transport replay: the cached MCP response is replayed
    # through the REST route and the route applies its own create
    # status (201).
    assert rest_response.status_code == 201, rest_response.text
    body = rest_response.json()
    assert body["campaign_id"] == mcp_campaign_id
    # F6 regression: the REST schema now forwards the wrapper's
    # ``idempotency_replay`` marker so clients can distinguish a
    # cached response from a fresh write.
    assert body["idempotency_replay"] is True


# ---------------------------------------------------------------------------
# Replay marker (F6) and in-progress reservation (F7)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_campaign_replay_sets_idempotency_replay_flag(
    api_client, auth_headers, persisted_user
) -> None:
    """A same-transport replay must also expose ``idempotency_replay=True``.

    The first call writes the row; the response carries the marker
    absent (``False``). The second call hits the cache and the
    marker must be ``True`` so REST clients can render "this was a
    duplicate retry" feedback without re-querying the campaign.
    """
    _ = persisted_user
    headers = {**auth_headers, "Idempotency-Key": "replay-marker-create"}
    payload = {"intake": _intake_payload("replay-marker")}

    first = await api_client.post("/api/campaigns", json=payload, headers=headers)
    assert first.status_code == 201, first.text
    first_body = first.json()
    assert first_body["idempotency_replay"] is False

    second = await api_client.post("/api/campaigns", json=payload, headers=headers)
    assert second.status_code == 201, second.text
    second_body = second.json()
    assert second_body["campaign_id"] == first_body["campaign_id"]
    assert second_body["idempotency_replay"] is True


@pytest.mark.asyncio
async def test_submit_results_replay_sets_idempotency_replay_flag(
    api_client, auth_headers, persisted_user
) -> None:
    """Same replay-marker contract for the submit-results route."""
    owner_id = str(persisted_user.id)
    campaign_id = await _create_campaign_for_owner(owner_id, "submit-replay-marker")
    headers = {**auth_headers, "Idempotency-Key": "replay-marker-submit"}
    body = {"results": [_result_row(0.5, 1.0)], "source": "api"}

    first = await api_client.post(f"/api/results/{campaign_id}", json=body, headers=headers)
    assert first.status_code == 201, first.text
    assert first.json()["idempotency_replay"] is False

    second = await api_client.post(f"/api/results/{campaign_id}", json=body, headers=headers)
    assert second.status_code == 201, second.text
    second_body = second.json()
    assert second_body["result_ids"] == first.json()["result_ids"]
    assert second_body["idempotency_replay"] is True


async def _seed_pending_reservation(
    *, tool_name: str, idempotency_key: str, request_hash: str
) -> None:
    """Insert a pending-reservation row directly into the cache.

    Mimics the state of "a concurrent retry is mid-flight": the row
    exists with the ``response_json`` sentinel set to the empty
    string (the pending marker used by
    :func:`bo_mcp_server.idempotency._try_reserve`) and an
    ``expires_at`` well in the future so the lazy-GC purge cannot
    delete it before the test re-enters the lookup. The reservation
    token is arbitrary — the matching ``request_hash`` is what
    triggers the in-progress short-circuit (a mismatched hash would
    take the conflict path instead, which is already covered by F2).
    """
    now = datetime.now(UTC)
    async with get_session() as session:
        session.add(
            IdempotencyCacheModel(
                tool_name=tool_name,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                # Not a credential — just a per-reservation UUID-shaped
                # sentinel matched by finalize / drop. S106 only fires
                # because the column name contains "token".
                reservation_token="seeded-reservation-token",  # noqa: S106
                response_json="",  # _PENDING_SENTINEL — pending reservation
                created_at=now,
                expires_at=now + timedelta(minutes=10),
            )
        )
        await session.commit()


@pytest.mark.asyncio
async def test_create_campaign_in_progress_reservation_returns_409(
    api_client, auth_headers, persisted_user
) -> None:
    """F7 regression: a pending reservation must surface as 409 / E014.

    The REST route relies on the "missing success-shape field"
    branch to handle the in-progress envelope, but no test pinned
    that path — a refactor of the route adapter could regress it
    silently. Seeding the reservation row directly is deterministic
    (no race window or sleep) and exercises the exact lookup branch
    a real concurrent retry would hit.
    """
    _ = persisted_user
    intake = _intake_payload("in-progress-create")
    # Seed a pending reservation under the same hash the REST route
    # will compute for this payload. The wrapper's lookup must find
    # it and short-circuit with the IDEMPOTENCY_IN_PROGRESS envelope.
    request_hash = canonical_request_hash(
        canonical_create_campaign_payload(
            intake_data=intake,
            owner_id=str(persisted_user.id),
        )
    )
    await _seed_pending_reservation(
        tool_name="bo_create_campaign",
        idempotency_key="in-progress-key-create",
        request_hash=request_hash,
    )

    response = await api_client.post(
        "/api/campaigns",
        json={"intake": intake},
        headers={**auth_headers, "Idempotency-Key": "in-progress-key-create"},
    )
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "E014", detail
    assert detail["retryable"] is True
    # Backoff hint is positive so HTTP retry middleware can honour it.
    assert detail["retry_after"] is not None
    assert detail["retry_after"] > 0
    assert detail["details"]["idempotency_in_progress"] is True
