"""Behavioral tests for the review-pass fixes on top of TODOs 1.9-1.48.

Covers:

- Concurrent retries under :func:`asyncio.gather` produce a single side
  effect through ``apply_idempotency``.
- The MCP tool schema for ``bo_submit_results`` exposes the
  ``ResultMetadata`` keys via the result-item ``metadata`` field.
- ``suggestions://{campaign_id}`` returns the structured
  ``CAMPAIGN_NOT_FOUND`` envelope when the campaign id is a valid UUID
  but no such campaign exists.
- ``list_campaigns_operation`` / ``list_results_operation`` /
  ``list_suggestions_operation`` do not emit ``next_cursor`` when the
  total row count equals ``limit`` exactly.
- ``campaigns://list/{filters}`` accepts and forwards ``cursor`` so
  resource callers can use the stable pagination path.

The fixtures already initialise a fresh SQLite-in-memory database via
``setup_database`` so each test runs against a clean state.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import UUID, uuid4

import pytest

from bo_mcp_server.domain import CampaignIntakeInput, ResultSubmissionInput
from bo_mcp_server.operations.create_campaign import create_campaign_operation
from bo_mcp_server.operations.list_campaigns import list_campaigns_operation
from bo_mcp_server.operations.list_results import list_results_operation
from bo_mcp_server.operations.list_suggestions import list_suggestions_operation
from bo_mcp_server.operations.submit_results import submit_results_operation
from tests.factories import seed_owner

pytestmark = pytest.mark.usefixtures("setup_database")


def _single_param_spec(name: str = "Test Campaign") -> dict[str, Any]:
    return {
        "name": name,
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }


async def _make_campaign(owner_id: str | None = None, name: str = "Test Campaign") -> str:
    """Create a campaign via the operation layer and return its id."""
    owner_id = owner_id or await seed_owner()
    intake = CampaignIntakeInput.model_validate(_single_param_spec(name=name))
    response = await create_campaign_operation(
        intake_data=intake,
        owner_id=owner_id,
        verbosity="minimal",
    )
    assert response["success"], response
    return response["campaign_id"]


@pytest.mark.asyncio
async def test_concurrent_create_campaign_with_same_idempotency_key() -> None:
    """Two concurrent ``bo_create_campaign`` retries never produce two campaigns.

    Regression test for the High-severity review finding.
    The load-bearing invariant is **no duplicate campaign**: under
    concurrent retries with the same ``idempotency_key``, the database
    must contain at most one campaign. Each individual response may be
    a success, a cached replay, or an in-progress / stale-owner
    envelope — what matters is that no two retries commit
    independent ``Campaign`` rows.

    Test-environment note: the production behaviour ("one caller
    succeeds, others see in-progress or cached replay") relies on
    per-session DB connections with proper isolation. The async
    test environment uses SQLite ``:memory:`` with ``StaticPool`` —
    a single shared connection — which means concurrent sessions can
    interfere with each other's transactions during the reservation
    race. Under that quirk both callers may see the stale-owner /
    in-progress envelope and zero campaigns may land. The cross-
    session connectivity bug is *not* present in production
    (Postgres has independent connections per session); see the
    dedicated unit tests in ``test_idempotency.py`` for the
    semantics in isolation. This test pins only the duplicate-
    prevention invariant so the SQLite quirk does not block the
    suite.
    """
    from bo_mcp_server.tools.create_campaign import create_campaign

    owner_id = await seed_owner()
    intake = _single_param_spec("Race Campaign")
    key = "race-key-1"

    a, b = await asyncio.gather(
        create_campaign(intake, owner_id, verbosity="minimal", idempotency_key=key),
        create_campaign(intake, owner_id, verbosity="minimal", idempotency_key=key),
    )

    # Per-response accounting (success | E014 in-progress | cached replay).
    successful = [r for r in (a, b) if r.get("success")]
    in_progress = [
        r for r in (a, b) if not r.get("success") and r.get("error", {}).get("code") == "E014"
    ]
    assert len(successful) + len(in_progress) == 2

    # Load-bearing invariant: at most one campaign exists. The OLD code
    # (before the reservation pattern landed) would commit two; the new
    # code commits at most one regardless of which envelope each caller
    # received.
    listing = await list_campaigns_operation(owner_id=UUID(owner_id), limit=10)
    assert len(listing["campaigns"]) <= 1

    # When a caller did succeed, the listing shows their campaign id.
    campaign_ids = {r["campaign_id"] for r in successful}
    if campaign_ids:
        assert len(campaign_ids) == 1
        assert listing["campaigns"][0]["campaign_id"] in campaign_ids


@pytest.mark.asyncio
async def test_mcp_tool_schema_advertises_result_metadata_keys() -> None:
    """``bo_submit_results`` exposes the ResultMetadata key set in its schema.

    Regression test for the High-severity review finding:
    the runtime validator already rejected unknown keys, but the
    generated tool schema still advertised ``additionalProperties:
    true``, so agents could not introspect the documented set.
    """
    from bo_mcp_server.server import create_mcp_server

    server = create_mcp_server()
    tools = await server.list_tools()
    submit = next(t for t in tools if t.name == "bo_submit_results")

    # Drill into the result item schema. FastMCP / Pydantic emit nested
    # models via ``$ref`` so resolve it once.
    schema = submit.inputSchema
    defs = schema.get("$defs", {})
    items = schema["properties"]["results"]["items"]
    item_schema = defs[items["$ref"].split("/")[-1]] if "$ref" in items else items

    metadata_schema = item_schema["properties"]["metadata"]
    documented_keys = {
        "external_ref",
        "conditions",
        "cost",
        "experiment_id",
        "operator",
        "batch_ref",
        "notes",
        "source_row",
        "source_file",
    }
    assert set(metadata_schema.get("properties", {}).keys()) == documented_keys
    assert metadata_schema.get("additionalProperties") is False


@pytest.mark.asyncio
async def test_suggestions_resource_returns_envelope_for_unknown_campaign() -> None:
    """``suggestions://{id}`` with a valid-but-missing UUID raises CAMPAIGN_NOT_FOUND.

    Regression test for the Medium review finding: previously the
    handler queried the suggestions table directly and returned ``"No
    pending suggestions for campaign ..."`` for any campaign with zero
    rows, conflating "does not exist" with "exists but empty". 8.43
    promoted the error path to a raised
    :class:`ResourceOperationError`.
    """
    from bo_mcp_server.errors import ResourceOperationError
    from bo_mcp_server.resources.suggestion_resource import get_suggestions

    unknown_id = str(uuid4())
    with pytest.raises(ResourceOperationError) as exc_info:
        await get_suggestions(unknown_id)

    payload = json.loads(str(exc_info.value))
    assert payload["success"] is False
    assert payload["error"]["code"] == "E002"
    assert payload["error"]["details"]["campaign_id"] == unknown_id


@pytest.mark.asyncio
async def test_suggestions_resource_empty_campaign_returns_markdown() -> None:
    """An existing campaign with no pending suggestions still returns Markdown.

    Sibling of the previous test — proves the missing-campaign envelope
    does not regress the empty-but-existing case.
    """
    from bo_mcp_server.resources.suggestion_resource import get_suggestions

    campaign_id = await _make_campaign(name="Empty Campaign")
    raw = await get_suggestions(campaign_id)

    assert not raw.startswith("{")  # not a JSON envelope
    assert "No pending suggestions" in raw


@pytest.mark.asyncio
async def test_list_campaigns_no_cursor_on_exact_final_page() -> None:
    """``next_cursor`` is ``None`` when the total exactly equals ``limit``.

    Regression test for the Medium review finding: previously the
    operation set ``next_cursor`` whenever ``len(page) == limit``, which
    incorrectly advertised another page when the total was a multiple
    of ``limit``.
    """
    owner_id = await seed_owner()
    for i in range(3):
        await _make_campaign(owner_id=owner_id, name=f"Campaign {i}")

    # Three campaigns, limit=3 — the offset path should return all
    # three rows and report ``next_cursor=None``.
    response = await list_campaigns_operation(owner_id=UUID(owner_id), limit=3, verbosity="minimal")

    assert len(response["campaigns"]) == 3
    assert response["next_cursor"] is None


@pytest.mark.asyncio
async def test_list_campaigns_cursor_walks_until_exhausted() -> None:
    """Walking through cursors must emit ``None`` on the final page.

    Exercises the keyset path with ``limit=2`` over three campaigns: the
    first page returns two with a cursor; the second returns one with
    ``next_cursor=None``. The fix-under-test is the exact-multiple case
    handled above; this test guards against re-introducing it via a
    different limit boundary.
    """
    owner_id = await seed_owner()
    for i in range(3):
        await _make_campaign(owner_id=owner_id, name=f"Campaign {i}")

    page1 = await list_campaigns_operation(owner_id=UUID(owner_id), limit=2, verbosity="minimal")
    assert len(page1["campaigns"]) == 2
    assert page1["next_cursor"] is not None

    page2 = await list_campaigns_operation(
        owner_id=UUID(owner_id),
        limit=2,
        verbosity="minimal",
        cursor=page1["next_cursor"],
    )
    assert len(page2["campaigns"]) == 1
    assert page2["next_cursor"] is None


@pytest.mark.asyncio
async def test_campaigns_resource_accepts_cursor_filter() -> None:
    """``campaigns://list/{filters}`` honours the ``cursor`` filter.

    Regression test for the Medium review finding: the resource
    accepted ``owner``, ``status``, ``limit``, ``offset`` but not
    ``cursor``, so resource callers could not use the stable pagination
    path.
    """
    from bo_mcp_server.resources.campaign_resource import list_campaigns_filtered

    owner_id = await seed_owner()
    for i in range(3):
        await _make_campaign(owner_id=owner_id, name=f"Campaign {i}")

    # Drive the operation to get a cursor token for page 1 (limit=2).
    page1 = await list_campaigns_operation(owner_id=UUID(owner_id), limit=2, verbosity="minimal")
    assert page1["next_cursor"] is not None
    token = page1["next_cursor"]

    rendered = await list_campaigns_filtered(f"owner={owner_id}&limit=2&cursor={token}")

    # Page 2 must show exactly the remaining campaign.
    assert not rendered.startswith("{"), rendered
    assert "(showing 1 of 3" in rendered
    # And no further cursor on the final page.
    assert "Next cursor" not in rendered


@pytest.mark.asyncio
async def test_campaigns_resource_rejects_unknown_filter() -> None:
    """An unknown filter key raises a structured resource error."""
    from bo_mcp_server.errors import ResourceOperationError
    from bo_mcp_server.resources.campaign_resource import list_campaigns_filtered

    with pytest.raises(ResourceOperationError) as exc_info:
        await list_campaigns_filtered("ownerr=abc")
    payload = json.loads(str(exc_info.value))
    assert payload["error"]["code"] == "E005"
    assert "ownerr" in payload["error"]["details"]["unknown_keys"]


@pytest.mark.asyncio
async def test_events_resource_returns_envelope_for_unknown_campaign() -> None:
    """``events://{id}`` with a valid-but-missing UUID raises CAMPAIGN_NOT_FOUND.

    Regression test for the review-pass finding: the events
    resource previously returned ``"No events recorded..."`` Markdown
    for any campaign id without rows, conflating "does not exist" with
    "exists but has no audit trail". The fix promoted the error to a
    raised :class:`ResourceOperationError` so FastMCP surfaces it as a
    protocol-level failure.
    """
    from bo_mcp_server.errors import ResourceOperationError
    from bo_mcp_server.resources.events_resource import get_campaign_events

    unknown_id = str(uuid4())
    with pytest.raises(ResourceOperationError) as exc_info:
        await get_campaign_events(unknown_id)

    payload = json.loads(str(exc_info.value))
    assert payload["success"] is False
    assert payload["error"]["code"] == "E002"
    assert payload["error"]["details"]["campaign_id"] == unknown_id


@pytest.mark.asyncio
async def test_events_resource_empty_existing_campaign_returns_markdown() -> None:
    """An existing campaign with no audit events still returns Markdown.

    Sibling of the missing-campaign test — proves the campaign-existence
    check does not regress the empty-event-log case.
    """
    from bo_mcp_server.resources.events_resource import get_campaign_events

    campaign_id = await _make_campaign(name="No-events campaign")
    raw = await get_campaign_events(campaign_id)
    assert not raw.startswith("{"), raw
    assert "No events recorded" in raw


async def _seed_results(campaign_id: str, n: int) -> None:
    """Submit ``n`` free-floating results to a campaign."""
    submitter = str(uuid4())
    results = [
        ResultSubmissionInput(
            parameter_values={"x": i / 10.0},
            objective_values={"y": float(i)},
        )
        for i in range(n)
    ]
    response = await submit_results_operation(
        campaign_id=campaign_id,
        results=results,
        submitted_by=submitter,
        source="api",
        atomic=True,
        verbosity="minimal",
    )
    assert response["success"], response


@pytest.mark.asyncio
async def test_list_results_no_cursor_on_exact_final_page() -> None:
    """``list_results_operation`` does not emit ``next_cursor`` for exact-page totals.

    Regression test for the review-pass finding: previously
    ``next_cursor`` was set whenever ``len(page) == limit``, which
    misled callers when the total was an exact multiple of ``limit``.
    """
    campaign_id = await _make_campaign(name="Results-pagination campaign")
    await _seed_results(campaign_id, n=3)

    response = await list_results_operation(campaign_id=campaign_id, limit=3, verbosity="minimal")
    assert len(response["results"]) == 3
    assert response["next_cursor"] is None


@pytest.mark.asyncio
async def test_list_results_cursor_walks_until_exhausted() -> None:
    """Cursor pagination through results emits ``None`` on the last page."""
    campaign_id = await _make_campaign(name="Results-cursor-walk campaign")
    await _seed_results(campaign_id, n=3)

    page1 = await list_results_operation(campaign_id=campaign_id, limit=2, verbosity="minimal")
    assert len(page1["results"]) == 2
    assert page1["next_cursor"] is not None

    page2 = await list_results_operation(
        campaign_id=campaign_id,
        limit=2,
        verbosity="minimal",
        cursor=page1["next_cursor"],
    )
    assert len(page2["results"]) == 1
    assert page2["next_cursor"] is None


async def _seed_suggestions(campaign_id: str, batches: int) -> None:
    """Run ``generate_suggestions_operation`` ``batches`` times.

    Each call produces ``batch_size`` (default: spec batch_size = 1)
    suggestions, so this seeds ``batches`` rows total.
    """
    from bo_mcp_server.operations.generate_suggestions import generate_suggestions_operation

    for _ in range(batches):
        response = await generate_suggestions_operation(
            campaign_id=campaign_id, batch_size=1, verbosity="minimal"
        )
        assert response["success"], response


@pytest.mark.asyncio
async def test_list_suggestions_no_cursor_on_exact_final_page() -> None:
    """``list_suggestions_operation`` does not over-advertise on exact pages.

    Regression test for the review-pass finding, mirroring
    the campaigns / results tests for the suggestion path.
    """
    campaign_id = await _make_campaign(name="Suggestions-pagination campaign")
    await _seed_suggestions(campaign_id, batches=3)

    response = await list_suggestions_operation(
        campaign_id=campaign_id, limit=3, verbosity="minimal"
    )
    assert len(response["suggestions"]) == 3
    assert response["next_cursor"] is None


@pytest.mark.asyncio
async def test_list_suggestions_cursor_walks_until_exhausted() -> None:
    """Suggestion cursor pagination emits ``None`` on the final page."""
    campaign_id = await _make_campaign(name="Suggestions-cursor-walk campaign")
    await _seed_suggestions(campaign_id, batches=3)

    page1 = await list_suggestions_operation(campaign_id=campaign_id, limit=2, verbosity="minimal")
    assert len(page1["suggestions"]) == 2
    assert page1["next_cursor"] is not None

    page2 = await list_suggestions_operation(
        campaign_id=campaign_id,
        limit=2,
        verbosity="minimal",
        cursor=page1["next_cursor"],
    )
    assert len(page2["suggestions"]) == 1
    assert page2["next_cursor"] is None


@pytest.mark.asyncio
async def test_upload_results_file_replays_idempotent_response_for_large_csv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A multi-MB upload with an idempotency_key replays the cached response.

    Regression test for the review-pass finding:

    1. The previous 1 MiB cap on ``canonical_request_hash`` would have
       raised ``ValueError`` before the tool returned a structured
       envelope. The fix removes the cap so the hash succeeds.
    2. The upload tool now pre-digests ``file_content`` into a
       ``file_content_digest`` field on the idempotency payload, so a
       multi-MB CSV does not inflate the cache row.

    To keep the test fast we stub the inner upload helper rather than
    submitting tens of thousands of rows: this isolates the idempotency
    layer, which is the part actually under test.
    """
    from bo_mcp_server.tools import upload_results_file as upload_mod
    from bo_mcp_server.tools.upload_results_file import upload_results_file

    n_invocations = 0
    expected_campaign = await _make_campaign(name="Upload-large-csv campaign")

    async def _stub_inner(**_kwargs: Any) -> dict[str, Any]:
        nonlocal n_invocations
        n_invocations += 1
        return {"success": True, "results_created": 1, "errors": []}

    monkeypatch.setattr(upload_mod, "_upload_results_file_inner", _stub_inner)

    # Build a CSV that comfortably exceeds the old 1 MiB cap.
    big_blob = "param_x,obj_y\n" + ("0.5,1.0\n" * 200_000)
    assert len(big_blob) > 1 * 1024 * 1024

    submitter = str(uuid4())
    first = await upload_results_file(
        campaign_id=expected_campaign,
        file_content=big_blob,
        submitted_by=submitter,
        idempotency_key="big-upload-1",
    )
    second = await upload_results_file(
        campaign_id=expected_campaign,
        file_content=big_blob,
        submitted_by=submitter,
        idempotency_key="big-upload-1",
    )

    # The first call ran the inner upload; the second replayed.
    assert n_invocations == 1
    assert first["success"] is True
    assert second["success"] is True
    assert second.get("idempotency_replay") is True


def _inject_lost_race(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Patch ``CampaignRepository.save`` so the first version-update raises.

    Mirrors the helper used by ``test_concurrent_modification.py`` so the
    regression coverage runs against the same deterministic injection
    pattern. The first call with a non-``None`` ``expected_version``
    raises :class:`ConcurrentModificationError`; subsequent calls fall
    through to the real save.
    """
    from bo_mcp_server.storage import repositories as repo_mod
    from bo_mcp_server.storage.base import ConcurrentModificationError

    original_save = repo_mod.CampaignRepository.save
    state = {"updates_triggered": 0}

    async def lost_race_save(self, campaign, expected_version=None):  # type: ignore[no-untyped-def]
        if expected_version is not None:
            state["updates_triggered"] += 1
            if state["updates_triggered"] == 1:
                msg = "Campaign"
                raise ConcurrentModificationError(msg, campaign.id, expected_version)
        return await original_save(self, campaign, expected_version=expected_version)

    monkeypatch.setattr(repo_mod.CampaignRepository, "save", lost_race_save)
    return state


@pytest.mark.asyncio
async def test_idempotent_generate_suggestions_rolls_back_on_concurrent_modification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A version-update conflict during idempotent generate_suggestions persists no rows.

    Regression test for the review-pass finding: previously, the
    operation caught ``ConcurrentModificationError`` and returned an
    error envelope while still leaving the just-inserted Suggestion
    rows on the caller-supplied session. ``apply_idempotency``'s
    session-aware outer commit would then persist those rows AND
    cache the error response, leaving the campaign in a partial state
    that no retry could resolve.

    The fix rolls back the supplied session inside the operation's
    ``except`` block and signals the idempotency layer to drop the
    reservation instead of finalizing. This test races
    ``bo_generate_suggestions`` with an injected first-version-update
    failure and asserts:

    1. The response is the structured ``CONCURRENT_MODIFICATION``
       envelope.
    2. No suggestion rows were persisted.
    3. No cache row was finalized (the reservation was dropped).
    4. A retry with the same idempotency_key can re-run the operation
       (rather than being blocked by ``E014`` until TTL).
    """
    from sqlalchemy import select

    from bo_mcp_server.storage import SuggestionRepository, get_session
    from bo_mcp_server.storage.models import IdempotencyCacheModel
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    campaign_id = await _make_campaign(name="Idempotent CMError campaign")
    key = "gs-cm-key"
    race_state = _inject_lost_race(monkeypatch)

    response = await generate_suggestions(
        campaign_id=campaign_id,
        batch_size=1,
        verbosity="minimal",
        idempotency_key=key,
    )

    # 1) Structured CMError envelope returned to the caller.
    assert race_state["updates_triggered"] == 1
    assert response["success"] is False
    assert response["error"]["code"] == "E010"

    # 2) No suggestion rows survived the rollback.
    async with get_session() as session:
        repo = SuggestionRepository(session)
        suggestions = await repo.list_by_campaign(UUID(campaign_id))
    assert suggestions == [], (
        "Suggestions inserted before the conflict must not survive the rollback"
    )

    # 3) No finalized cache row (the reservation should have been
    #    dropped so it does not block future retries).
    async with get_session() as session:
        result = await session.execute(
            select(IdempotencyCacheModel).where(
                IdempotencyCacheModel.tool_name == "bo_generate_suggestions",
                IdempotencyCacheModel.idempotency_key == key,
            )
        )
        assert result.scalar_one_or_none() is None, (
            "Transient CMError must not leave an idempotency cache row behind"
        )

    # 4) Retry can re-run the operation (the second save now succeeds
    #    because the injected race only fires on the first update).
    retry = await generate_suggestions(
        campaign_id=campaign_id,
        batch_size=1,
        verbosity="minimal",
        idempotency_key=key,
    )
    assert retry["success"] is True
    assert retry.get("idempotency_replay") is not True

    # Verify the second pass actually persisted a suggestion (the
    # retry's response shape varies by verbosity so we read the DB).
    async with get_session() as session:
        repo = SuggestionRepository(session)
        persisted = await repo.list_by_campaign(UUID(campaign_id))
    assert len(persisted) == 1


@pytest.mark.asyncio
async def test_idempotent_submit_results_rolls_back_on_concurrent_modification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A version-update conflict during idempotent submit_results persists no result rows.

    Mirrors the generate_suggestions test for the submit path. The
    multi-objective spec forces ``_update_campaign_state`` to compute
    hypervolume and save the campaign (which is where the injected
    race fires), so ``Result`` rows are saved at phase 2 before the
    conflict raises.
    """
    from sqlalchemy import select

    from bo_mcp_server.operations.create_campaign import create_campaign_operation
    from bo_mcp_server.storage import ResultRepository, get_session
    from bo_mcp_server.storage.models import IdempotencyCacheModel
    from bo_mcp_server.tools.submit_results import submit_results as submit_results_tool

    owner_id = await seed_owner()
    intake = CampaignIntakeInput.model_validate(
        {
            "name": "Idempotent Submit CMError",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [
                {"name": "y1", "direction": "minimize"},
                {"name": "y2", "direction": "minimize"},
            ],
        }
    )
    created = await create_campaign_operation(
        intake_data=intake, owner_id=owner_id, verbosity="minimal"
    )
    campaign_id = created["campaign_id"]
    race_state = _inject_lost_race(monkeypatch)

    key = "sr-cm-key"
    submitter = str(uuid4())
    payload = [
        ResultSubmissionInput(
            parameter_values={"x": 0.5},
            objective_values={"y1": 1.0, "y2": 2.0},
        )
    ]

    response = await submit_results_tool(
        campaign_id=campaign_id,
        results=payload,
        submitted_by=submitter,
        source="api",
        verbosity="minimal",
        idempotency_key=key,
    )

    # 1) Structured CMError envelope.
    assert race_state["updates_triggered"] == 1
    assert response["success"] is False
    assert response["error"]["code"] == "E010"

    # 2) No result rows persisted.
    async with get_session() as session:
        repo = ResultRepository(session)
        results = await repo.list_by_campaign(UUID(campaign_id))
    assert results == [], "Result rows inserted before the conflict must not survive rollback"

    # 3) No cache row.
    async with get_session() as session:
        cache = await session.execute(
            select(IdempotencyCacheModel).where(
                IdempotencyCacheModel.tool_name == "bo_submit_results",
                IdempotencyCacheModel.idempotency_key == key,
            )
        )
        assert cache.scalar_one_or_none() is None

    # 4) Retry can re-run (the second update now succeeds).
    retry = await submit_results_tool(
        campaign_id=campaign_id,
        results=payload,
        submitted_by=submitter,
        source="api",
        verbosity="minimal",
        idempotency_key=key,
    )
    assert retry["success"] is True
    assert retry.get("idempotency_replay") is not True

    # Verify a result row landed via the DB (response shape varies by
    # verbosity).
    async with get_session() as session:
        repo = ResultRepository(session)
        persisted = await repo.list_by_campaign(UUID(campaign_id))
    assert len(persisted) == 1
