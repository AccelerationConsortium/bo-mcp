"""Phase I MCP/LLM-ergonomics regression tests.

Covers the Phase-I findings consolidated in the May 2026 review:

- 8.40: ``campaigns://recent`` is a cheap discovery surface and a typo'd
  campaign id in ``campaign://{id}`` surfaces fuzzy-match suggestions
  under ``error.details.suggestions``.
- 8.42: cursor and offset are mutually exclusive on
  ``list_campaigns_operation``.
- 8.45: every registered MCP tool routes through the boundary
  envelope wrapper (the startup invariant), and an unwrapped tool
  manager fails the assertion.

The MCP resource-subscription / pagination references:
  https://modelcontextprotocol.io/specification/2025-06-18/server/resources
  https://modelcontextprotocol.io/specification/2025-06-18/server/tools
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID, uuid4

import pytest

from bo_mcp_server.domain import CampaignIntakeInput
from bo_mcp_server.operations.create_campaign import create_campaign_operation

pytestmark = pytest.mark.usefixtures("setup_database")


def _spec(name: str = "Test Campaign") -> dict[str, Any]:
    return {
        "name": name,
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }


async def _make_campaign(name: str = "Test Campaign") -> str:
    intake = CampaignIntakeInput.model_validate(_spec(name=name))
    response = await create_campaign_operation(
        intake_data=intake,
        owner_id=str(uuid4()),
        verbosity="minimal",
    )
    assert response["success"], response
    return response["campaign_id"]


class TestCampaignsRecentResource:
    """``campaigns://recent`` — cheap-discovery surface."""

    @pytest.mark.asyncio
    async def test_empty_listing_returns_placeholder(self) -> None:
        from bo_mcp_server.resources.campaign_resource import recent_campaigns

        rendered = await recent_campaigns()
        assert "No campaigns recorded yet." in rendered

    @pytest.mark.asyncio
    async def test_returns_newest_first(self) -> None:
        from bo_mcp_server.resources.campaign_resource import recent_campaigns

        # Insertion order matches creation order; resource must return
        # newest-first.
        await _make_campaign(name="Campaign A")
        await _make_campaign(name="Campaign B")
        rendered = await recent_campaigns()

        idx_a = rendered.find("Campaign A")
        idx_b = rendered.find("Campaign B")
        assert idx_a != -1
        assert idx_b != -1
        assert idx_b < idx_a, "Newest campaign must appear first"

    @pytest.mark.asyncio
    async def test_limit_filter_is_honoured(self) -> None:
        from bo_mcp_server.resources.campaign_resource import recent_campaigns_with_limit

        for i in range(4):
            await _make_campaign(name=f"Campaign {i}")
        rendered = await recent_campaigns_with_limit("limit=2")
        # Two campaigns rendered, not all four.
        assert "(showing 2)" in rendered

    @pytest.mark.asyncio
    async def test_unknown_filter_key_raises_structured_error(self) -> None:
        from bo_mcp_server.errors import ResourceOperationError
        from bo_mcp_server.resources.campaign_resource import recent_campaigns_with_limit

        with pytest.raises(ResourceOperationError) as exc_info:
            await recent_campaigns_with_limit("offset=5")
        payload = json.loads(str(exc_info.value))
        assert payload["error"]["code"] == "E005"
        assert "offset" in payload["error"]["details"]["unknown_keys"]


class TestCampaignNotFoundSuggestions:
    """Typo'd campaign ids surface fuzzy-match suggestions."""

    @pytest.mark.asyncio
    async def test_typo_uuid_returns_suggestions(self) -> None:
        from bo_mcp_server.errors import ResourceOperationError
        from bo_mcp_server.resources.campaign_resource import get_campaign

        true_id = await _make_campaign(name="Target")
        # Flip the last hex digit so the candidate is close-but-not-equal.
        last = true_id[-1]
        next_hex = "0" if last.lower() == "f" else hex((int(last, 16) + 1) % 16)[2:]
        typo = true_id[:-1] + next_hex
        with pytest.raises(ResourceOperationError) as exc_info:
            await get_campaign(typo)
        payload = json.loads(str(exc_info.value))
        assert payload["error"]["code"] == "E002"
        suggestions = payload["error"]["details"].get("suggestions", [])
        assert true_id in suggestions

    @pytest.mark.asyncio
    async def test_unrelated_uuid_omits_suggestions(self) -> None:
        from bo_mcp_server.errors import ResourceOperationError
        from bo_mcp_server.resources.campaign_resource import get_campaign

        await _make_campaign(name="Target")
        # Completely unrelated UUID — fuzzy match should drop below
        # the similarity threshold.
        unrelated = "ffffffff-eeee-dddd-cccc-bbbbbbbbbbbb"
        with pytest.raises(ResourceOperationError) as exc_info:
            await get_campaign(unrelated)
        payload = json.loads(str(exc_info.value))
        assert "suggestions" not in payload["error"]["details"]


class TestCursorOffsetMutualExclusion:
    """``list_campaigns_operation`` rejects ``cursor`` + ``offset>0``."""

    @pytest.mark.asyncio
    async def test_both_cursor_and_offset_returns_validation_error(self) -> None:
        from bo_mcp_server.operations.list_campaigns import list_campaigns_operation

        # Build a real cursor via a single round-trip — passing a
        # decoder-failing value would mask the mutual-exclusion check.
        owner = uuid4()
        for i in range(2):
            await _make_campaign(name=f"OwnedCampaign {i}")
        page = await list_campaigns_operation(owner_id=owner, limit=1, verbosity="minimal")
        # When there are no rows for this owner the cursor is None;
        # supply a synthetic-but-well-formed token instead.
        token = page.get("next_cursor") or _synthetic_cursor()

        response = await list_campaigns_operation(
            limit=10, offset=5, cursor=token, verbosity="minimal"
        )
        assert response["success"] is False
        assert response["error"]["code"] == "E005"
        assert "mutually exclusive" in response["error"]["message"]

    @pytest.mark.asyncio
    async def test_offset_only_remains_supported(self) -> None:
        from bo_mcp_server.operations.list_campaigns import list_campaigns_operation

        for i in range(3):
            await _make_campaign(name=f"Campaign {i}")
        response = await list_campaigns_operation(limit=1, offset=1, verbosity="minimal")
        assert response["success"] is True

    @pytest.mark.asyncio
    async def test_cursor_only_remains_supported(self) -> None:
        from bo_mcp_server.operations.list_campaigns import list_campaigns_operation

        # Seed multiple campaigns and walk one page via cursor.
        owner = UUID(int=42)
        for i in range(3):
            intake = CampaignIntakeInput.model_validate(_spec(name=f"Cursor {i}"))
            response = await create_campaign_operation(
                intake_data=intake,
                owner_id=str(owner),
                verbosity="minimal",
            )
            assert response["success"], response
        page = await list_campaigns_operation(owner_id=owner, limit=1, verbosity="minimal")
        assert page["success"] is True
        if page["next_cursor"]:
            next_page = await list_campaigns_operation(
                owner_id=owner,
                limit=1,
                cursor=page["next_cursor"],
                verbosity="minimal",
            )
            assert next_page["success"] is True

    @pytest.mark.asyncio
    async def test_cursor_pages_have_no_duplicates_and_no_skips(self) -> None:
        """Friend-review finding: keyset pagination must not duplicate or skip.

        Pre-fix the first page used ``list_filtered`` ordered DESC
        while the cursor pages used ``list_keyset`` ordered ASC with
        ``> cursor`` — page 1 returned the two newest campaigns, the
        cursor pointed at the *older* of those two, and page 2 fetched
        everything with ``created_at > older_page1`` ASC. Result:
        page 2 duplicated the newest campaign and silently skipped the
        oldest one. The fix aligns keyset ordering with the first-page
        order (DESC, walking backwards through time); this test pins
        the no-duplicate / no-skip invariant across all pages.
        """
        from bo_mcp_server.operations.list_campaigns import list_campaigns_operation

        owner = UUID(int=7)
        expected_ids: list[str] = []
        for i in range(3):
            intake = CampaignIntakeInput.model_validate(_spec(name=f"Walker {i}"))
            response = await create_campaign_operation(
                intake_data=intake,
                owner_id=str(owner),
                verbosity="minimal",
            )
            assert response["success"], response
            expected_ids.append(response["campaign_id"])

        page1 = await list_campaigns_operation(owner_id=owner, limit=2, verbosity="minimal")
        page1_ids = [c["campaign_id"] for c in page1["campaigns"]]
        assert page1["next_cursor"] is not None

        page2 = await list_campaigns_operation(
            owner_id=owner,
            limit=2,
            cursor=page1["next_cursor"],
            verbosity="minimal",
        )
        page2_ids = [c["campaign_id"] for c in page2["campaigns"]]

        # No duplicates across pages.
        assert set(page1_ids).isdisjoint(set(page2_ids)), (
            f"page1 {page1_ids} and page2 {page2_ids} overlap"
        )
        # No skips: union must contain every seeded id.
        assert set(page1_ids) | set(page2_ids) == set(expected_ids), (
            f"union {sorted(set(page1_ids) | set(page2_ids))} != expected {sorted(expected_ids)}"
        )
        # And the final page should not advertise another cursor.
        assert page2["next_cursor"] is None

    @pytest.mark.asyncio
    async def test_cursor_walk_with_equal_timestamps_has_no_duplicates_or_skips(
        self,
    ) -> None:
        """Friend-review follow-up: tie-breaker on ``id`` must protect equal-timestamp rows.

        Pre-fix the keyset query ordered by ``(created_at DESC, id
        DESC)`` but the offset/cursorless first page ordered by
        ``created_at DESC`` only. With rows that share a
        ``created_at`` value (bulk seeding, scripted creation, or any
        clock with millisecond-truncated timestamps under load), the
        first page picked an arbitrary order among the tied rows
        while the keyset comparator used ``id < cursor_id`` — opening
        the same duplicate / skip window the earlier ASC-DESC bug
        opened, just hidden behind a tie. The previous regression
        tests used natural creation timing so the timestamps were
        always distinct; this test pins the tie-breaker explicitly by
        creating four real campaigns through the operation layer
        (real specs, real owner row, valid FKs) and then forcing
        their ``created_at`` values to a shared timestamp with a
        direct ``UPDATE``. This keeps the test portable to FK-
        enforced SQLite and Postgres — the earlier approach went
        through :meth:`CampaignRepository.save` with random
        ``spec_id`` / ``owner_id`` UUIDs that only worked because the
        default SQLite fixture leaves ``PRAGMA foreign_keys`` off.
        """
        from datetime import UTC, datetime

        from sqlalchemy import update

        from bo_mcp_server.operations.list_campaigns import list_campaigns_operation
        from bo_mcp_server.storage import get_session
        from bo_mcp_server.storage.models import CampaignModel

        owner = UUID(int=21)
        shared_ts = datetime(2026, 5, 18, 12, 0, 0, tzinfo=UTC)

        # Create four real campaigns through the operation layer so
        # the ``campaign_specs`` row + ``users`` row + FK references
        # are all valid. ``create_campaign_operation`` upserts the
        # owner user when missing, so passing a fresh ``owner`` here
        # is enough — no test-only seeding required.
        seeded: list[str] = []
        for i in range(4):
            intake = CampaignIntakeInput.model_validate(_spec(name=f"Tied {i}"))
            response = await create_campaign_operation(
                intake_data=intake,
                owner_id=str(owner),
                verbosity="minimal",
            )
            assert response["success"], response
            seeded.append(response["campaign_id"])

        # Force a shared ``created_at`` on all four rows. The earlier
        # approach constructed Campaign entities with the desired
        # timestamp and ``save()``-d them, but that bypassed the
        # spec/user FK constraints — a bulk ``UPDATE`` after
        # operation-layer creation gives us the same equal-timestamp
        # surface while keeping every FK valid.
        async with get_session() as session:
            await session.execute(
                update(CampaignModel)
                .where(CampaignModel.id.in_(seeded))
                .values(created_at=shared_ts, updated_at=shared_ts)
            )
            await session.commit()

        # Walk with limit=2 and re-use the returned cursor for page 2.
        page1 = await list_campaigns_operation(owner_id=owner, limit=2, verbosity="minimal")
        page1_ids = [c["campaign_id"] for c in page1["campaigns"]]
        assert page1["next_cursor"] is not None
        page2 = await list_campaigns_operation(
            owner_id=owner,
            limit=2,
            cursor=page1["next_cursor"],
            verbosity="minimal",
        )
        page2_ids = [c["campaign_id"] for c in page2["campaigns"]]

        # No duplicates across pages even though all four rows share
        # a ``created_at`` value.
        assert set(page1_ids).isdisjoint(set(page2_ids)), (
            f"page1 {page1_ids} and page2 {page2_ids} overlap on tied timestamps"
        )
        # No skips — every seeded id must appear exactly once.
        union = set(page1_ids) | set(page2_ids)
        assert union == set(seeded), f"union {sorted(union)} != expected {sorted(seeded)}"
        assert page2["next_cursor"] is None

        # And a ``limit=1`` walk over the same tied rows visits each
        # exactly once — the harshest stress on the tiebreaker
        # because every page boundary lands on a tied row.
        visited: list[str] = []
        cursor: str | None = None
        for _ in range(8):  # hard cap
            page = await list_campaigns_operation(
                owner_id=owner, limit=1, cursor=cursor, verbosity="minimal"
            )
            assert page["success"] is True
            visited.extend(c["campaign_id"] for c in page["campaigns"])
            cursor = page["next_cursor"]
            if cursor is None:
                break
        assert len(visited) == len(set(visited)), f"duplicates in walk: {visited}"
        assert set(visited) == set(seeded), (
            f"visited {sorted(visited)} != expected {sorted(seeded)}"
        )

    @pytest.mark.asyncio
    async def test_cursor_walk_with_limit_one_covers_all_campaigns(self) -> None:
        """Edge case: tiny pages stress the cursor comparator hardest.

        ``limit=1`` forces every page boundary to land on a fresh row,
        so a direction bug here surfaces as either an infinite loop
        (skip the only row that can advance the cursor) or a tight
        duplicate cycle. Walking with limit=1 over five seeded
        campaigns must visit each exactly once.
        """
        from bo_mcp_server.operations.list_campaigns import list_campaigns_operation

        owner = UUID(int=11)
        expected_ids: list[str] = []
        for i in range(5):
            intake = CampaignIntakeInput.model_validate(_spec(name=f"Tiny {i}"))
            response = await create_campaign_operation(
                intake_data=intake,
                owner_id=str(owner),
                verbosity="minimal",
            )
            expected_ids.append(response["campaign_id"])

        visited: list[str] = []
        cursor: str | None = None
        for _ in range(10):  # hard cap to avoid an infinite loop
            page = await list_campaigns_operation(
                owner_id=owner, limit=1, cursor=cursor, verbosity="minimal"
            )
            assert page["success"] is True
            visited.extend(c["campaign_id"] for c in page["campaigns"])
            cursor = page["next_cursor"]
            if cursor is None:
                break

        assert len(visited) == len(set(visited)), f"duplicates in walk: {visited}"
        assert set(visited) == set(expected_ids), (
            f"visited {sorted(visited)} != expected {sorted(expected_ids)}"
        )


def _synthetic_cursor() -> str:
    """Return a well-formed (but arbitrary) cursor token for branch coverage."""
    from datetime import UTC, datetime

    from bo_mcp_server.pagination import cursor_from_row

    return cursor_from_row(datetime.now(UTC), str(uuid4()))


class TestToolBoundaryDefaults:
    """Every MCP tool routes through the envelope wrapper."""

    def test_assert_invariant_passes_after_create_mcp_server(self) -> None:
        from bo_mcp_server.server import create_mcp_server
        from bo_mcp_server.tool_boundary import assert_all_tools_routed_through_wrapper

        mcp = create_mcp_server()
        # The startup invariant is already called inside
        # ``create_mcp_server`` but re-running it must be a no-op.
        assert_all_tools_routed_through_wrapper(mcp)

    def test_assert_invariant_detects_unwrapped_manager(self) -> None:
        from bo_mcp_server.tool_boundary import assert_all_tools_routed_through_wrapper

        class _DummyManager:
            def __init__(self) -> None:
                self._tools: dict[str, Any] = {"bo_dummy": object()}

            async def call_tool(
                self, name: str, arguments: dict[str, Any], **_: Any
            ) -> dict[str, Any]:
                return {"name": name, "arguments": arguments}

        class _DummyMcp:
            def __init__(self) -> None:
                self._tool_manager = _DummyManager()

        with pytest.raises(RuntimeError, match="not installed"):
            assert_all_tools_routed_through_wrapper(_DummyMcp())  # ty: ignore[invalid-argument-type]

    @pytest.mark.asyncio
    async def test_unregistered_tool_still_gets_envelope_on_validation_error(self) -> None:
        """Tools omitted from the override map still receive the canonical envelope.

        Pre-8.45 only tools listed in ``_TOOL_ENVELOPE_DEFAULTS``
        received the structured response; the rest leaked raw
        ``ToolError`` text. The default-on contract means an
        unregistered tool (here, ``bo_list_campaigns``) still gets
        the canonical envelope on a Pydantic argument-validation
        failure.
        """
        from bo_mcp_server.server import create_mcp_server

        mcp = create_mcp_server()
        result = await mcp._tool_manager.call_tool(
            "bo_list_campaigns",
            arguments={"limit": "not-an-int"},
        )
        assert isinstance(result, dict)
        assert result["success"] is False
        # ``bo_list_campaigns`` is not in the override map, so the
        # envelope has no extra polish — just the canonical fields.
        assert result["error"]["code"] == "E005"
        assert "field_errors" in result
