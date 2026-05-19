"""Strict JSON-column decode contract (TODO 8.48).

The previous read path silently substituted an empty default
(``[]`` / ``{}`` / ``None``) when a JSON column was corrupted: a spec
with a damaged ``parameters_json`` would round-trip as a *zero-parameter*
campaign and the BO loop would silently misbehave. The new
``_strict_json_loads`` helper raises a typed
:class:`bo_mcp_server.storage.CorruptedJsonColumnError` so the failure
surfaces at the operation layer (mapped to ``DATA_INTEGRITY_ERROR``) instead
of becoming a wrong-but-quiet response.

Strategy
========

Application writes flow through ``json.dumps`` over Pydantic-validated
payloads, so the only realistic way to land a malformed JSON column in
production is an out-of-band write (manual SQL, partial migration,
admin script). We reproduce that here by issuing a raw ``UPDATE`` that
overwrites a previously-good column with non-JSON text, then exercise
the read path and assert it raises the typed error.

Coverage
========

* :class:`CampaignSpecModel.parsed_parameters` (most load-bearing — feeds
  every BO computation).
* :class:`CampaignModel.parsed_hypervolume_history` (the schema's only
  non-nullable defaulted JSON column).
* :class:`ResultModel.parsed_objective_values` (touches the suggestion
  path: a corrupt row used to deflate the dataset to an empty objective
  dict and produce nonsense observations).
* :class:`SuggestionModel.parsed_provenance` (forward provenance is
  consumed by the suggestion-explanation MCP tool).
* The exception payload carries the column context and the original
  decode error so dashboards and SREs can address the bad row directly.

References
==========

* SQLAlchemy "Working with Engines and Connections" — using raw SQL
  through ``session.execute(text(...))``:
  https://docs.sqlalchemy.org/en/20/core/connections.html#using-textual-sql
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncGenerator
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select, text

from bo_mcp_server.domain import (
    CampaignSpec,
    InputParameter,
    Objective,
    ParameterType,
    Result,
    ResultSource,
    Suggestion,
    SuggestionProvenance,
    User,
)
from bo_mcp_server.domain.campaign import Campaign, CampaignStatus
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    CorruptedJsonColumnError,
    ResultRepository,
    SuggestionRepository,
    UserRepository,
    close_database,
    get_session,
    init_database,
)
from bo_mcp_server.storage.models import (
    CampaignModel,
    CampaignSpecModel,
    ResultModel,
    SuggestionModel,
)


@pytest_asyncio.fixture
async def fresh_database() -> AsyncGenerator[None]:
    """Reset the engine + session factory between tests."""
    await close_database()
    await init_database()
    try:
        yield
    finally:
        await close_database()


async def _seed_full_campaign() -> dict[str, str]:
    """Persist a campaign with one result and one suggestion.

    Returns the persisted IDs so tests can corrupt specific rows via
    raw SQL without re-parsing the entire schema.
    """
    async with get_session() as session:
        unique = str(uuid4())
        user = User(
            name="Owner",
            email=f"owner-{unique}@example.com",
            api_key_hash=hashlib.sha256(unique.encode()).hexdigest(),
        )
        user = await UserRepository(session).save(user)

        spec = CampaignSpec(
            name="Corrupted-JSON Test",
            description="seed for raw-SQL corruption tests",
            parameters=(
                InputParameter(
                    name="x",
                    type=ParameterType.CONTINUOUS,
                    bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
                ),
            ),
            objectives=(Objective(name="y", direction="maximize"),),
            batch_size=1,
        )
        spec_id = uuid4()
        await CampaignSpecRepository(session).save(spec, spec_id=spec_id)
        campaign = Campaign(
            spec_id=spec_id,
            owner_id=user.id,
            status=CampaignStatus.RUNNING,
            hypervolume_history=[0.1, 0.2],
        )
        await CampaignRepository(session).save(campaign)
        suggestion = Suggestion(
            campaign_id=campaign.id,
            parameter_values={"x": 0.5},
            provenance=SuggestionProvenance(
                iteration=1,
                batch_index=0,
                generation_method="initial_design",
            ),
        )
        await SuggestionRepository(session).save(suggestion)
        result = Result(
            campaign_id=campaign.id,
            suggestion_id=suggestion.id,
            parameter_values={"x": 0.5},
            objective_values={"y": 1.0},
            source=ResultSource.API,
            submitted_by=user.id,
        )
        await ResultRepository(session).save(result)
        await session.commit()
        return {
            "spec_id": str(spec_id),
            "campaign_id": str(campaign.id),
            "suggestion_id": str(suggestion.id),
            "result_id": str(result.id),
        }


# Allowlist of (table, column) pairs the corruption helper may target.
# Restricting the helper to a fixed set keeps the f-string interpolation
# in :func:`_corrupt_column` safe — any caller that asks for an unlisted
# column raises ``ValueError`` before the SQL is built, so the S608 risk
# (SQL injection through dynamic identifiers) cannot be exercised from
# this test surface even if a future test passes attacker-controlled
# strings.
_CORRUPTIBLE_COLUMNS: frozenset[tuple[str, str]] = frozenset(
    {
        ("campaign_specs", "parameters_json"),
        ("campaign_specs", "objectives_json"),
        ("campaign_specs", "constraints_json"),
        ("campaigns", "turbo_state_json"),
        ("campaigns", "hypervolume_history_json"),
        ("suggestions", "parameter_values_json"),
        ("suggestions", "provenance_json"),
        ("results", "parameter_values_json"),
        ("results", "objective_values_json"),
        ("results", "metadata_json"),
        ("results", "suggestion_snapshot_json"),
    }
)


async def _corrupt_column(table: str, column: str, row_id: str, payload: str) -> None:
    """Overwrite a JSON column with non-JSON text via raw SQL.

    Application writes flow through ``json.dumps``; this helper simulates
    the only realistic production failure mode for these columns (manual
    SQL, partial migration, admin script writing bad data). The
    ``(table, column)`` pair is checked against :data:`_CORRUPTIBLE_COLUMNS`
    so the identifier interpolation that ruff S608 flags cannot be driven
    by an unknown / attacker-supplied value — any unlisted target fails
    fast with ``ValueError`` before SQL is constructed.
    """
    if (table, column) not in _CORRUPTIBLE_COLUMNS:
        raise ValueError(
            f"Refusing to corrupt unlisted column: ({table!r}, {column!r}). "
            f"Add it to _CORRUPTIBLE_COLUMNS if the test legitimately needs it."
        )
    # Identifiers come from the static ``_CORRUPTIBLE_COLUMNS`` allowlist
    # checked above, not from caller-supplied data; values stay
    # parameter-bound. The ``noqa: S608`` is intentional.
    stmt = text(f"UPDATE {table} SET {column} = :payload WHERE id = :row_id")  # noqa: S608
    async with get_session() as session:
        await session.execute(stmt, {"payload": payload, "row_id": row_id})
        await session.commit()


# ---------------------------------------------------------------------------
# Read-side guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_campaign_spec_parameters_decode_raises_on_corruption(
    fresh_database: None,
) -> None:
    """A non-JSON ``parameters_json`` value surfaces as the typed exception.

    Reproducer for the silent-empty-list hazard: previously the spec
    would round-trip as a zero-parameter campaign and the BO loop would
    miscompute. The new contract makes this a loud failure.
    """
    ids = await _seed_full_campaign()
    await _corrupt_column("campaign_specs", "parameters_json", ids["spec_id"], "not-json")

    async with get_session() as session:
        row = (
            await session.execute(
                select(CampaignSpecModel).where(CampaignSpecModel.id == ids["spec_id"])
            )
        ).scalar_one()
        with pytest.raises(CorruptedJsonColumnError) as exc_info:
            _ = row.parsed_parameters
    assert "CampaignSpec" in str(exc_info.value)
    assert "parameters" in str(exc_info.value)
    assert exc_info.value.raw_excerpt == "not-json"


@pytest.mark.asyncio
async def test_campaign_hypervolume_history_decode_raises_on_corruption(
    fresh_database: None,
) -> None:
    """Corrupted hypervolume history fails loud instead of returning ``[]``."""
    ids = await _seed_full_campaign()
    await _corrupt_column(
        "campaigns",
        "hypervolume_history_json",
        ids["campaign_id"],
        "{not-an-array}",
    )

    async with get_session() as session:
        row = (
            await session.execute(
                select(CampaignModel).where(CampaignModel.id == ids["campaign_id"])
            )
        ).scalar_one()
        with pytest.raises(CorruptedJsonColumnError):
            _ = row.parsed_hypervolume_history


@pytest.mark.asyncio
async def test_result_objective_values_decode_raises_on_corruption(
    fresh_database: None,
) -> None:
    """Corrupted objective values fail loud instead of yielding an empty dict.

    Most consequential failure mode of the legacy ``_safe_json_loads``:
    an empty ``{}`` would still pass through to the BO loop as a row
    with no observed objectives, polluting the dataset silently.
    """
    ids = await _seed_full_campaign()
    await _corrupt_column(
        "results",
        "objective_values_json",
        ids["result_id"],
        "}corrupted{",
    )

    async with get_session() as session:
        row = (
            await session.execute(select(ResultModel).where(ResultModel.id == ids["result_id"]))
        ).scalar_one()
        with pytest.raises(CorruptedJsonColumnError):
            _ = row.parsed_objective_values


@pytest.mark.asyncio
async def test_suggestion_provenance_decode_raises_on_corruption(
    fresh_database: None,
) -> None:
    """Suggestion provenance corruption surfaces loud, not as an empty dict."""
    ids = await _seed_full_campaign()
    await _corrupt_column(
        "suggestions",
        "provenance_json",
        ids["suggestion_id"],
        "abc",
    )

    async with get_session() as session:
        row = (
            await session.execute(
                select(SuggestionModel).where(SuggestionModel.id == ids["suggestion_id"])
            )
        ).scalar_one()
        with pytest.raises(CorruptedJsonColumnError):
            _ = row.parsed_provenance


# ---------------------------------------------------------------------------
# Write-side guard: legitimate writes still round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legitimate_writes_round_trip_through_strict_decoder(
    fresh_database: None,
) -> None:
    """The new strict decoder must not break the happy path."""
    ids = await _seed_full_campaign()
    async with get_session() as session:
        spec_row = (
            await session.execute(
                select(CampaignSpecModel).where(CampaignSpecModel.id == ids["spec_id"])
            )
        ).scalar_one()
        # Parameters / objectives / constraints decode without raising.
        assert spec_row.parsed_parameters[0]["name"] == "x"
        assert spec_row.parsed_objectives[0]["name"] == "y"
        assert spec_row.parsed_constraints == []

        campaign_row = (
            await session.execute(
                select(CampaignModel).where(CampaignModel.id == ids["campaign_id"])
            )
        ).scalar_one()
        assert campaign_row.parsed_hypervolume_history == [0.1, 0.2]
        assert campaign_row.parsed_turbo_state is None
