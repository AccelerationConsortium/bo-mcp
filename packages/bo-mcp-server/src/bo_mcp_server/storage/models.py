"""SQLAlchemy ORM models.

Immutability contract for the ``parsed_*`` cached properties
============================================================

Every ``@functools.cached_property`` on these ORM models deserializes a
JSON text column once per instance and caches the parsed Python object.
The cache is correct **only** while the backing JSON column and the
parsed value remain logically immutable. To enforce that:

1. Domain value objects emitted from these models (``InputParameter``,
   ``Objective``, ``Constraint``, ``CampaignSpec``, ...) are Pydantic
   models with ``model_config = ConfigDict(frozen=True)`` (see TODO 1.22).
2. Callers must never assign back into ``*_json`` columns or mutate the
   list/dict returned by a ``parsed_*`` property in place. Producing a
   new ORM instance via ``session.merge`` is the only supported edit
   path. Mutating the cached value would return stale reads on the
   next access with no error.
"""

import functools
import json
import logging
from datetime import datetime
from typing import Any, cast

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from bo_mcp_server.domain.campaign import CampaignStatus
from bo_mcp_server.domain.event import EventType
from bo_mcp_server.domain.result import ResultSource
from bo_mcp_server.domain.suggestion import SuggestionStatus
from bo_mcp_server.domain.utils import utcnow

logger = logging.getLogger(__name__)


class CorruptedJsonColumnError(RuntimeError):
    """Raised when a persisted JSON column cannot be decoded.

    The previous read path silently substituted an empty default and
    logged at ERROR, which made corruption invisible to callers: a
    spec with unreadable parameters would round-trip as an empty
    parameter list and a downstream BO run would silently misbehave.
    TODO 8.48 hardens this to a typed exception so the failure is
    addressable at the operation layer (mapped to ``DATA_INTEGRITY_ERROR``
    by ``make_corrupted_json_response`` at every transport boundary —
    distinct from ``DATABASE_ERROR`` because data corruption is
    deterministic and clients should not retry) instead of being
    swallowed.

    Writes go through ``json.dumps`` over Pydantic-validated payloads,
    so this exception only fires for rows that were corrupted *outside*
    the application — manual SQL edits, partial migrations, or a
    Postgres ``jsonb`` row tampered with by an admin script.
    """

    def __init__(self, context: str, raw: str, original: Exception) -> None:
        """Record the column context, an excerpt of the raw payload, and the cause."""
        self.context = context
        self.raw_excerpt = raw[:200]
        self.original = original
        super().__init__(f"Corrupted JSON in {context}: {original} (raw={self.raw_excerpt!r})")


def _strict_json_loads(raw: str, *, context: str) -> object:
    """Parse JSON, raising :class:`CorruptedJsonColumnError` on failure.

    Replaces the legacy ``_safe_json_loads`` that swallowed
    ``JSONDecodeError`` and returned a typed empty default. The new
    contract is "decode error means corruption, fail loud": empty
    columns either persist as the empty-JSON-literal (``"[]"`` or
    ``"{}"``) or stay ``NULL`` and are handled explicitly by the
    caller before reaching this helper. See TODO 8.48 for the audit
    rationale.
    """
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        # Log at ERROR so dashboards / alerting still see the
        # corrupted row even when the caller catches and re-raises
        # the typed exception further up.
        logger.exception("Corrupted JSON in %s (raw=%r)", context, raw[:200])
        raise CorruptedJsonColumnError(context, raw if isinstance(raw, str) else "", exc) from exc


# Foreign key constants to avoid duplicated literals
CAMPAIGNS_ID_FK = "campaigns.id"
SUGGESTIONS_ID_FK = "suggestions.id"

# Partial-index predicate for soft-delete-aware "active row" indexes.
# Centralized so the four ``ix_<table>_active`` declarations and the
# corresponding ``WHERE`` clauses in :mod:`bo_mcp_server.storage.repositories`
# cannot drift apart.
_ACTIVE_ROW_PREDICATE = "deleted_at IS NULL"

# Per-status partial-index predicates for ``campaigns.status`` (TODO 8.49).
# Mirrors migration ``015_campaigns_status_partials`` so
# ``alembic --autogenerate`` does not propose dropping the indexes.
# The status values are the *stored* enum names (uppercase), not the
# domain enum's lowercase ``.value`` strings. A round-trip drift test
# in ``tests/unit/test_storage/test_campaigns_status_partial_indexes.py``
# pins the stored representation **and** asserts the SQLite planner
# picks the partial index for the application's single-status equality
# query (an earlier IN-keyed iteration matched the audit's wording but
# not the actual repository query shape).
_CAMPAIGN_STATUS_PARTIAL_INDEXES: tuple[tuple[str, str], ...] = (
    ("ix_campaigns_status_created_active", "CREATED"),
    ("ix_campaigns_status_running_active", "RUNNING"),
    ("ix_campaigns_status_completed_active", "COMPLETED"),
    ("ix_campaigns_status_failed_active", "FAILED"),
)


def _campaign_status_predicate(status: str) -> str:
    """Return the partial-index WHERE clause for one stored status name."""
    return f"status = '{status}' AND deleted_at IS NULL"


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class UserModel(Base):
    """User ORM model."""

    __tablename__ = "users"

    # The hash must be globally unique: the auth path resolves a single
    # user with ``scalar_one_or_none()``, so duplicate provisioning
    # would surface as a 500 (``MultipleResultsFound``) on every login.
    # The uniqueness is declared here (rather than via column-level
    # ``unique=True``) so the ORM metadata names the index the same way
    # the Alembic migration does. Without this, ``alembic --autogenerate``
    # would propose dropping the migration's index and re-adding an
    # auto-named UNIQUE constraint each time.
    __table_args__ = (Index("ix_users_api_key_hash_unique", "api_key_hash", unique=True),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    api_key_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_active_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    #
    # All ORM relationships on this module use ``lazy="raise"`` so that
    # an accidental attribute access (``user.campaigns``, ``campaign.spec``,
    # …) raises :class:`sqlalchemy.exc.InvalidRequestError` instead of
    # silently emitting a per-row SELECT. Production code reads cross-table
    # data through explicit batch fetches in the repository layer
    # (``CampaignSpecRepository.get_by_ids`` / ``ResultRepository.count_by_campaigns``);
    # any future read path must follow the same pattern. See TODO 8.47.
    campaigns: Mapped[list["CampaignModel"]] = relationship(back_populates="owner", lazy="raise")


class CampaignSpecModel(Base):
    """Campaign specification ORM model."""

    __tablename__ = "campaign_specs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    parameters_json: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    objectives_json: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    constraints_json: Mapped[str] = mapped_column(Text, default="[]")  # JSON
    batch_size: Mapped[int] = mapped_column(Integer, default=1)
    max_iterations: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_observations: Mapped[int | None] = mapped_column(Integer, nullable=True)
    convergence_tolerance: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Per-campaign acquisition-optimizer overrides. Both columns are nullable
    # — when unset, bo-engine falls back to its dimension-adaptive defaults.
    acquisition_num_restarts: Mapped[int | None] = mapped_column(Integer, nullable=True)
    acquisition_raw_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)
    initial_design_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    random_seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    backend: Mapped[str] = mapped_column(String(50), default="botorch", server_default="botorch")
    # Per-backend native option surface (TODO 1.66). JSON-encoded because
    # each entry is opaque to the neutral spec. NULL means "no options".
    backend_options_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Advanced cross-backend knobs (TODO 1.66): acquisition_method,
    # use_input_warping, use_cost_aware, turbo_config, saasbo_config,
    # fidelity_parameter, transfer_learning, outcome_constraints. Stored
    # as a JSON blob because the values are consumed in-process and
    # adding a new advanced field should not require another migration.
    advanced_options_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # Relationships — see :class:`UserModel` for the ``lazy="raise"`` rationale.
    campaigns: Mapped[list["CampaignModel"]] = relationship(back_populates="spec", lazy="raise")

    @functools.cached_property
    def parsed_parameters(self) -> list[dict[str, Any]]:
        """Deserialize parameters JSON (cached per instance).

        Immutability contract: see the module docstring. Treat both
        ``parameters_json`` and the returned list as read-only.
        """
        ctx = f"CampaignSpec({self.id}).parameters"
        return cast("list[dict[str, Any]]", _strict_json_loads(self.parameters_json, context=ctx))

    @functools.cached_property
    def parsed_objectives(self) -> list[dict[str, Any]]:
        """Deserialize objectives JSON (cached per instance).

        Immutability contract: see the module docstring. Treat both
        ``objectives_json`` and the returned list as read-only.
        """
        ctx = f"CampaignSpec({self.id}).objectives"
        return cast("list[dict[str, Any]]", _strict_json_loads(self.objectives_json, context=ctx))

    @functools.cached_property
    def parsed_constraints(self) -> list[dict[str, Any]]:
        """Deserialize constraints JSON (cached per instance).

        Immutability contract: see the module docstring. Treat both
        ``constraints_json`` and the returned list as read-only.
        """
        ctx = f"CampaignSpec({self.id}).constraints"
        return cast("list[dict[str, Any]]", _strict_json_loads(self.constraints_json, context=ctx))

    # Backward-compat aliases for code using the old method names
    def get_parameters(self) -> list[dict[str, Any]]:
        """Return the parsed parameter list (alias for ``parsed_parameters``)."""
        return self.parsed_parameters

    def get_objectives(self) -> list[dict[str, Any]]:
        """Return the parsed objective list (alias for ``parsed_objectives``)."""
        return self.parsed_objectives

    def get_constraints(self) -> list[dict[str, Any]]:
        """Return the parsed constraint list (alias for ``parsed_constraints``)."""
        return self.parsed_constraints


class CampaignModel(Base):
    """Campaign ORM model."""

    __tablename__ = "campaigns"
    # ``ix_campaigns_active`` shadows the soft-delete partial filter
    # used by every read query (``deleted_at IS NULL``). The index is
    # also declared in migration ``013_soft_delete`` and mirrored here
    # so ``alembic --autogenerate`` does not propose dropping it.
    #
    # The per-status partial indexes are declared in migration
    # ``015_campaigns_status_partials`` and mirrored here for autogen
    # stability. Each partial predicate matches the application's
    # single-status equality query shape (``WHERE status = X``) so the
    # SQLite / Postgres planner actually picks them — an earlier
    # iteration used IN-keyed predicates that matched the audit's
    # wording but did not subsume the equality queries the repository
    # emits in ``list_filtered`` / ``list_keyset``.
    __table_args__ = (
        Index(
            "ix_campaigns_active",
            "id",
            sqlite_where=text(_ACTIVE_ROW_PREDICATE),
            postgresql_where=text(_ACTIVE_ROW_PREDICATE),
        ),
        *(
            Index(
                index_name,
                "status",
                sqlite_where=text(_campaign_status_predicate(status)),
                postgresql_where=text(_campaign_status_predicate(status)),
            )
            for index_name, status in _CAMPAIGN_STATUS_PARTIAL_INDEXES
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    spec_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("campaign_specs.id"), nullable=False, index=True
    )
    owner_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id"), nullable=False, index=True
    )
    status: Mapped[CampaignStatus] = mapped_column(
        Enum(CampaignStatus), default=CampaignStatus.CREATED
    )
    version: Mapped[int] = mapped_column(Integer, default=1)
    iteration: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # v1.2: TuRBO state for high-dimensional optimization
    turbo_state_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # v2.8: Hypervolume history for multi-objective convergence detection
    hypervolume_history_json: Mapped[str] = mapped_column(Text, default="[]")
    # Soft-delete marker (TODO 8.11). ``NULL`` means active;
    # repositories filter rows where this is non-null by default.
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships — see :class:`UserModel` for the ``lazy="raise"`` rationale.
    spec: Mapped["CampaignSpecModel"] = relationship(back_populates="campaigns", lazy="raise")
    owner: Mapped["UserModel"] = relationship(back_populates="campaigns", lazy="raise")
    suggestions: Mapped[list["SuggestionModel"]] = relationship(
        back_populates="campaign", lazy="raise"
    )
    results: Mapped[list["ResultModel"]] = relationship(back_populates="campaign", lazy="raise")

    @functools.cached_property
    def parsed_turbo_state(self) -> dict[str, Any] | None:
        """Deserialize TuRBO state JSON (cached per instance).

        Immutability contract: see the module docstring. Treat both
        ``turbo_state_json`` and the returned dict as read-only.
        """
        if self.turbo_state_json is None:
            return None
        ctx = f"Campaign({self.id}).turbo_state"
        return cast("dict[str, Any]", _strict_json_loads(self.turbo_state_json, context=ctx))

    @functools.cached_property
    def parsed_hypervolume_history(self) -> list[float]:
        """Deserialize hypervolume history JSON (cached per instance).

        Immutability contract: see the module docstring. Treat both
        ``hypervolume_history_json`` and the returned list as read-only.
        """
        if not self.hypervolume_history_json:
            return []
        ctx = f"Campaign({self.id}).hypervolume_history"
        return cast("list[float]", _strict_json_loads(self.hypervolume_history_json, context=ctx))

    def get_turbo_state(self) -> dict[str, Any] | None:
        """Return the parsed TuRBO state (alias for ``parsed_turbo_state``)."""
        return self.parsed_turbo_state

    def get_hypervolume_history(self) -> list[float]:
        """Return the parsed hypervolume history (alias for ``parsed_hypervolume_history``)."""
        return self.parsed_hypervolume_history


class SuggestionModel(Base):
    """Suggestion ORM model."""

    __tablename__ = "suggestions"
    # See ``CampaignModel.__table_args__`` for the soft-delete index
    # rationale; mirrored here so ``alembic --autogenerate`` is stable.
    __table_args__ = (
        Index(
            "ix_suggestions_active",
            "id",
            sqlite_where=text(_ACTIVE_ROW_PREDICATE),
            postgresql_where=text(_ACTIVE_ROW_PREDICATE),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    campaign_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(CAMPAIGNS_ID_FK, ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    parameter_values_json: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    status: Mapped[SuggestionStatus] = mapped_column(
        Enum(SuggestionStatus), default=SuggestionStatus.PENDING
    )
    provenance_json: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # Soft-delete marker (TODO 8.11). See :class:`CampaignModel.deleted_at`.
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships — see :class:`UserModel` for the ``lazy="raise"`` rationale.
    campaign: Mapped["CampaignModel"] = relationship(back_populates="suggestions", lazy="raise")
    result: Mapped["ResultModel | None"] = relationship(back_populates="suggestion", lazy="raise")

    @functools.cached_property
    def parsed_parameter_values(self) -> dict[str, Any]:
        """Deserialize parameter values JSON (cached per instance).

        Immutability contract: see the module docstring. Treat both
        ``parameter_values_json`` and the returned dict as read-only.
        """
        ctx = f"Suggestion({self.id}).parameter_values"
        return cast("dict[str, Any]", _strict_json_loads(self.parameter_values_json, context=ctx))

    @functools.cached_property
    def parsed_provenance(self) -> dict[str, Any]:
        """Deserialize provenance JSON (cached per instance).

        Immutability contract: see the module docstring. Treat both
        ``provenance_json`` and the returned dict as read-only.
        """
        ctx = f"Suggestion({self.id}).provenance"
        return cast("dict[str, Any]", _strict_json_loads(self.provenance_json, context=ctx))

    def get_parameter_values(self) -> dict[str, Any]:
        """Return the parsed parameter-value dict (alias for ``parsed_parameter_values``)."""
        return self.parsed_parameter_values

    def get_provenance(self) -> dict[str, Any]:
        """Return the parsed provenance dict (alias for ``parsed_provenance``)."""
        return self.parsed_provenance


class ResultModel(Base):
    """Result ORM model."""

    __tablename__ = "results"
    # Partial-unique index on ``suggestion_id``: at most one *active*
    # Result row can reference any given suggestion. NULL values
    # (free-floating results) are excluded so manual/imported rows can
    # coexist without forcing them to share a single global NULL slot.
    # ``deleted_at IS NULL`` is part of the predicate so a soft-deleted
    # result releases its suggestion back to the available pool — TODO
    # 8.11 treats soft-deleted rows as logically gone for application
    # reads, and the uniqueness contract follows the same semantics.
    # Postgres and SQLite both honour the ``WHERE`` clause; the
    # server-side phase-1 check
    # (:func:`_validate_suggestion_references`) catches duplicates
    # within a single batch before they reach this constraint.
    __table_args__ = (
        Index(
            "ix_results_suggestion_id_unique",
            "suggestion_id",
            unique=True,
            sqlite_where=text(f"suggestion_id IS NOT NULL AND {_ACTIVE_ROW_PREDICATE}"),
            postgresql_where=text(f"suggestion_id IS NOT NULL AND {_ACTIVE_ROW_PREDICATE}"),
        ),
        # Soft-delete partial index; see :class:`CampaignModel`.
        Index(
            "ix_results_active",
            "id",
            sqlite_where=text(_ACTIVE_ROW_PREDICATE),
            postgresql_where=text(_ACTIVE_ROW_PREDICATE),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    campaign_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(CAMPAIGNS_ID_FK, ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    # ``ondelete=SET NULL`` is retained intentionally: when a suggestion
    # is removed the result drops its pointer but
    # ``suggestion_snapshot_json`` already carries the parameter values
    # + provenance captured at submission time, so the audit trail
    # survives the broken FK.
    suggestion_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey(SUGGESTIONS_ID_FK, ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    parameter_values_json: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    objective_values_json: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    source: Mapped[ResultSource] = mapped_column(Enum(ResultSource), nullable=False)
    submitted_by: Mapped[str] = mapped_column(String(36), nullable=False)
    measurement_uncertainty_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")  # JSON
    # Snapshot of the originating suggestion at submission time (TODO 8.11).
    # ``None`` for free-floating results (no suggestion_id). For
    # suggestion-linked results this carries ``parameter_values`` and
    # ``provenance`` so the result reconstructs the BO context even if
    # the suggestion row is later removed.
    suggestion_snapshot_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # Soft-delete marker (TODO 8.11). See :class:`CampaignModel.deleted_at`.
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships — see :class:`UserModel` for the ``lazy="raise"`` rationale.
    campaign: Mapped["CampaignModel"] = relationship(back_populates="results", lazy="raise")
    suggestion: Mapped["SuggestionModel | None"] = relationship(
        back_populates="result", lazy="raise"
    )

    @functools.cached_property
    def parsed_parameter_values(self) -> dict[str, Any]:
        """Deserialize parameter values JSON (cached per instance).

        Immutability contract: see the module docstring. Treat both
        ``parameter_values_json`` and the returned dict as read-only.
        """
        ctx = f"Result({self.id}).parameter_values"
        return cast("dict[str, Any]", _strict_json_loads(self.parameter_values_json, context=ctx))

    @functools.cached_property
    def parsed_objective_values(self) -> dict[str, float]:
        """Deserialize objective values JSON (cached per instance).

        Immutability contract: see the module docstring. Treat both
        ``objective_values_json`` and the returned dict as read-only.
        """
        ctx = f"Result({self.id}).objective_values"
        return cast("dict[str, float]", _strict_json_loads(self.objective_values_json, context=ctx))

    @functools.cached_property
    def parsed_metadata(self) -> dict[str, Any]:
        """Deserialize metadata JSON (cached per instance).

        Immutability contract: see the module docstring. Treat both
        ``metadata_json`` and the returned dict as read-only.
        """
        ctx = f"Result({self.id}).metadata"
        return cast("dict[str, Any]", _strict_json_loads(self.metadata_json, context=ctx))

    @functools.cached_property
    def parsed_suggestion_snapshot(self) -> dict[str, Any] | None:
        """Deserialize the suggestion-provenance snapshot (TODO 8.11).

        ``None`` for free-floating rows (no suggestion_id at submission).
        Immutability contract: see the module docstring.
        """
        if self.suggestion_snapshot_json is None:
            return None
        ctx = f"Result({self.id}).suggestion_snapshot"
        return cast(
            "dict[str, Any]", _strict_json_loads(self.suggestion_snapshot_json, context=ctx)
        )

    def get_parameter_values(self) -> dict[str, Any]:
        """Return the parsed parameter values (alias for ``parsed_parameter_values``)."""
        return self.parsed_parameter_values

    def get_objective_values(self) -> dict[str, float]:
        """Return the parsed objective values (alias for ``parsed_objective_values``)."""
        return self.parsed_objective_values

    def get_metadata(self) -> dict[str, Any]:
        """Return the parsed metadata dict (alias for ``parsed_metadata``)."""
        return self.parsed_metadata

    def get_suggestion_snapshot(self) -> dict[str, Any] | None:
        """Return the parsed suggestion snapshot (alias for ``parsed_suggestion_snapshot``)."""
        return self.parsed_suggestion_snapshot


class EventModel(Base):
    """Audit event ORM model for MCP tool call logging."""

    __tablename__ = "events"
    # See ``CampaignModel.__table_args__`` for the soft-delete index.
    __table_args__ = (
        Index(
            "ix_events_active",
            "id",
            sqlite_where=text(_ACTIVE_ROW_PREDICATE),
            postgresql_where=text(_ACTIVE_ROW_PREDICATE),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    campaign_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey(CAMPAIGNS_ID_FK, ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    event_type: Mapped[EventType] = mapped_column(Enum(EventType), default=EventType.TOOL_CALL)
    tool_name: Mapped[str] = mapped_column(String(255), nullable=False)
    input_summary_json: Mapped[str] = mapped_column(Text, default="{}")
    output_summary_json: Mapped[str] = mapped_column(Text, default="{}")
    actor_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # Soft-delete marker (TODO 8.11). See :class:`CampaignModel.deleted_at`.
    # Events are append-only audit history; soft-delete should be reserved
    # for explicit retention purges. Read paths filter ``deleted_at IS
    # NULL`` for consistency.
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class IdempotencyCacheModel(Base):
    """Persisted ``(tool_name, idempotency_key) -> response`` cache.

    Backs the ``idempotency_key`` argument added to state-mutating MCP
    tools in TODO 1.46. The cache is shared across processes (rows live
    in the campaign DB) so retries that land on a different worker still
    short-circuit. Stale rows are pruned on read by
    :class:`bo_mcp_server.idempotency.IdempotencyStore`.

    ``reservation_token`` is set when the row is first inserted (during
    reservation) and re-matched on finalize/drop so a slow operation
    whose reservation expired cannot accidentally overwrite or delete
    a newer reservation taken by a concurrent retry.
    """

    __tablename__ = "idempotency_cache"

    tool_name: Mapped[str] = mapped_column(String(255), primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String(255), primary_key=True)
    # SHA256 of the canonical request payload so a re-used key paired
    # with a different payload is surfaced as a conflict rather than
    # silently masking a client bug.
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # Per-reservation UUID. Nullable for migration compatibility with
    # rows written before TODO 1.46 follow-up.
    reservation_token: Mapped[str | None] = mapped_column(String(36), nullable=True)
    response_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
