"""SQLAlchemy ORM models."""

import functools
import json
import logging
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from bo_mcp_server.domain.campaign import CampaignStatus
from bo_mcp_server.domain.event import EventType
from bo_mcp_server.domain.result import ResultSource
from bo_mcp_server.domain.suggestion import SuggestionStatus
from bo_mcp_server.domain.utils import utcnow

logger = logging.getLogger(__name__)


def _safe_json_loads(raw: str, *, default: Any, context: str = "") -> Any:
    """Parse JSON with error handling for corrupted data.

    Returns *default* if parsing fails, logging the error at ERROR level
    so corrupted rows are visible in logs without crashing the request.
    """
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.error("Corrupted JSON in %s: %s (raw=%r)", context, exc, raw[:200])
        return default


# Foreign key constants to avoid duplicated literals
CAMPAIGNS_ID_FK = "campaigns.id"
SUGGESTIONS_ID_FK = "suggestions.id"


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class UserModel(Base):
    """User ORM model."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    api_key_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_active_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    campaigns: Mapped[list["CampaignModel"]] = relationship(back_populates="owner")


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

    # Relationships
    campaigns: Mapped[list["CampaignModel"]] = relationship(back_populates="spec")

    @functools.cached_property
    def parsed_parameters(self) -> list[dict[str, Any]]:
        """Deserialize parameters JSON (cached per instance)."""
        ctx = f"CampaignSpec({self.id}).parameters"
        return _safe_json_loads(self.parameters_json, default=[], context=ctx)

    @functools.cached_property
    def parsed_objectives(self) -> list[dict[str, Any]]:
        """Deserialize objectives JSON (cached per instance)."""
        ctx = f"CampaignSpec({self.id}).objectives"
        return _safe_json_loads(self.objectives_json, default=[], context=ctx)

    @functools.cached_property
    def parsed_constraints(self) -> list[dict[str, Any]]:
        """Deserialize constraints JSON (cached per instance)."""
        ctx = f"CampaignSpec({self.id}).constraints"
        return _safe_json_loads(self.constraints_json, default=[], context=ctx)

    # Backward-compat aliases for code using the old method names
    def get_parameters(self) -> list[dict[str, Any]]:
        return self.parsed_parameters

    def get_objectives(self) -> list[dict[str, Any]]:
        return self.parsed_objectives

    def get_constraints(self) -> list[dict[str, Any]]:
        return self.parsed_constraints


class CampaignModel(Base):
    """Campaign ORM model."""

    __tablename__ = "campaigns"

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

    # Relationships
    spec: Mapped["CampaignSpecModel"] = relationship(back_populates="campaigns")
    owner: Mapped["UserModel"] = relationship(back_populates="campaigns")
    suggestions: Mapped[list["SuggestionModel"]] = relationship(back_populates="campaign")
    results: Mapped[list["ResultModel"]] = relationship(back_populates="campaign")

    @functools.cached_property
    def parsed_turbo_state(self) -> dict[str, Any] | None:
        """Deserialize TuRBO state JSON (cached per instance)."""
        if self.turbo_state_json is None:
            return None
        ctx = f"Campaign({self.id}).turbo_state"
        return _safe_json_loads(self.turbo_state_json, default=None, context=ctx)

    @functools.cached_property
    def parsed_hypervolume_history(self) -> list[float]:
        """Deserialize hypervolume history JSON (cached per instance)."""
        if not self.hypervolume_history_json:
            return []
        ctx = f"Campaign({self.id}).hypervolume_history"
        return _safe_json_loads(self.hypervolume_history_json, default=[], context=ctx)

    def get_turbo_state(self) -> dict[str, Any] | None:
        return self.parsed_turbo_state

    def get_hypervolume_history(self) -> list[float]:
        return self.parsed_hypervolume_history


class SuggestionModel(Base):
    """Suggestion ORM model."""

    __tablename__ = "suggestions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    campaign_id: Mapped[str] = mapped_column(
        String(36), ForeignKey(CAMPAIGNS_ID_FK), nullable=False, index=True
    )
    parameter_values_json: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    status: Mapped[SuggestionStatus] = mapped_column(
        Enum(SuggestionStatus), default=SuggestionStatus.PENDING
    )
    provenance_json: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # Relationships
    campaign: Mapped["CampaignModel"] = relationship(back_populates="suggestions")
    result: Mapped["ResultModel | None"] = relationship(back_populates="suggestion")

    @functools.cached_property
    def parsed_parameter_values(self) -> dict[str, Any]:
        ctx = f"Suggestion({self.id}).parameter_values"
        return _safe_json_loads(self.parameter_values_json, default={}, context=ctx)

    @functools.cached_property
    def parsed_provenance(self) -> dict[str, Any]:
        ctx = f"Suggestion({self.id}).provenance"
        return _safe_json_loads(self.provenance_json, default={}, context=ctx)

    def get_parameter_values(self) -> dict[str, Any]:
        return self.parsed_parameter_values

    def get_provenance(self) -> dict[str, Any]:
        return self.parsed_provenance


class ResultModel(Base):
    """Result ORM model."""

    __tablename__ = "results"
    # Partial-unique index on ``suggestion_id``: at most one Result row can
    # reference any given suggestion. NULL values (free-floating results)
    # are intentionally excluded so manual/imported rows can coexist
    # without forcing them to share a single global NULL slot. Postgres and
    # SQLite both honour the ``WHERE`` clause; the server-side phase-1
    # check (:func:`_validate_suggestion_references`) catches duplicates
    # within a single batch before they reach this constraint.
    __table_args__ = (
        Index(
            "ix_results_suggestion_id_unique",
            "suggestion_id",
            unique=True,
            sqlite_where=text("suggestion_id IS NOT NULL"),
            postgresql_where=text("suggestion_id IS NOT NULL"),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    campaign_id: Mapped[str] = mapped_column(
        String(36), ForeignKey(CAMPAIGNS_ID_FK), nullable=False, index=True
    )
    suggestion_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey(SUGGESTIONS_ID_FK), nullable=True, index=True
    )
    parameter_values_json: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    objective_values_json: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    source: Mapped[ResultSource] = mapped_column(Enum(ResultSource), nullable=False)
    submitted_by: Mapped[str] = mapped_column(String(36), nullable=False)
    measurement_uncertainty_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")  # JSON
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # Relationships
    campaign: Mapped["CampaignModel"] = relationship(back_populates="results")
    suggestion: Mapped["SuggestionModel | None"] = relationship(back_populates="result")

    @functools.cached_property
    def parsed_parameter_values(self) -> dict[str, Any]:
        ctx = f"Result({self.id}).parameter_values"
        return _safe_json_loads(self.parameter_values_json, default={}, context=ctx)

    @functools.cached_property
    def parsed_objective_values(self) -> dict[str, float]:
        ctx = f"Result({self.id}).objective_values"
        return _safe_json_loads(self.objective_values_json, default={}, context=ctx)

    @functools.cached_property
    def parsed_metadata(self) -> dict[str, Any]:
        ctx = f"Result({self.id}).metadata"
        return _safe_json_loads(self.metadata_json, default={}, context=ctx)

    def get_parameter_values(self) -> dict[str, Any]:
        return self.parsed_parameter_values

    def get_objective_values(self) -> dict[str, float]:
        return self.parsed_objective_values

    def get_metadata(self) -> dict[str, Any]:
        return self.parsed_metadata


class EventModel(Base):
    """Audit event ORM model for MCP tool call logging."""

    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    campaign_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey(CAMPAIGNS_ID_FK), nullable=True, index=True
    )
    event_type: Mapped[EventType] = mapped_column(Enum(EventType), default=EventType.TOOL_CALL)
    tool_name: Mapped[str] = mapped_column(String(255), nullable=False)
    input_summary_json: Mapped[str] = mapped_column(Text, default="{}")
    output_summary_json: Mapped[str] = mapped_column(Text, default="{}")
    actor_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
