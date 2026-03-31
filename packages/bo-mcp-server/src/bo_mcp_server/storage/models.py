"""SQLAlchemy ORM models."""

import functools
import json
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from bo_mcp_server.domain.campaign import CampaignStatus
from bo_mcp_server.domain.event import EventType
from bo_mcp_server.domain.result import ResultSource
from bo_mcp_server.domain.suggestion import SuggestionStatus
from bo_mcp_server.domain.utils import utcnow


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
    initial_design_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    random_seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    # Relationships
    campaigns: Mapped[list["CampaignModel"]] = relationship(back_populates="spec")

    @functools.cached_property
    def parsed_parameters(self) -> list[dict[str, Any]]:
        """Deserialize parameters JSON (cached per instance)."""
        return json.loads(self.parameters_json)

    @functools.cached_property
    def parsed_objectives(self) -> list[dict[str, Any]]:
        """Deserialize objectives JSON (cached per instance)."""
        return json.loads(self.objectives_json)

    @functools.cached_property
    def parsed_constraints(self) -> list[dict[str, Any]]:
        """Deserialize constraints JSON (cached per instance)."""
        return json.loads(self.constraints_json)

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
        return json.loads(self.turbo_state_json)

    @functools.cached_property
    def parsed_hypervolume_history(self) -> list[float]:
        """Deserialize hypervolume history JSON (cached per instance)."""
        if not self.hypervolume_history_json:
            return []
        return json.loads(self.hypervolume_history_json)

    def get_turbo_state(self) -> dict[str, Any] | None:
        return self.parsed_turbo_state

    def get_hypervolume_history(self) -> list[float]:
        return self.parsed_hypervolume_history


class SuggestionModel(Base):
    """Suggestion ORM model."""

    __tablename__ = "suggestions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    campaign_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("campaigns.id"), nullable=False, index=True
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
        return json.loads(self.parameter_values_json)

    @functools.cached_property
    def parsed_provenance(self) -> dict[str, Any]:
        return json.loads(self.provenance_json)

    def get_parameter_values(self) -> dict[str, Any]:
        return self.parsed_parameter_values

    def get_provenance(self) -> dict[str, Any]:
        return self.parsed_provenance


class ResultModel(Base):
    """Result ORM model."""

    __tablename__ = "results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    campaign_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("campaigns.id"), nullable=False, index=True
    )
    suggestion_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("suggestions.id"), nullable=True, index=True
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
        return json.loads(self.parameter_values_json)

    @functools.cached_property
    def parsed_objective_values(self) -> dict[str, float]:
        return json.loads(self.objective_values_json)

    @functools.cached_property
    def parsed_metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata_json)

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
        String(36), ForeignKey("campaigns.id"), nullable=True, index=True
    )
    event_type: Mapped[EventType] = mapped_column(Enum(EventType), default=EventType.TOOL_CALL)
    tool_name: Mapped[str] = mapped_column(String(255), nullable=False)
    input_summary_json: Mapped[str] = mapped_column(Text, default="{}")
    output_summary_json: Mapped[str] = mapped_column(Text, default="{}")
    actor_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
