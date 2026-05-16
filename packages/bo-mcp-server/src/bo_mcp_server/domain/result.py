"""Result entity."""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from bo_mcp_server.domain.utils import utcnow


class ResultSource(StrEnum):
    """Source of the result data."""

    GUI = "gui"  # Manual entry via web interface
    FILE_UPLOAD = "file_upload"  # CSV/Excel upload
    API = "api"  # Programmatic submission


class ExternalRef(BaseModel):
    """Pointer to the source system that produced this result.

    Used to round-trip a result back to the lab notebook / LIMS / file
    that produced it. ``system`` and ``id`` are required so the pointer
    is actually resolvable; ``url`` is optional for systems that do not
    expose deep links.
    """

    system: str = Field(..., min_length=1)
    id: str = Field(..., min_length=1)
    url: str | None = None

    model_config = ConfigDict(extra="forbid")


class ResultMetadata(BaseModel):
    """Validated metadata payload for :class:`Result`.

    Replaces the previously-freeform ``dict[str, Any]`` so callers and
    agents can introspect the schema instead of guessing. Each field is
    optional — only the keys a particular call actually carries must be
    present — but unknown keys are rejected with ``ConfigDict(extra=
    "forbid")`` so a misspelled key surfaces at the intake boundary
    rather than being silently dropped on the way to storage.

    The supported keys are:
        external_ref: Pointer to the source system (see ``ExternalRef``).
        conditions: Free-form environmental / equipment context that is
            not captured as a BO parameter (ambient temperature, operator
            initials, equipment tag, etc.). Allowed value types are
            primitives so the blob stays JSON-safe.
        cost: Evaluation cost for cost-aware acquisition (read by
            ``operations.helpers.results_to_observations``).
        experiment_id, operator, batch_ref, notes: human-facing
            audit-trail fields surfaced in the GUI and reports.
        source_row: CSV row number for file-upload results (set by
            ``tools.upload_results_file``).
    """

    external_ref: ExternalRef | None = None
    conditions: dict[str, str | int | float | bool | None] | None = None
    cost: float | None = Field(default=None, ge=0.0)
    experiment_id: str | None = Field(default=None, min_length=1)
    operator: str | None = Field(default=None, min_length=1)
    batch_ref: str | None = Field(default=None, min_length=1)
    notes: str | None = None
    source_row: int | None = Field(default=None, ge=1)

    model_config = ConfigDict(extra="forbid")


class Result(BaseModel):
    """Result entity representing an experimental observation.

    See :class:`ResultMetadata` for the validated metadata schema. The
    in-memory ``metadata`` field is still typed as ``dict[str, Any]`` to
    preserve compatibility with rows persisted before the schema was
    introduced; new writes flow through ``ResultSubmissionInput`` which
    rejects unknown keys at the boundary.
    """

    id: UUID = Field(default_factory=uuid4)
    campaign_id: UUID
    suggestion_id: UUID | None = None  # None for historical/manual data
    parameter_values: dict[str, Any]  # Parameter name -> value
    objective_values: dict[str, float]  # Objective name -> observed value
    source: ResultSource
    submitted_by: UUID  # User who submitted
    measurement_uncertainty: dict[str, float] | None = None  # Per-objective noise estimate (std)
    metadata: dict[str, Any] = Field(default_factory=dict)  # Validated via ResultMetadata at intake
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def is_from_suggestion(self) -> bool:
        """Check if result came from a suggestion."""
        return self.suggestion_id is not None

    def format_objectives(self) -> str:
        """Format objectives for display."""
        parts = []
        for name, value in self.objective_values.items():
            parts.append(f"{name}={value:.4g}")
        return ", ".join(parts)

    def format_parameters(self) -> str:
        """Format parameters for display."""
        parts = []
        for name, value in self.parameter_values.items():
            if isinstance(value, float):
                parts.append(f"{name}={value:.4g}")
            else:
                parts.append(f"{name}={value}")
        return ", ".join(parts)
