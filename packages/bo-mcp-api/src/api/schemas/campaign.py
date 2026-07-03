"""Campaign schemas."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from api.limits import MAX_BATCH_CAMPAIGN_IDS, MAX_COMPARE_CAMPAIGN_IDS
from api.schemas.common import ResponseEnvelope, VerbosityLevel
from api.schemas.intake import IntakeData
from bo_mcp_server.client import ValidateIntakeSpecSummary

# ``extra="forbid"`` is applied to every request schema in this module so
# an unknown field — typically a typo or a not-yet-supported key — raises
# 422 at the transport boundary instead of being silently dropped.
#
# Response schemas do not forbid extras (the MCP formatter splices a
# ``_metadata`` envelope into every payload). But the Pydantic default
# ``extra="ignore"`` still *drops* unknown keys, so a response whose
# operation returns verbosity-/backend-dependent passthrough keys must
# opt into ``extra="allow"`` (``CompareCampaignsResponse`` /
# ``TransferCandidatesResponse`` / ``BatchStatusResponse`` here, plus
# ``DiagnosticsResponse``) to forward them — and ``_metadata`` —
# verbatim; its route pairs that with
# ``response_model_exclude_unset=True`` so a verbosity tier that omits a
# declared key is echoed exactly (REST stays byte-equal to MCP).
_FORBID_EXTRA: ConfigDict = ConfigDict(extra="forbid")

LifecycleAction = Literal["pause", "resume", "terminate"]


class CampaignCreate(BaseModel):
    """Campaign creation request."""

    model_config = _FORBID_EXTRA

    intake: IntakeData


class CampaignResponse(BaseModel):
    """Campaign response schema."""

    id: str
    spec_id: str
    name: str
    description: str
    status: str
    iteration: int
    created_at: datetime
    updated_at: datetime
    n_parameters: int
    n_objectives: int


class CampaignConfigResponse(BaseModel):
    """Stable campaign setup snapshot for reproducibility/provenance."""

    campaign_id: str
    spec_id: str
    name: str
    description: str
    status: str
    iteration: int
    backend_requested: str | None = None
    backend_resolved: str | None = None
    batch_size: int
    max_iterations: int | None = None
    max_observations: int | None = None
    initial_design_size_requested: int | None = None
    initial_design_size: int | None = None
    initial_design_size_source: str | None = None
    random_seed: int | None = None
    convergence_tolerance: float | None = None
    parameters: list[dict[str, Any]] = Field(default_factory=list)
    objectives: list[dict[str, Any]] = Field(default_factory=list)
    constraints: list[dict[str, Any]] = Field(default_factory=list)
    outcome_constraints: list[dict[str, Any]] = Field(default_factory=list)
    acquisition_method: str | None = None
    acquisition_optimization: dict[str, Any] | None = None
    use_input_warping: bool
    use_cost_aware: bool
    turbo_config: dict[str, Any] | None = None
    saasbo_config: dict[str, Any] | None = None
    fidelity_parameter: dict[str, Any] | None = None
    transfer_learning: dict[str, Any] | None = None
    backend_options: dict[str, dict[str, Any]] | None = None
    acknowledge_degradations: list[str] = Field(default_factory=list)


class CampaignCreateResponse(ResponseEnvelope):
    """Campaign creation response.

    ``idempotency_replay`` is ``True`` when the response was served
    from the idempotency cache instead of executing a fresh
    mutation — same marker the MCP tool exposes. REST clients can
    distinguish a network retry's replayed response from a brand-new
    create and surface the distinction to their users (e.g. "Already
    created earlier, here's the same id").
    """

    success: bool
    campaign_id: str | None = None
    spec_id: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str]
    idempotency_replay: bool = False


class CampaignListResponse(ResponseEnvelope):
    """Campaign list response."""

    campaigns: list[CampaignResponse]
    total: int


class ValidateIntakeRequest(BaseModel):
    """Intake validation request (dry-run, no campaign created)."""

    model_config = _FORBID_EXTRA

    intake: IntakeData


class ValidateIntakeResponse(ResponseEnvelope):
    """Intake validation response."""

    valid: bool
    errors: list[str]
    warnings: list[str] = Field(default_factory=list)
    spec_summary: ValidateIntakeSpecSummary | None = None


class CapabilitiesResponse(ResponseEnvelope):
    """Backend capabilities response.

    ``supported_features`` lists features the backend can honour for
    *any* well-formed spec; ``conditional_features`` maps each
    feature that depends on spec shape to a short description of the
    precondition (e.g. BayBE's TRANSFER_LEARNING requires a
    TaskParameter). Together the two surfaces match the runtime
    contract so callers can plan ahead instead of hitting late
    rejections.
    """

    backend: str
    supported_features: list[str]
    conditional_features: dict[str, str] = Field(default_factory=dict)
    server_version: str


class CampaignQueryRequest(BaseModel):
    """Campaign query request with filtering and pagination.

    Pagination model: cursor-only is the supported path. The legacy
    ``offset`` field is preserved for callers that have not migrated
    but is marked ``deprecated`` so OpenAPI clients and the auto-
    generated docs surface the deprecation; it is mutually exclusive
    with ``cursor`` at the operation layer (supplying both yields a
    ``VALIDATION_FAILED`` envelope).
    """

    model_config = _FORBID_EXTRA

    status: str | None = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(
        default=0,
        ge=0,
        deprecated=(
            "Offset-based pagination is unstable under concurrent inserts. "
            "Use the cursor returned in next_cursor instead."
        ),
    )
    cursor: str | None = Field(
        default=None,
        description=(
            "Opaque cursor from a previous response's next_cursor field. "
            "Cursor-based pagination is stable under concurrent inserts. "
            "Mutually exclusive with offset."
        ),
    )
    verbosity: VerbosityLevel = VerbosityLevel.STANDARD


class CampaignQueryResponse(ResponseEnvelope):
    """Campaign query response with pagination envelope.

    ``next_cursor`` carries the opaque pagination pointer for the next
    page. ``offset`` is echoed back for callers still on the deprecated
    pagination model.
    """

    success: bool
    campaigns: list[dict[str, Any]] = Field(default_factory=list)
    total_count: int = 0
    limit: int = 20
    offset: int = 0
    next_cursor: str | None = None
    errors: list[str] = Field(default_factory=list)


class CampaignLifecycleRequest(BaseModel):
    """Lifecycle action request."""

    model_config = _FORBID_EXTRA

    action: LifecycleAction = Field(
        description=(
            'Lifecycle action to apply. Use "terminate" to end or complete a '
            'campaign; there is no separate "complete" action.'
        ),
        examples=["pause", "resume", "terminate"],
    )


class CampaignLifecycleResponse(ResponseEnvelope):
    """Lifecycle action response."""

    success: bool
    campaign_id: str
    status: str | None = None
    previous_status: str | None = None
    errors: list[str] = Field(default_factory=list)


class BatchStatusRequest(BaseModel):
    """Batch status request.

    ``campaign_ids`` is bounded by
    :data:`api.limits.MAX_BATCH_CAMPAIGN_IDS` to keep the read-only
    fan-out from being weaponised into a memory-heavy lookup storm.
    """

    model_config = _FORBID_EXTRA

    campaign_ids: list[str] = Field(..., min_length=1, max_length=MAX_BATCH_CAMPAIGN_IDS)
    verbosity: VerbosityLevel = VerbosityLevel.MINIMAL


class BatchStatusResponse(ResponseEnvelope):
    """Batch status response.

    The top-level shape is verbosity-stable (verbosity only varies the
    per-campaign values nested under ``campaigns``), so — unlike compare
    / transfer — this model is not tier-mismatched. ``extra="allow"`` is
    still required to forward the ``_metadata`` envelope the shared
    operation attaches (via ``with_response_metadata``); the route pairs
    it with ``response_model_exclude_unset=True`` so an error envelope —
    which omits ``campaigns`` / ``failed_ids`` — is not padded with empty
    defaults, keeping the body byte-equal to the MCP tool output.
    """

    model_config = ConfigDict(extra="allow")

    success: bool
    campaigns: dict[str, dict[str, Any]] = Field(default_factory=dict)
    failed_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class CompareCampaignsRequest(BaseModel):
    """Campaign comparison request.

    ``campaign_ids`` is bounded by
    :data:`api.limits.MAX_COMPARE_CAMPAIGN_IDS` because pairwise
    trajectory joins inside the comparison operation are quadratic
    in the number of supplied campaigns.
    """

    model_config = _FORBID_EXTRA

    campaign_ids: list[str] = Field(..., min_length=1, max_length=MAX_COMPARE_CAMPAIGN_IDS)
    verbosity: VerbosityLevel = VerbosityLevel.STANDARD


class CompareCampaignsResponse(ResponseEnvelope):
    """Campaign comparison response.

    The declared fields are the union of the minimal tier
    (``n_campaigns`` / ``best_performer`` / ``recommendation``) and the
    standard/detailed tier (``campaigns`` / ``comparison``), so one model
    serves every verbosity. ``extra="allow"`` forwards the ``_metadata``
    envelope the MCP formatter attaches; the route pairs it with
    ``response_model_exclude_unset=True`` so the other tier's
    declared-but-absent fields are not re-added as defaults — keeping the
    body byte-equal to the MCP tool output.
    """

    model_config = ConfigDict(extra="allow")

    success: bool
    campaigns: list[dict[str, Any]] = Field(default_factory=list)
    comparison: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)
    n_campaigns: int | None = None
    best_performer: str | None = None
    recommendation: str | None = None


class TransferCandidatesRequest(BaseModel):
    """Transfer candidate discovery request.

    ``parameter_aliases`` mirrors the MCP tool's optional alias map so
    REST callers can bridge parameter-name drift across campaigns
    (e.g. ``{"temperature": ["temp_c", "temp_celsius"]}`` unifies all
    three names when computing parameter-set and bounds overlap).
    """

    model_config = _FORBID_EXTRA

    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_candidates: int = Field(default=5, ge=1)
    verbosity: VerbosityLevel = VerbosityLevel.STANDARD
    parameter_aliases: dict[str, list[str]] | None = None


class TransferCandidatesResponse(ResponseEnvelope):
    """Transfer candidate discovery response.

    The declared fields cover the standard/detailed tier; the minimal
    tier's keys (``n_candidates`` / ``top_candidate_id`` /
    ``top_similarity`` / ``recommendation``) and the ``_metadata``
    envelope ride through via ``extra="allow"`` instead of being
    dropped. The route pairs this with
    ``response_model_exclude_unset=True`` so the standard-tier fields the
    minimal projection omits are not re-added as defaults — keeping the
    body byte-equal to the MCP tool output.
    """

    model_config = ConfigDict(extra="allow")

    success: bool
    target_campaign: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    overall_recommendation: str | None = None
    errors: list[str] = Field(default_factory=list)
    n_candidates: int | None = None
    top_candidate_id: str | None = None
    top_similarity: float | None = None
    recommendation: str | None = None
