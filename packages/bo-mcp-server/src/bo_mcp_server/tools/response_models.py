"""Response models for MCP tools that return raw dicts (no per-verbosity formatter).

Every tool used to declare ``-> dict[str, Any]``, which FastMCP renders as
``outputSchema = {"type": "object", "additionalProperties": true}`` -- no
information about the real response shape. These models give FastMCP real
field information to advertise instead.

All models use ``extra="allow"``: a given tool's response shape varies by
verbosity, dry-run vs. committed, and success vs. failure, and the models
here deliberately do not attempt to enumerate every branch precisely.
``extra="allow"`` guarantees the declared return type can never reject an
actual response at the FastMCP conversion boundary (``ToolManager.call_tool``
-> ``fn_metadata.convert_result``, which validates the return value against
the declared type) regardless of which branch produced it -- a stricter
model (``extra="forbid"``) would risk turning a previously-successful tool
call into a validation error the moment an undeclared key showed up (e.g. a
verbosity-only field, or the ``schema_version``/``_metadata`` keys every
response carries). Top-level fields are ``None``-defaulted so every
documented key is still visible in ``outputSchema`` for discovery.

``bo_list_campaigns``/``bo_list_suggestions``/``bo_list_results`` additionally
type their list items as ``CampaignSummaryItem``/``SuggestionSummaryItem``/
``ResultSummaryItem`` -- merged supersets of the MINIMAL/STANDARD/DETAILED
per-item projections built by the corresponding operation's
``_serialize_*``/``_build_*_summary`` helper, so ``outputSchema.properties
.<list>.items.properties`` advertises real field names instead of
``additionalProperties: true``. Fields that are themselves open-ended value
maps keyed by user-defined parameter/objective names (``parameter_values``,
``objective_values``, ``spec_summary``, ``metadata``, ...) stay
``dict[str, Any]`` deliberately -- there is no fixed schema for those
regardless of effort, since the keys come from the campaign's own spec.
The remaining tools (``bo_batch_get_status``, ``bo_export_campaign``, dry-run
``preview`` blocks, ...) still return untyped item/nested bodies; extending
this pattern to them is a follow-up.

Tool wrappers must annotate their return type as one of these models
alone, never as ``SomeModel | ErrorEnvelope``: FastMCP only exposes
``structuredContent`` unwrapped at the top level when the return
annotation resolves to a single object-schema type. A union of two
object schemas has no single top-level object shape, so FastMCP falls
back to wrapping the whole result under a ``{"result": ...}`` key --
silently breaking every caller that reads e.g. ``response["success"]``
directly. ``extra="allow"`` on the single model already accepts the
differently-shaped error envelope (``schema_version``/``error`` land as
allowed extra keys), so the union added no validation value while
triggering that wrapping.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _PermissiveResponse(BaseModel):
    """Base for tool-level response models: always tolerates unknown keys."""

    model_config = ConfigDict(extra="allow")


class CampaignSummaryItem(_PermissiveResponse):
    """One ``campaigns[]`` entry from ``bo_list_campaigns``.

    Superset of the MINIMAL / STANDARD / DETAILED per-item projections
    built by ``operations.list_campaigns._build_{minimal,standard,detailed}_summary``
    -- every field beyond ``campaign_id``/``name``/``status`` (the MINIMAL
    shape) is ``None`` unless the caller requested a richer verbosity.
    """

    campaign_id: str | None = None
    name: str | None = None
    status: str | None = None
    iteration: int | None = None
    n_results: int | None = None
    created_at: str | None = None
    owner_id: str | None = None
    spec_id: str | None = None
    spec_summary: dict[str, Any] | None = None
    has_hypervolume_history: bool | None = None
    has_backend_state: bool | None = None


class SuggestionSummaryItem(_PermissiveResponse):
    """One ``suggestions[]`` entry from ``bo_list_suggestions``.

    Superset of the MINIMAL / STANDARD / DETAILED per-item projections
    built by ``operations.list_suggestions._serialize_suggestion``.
    """

    suggestion_id: str | None = None
    status: str | None = None
    parameter_values: dict[str, Any] | None = None
    iteration: int | None = None
    generation_method: str | None = None
    created_at: str | None = None
    batch_index: int | None = None
    acquisition_function: str | None = None
    acquisition_value: float | None = None
    model_uncertainty: float | None = None
    model_type: str | None = None
    confidence_level: float | None = None
    predicted_objectives: dict[str, Any] | None = None
    predicted_std: dict[str, Any] | None = None
    updated_at: str | None = None


class ResultSummaryItem(_PermissiveResponse):
    """One ``results[]`` entry from ``bo_list_results``.

    Superset of the MINIMAL / STANDARD / DETAILED per-item projections
    built by ``operations.list_results._serialize_result``.
    """

    result_id: str | None = None
    objective_values: dict[str, Any] | None = None
    parameter_values: dict[str, Any] | None = None
    suggestion_id: str | None = None
    created_at: str | None = None
    source: str | None = None
    submitted_by: str | None = None
    measurement_uncertainty: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class CampaignListResponse(_PermissiveResponse):
    """Response shape for ``bo_list_campaigns``."""

    success: bool | None = None
    campaigns: list[CampaignSummaryItem] = Field(default_factory=list)
    total_count: int | None = None
    limit: int | None = None
    offset: int | None = None
    next_cursor: str | None = None
    errors: list[str] = Field(default_factory=list)


class SuggestionListResponse(_PermissiveResponse):
    """Response shape for ``bo_list_suggestions``."""

    success: bool | None = None
    suggestions: list[SuggestionSummaryItem] = Field(default_factory=list)
    total_count: int | None = None
    limit: int | None = None
    offset: int | None = None
    next_cursor: str | None = None
    errors: list[str] = Field(default_factory=list)


class ResultListResponse(_PermissiveResponse):
    """Response shape for ``bo_list_results``."""

    success: bool | None = None
    results: list[ResultSummaryItem] = Field(default_factory=list)
    total_count: int | None = None
    limit: int | None = None
    offset: int | None = None
    next_cursor: str | None = None
    errors: list[str] = Field(default_factory=list)


class ExportCampaignResponse(_PermissiveResponse):
    """Response shape for ``bo_export_campaign``."""

    success: bool | None = None
    format: str | None = None
    content: str | None = None
    campaign_name: str | None = None
    n_results: int | None = None
    n_results_included: int | None = None
    truncated: bool | None = None
    errors: list[str] = Field(default_factory=list)


class CapabilitiesResponse(_PermissiveResponse):
    """Response shape for ``bo_list_capabilities``."""

    backend: str | None = None
    supported_features: list[str] = Field(default_factory=list)
    conditional_features: dict[str, str] = Field(default_factory=dict)
    available_backends: list[str] = Field(default_factory=list)
    default_backend: str | None = None
    server_version: str | None = None


class HealthCheckResponse(_PermissiveResponse):
    """Response shape for ``bo_health_check``."""

    healthy: bool | None = None
    version: str | None = None
    database: str | None = None
    tools_available: int | None = None
    uptime_seconds: int | None = None
    backends: dict[str, Any] = Field(default_factory=dict)


class BatchStatusResponse(_PermissiveResponse):
    """Response shape for ``bo_batch_get_status``."""

    success: bool | None = None
    campaigns: dict[str, dict[str, Any]] = Field(default_factory=dict)
    failed_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class SuggestionExplanationResponse(_PermissiveResponse):
    """Response shape for ``bo_get_suggestion_explanation``."""

    success: bool | None = None
    explanation: str | None = None
    provenance: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)


class CampaignLifecycleToolResponse(_PermissiveResponse):
    """Response shape shared by the lifecycle tools.

    Covers ``bo_pause_campaign``, ``bo_resume_campaign``,
    ``bo_terminate_campaign``, ``bo_reopen_campaign`` -- the committed
    response, the ``noop`` retry-safe response (see the pause/resume
    idempotency fix), and the ``dry_run`` preview all share this shape.
    """

    success: bool | None = None
    campaign_id: str | None = None
    status: str | None = None
    previous_status: str | None = None
    noop: bool | None = None
    dry_run: bool | None = None
    preview: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)


class CheckProgressResponse(_PermissiveResponse):
    """Response shape for ``bo_check_progress``."""

    tracked: bool | None = None
    campaign_id: str | None = None
    phase: str | None = None
    message: str | None = None
    progress: float | None = None
    total: float | None = None
    failures: int | None = None
    last_failure_reason: str | None = None


class SuggestionStatusUpdateResponse(_PermissiveResponse):
    """Response shape for ``bo_update_suggestion_status``."""

    success: bool | None = None
    suggestion_id: str | None = None
    status: str | None = None
    previous_status: str | None = None
    dry_run: bool | None = None
    preview: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)


class UploadResultsResponse(_PermissiveResponse):
    """Response shape for ``bo_upload_results_file``."""

    success: bool | None = None
    results_created: int | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    duplicates_detected: list[dict[str, Any]] = Field(default_factory=list)
    dry_run: bool | None = None
    preview: dict[str, Any] | None = None


class DiscoverTransferCandidatesResponse(_PermissiveResponse):
    """Response shape for ``bo_discover_transfer_candidates``.

    Covers both the MINIMAL projection (``n_candidates``,
    ``top_candidate_id``, ``top_similarity``, ``recommendation``) and
    the STANDARD/DETAILED projection (``target_campaign``,
    ``candidates``, ``overall_recommendation``) -- which projection a
    given call returns depends on the caller's ``verbosity`` argument.
    """

    success: bool | None = None
    n_candidates: int | None = None
    top_candidate_id: str | None = None
    top_similarity: float | None = None
    recommendation: str | None = None
    target_campaign: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    overall_recommendation: str | None = None
    errors: list[str] = Field(default_factory=list)


class CreateCampaignResponse(_PermissiveResponse):
    """Response shape for ``bo_create_campaign``.

    Covers the MINIMAL/STANDARD/DETAILED projections (``campaign_id``,
    ``spec_id``, ``campaign_name``), the ``dry_run`` preview, and the
    pre-identity ``field_errors`` validation envelope.
    """

    success: bool | None = None
    campaign_id: str | None = None
    spec_id: str | None = None
    campaign_name: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    field_errors: dict[str, list[str]] = Field(default_factory=dict)
    dry_run: bool | None = None
    preview: dict[str, Any] | None = None


class SubmitResultsResponse(_PermissiveResponse):
    """Response shape for ``bo_submit_results``.

    Covers the MINIMAL projection (``n_submitted``), the STANDARD/
    DETAILED projections (``result_ids``, ``duplicates_detected`` /
    ``n_duplicates_detected``), and the ``dry_run`` preview.
    """

    success: bool | None = None
    n_submitted: int | None = None
    result_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    field_errors: dict[str, list[str]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    n_duplicates_detected: int | None = None
    duplicates_detected: list[dict[str, Any]] = Field(default_factory=list)
    dry_run: bool | None = None
    preview: dict[str, Any] | None = None


class ValidateIntakeResponse(_PermissiveResponse):
    """Response shape for ``bo_validate_intake``.

    Covers the MINIMAL projection (``valid``, ``errors``), the
    STANDARD projection (adds ``spec_summary``), and the DETAILED
    projection (adds the full ``spec``).
    """

    valid: bool | None = None
    errors: list[str] = Field(default_factory=list)
    field_errors: dict[str, list[str]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    spec_summary: dict[str, Any] | None = None
    spec: dict[str, Any] | None = None


class GetDiagnosticsResponse(_PermissiveResponse):
    """Response shape for ``bo_get_diagnostics``.

    Covers the MINIMAL projection (``status``, ``health``,
    ``progress``, ``key_metric``, ``converged``) and the STANDARD/
    DETAILED projection (``campaign_status``, ``n_pending_suggestions``,
    plus backend-specific per-section blocks admitted via
    ``extra="allow"``) -- which projection a given call returns
    depends on the caller's ``verbosity`` argument and requested
    ``sections``.
    """

    success: bool | None = None
    status: str | None = None
    campaign_status: str | None = None
    iteration: int | None = None
    n_results: int | None = None
    n_pending_suggestions: int | None = None
    health: str | None = None
    progress: str | None = None
    key_metric: dict[str, Any] = Field(default_factory=dict)
    converged: bool | None = None
    next_action: str | dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class GenerateSuggestionsResponse(_PermissiveResponse):
    """Response shape for ``bo_generate_suggestions``.

    Covers the MINIMAL projection (``suggestion_ids``, ``method``) and
    the STANDARD/DETAILED projection (full ``suggestions`` list,
    ``method_selection``), plus the ``dry_run`` preview.
    """

    success: bool | None = None
    suggestions: list[dict[str, Any]] = Field(default_factory=list)
    suggestion_ids: list[str | None] = Field(default_factory=list)
    iteration: int | None = None
    method: str | None = None
    method_selection: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    dry_run: bool | None = None
    preview: dict[str, Any] | None = None
