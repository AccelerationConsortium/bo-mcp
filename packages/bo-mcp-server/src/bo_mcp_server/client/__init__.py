"""Public facade for ``bo-mcp-server``.

Transport layers (FastAPI REST, MCP/stdio) import only from this module.
Internals — :mod:`bo_mcp_server.storage`, :mod:`bo_mcp_server.operations`,
:mod:`bo_mcp_server.domain` (raw) — are NOT public. As long as the
surface defined here is preserved, internal refactors stay invisible to
the transport layers.

The facade re-exports four kinds of entities:

1. **DTOs** — Pydantic / dataclass value objects passed across the
   boundary (``CampaignIntakeInput``, ``ResultSubmissionInput`` …).
2. **Operations** — async callables that implement the per-tool
   business logic; each returns either a structured success payload or
   the error envelope produced by :func:`make_error_response`.
3. **Authorization helpers** — owner-aware lookups
   (:func:`authorize_campaign`, :func:`list_campaign_suggestions` …)
   that raise typed :class:`ClientError` subclasses transports translate
   into HTTP / MCP errors.
4. **Lifecycle hooks** — :func:`init_database`, :func:`ping_database`
   for FastAPI startup / health checks.
"""

from __future__ import annotations

from bo_mcp_server.backend_context import campaign_backend_scope
from bo_mcp_server.client.auth import (
    DEV_API_KEY,
    DEV_USER_EMAIL,
    DEV_USER_NAME,
    AuthenticationConfigurationError,
    ClientError,
    InvalidIdentifierError,
    NotAuthorizedError,
    NotFoundError,
    authorize_campaign,
    authorize_suggestion,
    ensure_dev_user,
    ensure_mcp_startup_user,
    ensure_owned_campaigns,
    get_campaign_spec_by_id,
    get_campaign_with_spec,
    get_spec_for_user,
    get_user_by_api_key,
    list_campaign_results,
    list_campaign_suggestions,
    list_owner_campaigns_with_specs,
    parse_uuid,
    resolve_mcp_user,
)
from bo_mcp_server.client.lifecycle import (
    DatabasePingResult,
    init_database,
    ping_database,
    ping_database_detailed,
)
from bo_mcp_server.constants import MAX_GENERATION_BATCH_SIZE
from bo_mcp_server.domain import (
    AcquisitionMethod,
    AcquisitionOptimizationConfig,
    Campaign,
    CampaignIntakeInput,
    CampaignSpec,
    CampaignStatus,
    Constraint,
    ExternalRef,
    FidelityParameter,
    FiniteFloat,
    InputParameter,
    Objective,
    OutcomeConstraint,
    Result,
    ResultMetadata,
    ResultSource,
    ResultSubmissionInput,
    SaasboConfig,
    Suggestion,
    SuggestionProvenance,
    SuggestionSnapshot,
    SuggestionStatus,
    TransferLearningConfig,
    TurboConfig,
    User,
    field_docs,
)
from bo_mcp_server.errors import (
    ERROR_CODE_TO_HTTP_STATUS,
    ErrorCode,
    http_status_for_error,
    make_corrupted_json_response,
    make_error_response,
)
from bo_mcp_server.idempotency_gc import idempotency_gc_lifespan
from bo_mcp_server.metrics import (
    observe_suggestion_latency,
    record_campaign_created,
    record_diagnostics_cache,
    snapshot_db_pool,
)
from bo_mcp_server.operations.batch_status import batch_get_status_operation
from bo_mcp_server.operations.campaign_lifecycle import (
    manage_campaign_lifecycle_operation,
)
from bo_mcp_server.operations.compare_campaigns import compare_campaigns_operation
from bo_mcp_server.operations.create_campaign import create_campaign_operation
from bo_mcp_server.operations.export_campaign import export_campaign_operation
from bo_mcp_server.operations.generate_suggestions import (
    ensure_canonical_suggestion_keys,
    generate_suggestions_operation,
)
from bo_mcp_server.operations.get_diagnostics import get_diagnostics_operation
from bo_mcp_server.operations.idempotency_wrapper import (
    OperationExecutor,
    canonical_create_campaign_payload,
    canonical_generate_suggestions_payload,
    canonical_submit_results_payload,
    run_idempotent_operation,
)
from bo_mcp_server.operations.list_campaigns import MAX_LIMIT as MAX_CAMPAIGNS_LIST_LIMIT
from bo_mcp_server.operations.list_campaigns import list_campaigns_operation
from bo_mcp_server.operations.list_capabilities import list_capabilities_operation
from bo_mcp_server.operations.list_results import MAX_RESULTS_LIMIT, list_results_operation
from bo_mcp_server.operations.list_suggestions import (
    MAX_SUGGESTIONS_LIMIT,
    list_suggestions_operation,
)
from bo_mcp_server.operations.submit_results import submit_results_operation
from bo_mcp_server.operations.suggestion_explanation import (
    get_suggestion_explanation_operation,
)
from bo_mcp_server.operations.transfer_candidates import (
    discover_transfer_candidates_operation,
)
from bo_mcp_server.operations.update_suggestion_status import (
    update_suggestion_status_operation,
)
from bo_mcp_server.operations.validate_intake import (
    validate_intake_operation,
    validate_intake_with_capabilities,
)
from bo_mcp_server.response_formatter import (
    RESPONSE_SCHEMA_VERSION,
    ValidateIntakeSpecSummary,
    VerbosityLevel,
    format_validate_intake_response,
)
from bo_mcp_server.result_upload_parser import parse_named_result_rows
from bo_mcp_server.schema_extension import (
    augment_backend_options,
    augment_parameter_options,
)
from bo_mcp_server.storage.models import CorruptedJsonColumnError
from bo_mcp_server.trace_context import bind_trace_id, get_trace_id

__all__ = [
    "DEV_API_KEY",
    "DEV_USER_EMAIL",
    "DEV_USER_NAME",
    "ERROR_CODE_TO_HTTP_STATUS",
    # Shared request-size caps (transport-neutral)
    "MAX_CAMPAIGNS_LIST_LIMIT",
    "MAX_GENERATION_BATCH_SIZE",
    "MAX_RESULTS_LIMIT",
    "MAX_SUGGESTIONS_LIMIT",
    "RESPONSE_SCHEMA_VERSION",
    # DTOs
    "AcquisitionMethod",
    "AcquisitionOptimizationConfig",
    # Auth helpers / exceptions
    "AuthenticationConfigurationError",
    "Campaign",
    "CampaignIntakeInput",
    "CampaignSpec",
    "CampaignStatus",
    "ClientError",
    "Constraint",
    # Errors / formatting
    "CorruptedJsonColumnError",
    # Lifecycle
    "DatabasePingResult",
    "ErrorCode",
    "ExternalRef",
    "FidelityParameter",
    "FiniteFloat",
    "InputParameter",
    "InvalidIdentifierError",
    "NotAuthorizedError",
    "NotFoundError",
    "Objective",
    # Idempotency (transport-neutral)
    "OperationExecutor",
    "OutcomeConstraint",
    "Result",
    "ResultMetadata",
    "ResultSource",
    "ResultSubmissionInput",
    "SaasboConfig",
    "Suggestion",
    "SuggestionProvenance",
    "SuggestionSnapshot",
    "SuggestionStatus",
    "TransferLearningConfig",
    "TurboConfig",
    "User",
    "ValidateIntakeSpecSummary",
    "VerbosityLevel",
    "augment_backend_options",
    "augment_parameter_options",
    "authorize_campaign",
    "authorize_suggestion",
    # Operations
    "batch_get_status_operation",
    "bind_trace_id",
    "campaign_backend_scope",
    "canonical_create_campaign_payload",
    "canonical_generate_suggestions_payload",
    "canonical_submit_results_payload",
    "compare_campaigns_operation",
    "create_campaign_operation",
    "discover_transfer_candidates_operation",
    "ensure_canonical_suggestion_keys",
    "ensure_dev_user",
    "ensure_mcp_startup_user",
    "ensure_owned_campaigns",
    "export_campaign_operation",
    "field_docs",
    "format_validate_intake_response",
    "generate_suggestions_operation",
    "get_campaign_spec_by_id",
    "get_campaign_with_spec",
    "get_diagnostics_operation",
    "get_spec_for_user",
    "get_suggestion_explanation_operation",
    "get_trace_id",
    "get_user_by_api_key",
    "http_status_for_error",
    "idempotency_gc_lifespan",
    "init_database",
    "list_campaign_results",
    "list_campaign_suggestions",
    "list_campaigns_operation",
    "list_capabilities_operation",
    "list_owner_campaigns_with_specs",
    "list_results_operation",
    "list_suggestions_operation",
    "make_corrupted_json_response",
    "make_error_response",
    "manage_campaign_lifecycle_operation",
    # Metrics
    "observe_suggestion_latency",
    # File parsing
    "parse_named_result_rows",
    "parse_uuid",
    "ping_database",
    "ping_database_detailed",
    "record_campaign_created",
    "record_diagnostics_cache",
    "resolve_mcp_user",
    "run_idempotent_operation",
    "snapshot_db_pool",
    "submit_results_operation",
    "update_suggestion_status_operation",
    "validate_intake_operation",
    "validate_intake_with_capabilities",
]
