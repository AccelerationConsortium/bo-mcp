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

from bo_mcp_server.client.auth import (
    DEV_API_KEY,
    DEV_USER_EMAIL,
    DEV_USER_NAME,
    ClientError,
    InvalidIdentifierError,
    NotAuthorizedError,
    NotFoundError,
    authorize_campaign,
    authorize_suggestion,
    ensure_dev_user,
    ensure_owned_campaigns,
    get_campaign_spec_by_id,
    get_campaign_with_spec,
    list_campaign_results,
    list_campaign_suggestions,
    list_owner_campaigns_with_specs,
    parse_uuid,
)
from bo_mcp_server.client.lifecycle import (
    DatabasePingResult,
    init_database,
    ping_database,
    ping_database_detailed,
)
from bo_mcp_server.domain import (
    Campaign,
    CampaignIntakeInput,
    CampaignSpec,
    CampaignStatus,
    Constraint,
    ExternalRef,
    InputParameter,
    Objective,
    Result,
    ResultMetadata,
    ResultSource,
    ResultSubmissionInput,
    Suggestion,
    SuggestionProvenance,
    SuggestionStatus,
    User,
)
from bo_mcp_server.errors import (
    ErrorCode,
    http_status_for_error,
    make_error_response,
)
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
    generate_suggestions_operation,
)
from bo_mcp_server.operations.get_diagnostics import get_diagnostics_operation
from bo_mcp_server.operations.list_campaigns import list_campaigns_operation
from bo_mcp_server.operations.list_capabilities import list_capabilities_operation
from bo_mcp_server.operations.list_results import list_results_operation
from bo_mcp_server.operations.list_suggestions import list_suggestions_operation
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
from bo_mcp_server.operations.validate_intake import validate_intake_operation
from bo_mcp_server.response_formatter import (
    VerbosityLevel,
    format_validate_intake_response,
)
from bo_mcp_server.result_upload_parser import parse_named_result_rows

__all__ = [
    # DTOs
    "Campaign",
    "CampaignIntakeInput",
    "CampaignSpec",
    "CampaignStatus",
    "Constraint",
    "ExternalRef",
    "InputParameter",
    "Objective",
    "Result",
    "ResultMetadata",
    "ResultSource",
    "ResultSubmissionInput",
    "Suggestion",
    "SuggestionProvenance",
    "SuggestionStatus",
    "User",
    # Errors / formatting
    "ErrorCode",
    "VerbosityLevel",
    "format_validate_intake_response",
    "http_status_for_error",
    "make_error_response",
    # Operations
    "batch_get_status_operation",
    "compare_campaigns_operation",
    "create_campaign_operation",
    "discover_transfer_candidates_operation",
    "export_campaign_operation",
    "generate_suggestions_operation",
    "get_diagnostics_operation",
    "get_suggestion_explanation_operation",
    "list_campaigns_operation",
    "list_capabilities_operation",
    "list_results_operation",
    "list_suggestions_operation",
    "manage_campaign_lifecycle_operation",
    "submit_results_operation",
    "update_suggestion_status_operation",
    "validate_intake_operation",
    # Auth helpers / exceptions
    "ClientError",
    "DEV_API_KEY",
    "DEV_USER_EMAIL",
    "DEV_USER_NAME",
    "InvalidIdentifierError",
    "NotAuthorizedError",
    "NotFoundError",
    "authorize_campaign",
    "authorize_suggestion",
    "ensure_dev_user",
    "ensure_owned_campaigns",
    "get_campaign_spec_by_id",
    "get_campaign_with_spec",
    "list_campaign_results",
    "list_campaign_suggestions",
    "list_owner_campaigns_with_specs",
    "parse_uuid",
    # File parsing
    "parse_named_result_rows",
    # Lifecycle
    "DatabasePingResult",
    "init_database",
    "ping_database",
    "ping_database_detailed",
    # Metrics
    "observe_suggestion_latency",
    "record_campaign_created",
    "record_diagnostics_cache",
    "snapshot_db_pool",
]
