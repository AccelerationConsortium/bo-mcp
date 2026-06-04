"""Structured error system for MCP tools.

Provides error codes with recovery actions to help agents handle failures.
Each error includes a machine-readable code, human-readable message, and
actionable recovery guidance.

Usage:
    from bo_mcp_server.errors import ErrorCode, make_error_response

    # Simple error:
    return make_error_response(ErrorCode.CAMPAIGN_NOT_FOUND)

    # With custom message and details:
    return make_error_response(
        ErrorCode.CAMPAIGN_NOT_FOUND,
        message=f"Campaign {campaign_id} not found",
        details={"campaign_id": campaign_id},
    )
"""

import json
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal, NoReturn

from bo_engine.backend_base import (
    BackendError,
    BackendIncompatibilityError,
    BackendInputError,
    BackendInternalError,
    BackendTransientError,
)
from pydantic import BaseModel, ConfigDict, model_serializer

from bo_mcp_server.constants import CONCURRENT_MODIFICATION_RETRY_AFTER_SECONDS

if TYPE_CHECKING:
    from bo_mcp_server.storage.base import ConcurrentModificationError
    from bo_mcp_server.storage.models import CorruptedJsonColumnError

# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------


class OperationError(Exception):
    """Base exception for operation-layer errors.

    Carries an ``ErrorCode`` and optional details so that the transport
    layers (MCP tools, HTTP routes) can convert to the appropriate
    response format without re-interpreting strings.
    """

    def __init__(
        self,
        code: "ErrorCode",
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record the structured error code, human-readable message and detail dict."""
        self.code = code
        self.error_message = message  # avoid shadowing Exception.args
        self.details = details
        super().__init__(message or "")

    def to_response(self) -> dict[str, Any]:
        """Convert to the standard MCP error response dict."""
        return make_error_response(self.code, message=self.error_message, details=self.details)


class ValidationError(OperationError):
    """Raised for input validation failures (bad UUIDs, invalid enums, etc.)."""


class CampaignNotFoundError(OperationError):
    """Raised when a campaign is not found."""

    def __init__(self, campaign_id: str) -> None:
        """Build a CAMPAIGN_NOT_FOUND error for the given campaign id."""
        super().__init__(
            ErrorCode.CAMPAIGN_NOT_FOUND,
            message=f"Campaign {campaign_id} not found",
            details={"campaign_id": campaign_id},
        )


class InvalidStateError(OperationError):
    """Raised when a campaign is in the wrong state for the requested operation."""


class ErrorCode(StrEnum):
    """Error codes for MCP tool failures.

    Codes are organized by category:
    - E0xx: Validation errors (input, state, format issues)
    - E1xx: Processing errors (model fitting, database, computation)
    """

    # Validation errors (E0xx)
    INVALID_CAMPAIGN_ID = "E001"
    CAMPAIGN_NOT_FOUND = "E002"
    INVALID_STATE_TRANSITION = "E003"
    DUPLICATE_RESULT = "E004"
    VALIDATION_FAILED = "E005"
    MISSING_PARAMETERS = "E006"
    MISSING_OBJECTIVES = "E007"
    CONSTRAINT_VIOLATION = "E008"
    SUGGESTION_NOT_FOUND = "E009"
    CONCURRENT_MODIFICATION = "E010"
    SEARCH_SPACE_EXHAUSTED = "E011"
    BUDGET_EXCEEDED = "E012"
    CAMPAIGN_CONVERGED = "E013"
    IDEMPOTENCY_IN_PROGRESS = "E014"
    IDEMPOTENCY_CONFLICT = "E015"

    # Processing errors (E1xx)
    MODEL_FITTING_FAILED = "E101"
    ACQUISITION_OPTIMIZATION_FAILED = "E102"
    DATABASE_ERROR = "E103"
    INSUFFICIENT_DATA = "E104"
    BACKEND_TRANSIENT_ERROR = "E105"
    BACKEND_INCOMPATIBILITY = "E106"
    BACKEND_INTERNAL_ERROR = "E107"
    # Persisted data is unreadable / structurally invalid (e.g. a JSON
    # column that no longer decodes). Distinct from DATABASE_ERROR
    # because the failure is deterministic — the row stays broken until
    # an operator repairs it, so retrying the same request is a waste.
    # Mapped from CorruptedJsonColumnError at every transport boundary.
    DATA_INTEGRITY_ERROR = "E108"
    # Catch-all for unexpected transport-layer errors. The REST global
    # exception handler maps any exception not already classified by an
    # operation into this code so the client always sees the same
    # structured envelope shape — never a raw exception string.
    INTERNAL_ERROR = "E199"


class StructuredError(BaseModel):
    """Structured error with recovery guidance.

    Attributes:
        code: Error code from ErrorCode enum.
        message: Human-readable error description.
        recovery_action: Actionable guidance for the agent.
        retryable: Whether retrying the same request has a real
            chance of succeeding. Clients (and HTTP retry middleware)
            use this to decide between exponential backoff and a
            fail-fast surface to the user.
        retry_after: Suggested seconds to wait before retrying when
            ``retryable`` is True; ``None`` for terminal failures or
            when no specific backoff applies.
        details: Optional additional context about the error.
    """

    model_config = ConfigDict(extra="forbid")

    code: ErrorCode
    message: str
    recovery_action: str
    retryable: bool = False
    retry_after: float | None = None
    details: dict[str, Any] | None = None

    @model_serializer
    def _serialize(self) -> dict[str, Any]:
        """Render the wire shape clients depend on.

        ``retry_after`` is always present (``None`` for terminal
        failures); ``details`` is omitted entirely when empty rather than
        serialized as ``null`` — preserving the historical ``to_dict``
        contract. ``code`` is emitted as its string value.
        """
        result: dict[str, Any] = {
            "code": self.code.value,
            "message": self.message,
            "recovery_action": self.recovery_action,
            "retryable": self.retryable,
            "retry_after": self.retry_after,
        }
        if self.details:
            result["details"] = self.details
        return result

    def to_dict(self) -> dict[str, Any]:
        """Backward-compatible alias for :meth:`model_dump`."""
        return self.model_dump()


class ErrorEnvelope(BaseModel):
    """The standard MCP / REST error response envelope.

    Models the ``{schema_version, success, error, errors}`` shape that
    :func:`make_error_response` produces — the error-path analogue of the
    per-operation success models in
    :mod:`bo_mcp_server.response_formatter`. It is built once at the
    boundary and dumped to a plain ``dict`` so the downstream consumers
    that mutate it in place, persist it as JSON, and duck-type it
    (idempotency replay, :func:`http_status_for_error`, field-error
    splicing) keep operating on a dict unchanged.

    ``extra="allow"`` so operation-layer callers can splice
    operation-specific keys (``field_errors``, ``result_ids``, …) onto
    the envelope without them being rejected on any future
    re-validation; ``error.details`` stays an open ``dict`` for the same
    reason.
    """

    model_config = ConfigDict(extra="allow")

    schema_version: int
    success: Literal[False] = False
    error: StructuredError
    errors: list[str]


# Error recovery action catalog
# Maps each error code to a specific, actionable recovery instruction
ERROR_RECOVERY: dict[ErrorCode, str] = {
    ErrorCode.INVALID_CAMPAIGN_ID: (
        "Verify campaign_id is a valid UUID v4. Use campaigns://list resource to get valid IDs."
    ),
    ErrorCode.CAMPAIGN_NOT_FOUND: (
        "Use campaigns://list resource to verify campaign exists and get correct ID."
    ),
    ErrorCode.INVALID_STATE_TRANSITION: (
        "Check current status with campaign://{id} resource. "
        "Valid transitions: CREATED->RUNNING (on first suggestion), "
        "RUNNING->PAUSED (pause), PAUSED->RUNNING (resume), "
        "CREATED/RUNNING/PAUSED->COMPLETED (terminate)."
    ),
    ErrorCode.DUPLICATE_RESULT: (
        "Use force=True parameter to override duplicate detection, or skip this result."
    ),
    ErrorCode.VALIDATION_FAILED: (
        "Review the errors array, fix the issues, and retry bo_create_campaign."
    ),
    ErrorCode.MISSING_PARAMETERS: (
        "Add at least one parameter to the intake_data.parameters array."
    ),
    ErrorCode.MISSING_OBJECTIVES: (
        "Add at least one objective to the intake_data.objectives array."
    ),
    ErrorCode.CONSTRAINT_VIOLATION: (
        "Check that constraint parameters exist and bounds are valid."
    ),
    ErrorCode.SUGGESTION_NOT_FOUND: (
        "Verify suggestion_id is correct. "
        "Use bo_list_suggestions to list suggestions for the campaign."
    ),
    ErrorCode.CONCURRENT_MODIFICATION: (
        "Fetch the current campaign state via campaign://{id} or "
        "bo_list_campaigns to get the latest version, then retry the operation. "
        "Wait retry_after_seconds before retrying to avoid re-hitting the race."
    ),
    ErrorCode.SEARCH_SPACE_EXHAUSTED: (
        "Use bo_terminate_campaign: the finite (typically purely-categorical) "
        "search space has no unseen combinations left, so further experiments "
        "cannot provide new information. Verify via campaign://{id} before "
        "closing out."
    ),
    ErrorCode.BUDGET_EXCEEDED: (
        "Campaign reached its configured iteration or observation budget. "
        "Use bo_terminate_campaign to close it out, or increase the budget "
        "via a new campaign spec."
    ),
    ErrorCode.CAMPAIGN_CONVERGED: (
        "Use bo_terminate_campaign to accept the current best solution; "
        "recent improvement is below convergence_tolerance. "
        "Review the search space and submit a fresh campaign if a better "
        "solution is plausible."
    ),
    ErrorCode.IDEMPOTENCY_IN_PROGRESS: (
        "Another retry with the same idempotency_key is currently executing "
        "the operation. Wait a few hundred milliseconds and retry with the "
        "same key — the cached response will be available once the original "
        "call finishes. Do not reuse the key for a different payload."
    ),
    ErrorCode.IDEMPOTENCY_CONFLICT: (
        "The supplied idempotency_key was previously used with a different "
        "payload. Generate a fresh key for this distinct request, or look up "
        "the prior response that was cached against the existing key."
    ),
    ErrorCode.MODEL_FITTING_FAILED: (
        "Check data quality with bo_get_diagnostics. May need more observations (minimum 2)."
    ),
    ErrorCode.ACQUISITION_OPTIMIZATION_FAILED: (
        "Try reducing batch_size or check for constraint conflicts."
    ),
    ErrorCode.DATABASE_ERROR: (
        "Retry the operation. If persistent, check DATABASE_URL configuration."
    ),
    ErrorCode.DATA_INTEGRITY_ERROR: (
        "Do not retry; the persisted row is malformed and will keep failing. "
        "Use the request_id from details to locate the offending row in the "
        "server log and repair it manually or restore from backup."
    ),
    ErrorCode.INSUFFICIENT_DATA: (
        "Submit more results before generating suggestions. "
        "Need at least 2 observations for model fitting."
    ),
    ErrorCode.BACKEND_TRANSIENT_ERROR: (
        "Retry the request after the suggested backoff; the backend "
        "reported a transient failure (numerical hiccup, optimizer flake, "
        "or temporary resource pressure)."
    ),
    ErrorCode.BACKEND_INCOMPATIBILITY: (
        "The selected backend cannot handle this spec. Switch to "
        "backend='auto' or 'botorch', or remove the unsupported feature "
        "from the spec and resubmit."
    ),
    ErrorCode.BACKEND_INTERNAL_ERROR: (
        "An unexpected backend error occurred. Review the cause field "
        "in details, retry once to confirm reproducibility, and report "
        "persistent failures to the operator."
    ),
    ErrorCode.INTERNAL_ERROR: (
        "Retry the request once to confirm the failure is reproducible. "
        "If it persists, report the issue and quote the request_id "
        "from details — the server log records the full exception "
        "under that id."
    ),
}


# Default error messages for each error code
DEFAULT_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.INVALID_CAMPAIGN_ID: "Invalid campaign_id format",
    ErrorCode.CAMPAIGN_NOT_FOUND: "Campaign not found",
    ErrorCode.INVALID_STATE_TRANSITION: "Invalid campaign state for this operation",
    ErrorCode.DUPLICATE_RESULT: "Duplicate result detected",
    ErrorCode.VALIDATION_FAILED: "Intake validation failed",
    ErrorCode.MISSING_PARAMETERS: "At least one parameter is required",
    ErrorCode.MISSING_OBJECTIVES: "At least one objective is required",
    ErrorCode.CONSTRAINT_VIOLATION: "Constraint validation failed",
    ErrorCode.SUGGESTION_NOT_FOUND: "Suggestion not found",
    ErrorCode.CONCURRENT_MODIFICATION: (
        "Entity was modified by another request; your update lost the version race"
    ),
    ErrorCode.SEARCH_SPACE_EXHAUSTED: "Search space has no remaining unique combinations",
    ErrorCode.BUDGET_EXCEEDED: "Campaign exceeded its iteration or observation budget",
    ErrorCode.CAMPAIGN_CONVERGED: "Campaign has converged to its plateau",
    ErrorCode.IDEMPOTENCY_IN_PROGRESS: ("Operation with this idempotency_key is still executing"),
    ErrorCode.IDEMPOTENCY_CONFLICT: (
        "idempotency_key was previously used with a different payload"
    ),
    ErrorCode.MODEL_FITTING_FAILED: "Model fitting failed",
    ErrorCode.ACQUISITION_OPTIMIZATION_FAILED: "Acquisition optimization failed",
    ErrorCode.DATABASE_ERROR: "Database operation failed",
    ErrorCode.DATA_INTEGRITY_ERROR: "Persisted data is structurally invalid",
    ErrorCode.INSUFFICIENT_DATA: "Insufficient data for operation",
    ErrorCode.BACKEND_TRANSIENT_ERROR: "Backend reported a transient failure",
    ErrorCode.BACKEND_INCOMPATIBILITY: "Backend cannot handle this spec",
    ErrorCode.BACKEND_INTERNAL_ERROR: "Backend raised an unexpected internal error",
    ErrorCode.INTERNAL_ERROR: "An internal server error occurred",
}


def make_error_response(
    code: ErrorCode,
    message: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a standardized error response with recovery guidance.

    The response carries ``error.retryable`` (a stable bool clients
    can dispatch on) and ``error.retry_after`` (suggested backoff in
    seconds, or ``None`` for terminal failures). Both come from
    :data:`ERROR_CODE_RETRY_HINTS`; pass ``retry_after_seconds`` in
    ``details`` to override the default backoff at a call site.

    Args:
        code: Error code from ErrorCode enum.
        message: Optional custom message (uses default if not provided).
        details: Optional additional details about the error. A
            ``retry_after_seconds`` entry, when numeric, overrides
            the default backoff hint for this response only.

    Returns:
        Dictionary with:
            - success: False
            - error: Structured error dict with code, message,
              recovery_action, retryable, retry_after.
            - errors: List with the error message (for backward compatibility)
    """
    error_message = message or DEFAULT_MESSAGES.get(code, "Unknown error")
    recovery_action = ERROR_RECOVERY.get(code, "Contact support.")
    retryable, retry_after = retry_hint_for(code)

    # Allow per-call overrides of the default backoff so existing call
    # sites (e.g. the CONCURRENT_MODIFICATION envelope, which already
    # threads ``retry_after_seconds`` through ``details`` for clients
    # that read the legacy field) can keep emitting a custom value.
    if details is not None and "retry_after_seconds" in details:
        override = details["retry_after_seconds"]
        if isinstance(override, (int, float)):
            retry_after = float(override)

    error = StructuredError(
        code=code,
        message=error_message,
        recovery_action=recovery_action,
        retryable=retryable,
        retry_after=retry_after,
        details=details,
    )

    # Lazy import to break the otherwise-circular dependency: the
    # response_formatter pulls ``__version__`` from the package init,
    # which (transitively) imports this module via tool registration.
    from bo_mcp_server.response_formatter import (
        RESPONSE_SCHEMA_VERSION,
    )

    # Build through the typed envelope, then dump to a plain dict so the
    # many downstream consumers (in-place metadata splicing, JSON cache
    # persistence, idempotency replay, ``http_status_for_error``) keep
    # working on a dict. ``model_dump`` reproduces the historical shape
    # byte-for-byte: ``errors`` is the legacy single-message list.
    return ErrorEnvelope(
        schema_version=RESPONSE_SCHEMA_VERSION,
        error=error,
        errors=[error_message],
    ).model_dump()


# Maps error codes to HTTP status codes for use by the REST API layer.
ERROR_CODE_TO_HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.INVALID_CAMPAIGN_ID: 400,
    ErrorCode.CAMPAIGN_NOT_FOUND: 404,
    ErrorCode.INVALID_STATE_TRANSITION: 409,
    ErrorCode.DUPLICATE_RESULT: 409,
    ErrorCode.VALIDATION_FAILED: 400,
    ErrorCode.MISSING_PARAMETERS: 400,
    ErrorCode.MISSING_OBJECTIVES: 400,
    ErrorCode.CONSTRAINT_VIOLATION: 400,
    ErrorCode.SUGGESTION_NOT_FOUND: 404,
    ErrorCode.CONCURRENT_MODIFICATION: 409,
    ErrorCode.SEARCH_SPACE_EXHAUSTED: 409,
    ErrorCode.BUDGET_EXCEEDED: 409,
    ErrorCode.CAMPAIGN_CONVERGED: 409,
    ErrorCode.IDEMPOTENCY_IN_PROGRESS: 409,
    ErrorCode.IDEMPOTENCY_CONFLICT: 409,
    ErrorCode.MODEL_FITTING_FAILED: 500,
    ErrorCode.ACQUISITION_OPTIMIZATION_FAILED: 500,
    ErrorCode.DATABASE_ERROR: 500,
    ErrorCode.DATA_INTEGRITY_ERROR: 500,
    ErrorCode.INSUFFICIENT_DATA: 422,
    ErrorCode.BACKEND_TRANSIENT_ERROR: 503,
    ErrorCode.BACKEND_INCOMPATIBILITY: 400,
    ErrorCode.BACKEND_INTERNAL_ERROR: 500,
    ErrorCode.INTERNAL_ERROR: 500,
}


# Backoff hint surfaced to retryable transient backend failures. The
# value is intentionally larger than the OCC retry constant because
# numerical hiccups during a GP fit typically clear over seconds, not
# milliseconds; clients that retry immediately would just re-trigger
# the same transient.
BACKEND_TRANSIENT_RETRY_AFTER_SECONDS = 2.0


# Per-error retryability + suggested backoff. The single source of
# truth for "should clients retry this?" plus the recommended pause
# before they do. Operations that need a different backoff for a
# specific call site can override by passing ``retry_after_seconds``
# in ``details`` (the response merges it with this default).
ERROR_CODE_RETRY_HINTS: dict[ErrorCode, tuple[bool, float | None]] = {
    # Validation / state errors — deterministic given the same input.
    ErrorCode.INVALID_CAMPAIGN_ID: (False, None),
    ErrorCode.CAMPAIGN_NOT_FOUND: (False, None),
    ErrorCode.INVALID_STATE_TRANSITION: (False, None),
    ErrorCode.DUPLICATE_RESULT: (False, None),
    ErrorCode.VALIDATION_FAILED: (False, None),
    ErrorCode.MISSING_PARAMETERS: (False, None),
    ErrorCode.MISSING_OBJECTIVES: (False, None),
    ErrorCode.CONSTRAINT_VIOLATION: (False, None),
    ErrorCode.SUGGESTION_NOT_FOUND: (False, None),
    # Terminal lifecycle states.
    ErrorCode.SEARCH_SPACE_EXHAUSTED: (False, None),
    ErrorCode.BUDGET_EXCEEDED: (False, None),
    ErrorCode.CAMPAIGN_CONVERGED: (False, None),
    # Concurrency / idempotency — resolve on retry once the
    # contending caller settles.
    ErrorCode.CONCURRENT_MODIFICATION: (True, CONCURRENT_MODIFICATION_RETRY_AFTER_SECONDS),
    ErrorCode.IDEMPOTENCY_IN_PROGRESS: (True, CONCURRENT_MODIFICATION_RETRY_AFTER_SECONDS),
    # Programmer error: caller reused a key for a different payload.
    # Retrying the exact same call will keep failing.
    ErrorCode.IDEMPOTENCY_CONFLICT: (False, None),
    # Processing errors. Model fitting / acquisition failures and
    # generic "internal" surfaces are deterministic from the
    # caller's point of view — the input must change.
    ErrorCode.MODEL_FITTING_FAILED: (False, None),
    ErrorCode.ACQUISITION_OPTIMIZATION_FAILED: (False, None),
    ErrorCode.DATABASE_ERROR: (True, CONCURRENT_MODIFICATION_RETRY_AFTER_SECONDS),
    # Deterministic failure: the row stays broken until an operator
    # repairs it, so a client retry just burns the same request again.
    # Distinct from DATABASE_ERROR which covers transient connection
    # blips / deadlocks that *do* resolve on retry.
    ErrorCode.DATA_INTEGRITY_ERROR: (False, None),
    ErrorCode.INSUFFICIENT_DATA: (False, None),
    # Backend hierarchy — only the explicit "transient" subtype is
    # retryable; the others are terminal until the spec/backend
    # changes.
    ErrorCode.BACKEND_TRANSIENT_ERROR: (True, BACKEND_TRANSIENT_RETRY_AFTER_SECONDS),
    ErrorCode.BACKEND_INCOMPATIBILITY: (False, None),
    ErrorCode.BACKEND_INTERNAL_ERROR: (False, None),
    # Catch-all for unhandled transport-layer exceptions. Without a
    # known cause we cannot promise that a retry will succeed, so we
    # report ``retryable=False`` rather than risk a retry storm. A
    # client that can investigate the request_id can resubmit on its
    # own once the operator has confirmed the underlying issue.
    ErrorCode.INTERNAL_ERROR: (False, None),
}


def retry_hint_for(code: ErrorCode) -> tuple[bool, float | None]:
    """Look up ``(retryable, retry_after)`` for ``code``.

    Unknown codes default to ``(False, None)`` — better to surface a
    non-retryable failure to a client than to encourage retry storms
    on a code whose semantics we did not pin.
    """
    return ERROR_CODE_RETRY_HINTS.get(code, (False, None))


def make_concurrent_modification_response(
    err: "ConcurrentModificationError",
    extra_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the structured response for a lost optimistic-locking race.

    Converts the storage-layer ``ConcurrentModificationError`` into a
    ``CONCURRENT_MODIFICATION`` envelope that identifies the entity that
    was racing, the version the caller tried to overwrite, and a
    ``retry_after_seconds`` backoff hint so agents can retry safely.

    Args:
        err: The raised ``ConcurrentModificationError``.
        extra_details: Operation-specific details to merge into ``error.details``
            (e.g. ``{"campaign_id": ...}``) so callers do not have to repeat
            the entity-id fields that are already on ``err``.

    Returns:
        A ``make_error_response`` dict for ``ErrorCode.CONCURRENT_MODIFICATION``
        with ``details`` including the offending entity and a retry hint.
    """
    details: dict[str, Any] = {
        "entity_type": err.entity_type,
        "entity_id": str(err.entity_id),
        "expected_version": err.expected_version,
        "retry_after_seconds": CONCURRENT_MODIFICATION_RETRY_AFTER_SECONDS,
    }
    if extra_details:
        details.update(extra_details)

    message = (
        f"{err.entity_type} {err.entity_id} was modified by another request "
        f"while your update was in flight (expected version {err.expected_version}). "
        "Fetch the current version and retry."
    )
    return make_error_response(
        ErrorCode.CONCURRENT_MODIFICATION,
        message=message,
        details=details,
    )


def make_corrupted_json_response(
    err: "CorruptedJsonColumnError",
    extra_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the structured response for a malformed persisted JSON column.

    The storage-layer ``_strict_json_loads`` raises
    :class:`CorruptedJsonColumnError` when a persisted JSON column no
    longer decodes; this helper is the transport-boundary mapper that
    turns that exception into the canonical structured envelope so MCP
    and REST clients see a typed failure instead of an opaque
    ``RuntimeError`` leaking through.

    Uses :data:`ErrorCode.DATA_INTEGRITY_ERROR` rather than the
    broader :data:`ErrorCode.DATABASE_ERROR` for a deliberate
    semantic reason: the failure is **deterministic**, not transient.
    A retry against the same request hits the same broken row and
    will keep failing until an operator repairs the column. The
    matching retry hint in :data:`ERROR_CODE_RETRY_HINTS` is
    ``(retryable=False, retry_after=None)`` so clients (and especially
    LLM agents) do not enter a retry storm against unrecoverable
    state. ``DATABASE_ERROR`` retains ``retryable=True`` for connection
    blips, deadlocks, and other transient DB failures that *do*
    resolve on retry.

    The ``raw_excerpt`` from the exception is **not** included in the
    response: a corrupted column's content may carry user-submitted
    chemistry, formulation IP, or other sensitive bytes; surfacing the
    first 200 chars of an arbitrary blob to an untrusted client would
    be the wrong default. Operators still see the full excerpt in the
    server-side ERROR log emitted by ``_strict_json_loads``.
    """
    details: dict[str, Any] = {
        "column": err.context,
    }
    if extra_details:
        details.update(extra_details)

    message = (
        "Persisted JSON column could not be decoded. The row exists but "
        f"the value in {err.context} is no longer valid JSON. Do not retry "
        "the request — the row will keep failing until an operator repairs "
        "it; use the request_id from details to locate the offending row "
        "in the server log and repair it manually or restore from backup."
    )
    return make_error_response(
        ErrorCode.DATA_INTEGRITY_ERROR,
        message=message,
        details=details,
    )


class ResourceOperationError(Exception):
    """Raised by MCP resource handlers to signal a protocol-level failure.

    Surfacing happens in two layers:

    * **Direct callers** (tests, in-process consumers) catch
      :class:`ResourceOperationError` and read the structured envelope
      off ``exc.envelope`` (or from ``str(exc)`` for the JSON form).
    * **The MCP wire path**, when the server is constructed via
      :func:`bo_mcp_server.server.create_mcp_server`, routes through
      the read-boundary wrapper installed by
      :mod:`bo_mcp_server.resource_boundary`. That wrapper detects
      :class:`ResourceOperationError` on the cause chain (FastMCP's
      ``ResourceTemplate.create_resource`` and
      ``ResourceManager.get_resource`` each wrap raised exceptions in a
      generic ``ValueError("Error creating resource from template: …")``)
      and re-raises ``mcp.shared.exceptions.McpError`` with the
      structured envelope on ``error.data`` and a semantic JSON-RPC
      code (``INVALID_PARAMS`` for caller-supplied validation
      failures, ``INTERNAL_ERROR`` otherwise). Without the wrapper
      patch, FastMCP would still escalate the failure to the JSON-RPC
      layer, but the message would carry the double-prefix wrap and a
      generic ``code=0``.

    Either way, MCP clients see the read fail at the protocol level
    rather than receiving a success-shaped JSON body that secretly
    carries an error; the envelope remains a single parseable shape
    across tool and resource surfaces. See
    :func:`raise_resource_error` below for the standard entry point.
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Build the structured envelope and JSON-serialize it for the FastMCP layer."""
        self.code = code
        self.error_message = message
        self.details = details
        self.envelope = make_error_response(code, message=message, details=details)
        super().__init__(json.dumps(self.envelope, ensure_ascii=False))


def raise_resource_error(
    code: ErrorCode,
    message: str | None = None,
    details: dict[str, Any] | None = None,
) -> NoReturn:
    """Raise a :class:`ResourceOperationError` for an MCP resource path.

    Thin helper so call sites read naturally:
        raise_resource_error(ErrorCode.CAMPAIGN_NOT_FOUND, ...)

    Equivalent to ``raise ResourceOperationError(code, message, details)``
    but keeps the import surface and the call site small. The return
    annotation is :class:`NoReturn` so type checkers narrow callers
    correctly: code after the call is treated as unreachable.
    """
    raise ResourceOperationError(code, message=message, details=details)


def render_resource_error(
    code: ErrorCode,
    message: str | None = None,
    details: dict[str, Any] | None = None,
) -> str:
    """Render a structured error envelope as a JSON string for MCP resources.

    Retained for tests and any third-party code that depends on the
    string-body shape. New resource handlers raise
    :class:`ResourceOperationError` (via :func:`raise_resource_error`)
    so the MCP read path (via the wrapper installed by
    :mod:`bo_mcp_server.resource_boundary`) surfaces the failure as a
    structured :class:`McpError` at the JSON-RPC layer; the envelope
    still rides inside the exception message and on
    ``McpError.error.data`` for clients that prefer the body.

    The output is stable JSON (sorted keys disabled to preserve the
    canonical field ordering of ``success``, ``error``, ``errors``) so
    downstream string parsers and snapshot tests get deterministic
    output.
    """
    envelope = make_error_response(code, message=message, details=details)
    return json.dumps(envelope, ensure_ascii=False)


# Maps each :class:`bo_engine.backend_base.BackendError` subclass to
# the structured ``ErrorCode`` it surfaces as. Keeping the mapping in
# one place means new backend exception types only need a single
# entry here to flow through the MCP / REST envelope layers with the
# right retry semantics.
_BACKEND_ERROR_CODE_MAP: tuple[tuple[type[BackendError], "ErrorCode"], ...] = (
    (BackendTransientError, ErrorCode.BACKEND_TRANSIENT_ERROR),
    (BackendInputError, ErrorCode.VALIDATION_FAILED),
    (BackendIncompatibilityError, ErrorCode.BACKEND_INCOMPATIBILITY),
    (BackendInternalError, ErrorCode.BACKEND_INTERNAL_ERROR),
)


def error_code_for_backend_exception(exc: BackendError) -> ErrorCode:
    """Look up the ``ErrorCode`` for a :class:`BackendError` subclass.

    Subclasses inherit the mapping of their nearest declared base, so
    a custom subclass of :class:`BackendTransientError` still surfaces
    as ``BACKEND_TRANSIENT_ERROR`` and keeps its retry semantics.
    Unrecognised types fall back to ``BACKEND_INTERNAL_ERROR`` (the
    catch-all that operators triage on).
    """
    for cls, code in _BACKEND_ERROR_CODE_MAP:
        if isinstance(exc, cls):
            return code
    return ErrorCode.BACKEND_INTERNAL_ERROR


def make_backend_error_response(
    exc: BackendError,
    *,
    extra_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a :class:`BackendError` into a structured response.

    Selects the matching :class:`ErrorCode` via
    :func:`error_code_for_backend_exception` so the envelope carries
    the correct ``retryable`` flag and ``retry_after`` hint (see
    :data:`ERROR_CODE_RETRY_HINTS`). The original exception type is
    recorded in ``details.backend_exception`` so operators can
    distinguish transient flakes from incompatibility rejections
    without re-parsing the message.
    """
    code = error_code_for_backend_exception(exc)
    details: dict[str, Any] = {
        "backend_exception": type(exc).__name__,
    }
    if exc.__cause__ is not None:
        details["cause"] = f"{type(exc.__cause__).__name__}: {exc.__cause__}"
    if extra_details:
        details.update(extra_details)
    return make_error_response(code, message=str(exc) or None, details=details)


def http_status_for_error(error_response: dict[str, Any]) -> int:
    """Extract the HTTP status code from an operation error response.

    Looks up the error code in ERROR_CODE_TO_HTTP_STATUS.
    Returns 500 as the default if the code is missing or unrecognised.
    """
    error_dict = error_response.get("error", {})
    code_value = error_dict.get("code")
    if code_value is not None:
        try:
            code = ErrorCode(code_value)
            return ERROR_CODE_TO_HTTP_STATUS.get(code, 500)
        except ValueError:
            pass
    return 500
