"""Response formatting utilities for verbosity control.

This module provides verbosity-based response formatting to reduce token
consumption for AI agents. Three verbosity levels are supported:

- MINIMAL (~50 tokens): Success status + key metrics only. Use for tight loops.
- STANDARD (~200 tokens): Default. Excludes debugging fields like hyperparameters.
- DETAILED (~500+ tokens): All fields including LOO-CV metrics and hyperparameters.

Each ``format_*_response`` builds a per-operation, per-verbosity
:class:`pydantic.BaseModel` (one model per verbosity where the shape
differs) and returns ``model.model_dump()`` so the dict served to MCP /
REST callers has been validated at the transport boundary. ``None``
fields are kept in the dump on purpose: the MINIMAL projection's
documented keys (``success``, ``campaign_id``, …) stay present even
when the underlying value is missing, so agent code can address them
unconditionally. This addresses the "weak ``TypedDict``" pattern where
the formatter previously cast arbitrary ``dict[str, Any]`` into a
declared shape with no runtime guarantee.

Usage:
    from bo_mcp_server.response_formatter import (
        VerbosityLevel,
        format_diagnostics_response,
        format_suggestions_response,
    )

    # In tool function:
    full_response = compute_diagnostics(...)
    return format_diagnostics_response(full_response, VerbosityLevel(verbosity))
"""

import functools
from collections.abc import Callable, Coroutine
from enum import StrEnum
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from bo_mcp_server import __version__

_METADATA_FIELD = "_metadata"
_SCHEMA_VERSION_FIELD = "schema_version"

# Top-level response-envelope schema version.
#
# Every MCP-tool / REST response carries ``schema_version`` so older
# clients (including older LLM tool descriptions) can detect a
# backwards-incompatible envelope change without parsing the body.
# Bump rules:
#   * additive fields → keep the version unchanged (the contract is
#     forward-compatible by design — callers ignore unknown keys);
#   * removed / renamed / re-typed fields → bump by 1 and document
#     the change in the project changelog and the migration notes.
#
# Reference: the schema-version handshake is the same pattern Stripe
# documents at https://stripe.com/docs/api/versioning — a single
# integer the client compares against the version it was built
# against.
#
# Version history:
#   1 → 2: suggestion payloads dropped the ``id`` key; ``suggestion_id``
#          is the only identity key on generated / listed / queried
#          suggestions across MCP and REST.
RESPONSE_SCHEMA_VERSION: Final[int] = 2


class ResponseMetadata(BaseModel):
    """Per-response provenance / addressability metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: str
    protocol: str
    server_version: str
    # ``trace_id`` is echoed only when the caller bound one via
    # :func:`bo_mcp_server.trace_context.bind_trace_id` so the metadata
    # envelope stays compact for one-off calls.
    trace_id: str | None = None


def get_response_metadata(protocol: str = "mcp") -> ResponseMetadata:
    """Build metadata for a formatted response.

    Campaign-scoped operations stamp the *campaign's* backend (bound via
    :mod:`bo_mcp_server.backend_context` when the spec is resolved); only
    campaign-agnostic calls fall back to the default backend, which the
    server warms at startup
    (:func:`bo_mcp_server.backend.warm_default_backend`), so the lookup
    here is a cache hit and safe to run on the event loop.

    Args:
        protocol: The transport protocol ("mcp" or "rest").

    Returns:
        Metadata with backend, protocol, and server version.
    """
    from bo_mcp_server.backend import get_backend
    from bo_mcp_server.backend_context import get_campaign_backend
    from bo_mcp_server.trace_context import get_trace_id

    backend_name = get_campaign_backend() or get_backend().name
    return ResponseMetadata(
        backend=backend_name,
        protocol=protocol,
        server_version=__version__,
        trace_id=get_trace_id(),
    )


def attach_response_metadata(response: dict[str, Any]) -> dict[str, Any]:
    """Splice the ``_metadata`` envelope into a raw operation response.

    Operations that bypass the per-verbosity Pydantic formatter
    (lifecycle transitions, dry-run previews, suggestion-status
    updates, error envelopes) still need to echo the workflow
    ``trace_id`` under ``_metadata.trace_id`` so the contract the
    cookbook documents holds uniformly across every tool return.

    The function mutates ``response`` in place and returns it for
    chaining; the metadata key is left out entirely when no trace is
    bound, keeping the envelope compact for one-off calls (existing
    snapshot tests assert the exact key set). If ``response`` already
    carries a ``_metadata`` block — e.g. it was assembled by the
    Pydantic per-verbosity formatter — the existing entry is left
    untouched.

    Always stamps ``schema_version`` at the top level so callers can
    dispatch on the envelope contract without inspecting the body
    shape.
    """
    response.setdefault(_SCHEMA_VERSION_FIELD, RESPONSE_SCHEMA_VERSION)
    if _METADATA_FIELD in response:
        return response
    metadata_dump = get_response_metadata().model_dump()
    if metadata_dump.get("trace_id") is None:
        metadata_dump.pop("trace_id", None)
    response[_METADATA_FIELD] = metadata_dump
    return response


def with_response_metadata(
    fn: Callable[..., Coroutine[Any, Any, dict[str, Any]]],
) -> Callable[..., Coroutine[Any, Any, dict[str, Any]]]:
    """Async decorator that splices ``_metadata`` into every return.

    Use on operation entry points whose body returns raw dicts (no
    per-verbosity Pydantic formatter) so trace-id echo and other
    response metadata reach the caller without each early-return
    branch having to remember to call :func:`attach_response_metadata`.
    """

    @functools.wraps(fn)
    async def wrapper(*args: object, **kwargs: object) -> dict[str, Any]:
        return attach_response_metadata(await fn(*args, **kwargs))

    return wrapper


def _with_metadata(
    fn: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """Decorator that injects ``_metadata`` into the response dict.

    Runs after the format function has already validated its payload
    through a Pydantic model, so the wrapped result is guaranteed to
    match the per-operation contract.
    """

    @functools.wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> dict[str, Any]:
        result = fn(*args, **kwargs)
        metadata_dump = get_response_metadata().model_dump()
        # Strip ``trace_id`` when no workflow is active so the metadata
        # envelope stays compact (existing snapshot tests assert the
        # exact key set).
        if metadata_dump.get("trace_id") is None:
            metadata_dump.pop("trace_id", None)
        result[_METADATA_FIELD] = metadata_dump
        # Every formatted envelope carries the top-level schema_version
        # so older clients can dispatch on the contract without parsing
        # the body. ``setdefault`` so a per-operation formatter that
        # needs to advertise a higher version during a staged migration
        # can override.
        result.setdefault(_SCHEMA_VERSION_FIELD, RESPONSE_SCHEMA_VERSION)
        return result

    return wrapper


class VerbosityLevel(StrEnum):
    """Verbosity levels for MCP tool responses.

    Attributes:
        MINIMAL: ~50 tokens - success + key metric only
        STANDARD: ~200 tokens - current default (excludes debug fields)
        DETAILED: ~500+ tokens - all fields including debug info
    """

    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"


# ---------------------------------------------------------------------------
# Pydantic response models — one per (operation, verbosity) tuple where the
# shape differs. ``DETAILED`` variants set ``extra="allow"`` because the
# detailed payloads carry operation-specific keys whose shape varies across
# backends; the minimal/standard variants forbid extras so a typo in the
# format function surfaces at runtime instead of silently producing the
# wrong agent-visible field.
# ---------------------------------------------------------------------------


class _StrictResponse(BaseModel):
    """Base for fully-typed minimal/standard projections."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class _PassthroughResponse(BaseModel):
    """Base for detailed projections that forward unknown keys verbatim."""

    model_config = ConfigDict(frozen=True, extra="allow")


# ---------- Shared list-item projections ----------


class SuggestionSummaryItem(BaseModel):
    """One ``suggestions[]`` entry from ``bo_list_suggestions``.

    Superset of the MINIMAL / STANDARD / DETAILED per-item projections
    built by ``operations.list_suggestions._serialize_suggestion``.
    Every field is ``None``-defaulted because which keys are present
    depends on the requested verbosity.

    Defined here rather than in :mod:`bo_mcp_server.tools.response_models`
    so the client facade can re-export it without importing the MCP tool
    layer (``bo_mcp_server.tools`` registers every tool against the
    FastMCP server at import time): the REST ``SuggestionSummary`` schema
    subclasses it to pin the identity fields as required, while
    ``tools.response_models.SuggestionListResponse`` types its list items
    with it for the ``bo_list_suggestions`` ``outputSchema``.

    Not one of the projection bases above: unlike the frozen
    per-(operation, verbosity) models, this item model keeps the
    tool-response convention of ``extra="allow"`` alone so unknown keys
    pass through and subclasses stay mutable.
    """

    model_config = ConfigDict(extra="allow")

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
    # Qualitative bucket ("high" / "medium" / "low"), matching the
    # domain provenance model — not a numeric confidence score.
    confidence_level: str | None = None
    predicted_objectives: dict[str, Any] | None = None
    predicted_std: dict[str, Any] | None = None
    updated_at: str | None = None


# ---------- Diagnostics ----------


class DiagnosticsMinimalResponse(_StrictResponse):
    """MINIMAL projection of the diagnostics response."""

    success: bool | None = None
    status: str | None = None
    iteration: int | None = None
    n_results: int | None = None
    health: str | None = None
    progress: str | None = None
    key_metric: dict[str, Any] = Field(default_factory=dict)
    converged: bool = False
    # ``next_action_recommendation`` may be a plain string ("call X") or a
    # structured payload with ``action`` / ``urgency`` keys produced by
    # the agent-usability diagnostics. Both shapes are accepted at the
    # transport boundary.
    next_action: str | dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)


class DiagnosticsResponse(_PassthroughResponse):
    """STANDARD / DETAILED projection of the diagnostics response.

    The diagnostics tool surface returns a deeply nested dict whose
    detailed metric blocks (LOO-CV, hyperparameters, hypervolume history)
    vary per backend. Passthrough lets those blocks flow without forcing
    a giant union of every possible field; the explicitly named fields
    below pin the keys agents are documented to consume so a typo in the
    format projection still produces a validation error.
    """

    success: bool | None = None
    campaign_status: str | None = None
    iteration: int | None = None
    n_results: int | None = None
    n_pending_suggestions: int | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ---------- Suggestions ----------


class SuggestionsMinimalResponse(_StrictResponse):
    """MINIMAL projection of the generate-suggestions response."""

    success: bool | None = None
    iteration: int | None = None
    suggestion_ids: list[str | None] = Field(default_factory=list)
    method: str | None = None
    errors: list[str] = Field(default_factory=list)


class SuggestionsResponse(_PassthroughResponse):
    """STANDARD / DETAILED projection of the generate-suggestions response."""

    success: bool | None = None
    iteration: int | None = None
    suggestions: list[dict[str, Any]] = Field(default_factory=list)
    method_selection: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ---------- Compare campaigns ----------


class CompareCampaignsMinimalResponse(_StrictResponse):
    """MINIMAL projection of the compare-campaigns response."""

    success: bool | None = None
    n_campaigns: int = 0
    best_performer: str | None = None
    recommendation: str | None = None
    errors: list[str] = Field(default_factory=list)


class CompareCampaignsResponse(_PassthroughResponse):
    """STANDARD / DETAILED projection of the compare-campaigns response."""

    success: bool | None = None
    campaigns: list[dict[str, Any]] = Field(default_factory=list)
    comparison: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)


# ---------- Transfer candidates ----------


class TransferCandidatesMinimalResponse(_StrictResponse):
    """MINIMAL projection of the discover-transfer-candidates response."""

    success: bool | None = None
    n_candidates: int = 0
    top_candidate_id: str | None = None
    top_similarity: float | None = None
    recommendation: str | None = None
    errors: list[str] = Field(default_factory=list)


class TransferCandidatesResponse(_PassthroughResponse):
    """STANDARD / DETAILED projection of the discover-transfer-candidates response."""

    success: bool | None = None
    target_campaign: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    overall_recommendation: str | None = None
    errors: list[str] = Field(default_factory=list)


# ---------- Create campaign ----------


class CreateCampaignMinimalResponse(_StrictResponse):
    """MINIMAL projection of the create-campaign response."""

    success: bool | None = None
    campaign_id: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    field_errors: dict[str, list[str]] = Field(default_factory=dict)


class CreateCampaignStandardResponse(_StrictResponse):
    """STANDARD projection of the create-campaign response."""

    success: bool | None = None
    campaign_id: str | None = None
    spec_id: str | None = None
    campaign_name: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    field_errors: dict[str, list[str]] = Field(default_factory=dict)


class CreateCampaignDetailedResponse(_PassthroughResponse):
    """DETAILED projection of the create-campaign response."""

    success: bool | None = None
    campaign_id: str | None = None
    spec_id: str | None = None
    campaign_name: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    field_errors: dict[str, list[str]] = Field(default_factory=dict)


# ---------- Submit results ----------


class SubmitResultsMinimalResponse(_StrictResponse):
    """MINIMAL projection of the submit-results response."""

    success: bool | None = None
    n_submitted: int = 0
    errors: list[str] = Field(default_factory=list)
    field_errors: dict[str, list[str]] = Field(default_factory=dict)


class SubmitResultsStandardResponse(_StrictResponse):
    """STANDARD projection of the submit-results response.

    ``partial_results`` (index → result id or per-row error) is the
    whole point of a ``continue_on_error`` submission — without it the
    caller cannot tell which rows persisted — so the default
    projection must carry it, not just DETAILED. ``None`` for atomic
    submissions (the key stays present, matching ``_dump``'s
    stable-shape convention).
    """

    success: bool | None = None
    result_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    field_errors: dict[str, list[str]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    n_duplicates_detected: int = 0
    partial_results: dict[int, str | dict[str, str]] | None = None


class SubmitResultsDetailedResponse(_PassthroughResponse):
    """DETAILED projection of the submit-results response."""

    success: bool | None = None
    result_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    field_errors: dict[str, list[str]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    duplicates_detected: list[dict[str, Any]] = Field(default_factory=list)


# ---------- Validate intake ----------


class ValidateIntakeMinimalResponse(_StrictResponse):
    """MINIMAL projection of the validate-intake response."""

    valid: bool | None = None
    errors: list[str] = Field(default_factory=list)
    field_errors: dict[str, list[str]] = Field(default_factory=dict)


class ValidateIntakeSpecSummary(_StrictResponse):
    """Summary of a campaign spec returned by validate-intake."""

    name: str | None = None
    n_parameters: int = 0
    n_objectives: int = 0
    n_constraints: int = 0
    batch_size: int | None = None


class ValidateIntakeStandardResponse(_StrictResponse):
    """STANDARD projection of the validate-intake response."""

    valid: bool | None = None
    errors: list[str] = Field(default_factory=list)
    field_errors: dict[str, list[str]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    spec_summary: ValidateIntakeSpecSummary | None = None


class ValidateIntakeDetailedResponse(_PassthroughResponse):
    """DETAILED projection of the validate-intake response."""

    valid: bool | None = None
    errors: list[str] = Field(default_factory=list)
    field_errors: dict[str, list[str]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    spec: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Format functions
# ---------------------------------------------------------------------------


def _dump(model: BaseModel) -> dict[str, Any]:
    """Render a response model as the dict served to callers.

    ``exclude_none=False`` keeps the documented MINIMAL shape stable (the
    keys agents key on are always present, even when the value is null).
    The dict is mutable so the metadata decorator can splice ``_metadata``
    in afterwards.
    """
    return model.model_dump()


@_with_metadata
def format_diagnostics_response(
    full_response: dict[str, Any],
    verbosity: VerbosityLevel = VerbosityLevel.STANDARD,
) -> dict[str, Any]:
    """Format diagnostics response based on verbosity level.

    Args:
        full_response: Complete diagnostics response dictionary from get_diagnostics.
        verbosity: Desired verbosity level.

    Returns:
        Filtered response dictionary based on verbosity.
    """
    if verbosity == VerbosityLevel.MINIMAL:
        is_multi = full_response.get("pareto_front") is not None
        key_metric: dict[str, Any] = (
            {"hypervolume": full_response.get("hypervolume")}
            if is_multi
            else {"best_value": full_response.get("best_value")}
        )
        return _dump(
            DiagnosticsMinimalResponse(
                success=full_response.get("success"),
                status=full_response.get("campaign_status"),
                iteration=full_response.get("iteration"),
                n_results=full_response.get("n_results"),
                health=full_response.get("health_status"),
                progress=full_response.get("progress_status"),
                key_metric=key_metric,
                converged=full_response.get("convergence", {}).get("converged", False),
                next_action=full_response.get("next_action_recommendation"),
                errors=full_response.get("errors", []),
            )
        )

    if verbosity == VerbosityLevel.STANDARD:
        exclude_fields = {
            "hyperparameters",
            "loo_cv_metrics",
            "hypervolume_history",
            "outliers",
        }
        projected = {k: v for k, v in full_response.items() if k not in exclude_fields}
        return _dump(DiagnosticsResponse.model_validate(projected))

    # DETAILED returns everything, validated as the passthrough shape so
    # documented keys retain their declared types.
    return _dump(DiagnosticsResponse.model_validate(full_response))


@_with_metadata
def format_suggestions_response(
    full_response: dict[str, Any],
    verbosity: VerbosityLevel = VerbosityLevel.STANDARD,
) -> dict[str, Any]:
    """Format suggestions response based on verbosity level."""
    if verbosity == VerbosityLevel.MINIMAL:
        suggestions = full_response.get("suggestions", [])
        method_selection = full_response.get("method_selection", {})
        return _dump(
            SuggestionsMinimalResponse(
                success=full_response.get("success"),
                iteration=full_response.get("iteration"),
                suggestion_ids=[s.get("suggestion_id") for s in suggestions],
                method=method_selection.get("acquisition_function"),
                errors=full_response.get("errors", []),
            )
        )

    if verbosity == VerbosityLevel.STANDARD:
        exclude_fields = {"pending_points"}
        projected = {k: v for k, v in full_response.items() if k not in exclude_fields}
        return _dump(SuggestionsResponse.model_validate(projected))

    return _dump(SuggestionsResponse.model_validate(full_response))


@_with_metadata
def format_compare_campaigns_response(
    full_response: dict[str, Any],
    verbosity: VerbosityLevel = VerbosityLevel.STANDARD,
) -> dict[str, Any]:
    """Format compare_campaigns response based on verbosity level."""
    if verbosity == VerbosityLevel.MINIMAL:
        comparison = full_response.get("comparison", {}) or {}
        campaigns = full_response.get("campaigns", [])
        return _dump(
            CompareCampaignsMinimalResponse(
                success=full_response.get("success"),
                n_campaigns=len(campaigns),
                best_performer=(
                    comparison.get("best_sample_efficiency")
                    or comparison.get("best_single_objective")
                    or comparison.get("best_multi_objective")
                ),
                recommendation=comparison.get("recommendation"),
                errors=full_response.get("errors", []),
            )
        )

    if verbosity == VerbosityLevel.STANDARD:
        campaigns = full_response.get("campaigns", [])
        simplified_campaigns = [
            {
                "campaign_id": c.get("campaign_id"),
                "campaign_name": c.get("campaign_name"),
                "status": c.get("status"),
                "n_results": c.get("n_results"),
                "best_value": c.get("best_value"),
                "hypervolume": c.get("hypervolume"),
                "sample_efficiency": c.get("sample_efficiency"),
            }
            for c in campaigns
        ]
        return _dump(
            CompareCampaignsResponse(
                success=full_response.get("success"),
                campaigns=simplified_campaigns,
                comparison=full_response.get("comparison"),
                errors=full_response.get("errors", []),
            )
        )

    return _dump(CompareCampaignsResponse.model_validate(full_response))


@_with_metadata
def format_transfer_candidates_response(
    full_response: dict[str, Any],
    verbosity: VerbosityLevel = VerbosityLevel.STANDARD,
) -> dict[str, Any]:
    """Format discover_transfer_candidates response based on verbosity level."""
    if verbosity == VerbosityLevel.MINIMAL:
        candidates = full_response.get("candidates", [])
        top_candidate = candidates[0] if candidates else None
        return _dump(
            TransferCandidatesMinimalResponse(
                success=full_response.get("success"),
                n_candidates=len(candidates),
                top_candidate_id=top_candidate.get("campaign_id") if top_candidate else None,
                top_similarity=top_candidate.get("similarity_score") if top_candidate else None,
                recommendation=full_response.get("overall_recommendation"),
                errors=full_response.get("errors", []),
            )
        )

    if verbosity == VerbosityLevel.STANDARD:
        candidates = full_response.get("candidates", [])
        simplified_candidates = [
            {
                "campaign_id": c.get("campaign_id"),
                "name": c.get("name"),
                "n_results": c.get("n_results"),
                "similarity_score": c.get("similarity_score"),
                "recommendation": c.get("recommendation"),
            }
            for c in candidates
        ]
        return _dump(
            TransferCandidatesResponse(
                success=full_response.get("success"),
                target_campaign=full_response.get("target_campaign"),
                candidates=simplified_candidates,
                overall_recommendation=full_response.get("overall_recommendation"),
                errors=full_response.get("errors", []),
            )
        )

    return _dump(TransferCandidatesResponse.model_validate(full_response))


@_with_metadata
def format_create_campaign_response(
    full_response: dict[str, Any],
    verbosity: VerbosityLevel = VerbosityLevel.STANDARD,
) -> dict[str, Any]:
    """Format create_campaign response based on verbosity level."""
    if verbosity == VerbosityLevel.MINIMAL:
        return _dump(
            CreateCampaignMinimalResponse(
                success=full_response.get("success"),
                campaign_id=full_response.get("campaign_id"),
                warnings=full_response.get("warnings", []),
                errors=full_response.get("errors", []),
                field_errors=full_response.get("field_errors", {}),
            )
        )

    if verbosity == VerbosityLevel.STANDARD:
        return _dump(
            CreateCampaignStandardResponse(
                success=full_response.get("success"),
                campaign_id=full_response.get("campaign_id"),
                spec_id=full_response.get("spec_id"),
                campaign_name=full_response.get("campaign_name"),
                warnings=full_response.get("warnings", []),
                errors=full_response.get("errors", []),
                field_errors=full_response.get("field_errors", {}),
            )
        )

    return _dump(CreateCampaignDetailedResponse.model_validate(full_response))


@_with_metadata
def format_submit_results_response(
    full_response: dict[str, Any],
    verbosity: VerbosityLevel = VerbosityLevel.STANDARD,
) -> dict[str, Any]:
    """Format submit_results response based on verbosity level."""
    if verbosity == VerbosityLevel.MINIMAL:
        result_ids = full_response.get("result_ids", [])
        return _dump(
            SubmitResultsMinimalResponse(
                success=full_response.get("success"),
                n_submitted=len(result_ids),
                errors=full_response.get("errors", []),
                field_errors=full_response.get("field_errors", {}),
            )
        )

    if verbosity == VerbosityLevel.STANDARD:
        return _dump(
            SubmitResultsStandardResponse(
                success=full_response.get("success"),
                result_ids=full_response.get("result_ids", []),
                errors=full_response.get("errors", []),
                field_errors=full_response.get("field_errors", {}),
                warnings=full_response.get("warnings", []),
                n_duplicates_detected=len(full_response.get("duplicates_detected", [])),
                partial_results=full_response.get("partial_results"),
            )
        )

    return _dump(SubmitResultsDetailedResponse.model_validate(full_response))


@_with_metadata
def format_validate_intake_response(
    full_response: dict[str, Any],
    verbosity: VerbosityLevel = VerbosityLevel.STANDARD,
) -> dict[str, Any]:
    """Format validate_intake response based on verbosity level."""
    if verbosity == VerbosityLevel.MINIMAL:
        return _dump(
            ValidateIntakeMinimalResponse(
                valid=full_response.get("valid"),
                errors=full_response.get("errors", []),
                field_errors=full_response.get("field_errors", {}),
            )
        )

    if verbosity == VerbosityLevel.STANDARD:
        spec = full_response.get("spec")
        spec_summary: ValidateIntakeSpecSummary | None = None
        if spec:
            spec_summary = ValidateIntakeSpecSummary(
                name=spec.get("name"),
                n_parameters=len(spec.get("parameters", [])),
                n_objectives=len(spec.get("objectives", [])),
                n_constraints=len(spec.get("constraints", [])),
                batch_size=spec.get("batch_size"),
            )
        return _dump(
            ValidateIntakeStandardResponse(
                valid=full_response.get("valid"),
                errors=full_response.get("errors", []),
                field_errors=full_response.get("field_errors", {}),
                warnings=full_response.get("warnings", []),
                spec_summary=spec_summary,
            )
        )

    return _dump(ValidateIntakeDetailedResponse.model_validate(full_response))
