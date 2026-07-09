"""Submit results tool wrapper for MCP."""

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, cast

from pydantic import Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from bo_mcp_server.client import AuthenticationConfigurationError, resolve_mcp_user
from bo_mcp_server.domain import ResultSubmissionInput
from bo_mcp_server.domain.intake_models import RESULT_SUBMISSION_JSON_SCHEMA
from bo_mcp_server.field_errors import shape_envelope, validation_envelope
from bo_mcp_server.idempotency import apply_idempotency
from bo_mcp_server.operations.idempotency_wrapper import (
    canonical_submit_results_payload,
)
from bo_mcp_server.operations.submit_results import submit_results_operation
from bo_mcp_server.response_formatter import attach_response_metadata
from bo_mcp_server.server import mcp
from bo_mcp_server.tool_boundary import SUBMIT_RESULTS_ENVELOPE_EXTRA
from bo_mcp_server.tools.annotations import NON_IDEMPOTENT_MUTATION
from bo_mcp_server.tools.common import mcp_identity_error
from bo_mcp_server.tools.response_models import SubmitResultsResponse
from bo_mcp_server.trace_context import bind_trace_id

# Accept loose payloads at the MCP boundary and validate inside the
# tool so payload-shape errors render as our structured envelope (with
# ``field_errors`` keyed by ``results[i].<path>``) rather than the
# opaque ``ToolError`` text FastMCP would otherwise produce. Typed as
# ``Any`` (not ``list[dict[str, Any]]``) so FastMCP cannot intercept
# the failure when ``results`` is not a list or when an item is not a
# dict; the wrapper now owns both checks. The spliced schema keeps
# ``tools/list`` advertising the rich ``ResultSubmissionInput``
# structure to agent introspection.
_RESULT_ITEM_SCHEMA = {
    "type": "object",
    "properties": RESULT_SUBMISSION_JSON_SCHEMA.get("properties", {}),
    "required": RESULT_SUBMISSION_JSON_SCHEMA.get("required", []),
    "$defs": RESULT_SUBMISSION_JSON_SCHEMA.get("$defs", {}),
    "additionalProperties": False,
}

ResultsPayload = Annotated[
    Any,
    Field(
        description=(
            "Batch of result payloads. Each entry is validated against "
            "ResultSubmissionInput; validation failures are returned as "
            "a structured error envelope with ``field_errors`` keyed by "
            "``results[i].<field>`` so agents can target the bad row."
        ),
        json_schema_extra={
            "type": "array",
            "items": _RESULT_ITEM_SCHEMA,
        },
    ),
]

_RESULTS_BOUNDARY_DEFAULTS = SUBMIT_RESULTS_ENVELOPE_EXTRA


def _validate_result_rows(
    rows: object,
) -> list[ResultSubmissionInput] | dict[str, Any]:
    """Convert a raw ``results`` payload into validated rows or an envelope.

    Outer-shape checks (``results`` is a list; each item is a dict
    or an already-validated ``ResultSubmissionInput``) live here so a
    malformed batch never reaches the operation layer and never raises
    an opaque ``ToolError`` at the MCP boundary. Returns the list of
    validated rows on success, or a structured envelope (with
    ``field_errors`` keyed by ``results``/``results[i]``/
    ``results[i].<field>``) on the first failure. Item-level Pydantic
    ``loc`` paths are rebased onto the outer ``results[i]`` index so
    they read identically to operation-layer paths.
    """
    if isinstance(rows, str) or not isinstance(rows, Sequence):
        return shape_envelope(
            "results",
            f"Input should be a list, got {type(rows).__name__}",
            extra=_RESULTS_BOUNDARY_DEFAULTS,
        )

    try:
        validated: list[ResultSubmissionInput] = []
        for index, row in enumerate(rows):
            if isinstance(row, ResultSubmissionInput):
                validated.append(row)
                continue
            if not isinstance(row, Mapping):
                # Row-level shape failure (e.g. ``results[0] = "string"``)
                # is reported as a synthesized ``field_errors`` entry
                # rather than letting Pydantic's per-row validate trip
                # over a non-mapping input.
                return shape_envelope(
                    f"results[{index}]",
                    f"Input should be an object, got {type(row).__name__}",
                    extra=_RESULTS_BOUNDARY_DEFAULTS,
                )
            try:
                validated.append(ResultSubmissionInput.model_validate(row))
            except ValidationError as exc:
                # Rewrite ``loc`` so paths read as ``results[i].<field>``
                # instead of being rooted at the inner field. Pydantic
                # cannot annotate ``loc`` with the outer index because
                # we validate each row in isolation.
                raise _rebase_row_validation_error(exc, index) from exc
    except ValidationError as exc:
        return validation_envelope(exc, extra=_RESULTS_BOUNDARY_DEFAULTS)
    return validated


def _rebase_row_validation_error(error: ValidationError, row_index: int) -> ValidationError:
    """Prepend ``("results", row_index)`` to every error's ``loc`` path.

    ``ValidationError`` is opaque on the public API but
    ``Pydantic.ValidationError.from_exception_data`` lets us rebuild
    one with rewritten line items. The rewritten error is what the
    outer ``try`` catches and converts to the structured envelope, so
    the resulting ``field_errors`` keys read ``results[i].<field>``
    just like the operation-layer paths.
    """
    # Pydantic's ``InitErrorDetails`` accepts a tuple ``loc`` only via the
    # dict-based ``from_exception_data`` constructor.
    line_errors = [
        {
            "type": err["type"],
            "loc": ("results", row_index, *err.get("loc", ())),
            "msg": err.get("msg", ""),
            "input": err.get("input"),
            "ctx": err.get("ctx", {}),
        }
        for err in error.errors()
    ]
    return ValidationError.from_exception_data(
        title=error.title,
        line_errors=line_errors,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    )


def _mcp_identity_error(exc: AuthenticationConfigurationError) -> dict[str, Any]:
    return mcp_identity_error(exc)


def _validate_results_before_identity(
    results: ResultsPayload,
    trace_id: str | None,
) -> dict[str, Any] | None:
    """Return a validation envelope before requiring MCP identity, if invalid."""
    with bind_trace_id(trace_id):
        validated_results = _validate_result_rows(results)
        if isinstance(validated_results, dict):
            return attach_response_metadata(validated_results)
    return None


async def _submit_results_for_user(
    campaign_id: str,
    results: ResultsPayload,
    submitted_by: str,
    source: str = "api",
    force: bool = False,
    atomic: bool = True,
    continue_on_error: bool = False,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
    idempotency_key: str | None = None,
    dry_run: bool = False,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Submit results for a concrete user id.

    This helper intentionally remains importable for internal scripts/tests
    that exercise the operation layer directly. The registered MCP tool below
    resolves ``submitted_by`` internally and does not expose it to agents.
    """
    # ``bind_trace_id`` wraps the entire wrapper body — including the
    # pre-operation row-shape validation — so a malformed payload still
    # emits an envelope whose ``_metadata.trace_id`` echoes the bound
    # workflow id. Otherwise validation envelopes would be the only
    # tool returns that drop the trace, undermining the cookbook
    # contract.
    with bind_trace_id(trace_id):
        validated_results = _validate_result_rows(results)
        if isinstance(validated_results, dict):
            # Per-row payload validation failed -- short-circuit with the
            # structured envelope before reserving any idempotency slot.
            return attach_response_metadata(validated_results)

        if dry_run:
            return await submit_results_operation(
                campaign_id=campaign_id,
                results=validated_results,
                submitted_by=submitted_by,
                source=source,
                force=force,
                atomic=atomic,
                continue_on_error=continue_on_error,
                verbosity=verbosity,
                dry_run=True,
            )

        # Canonical builder produces the same shape REST emits so a
        # retry on either transport replays the cached response from
        # the original mutation.
        request_payload = canonical_submit_results_payload(
            campaign_id=campaign_id,
            results=validated_results,
            submitted_by=submitted_by,
            source=source,
            force=force,
            atomic=atomic,
            continue_on_error=continue_on_error,
            verbosity=verbosity,
        )

        async def run(session: AsyncSession) -> dict[str, Any]:
            return await submit_results_operation(
                campaign_id=campaign_id,
                results=validated_results,
                submitted_by=submitted_by,
                source=source,
                force=force,
                atomic=atomic,
                continue_on_error=continue_on_error,
                verbosity=verbosity,
                session=session,
            )

        return await apply_idempotency(
            tool_name="bo_submit_results",
            idempotency_key=idempotency_key,
            request_payload=request_payload,
            executor=run,
        )


async def submit_results(
    campaign_id: str,
    results: ResultsPayload,
    submitted_by: str,
    source: str = "api",
    force: bool = False,
    atomic: bool = True,
    continue_on_error: bool = False,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
    idempotency_key: str | None = None,
    dry_run: bool = False,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Compatibility helper for Python callers that already have a user id."""
    return await _submit_results_for_user(
        campaign_id=campaign_id,
        results=results,
        submitted_by=submitted_by,
        source=source,
        force=force,
        atomic=atomic,
        continue_on_error=continue_on_error,
        verbosity=verbosity,
        idempotency_key=idempotency_key,
        dry_run=dry_run,
        trace_id=trace_id,
    )


@mcp.tool(name="bo_submit_results", annotations=NON_IDEMPOTENT_MUTATION)
async def _submit_results_tool(
    campaign_id: str,
    results: ResultsPayload,
    source: str = "api",
    force: bool = False,
    atomic: bool = True,
    continue_on_error: bool = False,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
    idempotency_key: str | None = None,
    dry_run: bool = False,
    trace_id: str | None = None,
) -> SubmitResultsResponse:
    """Submit experimental results for a campaign.

    Workflow: Call after running experiments from bo_generate_suggestions.
    Follow up with bo_get_diagnostics to check progress and convergence.

    The MCP transport resolves ``submitted_by`` internally from the current
    BO-MCP user identity. Agents must not provide database user ids.
    """
    validation_error = _validate_results_before_identity(
        results=results,
        trace_id=trace_id,
    )
    if validation_error is not None:
        return cast(SubmitResultsResponse, validation_error)

    try:
        user = await resolve_mcp_user()
    except AuthenticationConfigurationError as exc:
        return cast(SubmitResultsResponse, _mcp_identity_error(exc))

    return cast(
        SubmitResultsResponse,
        await _submit_results_for_user(
            campaign_id=campaign_id,
            results=results,
            submitted_by=str(user.id),
            source=source,
            force=force,
            atomic=atomic,
            continue_on_error=continue_on_error,
            verbosity=verbosity,
            idempotency_key=idempotency_key,
            dry_run=dry_run,
            trace_id=trace_id,
        ),
    )
