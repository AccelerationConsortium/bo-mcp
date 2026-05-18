"""Submit results operation — protocol-neutral business logic.

Backend calls that touch the BO engine (hypervolume, TuRBO state updates)
are offloaded via ``asyncio.to_thread`` so they cannot block the FastAPI /
MCP event loop during concurrent result submissions.
"""

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID

from bo_engine.backend import BOBackend
from bo_engine.constants import DUPLICATE_DETECTION_TOLERANCE
from sqlalchemy.ext.asyncio import AsyncSession

from bo_mcp_server.backend import get_backend
from bo_mcp_server.converters import campaign_spec_to_optimization_spec
from bo_mcp_server.domain import (
    Campaign,
    CampaignSpec,
    Result,
    ResultSource,
    ResultSubmissionInput,
    SuggestionStatus,
)
from bo_mcp_server.domain.campaign_spec import InputParameter, ParameterType
from bo_mcp_server.errors import (
    ErrorCode,
    make_concurrent_modification_response,
    make_error_response,
)
from bo_mcp_server.field_errors import add_row_field_error
from bo_mcp_server.idempotency import session_scope
from bo_mcp_server.operations.helpers import (
    parse_campaign_id,
    parse_verbosity,
    results_to_observations,
)
from bo_mcp_server.response_formatter import (
    VerbosityLevel,
    format_submit_results_response,
    with_response_metadata,
)
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    ConcurrentModificationError,
    ResultRepository,
    SuggestionRepository,
)

logger = logging.getLogger(__name__)


def _make_submit_error(
    code: ErrorCode,
    message: str | None = None,
    details: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    duplicates: list[dict[str, Any]] | None = None,
    field_errors: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Build a standardized error response for submit_results."""
    response = make_error_response(code, message=message, details=details)
    response.update(
        {
            "result_ids": [],
            "warnings": warnings or [],
            "duplicates_detected": duplicates or [],
            "field_errors": field_errors or {},
        }
    )
    return response


def _build_submit_dry_run_response(
    campaign_id: str,
    submitted_rows: int,
    persisted_rows: int,
    tracking: "_SubmitTracking",
) -> dict[str, Any]:
    """Compose the dry-run preview envelope for ``submit_results``."""
    logger.info(
        "Dry-run submit for campaign %s: %d row(s) would persist",
        campaign_id,
        persisted_rows,
    )
    return {
        "success": True,
        "dry_run": True,
        "result_ids": [],
        "errors": tracking.errors,
        "field_errors": tracking.field_errors,
        "warnings": tracking.warnings,
        "duplicates_detected": tracking.duplicates_detected,
        "preview": {
            "rows_submitted": submitted_rows,
            "rows_would_persist": persisted_rows,
            "rows_filtered": submitted_rows - persisted_rows,
        },
    }


def _short_circuit_submit(
    campaign_id: str,
    atomic: bool,
    force: bool,
    continue_on_error: bool,
    dry_run: bool,
    result_entities: list[Result],
    submitted_rows: int,
    tracking: "_SubmitTracking",
) -> dict[str, Any] | None:
    """Return the response envelope iff submit_results must exit before save.

    Handles three short-circuit paths in one place so the parent operation
    can stay below ruff's 6-return ceiling: atomic-batch validation
    failure, an empty result set after filtering, and ``dry_run=True``.
    """
    atomic_error = _check_atomic_failures(atomic, force, continue_on_error, campaign_id, tracking)
    if atomic_error is not None:
        return atomic_error

    if not result_entities:
        response_data: dict[str, Any] = {
            "success": False,
            "result_ids": [],
            "errors": (tracking.errors if tracking.errors else ["No valid results to submit"]),
            "field_errors": tracking.field_errors,
            "warnings": tracking.warnings,
            "duplicates_detected": tracking.duplicates_detected,
        }
        if not atomic and continue_on_error:
            response_data["partial_results"] = tracking.partial_results
        return response_data

    if dry_run:
        return _build_submit_dry_run_response(
            campaign_id, submitted_rows, len(result_entities), tracking
        )

    return None


def _validate_submit_inputs(
    campaign_id: str,
    submitted_by: str,
    source: str,
    results: list[ResultSubmissionInput],
    verbosity: str,
) -> tuple[VerbosityLevel, UUID, UUID, ResultSource] | dict[str, Any]:
    """Validate all submit_results inputs.

    Returns (verbosity_level, campaign_uuid, submitter_uuid, result_source)
    on success, or an error response dict.
    """
    verbosity_result = parse_verbosity(verbosity)
    if isinstance(verbosity_result, dict):
        return verbosity_result
    verbosity_level = verbosity_result

    campaign_id_result = parse_campaign_id(campaign_id)
    if isinstance(campaign_id_result, dict):
        # Preserve submit_results-specific error fields
        campaign_id_result.update(
            {
                "result_ids": [],
                "warnings": [],
                "duplicates_detected": [],
                "field_errors": {"campaign_id": ["invalid UUID format"]},
            }
        )
        return campaign_id_result

    campaign_uuid = campaign_id_result

    try:
        submitter_uuid = UUID(submitted_by)
    except ValueError:
        return _make_submit_error(
            ErrorCode.VALIDATION_FAILED,
            message="Invalid submitted_by format",
            details={"submitted_by": submitted_by},
            field_errors={"submitted_by": ["invalid UUID format"]},
        )

    try:
        result_source = ResultSource(source)
    except ValueError:
        return _make_submit_error(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid source '{source}', must be gui, file_upload, or api",
            details={"source": source},
            field_errors={"source": [f"must be gui, file_upload, or api; got '{source}'"]},
        )

    if not results:
        return _make_submit_error(
            ErrorCode.VALIDATION_FAILED,
            message="At least one result is required",
            field_errors={"results": ["at least one result is required"]},
        )

    return verbosity_level, campaign_uuid, submitter_uuid, result_source


@dataclass(frozen=True)
class _RowError:
    """A single row-level validation error with its field path.

    ``field_path`` is dotted relative to the offending row (no
    ``results[i]`` prefix); the prefix is added by
    :func:`_record_row_error` when the error is recorded onto tracking.
    ``message`` is the human-facing string surfaced through the legacy
    ``errors: list[str]`` envelope and the per-row ``partial_results``
    payload — preserving the existing wording so older callers and
    assertions stay green.
    """

    field_path: str
    message: str


def _check_duplicates_for_result(
    index: int,
    params: dict[str, Any],
    existing_params: list[dict[str, Any]],
    batch_so_far_params: list[dict[str, Any]],
    backend: BOBackend,
    warnings: list[str],
    duplicates_detected: list[dict[str, Any]],
) -> _RowError | None:
    """Run duplicate detection for a single result. Returns error string or None.

    Detection runs against two baselines so a duplicate cannot slip
    through by hiding inside the same submission:

    * ``existing_params`` — parameter dicts of already-stored results
      (captured before phase 1 began).
    * ``batch_so_far_params`` — parameter dicts of rows in this same
      submission that have already cleared validation. Without this
      list, two new rows with identical ``parameter_values`` and distinct
      ``suggestion_id``s would both be accepted in a single request.

    Exact duplicates are always treated as a hard row error when the
    caller has not passed ``force=True`` — the row is excluded from
    ``valid_submissions`` and therefore never written. The atomic mode
    decides only the *envelope shape* (standardized
    ``ErrorCode.DUPLICATE_RESULT`` envelope vs. generic
    ``success=False / errors=[...]`` shape, see
    :func:`_check_atomic_failures`); both modes refuse to persist the
    duplicate without an explicit override. Near-duplicates remain
    advisory warnings in every mode.

    Each entry in ``duplicates_detected`` carries a ``duplicate_source``
    of either ``"stored_result"`` or ``"in_flight_batch"`` so clients can
    tell whether the duplicate is against a persisted observation or an
    earlier row in the same request.
    """
    if not existing_params and not batch_so_far_params:
        return None
    combined_baseline = list(existing_params) + list(batch_so_far_params)
    n_stored = len(existing_params)
    duplicates = backend.detect_duplicates(
        new_params=params,
        existing_params=combined_baseline,
        tolerance=DUPLICATE_DETECTION_TOLERANCE,
    )
    if not duplicates:
        return None

    result_error: _RowError | None = None
    for dup in duplicates:
        is_in_batch = dup.index >= n_stored
        relative_index = dup.index - n_stored if is_in_batch else dup.index
        source_key = "in_flight_batch" if is_in_batch else "stored_result"
        duplicates_detected.append(
            {
                "result_index": index,
                "duplicate_of_index": relative_index,
                "duplicate_source": source_key,
                "is_exact": dup.is_exact,
                "parameter_distance": dup.parameter_distance,
            }
        )
        target_descr = (
            f"earlier row at batch index {relative_index} in this submission"
            if is_in_batch
            else f"existing result at index {relative_index}"
        )
        if dup.is_exact:
            warnings.append(
                f"Result {index} appears to be an exact duplicate of "
                f"{target_descr}. Use force=True to submit anyway."
            )
            result_error = _RowError(
                field_path="parameter_values",
                message=f"Result {index} is exact duplicate. Use force=True to override.",
            )
        else:
            warnings.append(
                f"Result {index} is very close to {target_descr} "
                f"(distance={dup.parameter_distance:.6f}). "
                "This may indicate a duplicate measurement."
            )
    return result_error


async def _resolve_suggestion_id(
    suggestion_id_str: str | None,
    index: int,
    campaign_uuid: UUID,
    suggestion_repo: SuggestionRepository,
    warnings: list[str],
    actionable_ids: set[str],
    *,
    dry_run: bool = False,
) -> tuple[UUID | None, dict[str, Any] | None]:
    """Resolve a ``suggestion_id`` and return the snapshot to persist with the result.

    The second tuple element is a JSON-safe snapshot of the originating
    suggestion's parameter values and provenance (TODO 8.11). The
    snapshot is copied onto the persisted ``Result`` row so the BO
    context survives even if the suggestion is later removed —
    ``results.suggestion_id`` is ``ON DELETE SET NULL`` and the
    ORM-level model lets a future cleanup hard-delete the suggestion
    without losing audit context.

    Setting ``dry_run=True`` skips the ``SuggestionStatus.COMPLETED``
    write so the caller can preview a submission without mutating
    suggestion state. The id (and snapshot) is still returned so the
    dry-run preview reflects the row that *would* link to the
    suggestion; only the persistence side-effect is suppressed.

    ``actionable_ids`` is the set of suggestion IDs that phase 1
    classified as actionable (PENDING / ACCEPTED) for this campaign.
    If the caller supplied an id that *was* in that set but the
    re-read here finds the row missing (typically because an admin
    soft-delete landed between phase 1 and phase 2),
    :class:`ConcurrentModificationError` is raised so the phase-2
    loop can route the conflict per submission mode. A
    ``None``-from-``get`` for an id that was *never* in
    ``actionable_ids`` is the historic "caller supplied a bad id"
    path and still falls through to the warning-only free-floating
    behaviour.
    """
    if not suggestion_id_str:
        return None, None
    try:
        suggestion_id = UUID(suggestion_id_str)
    except ValueError:
        warnings.append(f"Result {index}: invalid suggestion_id format")
        return None, None

    suggestion = await suggestion_repo.get(suggestion_id)
    if suggestion is None:
        if suggestion_id_str in actionable_ids:
            # Phase 1 saw this row as actionable, but the re-read
            # found nothing — an admin soft-delete (the only path
            # that can vanish a row under the `RESTRICT` FK) landed
            # between phases. Treat it like the transition race:
            # under ``atomic`` the whole batch aborts, under
            # ``continue_on_error`` the row is skipped.
            raise ConcurrentModificationError("Suggestion", suggestion_id, -1)
        warnings.append(f"Result {index}: suggestion {suggestion_id_str} not found")
        return None, None
    if suggestion.campaign_id != campaign_uuid:
        warnings.append(f"Result {index}: suggestion belongs to different campaign")
        return None, None
    if not dry_run:
        # Atomic ``UPDATE … WHERE status IN (PENDING, ACCEPTED) AND
        # deleted_at IS NULL`` so a concurrent manual status update
        # (PENDING → REJECTED / EXPIRED, ACCEPTED → REJECTED /
        # EXPIRED), a competing ``submit_results`` already completing
        # the same suggestion, or an admin soft-delete cannot be
        # silently clobbered.
        #
        # ``False`` here means the suggestion is no longer actionable.
        # We *cannot* silently link the new result to the now-stale
        # suggestion: the uniqueness index would reject a duplicate
        # COMPLETED→COMPLETED INSERT, and an EXPIRED/REJECTED/
        # soft-deleted suggestion attached to a fresh active result
        # makes the audit trail incoherent. Raising
        # :class:`ConcurrentModificationError` lets the caller
        # decide per submission mode: ``submit_results``'s phase-2
        # loop maps it to a structured conflict envelope under
        # ``atomic=True`` / ``continue_on_error=False``, and to a
        # row-level skip-with-error under ``continue_on_error=True``.
        if not await suggestion_repo.transition_status(
            suggestion_id,
            (SuggestionStatus.PENDING, SuggestionStatus.ACCEPTED),
            SuggestionStatus.COMPLETED,
        ):
            raise ConcurrentModificationError("Suggestion", suggestion_id, -1)
    snapshot = {
        "suggestion_id": str(suggestion.id),
        "parameter_values": dict(suggestion.parameter_values),
        "provenance": suggestion.provenance.model_dump(mode="json"),
        "suggestion_created_at": suggestion.created_at.isoformat(),
    }
    return suggestion_id, snapshot


@dataclass
class _SubmitTracking:
    """Mutable state accumulated during result validation."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duplicates_detected: list[dict[str, Any]] = field(default_factory=list)
    partial_results: dict[int, str | dict[str, str]] = field(default_factory=dict)
    field_errors: dict[str, list[str]] = field(default_factory=dict)


def _record_row_error(
    tracking: _SubmitTracking,
    row_index: int,
    field_path: str,
    message: str,
) -> None:
    """Record a row error in both the legacy ``errors`` list and ``field_errors``.

    The legacy list keeps the human-readable, prefixed message shape
    (``Result 5: ...``) so existing callers and assertions are
    untouched. The ``field_errors`` map indexes the same finding by
    dotted path so agents can target the field directly.
    """
    tracking.errors.append(message)
    add_row_field_error(tracking.field_errors, row_index, field_path, message)


async def _classify_suggestion_reference(
    index: int,
    sid: str | None,
    seen_suggestion_ids: set[str],
    actionable_ids: set[str],
    campaign_uuid: UUID,
    suggestion_repo: SuggestionRepository,
) -> _RowError | None:
    """Validate a row's ``suggestion_id`` against the in-flight batch + DB.

    Returns a :class:`_RowError` (with ``field_path="suggestion_id"``)
    when the reference is unusable, otherwise None. Cases:

    * ``sid is None`` → no-op (free-floating row).
    * ``sid`` already used by an earlier row in this batch → duplicate-id
      error.
    * ``sid`` is in the actionable (PENDING/ACCEPTED) set → accepted; the
      caller adds the id to ``seen_suggestion_ids``.
    * ``sid`` exists in the same campaign but is stale (REJECTED /
      EXPIRED / COMPLETED) → stale-status error.
    * ``sid`` does not exist or belongs to a different campaign → no-op
      (the existing warning-only path in :func:`_resolve_suggestion_id`
      handles these and the row stays in the batch as free-floating).
    """
    if sid is None:
        return None
    if sid in seen_suggestion_ids:
        return _RowError(
            field_path="suggestion_id",
            message=f"Result {index}: duplicate suggestion_id '{sid}' within batch",
        )
    if sid in actionable_ids:
        return None
    stale_status = await _classify_stale_reference(sid, campaign_uuid, suggestion_repo)
    if stale_status is None:
        return None
    return _RowError(
        field_path="suggestion_id",
        message=(
            f"Result {index}: suggestion {sid} is not actionable "
            f"(status={stale_status}); only pending or accepted "
            "suggestions can be completed."
        ),
    )


async def _classify_stale_reference(
    sid: str,
    campaign_uuid: UUID,
    suggestion_repo: SuggestionRepository,
) -> str | None:
    """Return the stale-status reason for ``sid``, or None if it is not stale.

    Helper for :func:`_classify_suggestion_reference`. A ``None`` return
    means the reference is not actionable but is also not a hard error
    (invalid format, missing, or different campaign) -- the row falls
    through to the warning-only path in :func:`_resolve_suggestion_id`.
    """
    try:
        lookup_uuid = UUID(sid)
    except ValueError:
        return None
    suggestion = await suggestion_repo.get(lookup_uuid)
    if suggestion is None or suggestion.campaign_id != campaign_uuid:
        return None
    return suggestion.status.value


@dataclass
class _Phase1Context:
    """Mutable per-batch state threaded into the per-row validators.

    Keeps :func:`_validate_and_create_results` short enough to stay under
    the cognitive-complexity budget while still letting each helper see
    the running set of consumed ``suggestion_id``s. The in-batch
    duplicate baseline is *not* tracked here: that check now runs
    post-budget so a row dropped by the budget guard cannot pollute the
    baseline.
    """

    param_names: set[str]
    objective_names: set[str]
    parameters: list[InputParameter]
    existing_params: list[dict[str, Any]]
    seen_suggestion_ids: set[str]
    actionable_ids: set[str]
    campaign_uuid: UUID
    suggestion_repo: SuggestionRepository
    backend: BOBackend
    force: bool
    tracking: _SubmitTracking


async def _phase1_row_error(
    index: int,
    r: ResultSubmissionInput,
    ctx: _Phase1Context,
) -> _RowError | None:
    """Run phase-1 row checks: shape, suggestion-ref, stored-duplicate.

    The *stored*-duplicate check stays in phase 1 because it depends only
    on data that is fixed for the entire request -- there is no benefit
    to deferring it. The *in-batch* duplicate check is deferred to
    :func:`_apply_in_batch_duplicate_filter` so it can run after the
    observation-budget guard has trimmed rows that will not persist;
    otherwise a free-floating row that ends up dropped by budget could
    pollute the baseline and shadow a later reservation-consuming row.
    """
    shape_error = _validate_single_result(
        index,
        r,
        ctx.param_names,
        ctx.objective_names,
        ctx.tracking.warnings,
        parameters=ctx.parameters,
    )
    if shape_error is not None:
        return shape_error

    ref_error = await _classify_suggestion_reference(
        index,
        r.suggestion_id,
        ctx.seen_suggestion_ids,
        ctx.actionable_ids,
        ctx.campaign_uuid,
        ctx.suggestion_repo,
    )
    if ref_error is not None:
        return ref_error

    if not ctx.force:
        dup_error = _check_duplicates_for_result(
            index,
            r.parameter_values,
            ctx.existing_params,
            [],  # in-batch baseline deferred to post-budget pass
            ctx.backend,
            ctx.tracking.warnings,
            ctx.tracking.duplicates_detected,
        )
        if dup_error is not None:
            return dup_error

    return None


def _partition_reserved(
    valid_submissions: list[tuple[int, ResultSubmissionInput]],
    actionable_ids: set[str],
) -> tuple[
    list[tuple[int, ResultSubmissionInput]],
    list[tuple[int, ResultSubmissionInput]],
]:
    """Split rows into (reservation-consuming, free-floating) in input order.

    A row is reservation-consuming when its ``suggestion_id`` matches an
    actionable suggestion that has not already been claimed by an earlier
    row in this batch. The first row referencing a given ``suggestion_id``
    wins the reservation; subsequent duplicates fall into the
    free-floating group so they compete for slack instead of
    double-consuming a single slot.
    """
    reserved: list[tuple[int, ResultSubmissionInput]] = []
    free_floating: list[tuple[int, ResultSubmissionInput]] = []
    claimed: set[str] = set()
    for idx, row in valid_submissions:
        sid = row.suggestion_id
        if sid is not None and sid in actionable_ids and sid not in claimed:
            reserved.append((idx, row))
            claimed.add(sid)
        else:
            free_floating.append((idx, row))
    return reserved, free_floating


@dataclass
class _FilterState:
    """Mutable per-batch accumulators for the dup + budget filter."""

    accepted_params: list[dict[str, Any]]
    kept: list[tuple[int, ResultSubmissionInput]]


def _check_in_batch_duplicate(
    idx: int,
    row: ResultSubmissionInput,
    state: _FilterState,
    backend: BOBackend,
    force: bool,
    atomic: bool,
    continue_on_error: bool,
    tracking: "_SubmitTracking",
) -> bool:
    """Return True iff the row clears the in-batch duplicate check.

    On failure the row error is recorded on ``tracking`` (both the
    legacy ``errors`` list and the keyed ``field_errors`` map) and, in
    non-atomic + continue mode, on ``partial_results``.
    """
    if force:
        return True
    dup_error = _check_duplicates_for_result(
        idx,
        row.parameter_values,
        [],  # stored already checked in phase 1
        state.accepted_params,
        backend,
        tracking.warnings,
        tracking.duplicates_detected,
    )
    if dup_error is None:
        return True
    _record_row_error(tracking, idx, dup_error.field_path, dup_error.message)
    if not atomic and continue_on_error:
        tracking.partial_results[idx] = {"error": dup_error.message}
    return False


def _accept_row(
    idx: int,
    row: ResultSubmissionInput,
    state: _FilterState,
) -> None:
    """Add a passing row to the kept list and the duplicate baseline."""
    state.kept.append((idx, row))
    state.accepted_params.append(row.parameter_values)


def _reject_free_floating(
    idx: int,
    spec: CampaignSpec,
    existing_count: int,
    n_actionable: int,
    atomic: bool,
    continue_on_error: bool,
    tracking: "_SubmitTracking",
) -> None:
    """Record a budget-overflow rejection for a free-floating row."""
    err = (
        f"Result {idx}: would exceed max_observations="
        f"{spec.max_observations} "
        f"(existing={existing_count}, pending_reserved={n_actionable})."
    )
    # Budget violation is a row-level (not field-level) constraint, so
    # the path bottoms out at ``results[idx]`` itself.
    _record_row_error(tracking, idx, "", err)
    if not atomic and continue_on_error:
        tracking.partial_results[idx] = {"error": err}


def _apply_dup_and_budget_filter(
    valid_submissions: list[tuple[int, ResultSubmissionInput]],
    spec: CampaignSpec,
    *,
    existing_count: int,
    actionable_ids: set[str],
    backend: BOBackend,
    force: bool,
    atomic: bool,
    continue_on_error: bool,
    tracking: "_SubmitTracking",
) -> list[tuple[int, ResultSubmissionInput]]:
    """Combined in-batch duplicate + observation-budget filter.

    Order of operations:

    1. **Partition** rows into reservation-consuming (``suggestion_id``
       matches an actionable PENDING/ACCEPTED suggestion, first claim
       wins) vs free-floating.
    2. **Pass A — reserved rows first.** Reservation rows are processed
       in input order. They are accepted unless they duplicate an
       earlier accepted reservation. This guarantees a reservation
       always wins a duplicate tie against a later or earlier
       free-floating row — committing the reserved result transitions
       the suggestion to ``COMPLETED`` and closes the user's commitment
       cleanly. (Letting a manual duplicate win would orphan the
       reservation in PENDING/ACCEPTED forever.)
    3. **Pass B — free-floating rows.** Walked in input order. Each
       row is rejected when it duplicates an already-accepted row
       (reserved or earlier free-floating) and otherwise competes for
       the unreserved slack ``cap - existing - len(actionable_ids)``;
       overflow rows fail with the budget error. A duplicate row never
       consumes slack, so a free-floating duplicate cannot starve a
       later unique free-floating row.

    Notes:
    * The stored-duplicate check happens earlier, per-row, in
      :func:`_phase1_row_error`. This function only handles in-batch
      duplicates so a row that is itself going to be dropped by the
      budget cannot pollute the duplicate baseline.
    * ``force=True`` skips the duplicate check entirely (the documented
      override) but the budget guard still applies.
    * When ``spec.max_observations`` is ``None`` the budget guard is a
      no-op; only the duplicate check runs.
    """
    reserved, free_floating = _partition_reserved(valid_submissions, actionable_ids)
    n_actionable = len(actionable_ids)
    has_cap = spec.max_observations is not None
    state = _FilterState(accepted_params=[], kept=[])

    # Pass A: reservation-consuming rows. They are not slack-constrained
    # (their reservation is the slot) and they get priority over
    # free-floating rows on duplicate ties so committing them
    # transitions their suggestion to COMPLETED rather than orphaning
    # the reservation.
    for idx, row in reserved:
        if _check_in_batch_duplicate(
            idx, row, state, backend, force, atomic, continue_on_error, tracking
        ):
            _accept_row(idx, row, state)

    # Pass B: free-floating rows compete for the unreserved slack.
    if spec.max_observations is None:
        slack = 0
    else:
        slack = max(int(spec.max_observations) - existing_count - n_actionable, 0)
    for idx, row in free_floating:
        if not _check_in_batch_duplicate(
            idx, row, state, backend, force, atomic, continue_on_error, tracking
        ):
            continue
        if has_cap and slack <= 0:
            _reject_free_floating(
                idx, spec, existing_count, n_actionable, atomic, continue_on_error, tracking
            )
            continue
        if has_cap:
            slack -= 1
        _accept_row(idx, row, state)

    # Output in input order so downstream ``entity_to_input_index``
    # mapping stays consistent with the caller's view.
    return sorted(state.kept, key=lambda pair: pair[0])


async def _validate_and_create_results(
    results: list[ResultSubmissionInput],
    spec: CampaignSpec,
    campaign_uuid: UUID,
    submitter_uuid: UUID,
    result_source: ResultSource,
    backend: BOBackend,
    existing_params: list[dict[str, Any]],
    suggestion_repo: SuggestionRepository,
    force: bool,
    atomic: bool,
    continue_on_error: bool,
    tracking: _SubmitTracking,
    *,
    dry_run: bool = False,
) -> tuple[list[Result], dict[int, int]]:
    """Validate inputs and materialize Result entities.

    The function runs in two phases so that the ``atomic`` flag can honor its
    all-or-nothing contract. Phase 1 is purely read-only: it inspects each
    submission against the spec and the existing parameter set, collecting
    errors, warnings and duplicate findings on ``tracking`` without issuing
    any DB writes. In atomic mode, a non-empty error list after phase 1 skips
    phase 2 entirely — the surrounding session therefore has nothing to commit
    and the campaign state is unchanged. Phase 2 only runs for submissions
    that passed phase 1 and is where suggestion-status updates (a write) and
    ``Result`` entity construction happen.

    Returns (result_entities, entity_to_input_index).
    """
    param_names = {p.name for p in spec.parameters}
    objective_names = {o.name for o in spec.objectives}
    parameters = list(spec.parameters)

    # Fetch the actionable (PENDING+ACCEPTED) set upfront. It feeds the
    # inline suggestion-reference check below *and* the budget guard later.
    actionable_suggestions = await suggestion_repo.list_actionable_by_campaign(campaign_uuid)
    actionable_ids = {str(s.id) for s in actionable_suggestions}

    # Phase 1 — pure validation; no writes hit the session.
    #
    # Per-row pipeline (order matters):
    #   1. Basic shape validation (param/objective presence, bounds,
    #      measurement-uncertainty finiteness).
    #   2. Suggestion-reference validation (duplicate actionable-id
    #      within batch, stale status). Only actionable IDs are tracked
    #      in ``seen_suggestion_ids`` so two rows sharing a missing or
    #      foreign-campaign id both fall through as free-floating.
    #   3. Stored-result duplicate detection against ``existing_params``.
    #
    # In-batch duplicate detection and budget filtering run later in
    # ``_apply_dup_and_budget_filter`` so a row dropped by either check
    # cannot pollute the cross-row baselines used by the other.
    valid_submissions: list[tuple[int, ResultSubmissionInput]] = []
    seen_suggestion_ids: set[str] = set()
    ctx = _Phase1Context(
        param_names=param_names,
        objective_names=objective_names,
        parameters=parameters,
        existing_params=existing_params,
        seen_suggestion_ids=seen_suggestion_ids,
        actionable_ids=actionable_ids,
        campaign_uuid=campaign_uuid,
        suggestion_repo=suggestion_repo,
        backend=backend,
        force=force,
        tracking=tracking,
    )

    for i, r in enumerate(results):
        row_error = await _phase1_row_error(i, r, ctx)
        if row_error is not None:
            _record_row_error(tracking, i, row_error.field_path, row_error.message)
            if not atomic and continue_on_error:
                tracking.partial_results[i] = {"error": row_error.message}
            continue
        valid_submissions.append((i, r))
        # Only claim actionable suggestion_ids in the seen-set. Missing,
        # invalid, or different-campaign IDs fall through the
        # warning-only branch in ``_classify_suggestion_reference`` and
        # must NOT trip the duplicate-id check on the next row sharing
        # the same bogus value; two rows with the same typo are both
        # free-floating and should commit (subject to budget +
        # parameter-duplicate rules).
        if r.suggestion_id is not None and r.suggestion_id in actionable_ids:
            seen_suggestion_ids.add(r.suggestion_id)

    # Combined in-batch duplicate + observation-budget filter. Running
    # both checks in a single input-order walk avoids two phantom-shadow
    # failure modes:
    #   * a budget-dropped row polluting the duplicate baseline (which
    #     would shadow a later reserved row with the same params);
    #   * a duplicate row consuming free-floating slack before being
    #     filtered out (which would force-reject a later unique row that
    #     could otherwise fit).
    # See :func:`_apply_dup_and_budget_filter`.
    valid_submissions = _apply_dup_and_budget_filter(
        valid_submissions,
        spec,
        existing_count=len(existing_params),
        actionable_ids=actionable_ids,
        backend=backend,
        force=force,
        atomic=atomic,
        continue_on_error=continue_on_error,
        tracking=tracking,
    )

    # Abort before phase 2 unless the caller has explicitly opted into
    # partial-write semantics (``atomic=False`` AND ``continue_on_error=True``).
    # Without that opt-in, errors are correctness signals, not "skip-and-keep"
    # signals -- writing the surviving rows while returning ``success=False``
    # leaves the caller with mutated state and no ``partial_results`` to
    # reconcile from.
    if tracking.errors and not (not atomic and continue_on_error):
        return [], {}

    return await _phase2_build_results(
        valid_submissions,
        campaign_uuid,
        submitter_uuid,
        result_source,
        suggestion_repo,
        tracking,
        actionable_ids=actionable_ids,
        atomic=atomic,
        continue_on_error=continue_on_error,
        dry_run=dry_run,
    )


async def _phase2_build_results(
    valid_submissions: list[tuple[int, ResultSubmissionInput]],
    campaign_uuid: UUID,
    submitter_uuid: UUID,
    result_source: ResultSource,
    suggestion_repo: SuggestionRepository,
    tracking: _SubmitTracking,
    *,
    actionable_ids: set[str],
    atomic: bool,
    continue_on_error: bool,
    dry_run: bool,
) -> tuple[list[Result], dict[int, int]]:
    """Phase 2 — build ``Result`` entities and atomically mark suggestions COMPLETED.

    All writes share the caller's session so save_batch and any
    downstream campaign-state update commit or roll back together.

    ``actionable_ids`` is the phase-1 set of suggestion IDs that
    were classified as actionable (PENDING / ACCEPTED) for this
    campaign. It's threaded into :func:`_resolve_suggestion_id` so
    a ``get()`` that returns ``None`` for an id phase 1 had
    accepted (an admin soft-delete that landed between phases) is
    raised as :class:`ConcurrentModificationError` instead of
    silently degrading the row to free-floating.

    ``_resolve_suggestion_id`` raises ``ConcurrentModificationError``
    in two situations:

    * The phase-1-actionable row disappeared between phases (the
      ``get() -> None`` case above).
    * The atomic ``transition_status`` lost a race — the suggestion
      was concurrently completed by another submit, rejected /
      expired by a status update, or soft-deleted after the re-read.

    Under ``atomic=True`` / ``continue_on_error=False`` we re-raise
    so the outer handler returns a structured
    ``CONCURRENT_MODIFICATION`` envelope; under
    ``continue_on_error=True`` we drop the row from the batch and
    record a row-level error so the rest of the batch can still
    commit. We never link a fresh active result to a suggestion that
    is no longer actionable, and we never silently demote a
    suggestion-linked submission to free-floating because of a
    concurrent change.
    """
    result_entities: list[Result] = []
    entity_to_input_index: dict[int, int] = {}
    for i, r in valid_submissions:
        try:
            suggestion_id, suggestion_snapshot = await _resolve_suggestion_id(
                r.suggestion_id,
                i,
                campaign_uuid,
                suggestion_repo,
                tracking.warnings,
                actionable_ids,
                dry_run=dry_run,
            )
        except ConcurrentModificationError:
            if atomic or not continue_on_error:
                raise
            conflict_msg = (
                f"Result {i}: suggestion {r.suggestion_id} status changed "
                "concurrently (rejected, expired, completed, or soft-deleted) "
                "before this result could be linked; row skipped"
            )
            _record_row_error(tracking, i, "suggestion_id", conflict_msg)
            tracking.partial_results[i] = {"error": conflict_msg}
            continue

        result = Result(
            campaign_id=campaign_uuid,
            suggestion_id=suggestion_id,
            parameter_values=r.parameter_values,
            objective_values=r.objective_values,
            source=result_source,
            submitted_by=submitter_uuid,
            measurement_uncertainty=r.measurement_uncertainty,
            metadata=r.metadata,
            suggestion_snapshot=suggestion_snapshot,
        )
        entity_to_input_index[len(result_entities)] = i
        result_entities.append(result)

    return result_entities, entity_to_input_index


def _check_numeric_bounds(
    param: InputParameter,
    value: int | float | str,
    index: int,
    warnings: list[str],
) -> None:
    """Check a numeric value against parameter bounds."""
    if param.bounds is None:
        return
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        pname = param.name
        vtype = type(value).__name__
        warnings.append(f"Result {index}: parameter '{pname}' expected numeric, got {vtype}")
        return
    if numeric < param.bounds.lower or numeric > param.bounds.upper:
        warnings.append(
            f"Result {index}: parameter '{param.name}' value {numeric} "
            f"is outside spec bounds "
            f"[{param.bounds.lower}, {param.bounds.upper}]"
        )


def _validate_parameter_value(
    param: InputParameter,
    value: int | float | str,
    index: int,
    warnings: list[str],
) -> None:
    """Check a single parameter value against its spec definition.

    Out-of-bounds values are reported as warnings rather than hard errors
    because real experiments may intentionally exceed spec bounds.
    """
    if param.type == ParameterType.CONTINUOUS:
        _check_numeric_bounds(param, value, index, warnings)
    elif param.type == ParameterType.DISCRETE:
        if param.values is not None and value not in param.values:
            warnings.append(
                f"Result {index}: parameter '{param.name}' value {value} "
                f"is not in allowed discrete values {param.values}"
            )
        else:
            _check_numeric_bounds(param, value, index, warnings)
    elif param.type == ParameterType.CATEGORICAL:
        if param.categories is not None and value not in param.categories:
            warnings.append(
                f"Result {index}: parameter '{param.name}' value "
                f"'{value}' is not in allowed categories "
                f"{param.categories}"
            )


def _validate_measurement_uncertainty(
    uncertainty: dict[str, float],
    objective_names: set[str],
    index: int,
    warnings: list[str],
) -> _RowError | None:
    """Validate measurement uncertainty keys and values.

    Unknown objective keys are surfaced as warnings (they are dropped on the
    way into bo-engine and do not corrupt the GP).

    Numerical defects on declared-objective values are hard errors: the
    bo-engine squares the stddev into ``train_yvar`` and routes the GP onto a
    ``FixedNoiseGaussianLikelihood``, so a negative value would silently
    become a positive variance and NaN/inf would propagate into MLL. The
    returned :class:`_RowError` pins the offending key in its
    ``field_path`` so the caller can surface it in ``field_errors``.
    """
    invalid_keys = set(uncertainty.keys()) - objective_names
    if invalid_keys:
        warnings.append(
            f"Result {index}: measurement_uncertainty has unknown "
            f"objective keys: {sorted(invalid_keys)}"
        )
    for obj_name, unc_val in uncertainty.items():
        # Unknown objective keys are stripped before reaching the engine, so
        # we tolerate odd values on them with the warning above. Declared
        # objectives, however, drive the GP and must be sane.
        if obj_name not in objective_names:
            continue
        if not isinstance(unc_val, (int, float)) or math.isnan(unc_val) or math.isinf(unc_val):
            return _RowError(
                field_path=f"measurement_uncertainty['{obj_name}']",
                message=(
                    f"Result {index}: measurement_uncertainty['{obj_name}'] "
                    f"is not a finite number: {unc_val}"
                ),
            )
        if unc_val < 0:
            return _RowError(
                field_path=f"measurement_uncertainty['{obj_name}']",
                message=(
                    f"Result {index}: measurement_uncertainty['{obj_name}'] "
                    f"is negative ({unc_val}); expected non-negative std"
                ),
            )
    return None


def _validate_single_result(
    index: int,
    r: ResultSubmissionInput,
    param_names: set[str],
    objective_names: set[str],
    warnings: list[str],
    parameters: list[InputParameter] | None = None,
) -> _RowError | None:
    """Validate a single result's parameter and objective *shape*.

    Shape-only checks: parameter presence, parameter spec validation
    (bounds/categories), objective presence, measurement-uncertainty
    finiteness. Cross-row checks (duplicate parameters, duplicate or
    stale ``suggestion_id``) live in the per-row loop in
    :func:`_validate_and_create_results` so they can see the running
    state of accepted rows; running them here would let a later row be
    rejected as the duplicate of an earlier row that was itself dropped
    by some other validator.
    """
    if missing_params := (param_names - set(r.parameter_values.keys())):
        return _RowError(
            field_path="parameter_values",
            message=f"Result {index} missing parameters: {missing_params}",
        )

    # Validate parameter values against spec bounds/categories
    if parameters is not None:
        param_by_name = {p.name: p for p in parameters}
        for pname, pvalue in r.parameter_values.items():
            if pname in param_by_name:
                _validate_parameter_value(param_by_name[pname], pvalue, index, warnings)

    if missing_objectives := (objective_names - set(r.objective_values.keys())):
        return _RowError(
            field_path="objective_values",
            message=f"Result {index} missing objectives: {missing_objectives}",
        )

    if r.measurement_uncertainty is not None:
        unc_error = _validate_measurement_uncertainty(
            r.measurement_uncertainty, objective_names, index, warnings
        )
        if unc_error is not None:
            return unc_error

    return None


async def _update_campaign_state(
    campaign: Campaign,
    spec: CampaignSpec,
    backend: BOBackend,
    campaign_uuid: UUID,
    campaign_id: str,
    result_entities: list[Result],
    result_repo: ResultRepository,
    campaign_repo: CampaignRepository,
    warnings: list[str],
) -> None:
    """Update hypervolume history and backend state after results are saved."""
    updated_campaign = campaign
    opt_spec = campaign_spec_to_optimization_spec(spec)

    if len(spec.objectives) >= 2:
        try:
            all_results = await result_repo.list_by_campaign(campaign_uuid)
            all_observations = results_to_observations(all_results)
            hv = await asyncio.to_thread(backend.compute_hypervolume, opt_spec, all_observations)
            if hv is not None:
                updated_campaign = updated_campaign.with_hypervolume(hv)
                logger.debug(
                    "Updated hypervolume history for campaign %s: hv=%.4f",
                    campaign_id,
                    hv,
                )
        except (RuntimeError, ValueError, TypeError) as e:
            logger.debug("Failed to compute hypervolume: %s", e)
            warnings.append(f"Could not compute hypervolume: {e}")

    if campaign.backend_state is not None:
        try:
            new_obs = results_to_observations(result_entities)
            new_state = await asyncio.to_thread(
                backend.update_state_after_results,
                opt_spec,
                new_obs,
                campaign.backend_state,
            )
            if new_state is not None:
                updated_campaign = updated_campaign.with_backend_state(new_state)
        except (RuntimeError, ValueError, TypeError) as e:
            logger.debug("Failed to update backend state: %s", e)
            warnings.append(f"Could not update backend state: {e}")

    if updated_campaign.version != campaign.version:
        await campaign_repo.save(updated_campaign, expected_version=campaign.version)


async def _fetch_campaign_and_spec(
    campaign_id: str,
    campaign_uuid: UUID,
    campaign_repo: CampaignRepository,
    spec_repo: CampaignSpecRepository,
) -> tuple[Campaign, CampaignSpec] | dict[str, Any]:
    """Fetch and validate campaign and spec. Returns (campaign, spec) or error dict."""
    campaign = await campaign_repo.get(campaign_uuid)
    if campaign is None:
        return _make_submit_error(
            ErrorCode.CAMPAIGN_NOT_FOUND,
            message=f"Campaign {campaign_id} not found",
            details={"campaign_id": campaign_id},
        )
    if not campaign.can_submit_results:
        return _make_submit_error(
            ErrorCode.INVALID_STATE_TRANSITION,
            message=f"Campaign status is {campaign.status.value}, cannot submit results",
            details={"current_status": campaign.status.value},
        )
    spec = await spec_repo.get(campaign.spec_id)
    if spec is None:
        return _make_submit_error(
            ErrorCode.DATABASE_ERROR,
            message="Campaign spec not found",
            details={"spec_id": str(campaign.spec_id)},
        )
    return campaign, spec


def _check_atomic_failures(
    atomic: bool,
    force: bool,
    continue_on_error: bool,
    campaign_id: str,
    tracking: _SubmitTracking,
) -> dict[str, Any] | None:
    """Check for all-or-nothing failures (validation errors, exact duplicates).

    Atomic mode is always all-or-nothing. Non-atomic mode is also all-or-
    nothing unless the caller opted into partial writes by passing
    ``continue_on_error=True`` — without that opt-in we must not commit
    rows while returning ``success=False``.

    Returns error response dict or None.
    """
    # Exact duplicates get the standardized ``ErrorCode.DUPLICATE_RESULT``
    # envelope (with recovery_action) before the generic-errors branch
    # fires. Without this ordering, atomic mode would return a plain
    # ``success=False / errors=[...]`` shape because
    # ``_check_duplicates_for_result`` also appends a row-level error to
    # ``tracking.errors`` -- clients would lose the duplicate-specific
    # response shape and the ``Use force=True`` recovery hint.
    exact_duplicates = [d for d in tracking.duplicates_detected if d.get("is_exact")]
    if atomic and exact_duplicates and not force:
        logger.warning(
            "Exact duplicates detected for campaign %s: %s",
            campaign_id,
            exact_duplicates,
        )
        return _make_submit_error(
            ErrorCode.DUPLICATE_RESULT,
            message="Exact duplicate results detected. Use force=True to override.",
            details={"duplicate_count": len(exact_duplicates)},
            warnings=tracking.warnings,
            duplicates=tracking.duplicates_detected,
            field_errors=tracking.field_errors,
        )

    all_or_nothing = atomic or not continue_on_error
    if all_or_nothing and tracking.errors:
        mode = "atomic mode" if atomic else "non-atomic, continue_on_error=False"
        logger.warning("Result submission validation failed (%s): %s", mode, tracking.errors)
        return {
            "success": False,
            "result_ids": [],
            "errors": tracking.errors,
            "field_errors": tracking.field_errors,
            "warnings": tracking.warnings,
            "duplicates_detected": tracking.duplicates_detected,
        }

    return None


def _build_submit_response(
    result_ids: list[str],
    tracking: _SubmitTracking,
    atomic: bool,
    continue_on_error: bool,
) -> dict[str, Any]:
    """Build the final success/partial-success response dict."""
    has_errors = len(tracking.errors) > 0
    partial_success_allowed = not atomic and continue_on_error
    overall_success = len(result_ids) > 0 and (not has_errors or partial_success_allowed)

    response_data: dict[str, Any] = {
        "success": overall_success,
        "result_ids": result_ids,
        "errors": tracking.errors,
        "field_errors": tracking.field_errors,
        "warnings": tracking.warnings,
        "duplicates_detected": tracking.duplicates_detected,
    }
    if not atomic and continue_on_error:
        response_data["partial_results"] = tracking.partial_results
    return response_data


async def _handle_submit_results_cm_error(
    err: ConcurrentModificationError,
    campaign_id: str,
    session: AsyncSession | None,
    tracking: _SubmitTracking,
) -> dict[str, Any]:
    """Build the structured envelope for an optimistic-lock conflict.

    When a caller (typically ``apply_idempotency``'s session-aware
    path) supplied the session, ``session_scope`` only yields it —
    partial writes that landed before the conflict (Result rows saved
    via ``result_repo.save_batch`` at phase 2 before
    ``_update_campaign_state`` raised) would otherwise survive the
    outer commit. Roll back explicitly so the conflict leaves no
    observable state.
    """
    if session is not None:
        await session.rollback()
    logger.warning(
        "Concurrent modification while submitting results for campaign %s: %s",
        campaign_id,
        err,
    )
    response = make_concurrent_modification_response(
        err, extra_details={"campaign_id": campaign_id}
    )
    response.update(
        {
            "result_ids": [],
            "warnings": tracking.warnings,
            "duplicates_detected": tracking.duplicates_detected,
            "field_errors": tracking.field_errors,
        }
    )
    return response


@with_response_metadata
async def submit_results_operation(
    campaign_id: str,
    results: list[ResultSubmissionInput],
    submitted_by: str,
    source: str = "api",
    force: bool = False,
    atomic: bool = True,
    continue_on_error: bool = False,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
    *,
    session: AsyncSession | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Submit experimental results for a campaign.

    Atomicity contract: when ``atomic=True`` (the default) the operation
    pre-validates every submission before issuing any DB write. If any
    submission fails validation — missing parameters/objectives, spec-bound
    violations, exact duplicate detection — the entire batch is rejected
    before the first suggestion status update or result row is persisted,
    so the campaign's observable state (results, suggestion statuses,
    hypervolume history, backend state) is unchanged. When ``atomic=False``
    and ``continue_on_error=True`` each result is saved independently and
    partial persistence is expected; callers MUST inspect
    ``partial_results`` to see which indices succeeded.

    When ``dry_run`` is true the operation runs full validation
    (parameter bounds, duplicate detection, suggestion-id resolution,
    atomic-failure gates) but returns before persisting any row or
    advancing campaign / backend state. The response carries
    ``dry_run: True`` plus a ``preview`` block summarizing how many
    rows would persist and how many were filtered.

    Args:
        campaign_id: UUID of the campaign
        results: List of result payloads
        submitted_by: UUID of the user submitting results
        source: Result source ("gui", "file_upload", or "api")
        force: If True, skip duplicate detection
        atomic: If True, reject the entire batch on any validation error;
            no partial writes occur. If False, invalid rows are skipped and
            valid rows are persisted (subject to ``continue_on_error``).
        continue_on_error: If True, continue after errors (ignored if atomic)
        verbosity: Response detail level
        dry_run: If True, perform full validation and return a preview
            without persisting anything.

    Returns:
        Dictionary with success, result_ids, errors, warnings, duplicates_detected.
    """
    logger.info(
        "Submitting %d results for campaign_id=%s by user=%s, verbosity=%s",
        len(results),
        campaign_id,
        submitted_by,
        verbosity,
    )

    validated = _validate_submit_inputs(campaign_id, submitted_by, source, results, verbosity)
    if isinstance(validated, dict):
        return validated
    verbosity_level, campaign_uuid, submitter_uuid, result_source = validated

    tracking = _SubmitTracking()

    try:
        async with session_scope(session) as db:
            campaign_repo = CampaignRepository(db)
            spec_repo = CampaignSpecRepository(db)
            result_repo = ResultRepository(db)
            suggestion_repo = SuggestionRepository(db)

            fetched = await _fetch_campaign_and_spec(
                campaign_id,
                campaign_uuid,
                campaign_repo,
                spec_repo,
            )
            if isinstance(fetched, dict):
                return fetched
            campaign, spec = fetched
            backend = get_backend(spec.backend)

            existing_results = await result_repo.list_by_campaign(campaign_uuid)
            existing_params = [r.parameter_values for r in existing_results]

            result_entities, entity_to_input_index = await _validate_and_create_results(
                results,
                spec,
                campaign_uuid,
                submitter_uuid,
                result_source,
                backend,
                existing_params,
                suggestion_repo,
                force,
                atomic,
                continue_on_error,
                tracking,
                dry_run=dry_run,
            )

            short_circuit = _short_circuit_submit(
                campaign_id,
                atomic,
                force,
                continue_on_error,
                dry_run,
                result_entities,
                len(results),
                tracking,
            )
            if short_circuit is not None:
                return short_circuit

            saved_results = await result_repo.save_batch(result_entities)
            result_ids = [str(r.id) for r in saved_results]

            if not atomic and continue_on_error:
                for entity_idx, saved in enumerate(saved_results):
                    tracking.partial_results[entity_to_input_index[entity_idx]] = str(saved.id)

            await _update_campaign_state(
                campaign,
                spec,
                backend,
                campaign_uuid,
                campaign_id,
                result_entities,
                result_repo,
                campaign_repo,
                tracking.warnings,
            )

            logger.info(
                "Successfully submitted %d results for campaign %s",
                len(result_ids),
                campaign_id,
            )

            return format_submit_results_response(
                _build_submit_response(result_ids, tracking, atomic, continue_on_error),
                verbosity_level,
            )
    except ConcurrentModificationError as err:
        return await _handle_submit_results_cm_error(err, campaign_id, session, tracking)
