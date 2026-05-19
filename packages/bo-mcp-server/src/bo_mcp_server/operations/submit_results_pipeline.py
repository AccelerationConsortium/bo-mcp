"""Two-phase row pipeline for ``submit_results``.

Split from :mod:`bo_mcp_server.operations.submit_results` so the
phase-1 / phase-2 orchestration plus its supporting helpers (duplicate
detection against stored + in-flight baselines, suggestion-id
classification and atomic resolution, reservation partitioning, and the
combined in-batch-duplicate / observation-budget filter) live in one
module.

The public operation entry point in :mod:`.submit_results` calls into
:func:`_validate_and_create_results` after shape-validating its inputs;
this module never opens its own database session — it consumes the
repositories handed in by the caller's ``session_scope`` block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from bo_engine.backend import BOBackend
from bo_engine.constants import DUPLICATE_DETECTION_TOLERANCE

from bo_mcp_server.domain import (
    CampaignSpec,
    Result,
    ResultSource,
    ResultSubmissionInput,
    SuggestionStatus,
)
from bo_mcp_server.domain.campaign_spec import InputParameter
from bo_mcp_server.operations.submit_results_validation import (
    _record_row_error,
    _RowError,
    _SubmitTracking,
    _validate_single_result,
)
from bo_mcp_server.storage import (
    ConcurrentModificationError,
    SuggestionRepository,
)


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
    tracking: _SubmitTracking,
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
    tracking: _SubmitTracking,
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
    tracking: _SubmitTracking,
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
