"""Generate suggestions operation — protocol-neutral business logic.

Extracted from the MCP tool so it can be reused by any transport
(MCP, REST, CLI, tests) without importing the MCP server object.

Uses the BOBackend protocol for all BO-engine interactions,
keeping the server decoupled from specific backend implementations.

Heavy BO-engine work (GP fitting, acquisition optimization, Sobol sampling)
is offloaded via ``asyncio.to_thread`` so it does not block the FastAPI /
MCP event loop while other requests are served concurrently.
"""

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from bo_engine.backend import BOBackend
from bo_engine.backend_base import BackendError, BackendTransientError
from bo_engine.constants import (
    PENDING_SUGGESTION_MAX_AGE_HOURS,
    resolve_initial_design_size,
)
from bo_engine.convergence import (
    StoppingDecision,
    StoppingReason,
    evaluate_stopping_decision,
)
from bo_engine.initial_design import SearchSpaceExhaustedError
from bo_engine.pending_points import filter_pending_points
from bo_engine.progress import ProgressCallback
from bo_engine.types import ObservationData, OptimizationSpec
from bo_mcp_server.backend import get_backend_async
from bo_mcp_server.constants import MAX_GENERATION_BATCH_SIZE
from bo_mcp_server.converters import campaign_spec_to_optimization_spec
from bo_mcp_server.domain import (
    Campaign,
    CampaignSpec,
    CampaignStatus,
    Suggestion,
    SuggestionProvenance,
    SuggestionStatus,
)
from bo_mcp_server.errors import (
    ErrorCode,
    make_backend_error_response,
    make_concurrent_modification_response,
    make_error_response,
)
from bo_mcp_server.idempotency import reservation_heartbeat, session_scope
from bo_mcp_server.metrics import observe_suggestion_latency
from bo_mcp_server.operations.backend_output import (
    BackendOutputError,
    BackendStateTooLargeError,
    validate_backend_batch,
)
from bo_mcp_server.operations.helpers import (
    parse_campaign_id,
    parse_verbosity,
    results_to_observations,
)
from bo_mcp_server.response_formatter import (
    VerbosityLevel,
    format_suggestions_response,
    with_response_metadata,
)
from bo_mcp_server.settings import (
    get_bo_compute_timeout_seconds,
    get_max_backend_state_bytes,
)
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    ConcurrentModificationError,
    ResultRepository,
    SuggestionRepository,
)
from bo_mcp_server.storage.models import ResultModel, SuggestionModel
from bo_mcp_server.subscriptions import notify_campaign_updated_after_commit

logger = logging.getLogger(__name__)

# Type alias for suggestion data tuples (params, provenance)
SuggestionDataList = list[tuple[dict[str, Any], dict[str, Any]]]


@dataclass
class _Repositories:
    """Repository instances for a single database session."""

    campaign: CampaignRepository
    spec: CampaignSpecRepository
    result: ResultRepository
    suggestion: SuggestionRepository


async def _create_and_save_suggestions(
    suggestion_data: SuggestionDataList,
    campaign_id: UUID,
    suggestion_repo: SuggestionRepository,
) -> list[Suggestion]:
    """Create Suggestion entities from data and save to database."""
    suggestions = []
    for params, prov_data in suggestion_data:
        provenance = SuggestionProvenance(**prov_data) if isinstance(prov_data, dict) else prov_data

        suggestion = Suggestion(
            campaign_id=campaign_id,
            parameter_values=params,
            status=SuggestionStatus.PENDING,
            provenance=provenance,
        )
        saved = await suggestion_repo.save(suggestion)
        suggestions.append(saved)
    return suggestions


def _make_suggestions_error(
    code: ErrorCode,
    message: str,
    iteration: int | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an error response with suggestion-specific fields."""
    response = make_error_response(code, message=message, details=details)
    response.update({"suggestions": [], "iteration": iteration})
    return response


def _make_exhaustion_response(err: SearchSpaceExhaustedError, campaign_id: str) -> dict[str, Any]:
    """Build the SEARCH_SPACE_EXHAUSTED envelope, subsample-aware.

    On the BayBE subsampled path the exhausted candidate set is a bounded
    sample, not the full space — ``n_total_combinations`` counts the
    subsample while the real space holds ``n_full_combinations``. The
    default recovery text ("no unseen combinations left, terminate")
    would then be factually wrong and close a viable campaign, so the
    envelope recommends raising
    ``backend_options['baybe'].max_candidates`` instead.
    """
    details: dict[str, Any] = {
        "campaign_id": campaign_id,
        "n_requested": err.n_requested,
        "n_available": err.n_available,
        "n_total_combinations": err.n_total_combinations,
        "next_action_recommendation": "terminate_campaign",
    }
    if err.subsampled:
        details.update(
            {
                "searchspace_subsampled": True,
                "n_full_combinations": err.n_full_combinations,
                "max_candidates": err.max_candidates,
                "next_action_recommendation": "increase_max_candidates",
            }
        )
    response = _make_suggestions_error(
        ErrorCode.SEARCH_SPACE_EXHAUSTED,
        message=str(err),
        details=details,
    )
    if err.subsampled:
        response["error"]["recovery_action"] = (
            "The exhausted candidate set is a bounded subsample "
            f"({err.n_total_combinations} candidates) of a "
            f"~{err.n_full_combinations}-combination space, so the campaign is "
            "not actually out of unseen combinations. Raise "
            "backend_options['baybe'].max_candidates (and, if needed, "
            "max_searchspace_state_bytes) or switch to backend='botorch' "
            "before considering termination."
        )
    return response


_STOPPING_REASON_TO_CODE: dict[StoppingReason, ErrorCode] = {
    StoppingReason.BUDGET_EXCEEDED_ITERATIONS: ErrorCode.BUDGET_EXCEEDED,
    StoppingReason.BUDGET_EXCEEDED_OBSERVATIONS: ErrorCode.BUDGET_EXCEEDED,
    StoppingReason.CONVERGED: ErrorCode.CAMPAIGN_CONVERGED,
}


def _build_stopping_response(
    stopping: StoppingDecision,
    iteration: int,
    campaign_id: str,
) -> dict[str, Any]:
    """Render a :class:`StoppingDecision` as a structured response envelope.

    The shape mirrors ``SEARCH_SPACE_EXHAUSTED`` so agent loops can route on
    ``next_action_recommendation == "terminate_campaign"`` without parsing
    free-form text.
    """
    if stopping.reason is None:
        msg = "StoppingDecision must have a reason when rendered as a stopping envelope"
        raise ValueError(msg)
    code = _STOPPING_REASON_TO_CODE[stopping.reason]
    details = {
        "campaign_id": campaign_id,
        "stopping_reason": stopping.reason.value,
        **stopping.details,
    }
    return _make_suggestions_error(
        code,
        message=stopping.message,
        iteration=iteration,
        details=details,
    )


def _status_breakdown(suggestions: list[Suggestion]) -> dict[str, int]:
    """Split a list of actionable suggestions by ``PENDING``/``ACCEPTED``.

    Used wherever a response previously surfaced a single ``n_pending`` /
    ``valid_pending`` count: clients now also see how many of those slots
    are user-approved (``accepted``) vs awaiting acknowledgement
    (``pending``). Statuses outside ``is_actionable`` are not expected here
    -- the caller already filters them -- but are bucketed under ``other``
    so a regression cannot silently disappear.
    """
    breakdown = {"pending": 0, "accepted": 0, "other": 0}
    for sugg in suggestions:
        if sugg.status == SuggestionStatus.PENDING:
            breakdown["pending"] += 1
        elif sugg.status == SuggestionStatus.ACCEPTED:
            breakdown["accepted"] += 1
        else:
            breakdown["other"] += 1
    return breakdown


def classify_pending_suggestions(
    pending: list[Suggestion],
) -> tuple[list[Suggestion], list[Suggestion]]:
    """Read-only split of actionable suggestions into ``valid`` and ``stale``.

    Auto-staleness applies only to ``PENDING`` rows: an ``ACCEPTED``
    suggestion is a user-approved commitment and must not be dropped just
    because an experiment is taking a long time, so it always counts as a
    reservation regardless of age. Pure function so dry-run callers can
    compute the actionable-vs-stale split without writing any
    ``EXPIRED`` rows.

    Public because the submit path reuses the same classification for its
    observation-budget guard — both operations must agree on which pending
    rows still hold budget slots, or a free-floating result gets rejected
    for slots the next generate would free.
    """
    if not pending:
        return [], []
    pending_params = [p.parameter_values for p in pending]
    pending_times = [p.created_at for p in pending]
    now = datetime.now(UTC)
    _, point_info = filter_pending_points(
        pending_params,
        pending_times,
        max_age_hours=PENDING_SUGGESTION_MAX_AGE_HOURS,
        now=now,
    )
    valid: list[Suggestion] = []
    stale: list[Suggestion] = []
    for sugg, info in zip(pending, point_info, strict=True):
        if info.is_stale and sugg.status == SuggestionStatus.PENDING:
            stale.append(sugg)
        else:
            valid.append(sugg)
    return valid, stale


def ensure_canonical_suggestion_keys(response: dict[str, Any]) -> dict[str, Any]:
    """Backfill ``suggestion_id`` on suggestion dicts that only carry ``id``.

    Idempotency-cache entries persisted before ``suggestion_id`` became
    the canonical key replay verbatim for the cache lifetime, so every
    transport must normalize replayed responses before returning them.
    Mutates ``response`` in place and returns it for call-site chaining.
    """
    for suggestion in response.get("suggestions", []):
        if (
            isinstance(suggestion, dict)
            and "suggestion_id" not in suggestion
            and "id" in suggestion
        ):
            suggestion["suggestion_id"] = suggestion["id"]
    return response


def _build_success_response(
    suggestions: list[Suggestion],
    iteration: int,
    method_selection: dict[str, Any],
    warnings: list[str],
    pending_info: dict[str, Any] | None,
    diversity_info: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the full success response dict."""
    # "suggestion_id" is the canonical identity key (matches the key
    # result submission consumes); "id" is a deprecated alias kept so
    # existing clients keep working.
    suggestion_dicts = [
        {
            "id": str(s.id),
            "suggestion_id": str(s.id),
            "parameter_values": s.parameter_values,
            "provenance": s.provenance.model_dump(),
            "created_at": s.created_at.isoformat(),
        }
        for s in suggestions
    ]

    response: dict[str, Any] = {
        "success": True,
        "suggestions": suggestion_dicts,
        "iteration": iteration,
        "errors": [],
        "warnings": warnings or [],
        "method_selection": method_selection,
    }

    if pending_info:
        response["pending_points"] = pending_info

    if diversity_info:
        response["batch_diversity"] = diversity_info

    return response


def _validate_batch_size(batch_size: int | None) -> dict[str, Any] | None:
    """Reject an out-of-range ``batch_size`` before any DB or backend work.

    ``None`` (use the campaign's configured batch size) is valid. Zero is
    an explicit error rather than a silent fallback to the spec default,
    negatives must never reach the backend, and the upper bound keeps a
    single request from pinning a worker in a superlinear q-batch
    acquisition solve. Lives in the shared operation so every transport
    (MCP, REST, CLI) enforces the same bounds.
    """
    if batch_size is None or 1 <= batch_size <= MAX_GENERATION_BATCH_SIZE:
        return None
    message = f"batch_size must be between 1 and {MAX_GENERATION_BATCH_SIZE}, got {batch_size}."
    response = make_error_response(
        ErrorCode.VALIDATION_FAILED,
        message=message,
        details={
            "batch_size": batch_size,
            "max_batch_size": MAX_GENERATION_BATCH_SIZE,
        },
    )
    response["field_errors"] = {"batch_size": [message]}
    return response


@with_response_metadata
async def generate_suggestions_operation(
    campaign_id: str,
    batch_size: int | None = None,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
    progress_callback: ProgressCallback | None = None,
    *,
    session: AsyncSession | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate next batch of experiment suggestions.

    Protocol-neutral entry point — contains all business logic
    formerly housed in the MCP tool handler. Uses the BOBackend
    protocol for all BO-engine interactions.

    Args:
        campaign_id: UUID of the campaign
        batch_size: Number of suggestions (default: campaign's)
        verbosity: Response detail level
        progress_callback: Optional progress hook. When supplied (e.g.
            built from an MCP :class:`Context` via
            :func:`bo_mcp_server.progress_bridge.make_progress_callback_from_context`),
            the backend's coarse milestones are forwarded to the client.
        session: Optional SQLAlchemy session to reuse. When supplied,
            ``apply_idempotency``'s session-aware path commits the
            generated suggestions atomically with the idempotency
            cache row.
        dry_run: If True, run preflight validation (campaign exists,
            status allows generation, stopping criteria, budget) and
            return a preview describing *which* iteration and batch
            size would run, **without** executing the BO algorithm,
            expiring stale pending rows, persisting suggestion rows,
            or advancing campaign state. The model fit and acquisition
            optimization are the expensive part of this tool — a true
            dry-run that produces candidate points would not save any
            time over a real call, so the preview is intentionally
            cheap.

    Returns:
        Dictionary with success, suggestions, iteration, errors.
    """
    logger.info(
        "Generating suggestions for campaign_id=%s, batch_size=%s, verbosity=%s, dry_run=%s",
        campaign_id,
        batch_size,
        verbosity,
        dry_run,
    )

    # --- Validate inputs ---
    verbosity_result = parse_verbosity(verbosity)
    if isinstance(verbosity_result, dict):
        return verbosity_result
    verbosity_level = verbosity_result

    campaign_id_result = parse_campaign_id(campaign_id)
    if isinstance(campaign_id_result, dict):
        return campaign_id_result
    campaign_uuid = campaign_id_result

    batch_size_error = _validate_batch_size(batch_size)
    if batch_size_error is not None:
        return batch_size_error

    if dry_run:
        return await _preview_generation(campaign_id, campaign_uuid, batch_size)

    # --- Database session scope ---
    # Re-use the caller's session when provided; otherwise open and
    # commit our own. Used by ``apply_idempotency``'s session-aware
    # path so the new suggestion rows and the cache finalize commit
    # together. The latency observation lives inside
    # ``_run_three_phase_generation`` so it only fires on the success
    # path that actually ran the BO compute.
    started = time.perf_counter()
    try:
        return await _run_three_phase_generation(
            campaign_id,
            campaign_uuid,
            batch_size,
            verbosity_level,
            session,
            started=started,
            progress_callback=progress_callback,
        )
    except (
        ConcurrentModificationError,
        SearchSpaceExhaustedError,
        BackendOutputError,
        BackendError,
    ) as err:
        return await _handle_generation_failure(err, campaign_id, session)


async def _run_three_phase_generation(
    campaign_id: str,
    campaign_uuid: UUID,
    batch_size: int | None,
    verbosity_level: VerbosityLevel,
    session: AsyncSession | None,
    *,
    started: float,
    progress_callback: ProgressCallback | None,
) -> dict[str, Any]:
    """Drive the snapshot → compute → persist split.

    Phase 1 (``_load_generation_snapshot``) acquires its own short
    read transaction even when the caller supplies an outer session,
    so the snapshot's locks are released before the BO compute runs.
    Phase 2 (``_compute_generation_batch``) does the heavy backend
    work with no DB session open — concurrent ``submit_results``,
    status updates, and other reads on the same campaign are no
    longer blocked. Phase 3 (``_persist_generation_batch``) reopens
    a write transaction (the caller's, or its own) and uses the
    existing ``CampaignRepository.save(expected_version=...)`` OCC
    guard to abort cleanly if the campaign moved on during the
    compute window.

    The pre-flight early-exits (campaign not found, invalid status,
    stopping criteria, budget exhaustion) all live in phase 1 and
    return their own envelope; the latency observation only fires
    on the success path so callers tracking ``observe_suggestion_latency``
    keep a clean histogram for "the compute actually ran".
    """
    phase1 = await _load_generation_snapshot(campaign_id, campaign_uuid, batch_size)
    if not isinstance(phase1, _GenerationSnapshot):
        return phase1

    backend = await get_backend_async(phase1.spec.backend)
    compute = await _compute_generation_batch(
        snapshot=phase1,
        backend=backend,
        prior_backend_state=phase1.campaign.backend_state,
        progress_callback=progress_callback,
    )

    async with session_scope(session) as db:
        response = await _persist_generation_batch(
            db=db,
            campaign_id=campaign_id,
            campaign_uuid=campaign_uuid,
            snapshot=phase1,
            compute=compute,
            backend=backend,
            verbosity_level=verbosity_level,
        )

    observe_suggestion_latency(
        response.get("_metadata", {}).get("backend"),
        time.perf_counter() - started,
    )
    return response


async def _handle_generation_failure(
    err: ConcurrentModificationError
    | SearchSpaceExhaustedError
    | BackendOutputError
    | BackendError,
    campaign_id: str,
    session: AsyncSession | None,
) -> dict[str, Any]:
    """Map a generation-loop exception to a structured error envelope.

    Extracted from :func:`generate_suggestions_operation` so the
    function stays under the lint-enforced cyclomatic-complexity
    budget (one ``except`` branch per failure mode would otherwise
    blow the return-statement count).

    When the caller supplied the session (e.g.
    ``apply_idempotency``'s session-aware path), partial writes that
    landed before the failure (new suggestion rows saved before the
    campaign-version save raised) would otherwise survive the outer
    commit. Roll back here so the failure produces no observable
    state.
    """
    if isinstance(err, ConcurrentModificationError):
        if session is not None:
            await session.rollback()
        logger.warning(
            "Concurrent modification while generating suggestions for campaign %s: %s",
            campaign_id,
            err,
        )
        response = make_concurrent_modification_response(
            err, extra_details={"campaign_id": campaign_id}
        )
        response.update({"suggestions": [], "iteration": None})
        return response
    if isinstance(err, SearchSpaceExhaustedError):
        logger.info(
            "Search space exhausted for campaign %s: %s",
            campaign_id,
            err,
        )
        return _make_exhaustion_response(err, campaign_id)
    if isinstance(err, BackendError):
        # The backend (BoTorch / BayBE) raised one of the typed
        # :class:`BackendError` subclasses defined in
        # ``bo_engine.backend_base``. Route it through
        # :func:`make_backend_error_response` so the envelope carries
        # the right ``retryable`` flag and ``retry_after`` hint without
        # this layer having to know the per-subclass mapping.
        if session is not None:
            await session.rollback()
        logger.warning(
            "Backend failure during suggestion generation for campaign %s: %s",
            campaign_id,
            err,
        )
        response = make_backend_error_response(
            err,
            extra_details={"campaign_id": campaign_id},
        )
        response.update({"suggestions": [], "iteration": None})
        return response
    # BackendStateTooLargeError: the backend produced a valid batch whose
    # serialized state exceeds the pre-persistence limit. Raised in the
    # compute phase, before any write — the campaign row keeps its
    # version and no suggestions are inserted. Deterministic for the
    # campaign shape, so not retryable; the recovery text points at the
    # mitigations (encoding='INT', fewer categories, backend='botorch',
    # raising the limit).
    if isinstance(err, BackendStateTooLargeError):
        if session is not None:
            await session.rollback()
        logger.error(
            "Backend state for campaign %s serializes to %d bytes, above the "
            "%d-byte persistence limit; aborting before the database write.",
            campaign_id,
            err.state_bytes,
            err.limit_bytes,
        )
        return _make_suggestions_error(
            ErrorCode.BACKEND_STATE_TOO_LARGE,
            message=str(err),
            details={
                "campaign_id": campaign_id,
                "state_bytes": err.state_bytes,
                "limit_bytes": err.limit_bytes,
            },
        )
    # BackendOutputError: the backend returned a malformed
    # SuggestionBatch. The fault is in the backend, not the caller,
    # so this is not retryable — surface the structured Pydantic
    # error list so an operator can identify the offending field.
    if session is not None:
        await session.rollback()
    logger.exception(
        "Backend returned malformed SuggestionBatch for campaign %s: %s",
        campaign_id,
        err.errors,
    )
    return _make_suggestions_error(
        ErrorCode.ACQUISITION_OPTIMIZATION_FAILED,
        message=str(err),
        details={
            "campaign_id": campaign_id,
            "validation_errors": err.errors,
        },
    )


def _init_repositories(session: AsyncSession) -> _Repositories:
    """Initialize all repository instances for a session."""
    return _Repositories(
        campaign=CampaignRepository(session),
        spec=CampaignSpecRepository(session),
        result=ResultRepository(session),
        suggestion=SuggestionRepository(session),
    )


@dataclass
class _GenerationPreflight:
    """Read-only outcome of the budget/stopping/pending checks.

    Surfaces every signal a dry-run needs to report — actionable
    pending count, stale-pending count, initial-design issuance history,
    observation-budget clamp, next iteration / planned batch — without
    writing any ``EXPIRED`` rows or running the BO algorithm.
    """

    next_iteration: int
    planned_batch_size: int
    n_results: int
    observations: list[ObservationData]
    valid_pending: list[Suggestion]
    stale_pending: list[Suggestion]
    initial_design_history: list[dict[str, Any]]
    budget_remaining: int | None
    batch_clamped: bool


@dataclass
class _GenerationSnapshot:
    """Phase-1 snapshot consumed by the compute phase.

    Captures every value the BO backend needs to generate a batch so
    the heavy ``backend.generate_suggestions`` call can run with no
    DB session open. ``campaign_version`` is recorded here and
    re-checked inside the phase-3 write transaction via
    ``CampaignRepository.save(..., expected_version=...)``.

    ``result_count`` is the child-table invariant used by phase 3 to
    detect concurrent ``submit_results`` calls that did *not* bump
    ``campaign.version`` (the single-objective / no-backend-state path
    in ``_update_campaign_state`` skips the campaign save when the
    hypervolume history is unchanged). Without this count, a result
    inserted during the compute window would be invisible to the OCC
    guard and the new suggestions would commit against stale
    observations.

    ``stale_pending`` carries the rows that the phase-3 writer will
    mark ``EXPIRED``; phase 1 only classifies them and never writes.
    ``initial_design_history`` holds the parameter dicts of *every*
    initial-design point already issued (observed, actionable, rejected,
    expired, stale, soft-deleted) so a deterministic warm-up backend can
    continue its Sobol sequence past them without rewinding.
    """

    campaign: Campaign
    campaign_version: int
    spec: CampaignSpec
    actual_batch_size: int
    new_iteration: int
    observations: list[ObservationData]
    valid_pending: list[Suggestion]
    stale_pending: list[Suggestion]
    initial_design_history: list[dict[str, Any]]
    pending_info: dict[str, Any] | None
    result_count: int


@dataclass
class _GenerationComputeResult:
    """Output of the phase-2 backend compute.

    Carries everything phase 3 needs to persist + format the response
    without touching the backend again.
    """

    suggestion_data: SuggestionDataList
    new_backend_state: dict[str, Any] | None
    warnings: list[str]
    method_selection: dict[str, Any]
    diversity_info: dict[str, Any] | None


def _compute_preflight(
    campaign: Campaign,
    spec: CampaignSpec,
    results: list[Any],
    pending: list[Suggestion],
    initial_design_history: list[dict[str, Any]],
    batch_size: int | None,
) -> _GenerationPreflight | dict[str, Any]:
    """Run the budget / stopping / pending preflight in read-only form.

    Returns either a fully-populated ``_GenerationPreflight`` describing
    what *would* happen on the real path, or — when a stopping criterion
    or budget exhaustion would short-circuit the generation — the same
    structured envelope the real path emits via ``_build_stopping_response``.

    The function is intentionally pure (no DB writes, no model fit) so
    both ``_preview_generation`` (dry-run) and ``_load_generation_snapshot``
    can call it before doing anything irreversible.
    """
    opt_spec = campaign_spec_to_optimization_spec(spec)
    valid_pending, stale_pending = classify_pending_suggestions(pending)
    observations = results_to_observations(results)
    next_iteration = campaign.iteration + 1

    stopping = evaluate_stopping_decision(opt_spec, observations, next_iteration)
    if stopping.should_stop:
        return _build_stopping_response(stopping, campaign.iteration, str(campaign.id))

    planned = batch_size if batch_size is not None else spec.batch_size
    budget_remaining: int | None = None
    clamped = False
    if opt_spec.max_observations is not None:
        budget_remaining = int(opt_spec.max_observations) - len(observations) - len(valid_pending)
        if budget_remaining <= 0:
            actionable_breakdown = _status_breakdown(valid_pending)
            stop = StoppingDecision(
                should_stop=True,
                reason=StoppingReason.BUDGET_EXCEEDED_OBSERVATIONS,
                message=(
                    f"max_observations={opt_spec.max_observations} already "
                    "covered by stored results plus actionable suggestions "
                    f"(pending={actionable_breakdown['pending']}, "
                    f"accepted={actionable_breakdown['accepted']})."
                ),
                details={
                    "n_observations": len(observations),
                    "n_pending": len(valid_pending),
                    "actionable_breakdown": actionable_breakdown,
                    "max_observations": int(opt_spec.max_observations),
                    "next_action_recommendation": "terminate_campaign",
                },
            )
            return _build_stopping_response(stop, campaign.iteration, str(campaign.id))
        if budget_remaining < planned:
            logger.info(
                "Clamping batch_size %d -> %d to respect max_observations=%d "
                "(n_observations=%d, n_pending=%d)",
                planned,
                budget_remaining,
                opt_spec.max_observations,
                len(observations),
                len(valid_pending),
            )
            clamped = True
            planned = budget_remaining

    return _GenerationPreflight(
        next_iteration=next_iteration,
        planned_batch_size=planned,
        n_results=len(results),
        observations=observations,
        valid_pending=valid_pending,
        stale_pending=stale_pending,
        initial_design_history=initial_design_history,
        budget_remaining=budget_remaining,
        batch_clamped=clamped,
    )


async def _load_preflight_inputs(
    campaign_id: str,
    campaign_uuid: UUID,
) -> (
    tuple[
        Any,
        CampaignSpec,
        list[Any],
        list[Suggestion],
        list[dict[str, Any]],
    ]
    | dict[str, Any]
):
    """Fetch the read-only inputs the preflight needs in a single session.

    Returns ``(campaign, spec, results, pending, initial_design_history)``
    on success, or the structured error envelope the real path would emit
    for a missing campaign/spec or an invalid state transition.

    ``initial_design_history`` is the parameter dicts of *every* initial-design
    point ever issued for the campaign (any status, **including soft-deleted
    rows**), collected only while the campaign is still in its warm-up window
    (``len(results) < min_data``). It is the durable, monotonic record of
    consumed Sobol positions — sourced from immutable suggestion issuance
    rather than mutable observations — so a soft-deleted result/suggestion
    cannot rewind a deterministic backend's continuation and a freestanding
    result cannot inflate it.
    """
    async with session_scope(None) as db:
        repos = _init_repositories(db)
        campaign = await repos.campaign.get(campaign_uuid)
        if campaign is None:
            logger.warning("Campaign not found: %s", campaign_id)
            return make_error_response(
                ErrorCode.CAMPAIGN_NOT_FOUND,
                message=f"Campaign {campaign_id} not found",
                details={"campaign_id": campaign_id},
            )
        if not campaign.can_generate_suggestions:
            logger.warning(
                "Cannot generate suggestions for campaign %s: status=%s",
                campaign_id,
                campaign.status.value,
            )
            response = make_error_response(
                ErrorCode.INVALID_STATE_TRANSITION,
                message=(
                    f"Campaign status is {campaign.status.value}, cannot generate suggestions"
                ),
                details={"current_status": campaign.status.value},
            )
            response["iteration"] = campaign.iteration
            return response
        spec = await repos.spec.get(campaign.spec_id)
        if spec is None:
            response = make_error_response(
                ErrorCode.DATABASE_ERROR,
                message="Campaign spec not found",
                details={"spec_id": str(campaign.spec_id)},
            )
            response["iteration"] = campaign.iteration
            return response
        results = await repos.result.list_by_campaign(campaign_uuid)
        pending = await repos.suggestion.list_actionable_by_campaign(campaign_uuid)
        initial_design_history: list[dict[str, Any]] = []
        opt_spec = campaign_spec_to_optimization_spec(spec)
        min_data = resolve_initial_design_size(
            opt_spec.n_parameters,
            opt_spec.initial_design_size,
        )
        # Only while still in warm-up does a deterministic backend need the
        # issuance history; once model-guided BO begins the Sobol offset is
        # irrelevant, so we skip the hydration entirely. ``include_deleted=True``
        # keeps the count monotonic: a soft-deleted suggestion still consumed a
        # Sobol position and must not reappear as a fresh candidate.
        if len(results) < min_data:
            all_suggestions = await repos.suggestion.list_by_campaign(
                campaign_uuid, include_deleted=True
            )
            initial_design_history = [
                suggestion.parameter_values
                for suggestion in all_suggestions
                if suggestion.provenance.generation_method == "initial_design"
            ]
    return campaign, spec, results, pending, initial_design_history


async def _preview_generation(
    campaign_id: str,
    campaign_uuid: UUID,
    batch_size: int | None,
) -> dict[str, Any]:
    """Return a preview of a generate_suggestions call.

    Runs the same read-only preflight the real path uses
    (:func:`_compute_preflight`): campaign existence + state, stopping
    criteria, observation budget, pending-suggestion classification.
    When a stopping criterion would short-circuit the real call the
    dry-run returns the same stopping envelope. Otherwise the response
    reports the actual planned batch (clamped by budget when needed),
    actionable-vs-stale pending counts, and remaining budget — without
    running the BO algorithm, expiring stale pending rows, or
    persisting any new suggestion.
    """
    loaded = await _load_preflight_inputs(campaign_id, campaign_uuid)
    if isinstance(loaded, dict):
        return loaded
    campaign, spec, results, pending, initial_design_history = loaded

    preflight = _compute_preflight(
        campaign,
        spec,
        results,
        pending,
        initial_design_history,
        batch_size,
    )
    if not isinstance(preflight, _GenerationPreflight):
        # Stopping criterion or budget exhaustion would short-circuit
        # the real call; surface the same envelope under dry-run so
        # callers can route on ``code`` / ``next_action_recommendation``.
        stopping_envelope: dict[str, Any] = preflight
        stopping_envelope["dry_run"] = True
        return stopping_envelope

    actionable = _status_breakdown(preflight.valid_pending)
    logger.info(
        "Generate-suggestions dry-run: campaign=%s, planned_batch=%d, "
        "next_iteration=%d, n_pending=%d, n_stale_pending=%d, clamped=%s",
        campaign_id,
        preflight.planned_batch_size,
        preflight.next_iteration,
        len(preflight.valid_pending),
        len(preflight.stale_pending),
        preflight.batch_clamped,
    )
    return {
        "success": True,
        "dry_run": True,
        "suggestions": [],
        "iteration": preflight.next_iteration,
        "errors": [],
        "preview": {
            "campaign_id": campaign_id,
            "current_status": campaign.status.value,
            "next_iteration": preflight.next_iteration,
            "planned_batch_size": preflight.planned_batch_size,
            "n_results": preflight.n_results,
            "n_pending": len(preflight.valid_pending),
            "n_stale_pending": len(preflight.stale_pending),
            "actionable_breakdown": actionable,
            "budget_remaining": preflight.budget_remaining,
            "batch_clamped_by_budget": preflight.batch_clamped,
        },
    }


async def _save_campaign_after_generation(
    campaign: Campaign,
    campaign_uuid: UUID,
    new_backend_state: dict[str, Any] | None,
    campaign_repo: CampaignRepository,
    db: AsyncSession,
) -> None:
    """Persist the campaign post-suggestion and arm a post-commit notification.

    The CREATED→RUNNING transition is the one state change subscribers
    care about on the suggestion path. We deliberately do not push for
    the iteration bump because that would flood subscribers with one
    notification per generated batch.

    The notification is registered as an ``after_commit`` hook on
    ``db`` rather than fired inline so subscribers see the transition
    only after the row is durable -- and never see it at all if a
    later optimistic-lock conflict (or the idempotency external-
    session path) rolls the transaction back.
    """
    updated_campaign = campaign.advance_iteration()
    status_changed = campaign.status == CampaignStatus.CREATED
    if status_changed:
        updated_campaign = updated_campaign.with_status(CampaignStatus.RUNNING)
    if new_backend_state is not None:
        updated_campaign = updated_campaign.with_backend_state(new_backend_state)
    await campaign_repo.save(
        updated_campaign,
        expected_version=campaign.version,
    )
    if status_changed:
        notify_campaign_updated_after_commit(db, campaign_uuid)


async def _load_generation_snapshot(
    campaign_id: str,
    campaign_uuid: UUID,
    batch_size: int | None,
) -> _GenerationSnapshot | dict[str, Any]:
    """Phase 1 — short read transaction → snapshot.

    Opens its own session (via :func:`_load_preflight_inputs`) so the
    read locks are released *before* the BO compute starts, then runs
    the same :func:`_compute_preflight` the dry-run preview uses — the
    stopping decision, observation budget, and batch clamp cannot
    drift between the two paths. Returns either a fully-populated
    snapshot or the structured envelope for the early-exit cases
    (campaign not found, invalid status, stopping criteria fired,
    budget exhausted). Phase 1 is read-only; the EXPIRED writes for
    stale pending rows are deferred to phase 3 so a long compute that
    ultimately conflicts on OCC does not leak a half-finished
    expiration set.
    """
    loaded = await _load_preflight_inputs(campaign_id, campaign_uuid)
    if isinstance(loaded, dict):
        return loaded
    campaign, spec, results, pending, initial_design_history = loaded

    preflight = _compute_preflight(
        campaign,
        spec,
        results,
        pending,
        initial_design_history,
        batch_size,
    )
    if not isinstance(preflight, _GenerationPreflight):
        return preflight

    pending_info = _build_pending_info(
        pending, preflight.valid_pending, len(preflight.stale_pending)
    )

    logger.debug(
        "Generation snapshot: n_results=%d, batch_size=%d, iteration=%d, n_pending=%d",
        preflight.n_results,
        preflight.planned_batch_size,
        preflight.next_iteration,
        len(preflight.valid_pending),
    )

    return _GenerationSnapshot(
        campaign=campaign,
        campaign_version=campaign.version,
        spec=spec,
        actual_batch_size=preflight.planned_batch_size,
        new_iteration=preflight.next_iteration,
        observations=preflight.observations,
        valid_pending=preflight.valid_pending,
        stale_pending=preflight.stale_pending,
        initial_design_history=preflight.initial_design_history,
        pending_info=pending_info,
        result_count=preflight.n_results,
    )


def _build_pending_info(
    pending: list[Suggestion],
    valid_pending: list[Suggestion],
    stale_count: int,
) -> dict[str, Any] | None:
    """Render the read-only pending-info block reported with the response.

    Replaces the side-effect-bearing :func:`_handle_pending_suggestions`
    on the phase-1 path: classification only — the EXPIRED writes for
    stale rows happen in phase 3 alongside the new-suggestion writes
    so a failed compute leaves no half-finished expiration set behind.
    """
    if not pending:
        return None
    breakdown = _status_breakdown(valid_pending)
    return {
        "total_pending": len(pending),
        "valid_pending": len(valid_pending),
        "stale_expired": stale_count,
        "actionable_breakdown": breakdown,
        "note": (
            f"{len(valid_pending)} actionable experiments "
            f"(pending={breakdown['pending']}, accepted={breakdown['accepted']}) "
            "considered for diversity. "
            f"{stale_count} stale suggestions expired."
            if valid_pending
            else "No valid pending experiments."
        ),
    }


async def _await_compute_with_timeout[T](compute: Awaitable[T]) -> T:
    """Await a backend compute, optionally bounded by ``BO_COMPUTE_TIMEOUT_SECONDS``.

    The compute runs on a worker thread (``asyncio.to_thread``) and is therefore
    *not* cancellable: when the budget elapses ``asyncio.wait_for`` cancels the
    awaiting coroutine and raises, but the worker thread keeps running until the
    GP fit finishes (its result is then discarded). The bound's value is freeing
    the request — and its idempotency reservation slot — from a hung backend, not
    killing the thread. A timeout surfaces as a retryable
    :class:`BackendTransientError` so the existing error routing returns a
    ``BACKEND_TRANSIENT_ERROR`` envelope and the reservation is dropped for retry.
    The budget defaults to a conservative 30 minutes (see
    ``BO_COMPUTE_TIMEOUT_SECONDS``); ``0`` disables the timeout and waits
    indefinitely.
    """
    timeout = get_bo_compute_timeout_seconds()
    if timeout <= 0:
        return await compute
    try:
        return await asyncio.wait_for(compute, timeout=timeout)
    except TimeoutError as exc:
        msg = f"Backend compute exceeded the {timeout:.0f}s timeout"
        raise BackendTransientError(msg, cause=exc) from exc


async def _compute_generation_batch(
    snapshot: _GenerationSnapshot,
    backend: BOBackend,
    prior_backend_state: dict[str, Any] | None,
    progress_callback: ProgressCallback | None,
) -> _GenerationComputeResult:
    """Phase 2 — heavy BO compute with no DB session open.

    Runs ``backend.generate_suggestions`` (and the follow-up diversity
    metric) outside any transaction so a concurrent
    ``submit_results`` / status update / dashboard read can acquire
    DB locks while the GP fit + acquisition optimization is running.
    """
    opt_spec = campaign_spec_to_optimization_spec(snapshot.spec)
    # X_pending must stay strictly actionable. Rejected / expired / stale
    # initial points also consumed Sobol positions, but they are *not*
    # in-flight experiments: routing them through X_pending would make a
    # model-guided backend (e.g. BayBE, which switches to model-guided
    # independently of our initial-design threshold) condition acquisition on
    # abandoned points. Their effect on Sobol continuation is carried instead
    # by ``snapshot.initial_design_history``, which only a deterministic
    # warm-up backend consumes.
    pending_parameter_values = [p.parameter_values for p in snapshot.valid_pending]

    suggestion_data, new_backend_state, warnings, live_method_info = await _generate_via_backend(
        backend,
        opt_spec,
        snapshot.observations,
        snapshot.actual_batch_size,
        snapshot.new_iteration,
        prior_backend_state,
        pending_parameter_values,
        initial_design_history=snapshot.initial_design_history,
        progress_callback=progress_callback,
    )

    diversity_info = await _compute_diversity_info_from_params(
        backend, opt_spec, [params for params, _ in suggestion_data]
    )

    # Prefer the live ``method_info`` the backend recorded during the
    # actual run; fall back to ``select_methods`` only when the backend
    # emits nothing (e.g. a third-party implementation that leaves the
    # field empty). Recomputing ``select_methods`` after the fact used
    # to mis-report: BayBE's static path explicitly tags every label
    # with ``(fallback)`` so the response would carry a fallback label
    # even on a successful BO run.
    if live_method_info:
        method_selection = live_method_info
        method_selection.setdefault("is_fallback", False)
    else:
        method_selection = backend.select_methods(
            opt_spec, n_observations=len(snapshot.observations)
        )
        method_selection["is_fallback"] = True
        method_selection.setdefault("confidence", "low")

    return _GenerationComputeResult(
        suggestion_data=suggestion_data,
        new_backend_state=new_backend_state,
        warnings=warnings,
        method_selection=method_selection,
        diversity_info=diversity_info,
    )


async def _compute_diversity_info_from_params(
    backend: BOBackend,
    opt_spec: OptimizationSpec,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Diversity-metric helper that consumes raw parameter dicts.

    Takes raw parameter dicts from the phase-2 backend output (which
    has not been persisted yet) so the diversity check can run
    without the side-effect of a database write.
    """
    if len(candidates) <= 1:
        return None
    try:
        metrics = await asyncio.to_thread(backend.compute_batch_diversity, opt_spec, candidates)
        if metrics is None:
            return None
        return {
            "min_pairwise_distance": round(metrics.min_pairwise_distance, 4),
            "mean_pairwise_distance": round(metrics.mean_pairwise_distance, 4),
            "diversity_score": round(metrics.diversity_score, 4),
            "is_diverse": metrics.is_diverse,
        }
    except (RuntimeError, ValueError, TypeError) as e:
        logger.debug("Could not compute batch diversity: %s", e)
        return None


async def _persist_generation_batch(
    db: AsyncSession,
    campaign_id: str,
    campaign_uuid: UUID,
    snapshot: _GenerationSnapshot,
    compute: _GenerationComputeResult,
    backend: BOBackend,
    verbosity_level: VerbosityLevel,
) -> dict[str, Any]:
    """Phase 3 — short write transaction with OCC + child-table recheck.

    Two complementary guards protect against concurrent state changes
    that landed during the compute window:

    1. ``CampaignRepository.save(..., expected_version=...)`` catches
       campaign-row mutations (status flips from a lifecycle call,
       multi-objective hypervolume bumps from ``submit_results``).
    2. A child-table invariant check catches mutations that don't
       touch ``campaign.version`` — most importantly a single-objective
       ``submit_results`` (no hypervolume to bump) and direct
       suggestion-status updates. Phase 1 captured the result count
       and the actionable-suggestion count; if either changed by the
       time phase 3 runs, the snapshot is stale and we raise
       ``ConcurrentModificationError`` so the existing machinery
       routes a retryable envelope back to the caller.
    """
    _ = campaign_id  # surfaced in log lines below
    repos = _init_repositories(db)

    fresh_result_count = (
        await db.execute(
            select(func.count())
            .select_from(ResultModel)
            .where(
                ResultModel.campaign_id == str(campaign_uuid),
                ResultModel.deleted_at.is_(None),
            )
        )
    ).scalar_one()
    fresh_actionable_count = (
        await db.execute(
            select(func.count())
            .select_from(SuggestionModel)
            .where(
                SuggestionModel.campaign_id == str(campaign_uuid),
                SuggestionModel.status.in_((SuggestionStatus.PENDING, SuggestionStatus.ACCEPTED)),
                SuggestionModel.deleted_at.is_(None),
            )
        )
    ).scalar_one()
    snapshot_actionable_count = len(snapshot.valid_pending) + len(snapshot.stale_pending)
    if (
        fresh_result_count != snapshot.result_count
        or fresh_actionable_count != snapshot_actionable_count
    ):
        logger.warning(
            "Snapshot invalidated for campaign %s during compute: results %d→%d, actionable %d→%d",
            campaign_uuid,
            snapshot.result_count,
            fresh_result_count,
            snapshot_actionable_count,
            fresh_actionable_count,
        )
        msg = "Campaign"
        raise ConcurrentModificationError(msg, campaign_uuid, snapshot.campaign_version)

    # Expire stale pending suggestions. The classification was done in
    # phase 1 but the write is deferred to phase 3 so a failed compute
    # leaves no half-finished EXPIRED set behind. The expire path uses
    # ``expire_if_pending`` (atomic ``UPDATE ... WHERE status = PENDING``)
    # so a concurrent ``PENDING -> ACCEPTED`` transition during the
    # compute window — which the count-only invariant cannot see,
    # since both PENDING and ACCEPTED count as actionable — does not
    # get clobbered with ``EXPIRED`` by the snapshot's view of the
    # world.
    #
    # ``expire_if_pending == False`` indicates the row changed status
    # during the compute (typically ``ACCEPTED`` via
    # ``update_suggestion_status``). Phase 1 excluded the stale row
    # from ``valid_pending``, so the new batch's budget and X_pending
    # were computed without it — committing those suggestions now
    # would understate the in-flight count and could exceed
    # ``max_observations`` or cluster against the newly-accepted
    # point. Raise :class:`ConcurrentModificationError` so the
    # existing machinery returns a retryable envelope; the retry's
    # phase 1 will see the row as ACCEPTED and account for it
    # correctly.
    for sugg in snapshot.stale_pending:
        if not await repos.suggestion.expire_if_pending(sugg.id):
            logger.warning(
                "Stale suggestion %s changed during compute; aborting phase 3 "
                "to let the caller retry with a fresh snapshot",
                sugg.id,
            )
            msg = "Suggestion"
            raise ConcurrentModificationError(msg, sugg.id, snapshot.campaign_version)

    suggestions = await _create_and_save_suggestions(
        compute.suggestion_data, campaign_uuid, repos.suggestion
    )

    await _save_campaign_after_generation(
        snapshot.campaign,
        campaign_uuid,
        compute.new_backend_state,
        repos.campaign,
        db,
    )

    logger.info(
        "Generated %d suggestions for campaign %s, iteration=%d",
        len(suggestions),
        campaign_uuid,
        snapshot.new_iteration,
    )

    _ = backend  # backend is captured in compute.method_selection; keep parity with phase-1 / 2.
    full_response = _build_success_response(
        suggestions,
        snapshot.new_iteration,
        compute.method_selection,
        compute.warnings,
        snapshot.pending_info,
        compute.diversity_info,
    )
    return format_suggestions_response(full_response, verbosity_level)


def _backend_accepts_initial_design_history(backend: BOBackend) -> bool:
    """Whether ``backend.generate_suggestions`` accepts ``initial_design_history``.

    Guards backward compatibility for third-party backend plugins written
    against the pre-``initial_design_history`` protocol: passing an unexpected
    keyword to such a backend would raise a raw ``TypeError`` outside the typed
    backend-error path. A backend qualifies if it names the parameter explicitly
    or accepts arbitrary ``**kwargs``. If the signature cannot be introspected
    (e.g. a C-implemented callable), assume the current contract and forward it.
    """
    try:
        signature = inspect.signature(backend.generate_suggestions)
    except (TypeError, ValueError):
        return True
    parameters = signature.parameters.values()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        return True
    return "initial_design_history" in signature.parameters


async def _generate_via_backend(
    backend: BOBackend,
    opt_spec: OptimizationSpec,
    observations: list[ObservationData],
    batch_size: int,
    iteration: int,
    prior_backend_state: dict[str, Any] | None,
    pending_parameter_values: list[dict[str, Any]] | None = None,
    initial_design_history: list[dict[str, Any]] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[
    SuggestionDataList,
    dict[str, Any] | None,
    list[str],
    dict[str, Any],
]:
    """Generate a batch of suggestions via the backend.

    Previously this function branched on ``len(results) == 0`` to call
    ``backend.generate_initial_design`` directly.  That dual-gate created
    a duplicate-suggestion bug: each operation-level call reseeded Sobol
    (via the backend) while the engine-level fallback in
    :func:`bo_engine.suggestions.generate_next_batch` independently decided
    when to reseed as well, so the two gates disagreed on which Sobol
    stream to continue.  Routing every call through
    ``backend.generate_suggestions`` makes the engine the single source of
    truth for the initial-design threshold, the Sobol continuation, and the
    exhaustion check for finite categorical spaces.

    ``pending_parameter_values`` carries **only genuinely actionable**
    suggestions that are not yet observed; they reach the backend as
    ``X_pending`` for parallel / batch BO. The Sobol continuation history —
    every issued initial-design point, including rejected / expired / stale /
    soft-deleted ones — travels separately as ``initial_design_history`` so a
    deterministic warm-up backend cannot rewind while a model-guided backend's
    acquisition stays free of abandoned points.

    Both paths are CPU-bound (Sobol sampling, GP fitting, acquisition
    optimization, MCMC) and are offloaded via ``asyncio.to_thread`` so
    concurrent requests do not stall the event loop.

    Returns ``(suggestion_data, backend_state, warnings, method_info)``.
    ``method_info`` is the live metadata the backend recorded during
    the actual run (which recommender phase / strategy / acquisition
    fired). Routing it through here lets ``_build_success_response``
    surface the live labels instead of the static ``select_methods``
    fallback the operation previously recomputed.
    """
    # The heartbeat is a no-op when there is no active idempotency
    # reservation (the typical direct-call path). When invoked inside
    # ``apply_idempotency``'s session-aware branch it keeps the
    # ``pending`` row alive past the default 10-min reservation TTL so
    # legitimately-slow runs (SAASBO MCMC, large batches) finish
    # without surrendering their slot to a concurrent retry storm.
    call_kwargs: dict[str, Any] = {
        "spec": opt_spec,
        "observations": observations,
        "batch_size": batch_size,
        "iteration": iteration,
        "backend_state": prior_backend_state,
        "pending_points": pending_parameter_values,
        "progress_callback": progress_callback,
    }
    # ``initial_design_history`` was added to the backend contract after the
    # first public ``BOBackend`` protocol. A third-party plugin compiled against
    # the older signature would raise a raw ``TypeError`` (unexpected keyword)
    # that escapes the structured backend-error path, so only forward the
    # keyword to backends that actually accept it. Backends that do not are still
    # driven correctly for continuous warm-up via observation/pending exclusion;
    # they simply forgo the durable-history refinement.
    if _backend_accepts_initial_design_history(backend):
        call_kwargs["initial_design_history"] = initial_design_history
    async with reservation_heartbeat():
        compute = asyncio.to_thread(backend.generate_suggestions, **call_kwargs)
        batch = await _await_compute_with_timeout(compute)
    # Re-validate the batch shape against the documented contract.
    # ``bo-engine`` is Pydantic-free for third-party backend plugins, so
    # a misbehaving backend can hand us a partial dict that would otherwise
    # surface downstream as an opaque ``KeyError`` or as a malformed
    # provenance row in storage. Convert the contract violation to a
    # typed :class:`BackendOutputError` here.
    batch = validate_backend_batch(batch, max_backend_state_bytes=get_max_backend_state_bytes())
    suggestion_data = [(item["parameter_values"], item["provenance"]) for item in batch.suggestions]
    return (
        suggestion_data,
        batch.backend_state,
        batch.warnings,
        dict(batch.method_info or {}),
    )
