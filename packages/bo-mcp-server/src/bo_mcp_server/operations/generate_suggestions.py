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
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from bo_engine.backend import BOBackend
from bo_engine.constants import PENDING_SUGGESTION_MAX_AGE_HOURS
from bo_engine.convergence import (
    StoppingDecision,
    StoppingReason,
    evaluate_stopping_decision,
)
from bo_engine.initial_design import SearchSpaceExhaustedError
from bo_engine.pending_points import filter_pending_points
from bo_engine.progress import ProgressCallback
from bo_engine.types import ObservationData, OptimizationSpec
from sqlalchemy.ext.asyncio import AsyncSession

from bo_mcp_server.backend import get_backend
from bo_mcp_server.converters import campaign_spec_to_optimization_spec
from bo_mcp_server.domain import (
    CampaignSpec,
    CampaignStatus,
    Suggestion,
    SuggestionProvenance,
    SuggestionStatus,
)
from bo_mcp_server.errors import (
    ErrorCode,
    make_concurrent_modification_response,
    make_error_response,
)
from bo_mcp_server.idempotency import session_scope
from bo_mcp_server.operations.backend_output import (
    BackendOutputError,
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
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    ConcurrentModificationError,
    ResultRepository,
    SuggestionRepository,
)
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
    assert stopping.reason is not None  # narrows the optional for the type checker
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


def _classify_pending_suggestions(
    pending: list[Suggestion],
) -> tuple[list[Suggestion], list[Suggestion]]:
    """Read-only split of actionable suggestions into ``valid`` and ``stale``.

    Auto-staleness applies only to ``PENDING`` rows: an ``ACCEPTED``
    suggestion is a user-approved commitment and must not be dropped just
    because an experiment is taking a long time, so it always counts as a
    reservation regardless of age. Pure function so dry-run callers can
    compute the actionable-vs-stale split without writing any
    ``EXPIRED`` rows.
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


async def _handle_pending_suggestions(
    pending: list[Suggestion],
    suggestion_repo: SuggestionRepository,
) -> tuple[dict[str, Any] | None, list[Suggestion]]:
    """Filter actionable suggestions, expire stale ones.

    Auto-staleness applies only to ``PENDING`` rows: an ``ACCEPTED``
    suggestion is a user-approved commitment and must not be dropped just
    because an experiment is taking a long time, so it always counts as a
    reservation regardless of age.

    Returns (pending_info dict or None, valid pending list).
    """
    if not pending:
        return None, []

    valid_pending, stale = _classify_pending_suggestions(pending)
    stale_count = len(stale)
    for sugg in stale:
        await suggestion_repo.save(sugg.with_status(SuggestionStatus.EXPIRED))

    # Split the actionable set into PENDING vs ACCEPTED so clients can
    # distinguish "generated, awaiting acknowledgement" from "user-approved,
    # awaiting result". The legacy ``total_pending``/``valid_pending`` keys
    # remain for back-compat but already include both statuses since
    # ``list_actionable_by_campaign`` started returning ACCEPTED rows; the
    # explicit breakdown makes that semantics visible.
    breakdown = _status_breakdown(valid_pending)
    pending_info = {
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
    logger.debug(
        "Pending point handling: total=%d, valid=%d, expired=%d",
        len(pending),
        len(valid_pending),
        stale_count,
    )
    return pending_info, valid_pending


async def _compute_diversity_info(
    backend: BOBackend,
    opt_spec: OptimizationSpec,
    suggestions: list[Suggestion],
) -> dict[str, Any] | None:
    """Compute batch diversity metrics via the backend.

    Returns diversity info dict, or None if fewer than 2
    suggestions or on failure. The backend call is offloaded to a thread
    so it cannot block the event loop.
    """
    if len(suggestions) <= 1:
        return None

    try:
        candidates = [s.parameter_values for s in suggestions]
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


def _build_success_response(
    suggestions: list[Suggestion],
    iteration: int,
    method_selection: dict[str, Any],
    warnings: list[str],
    pending_info: dict[str, Any] | None,
    diversity_info: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the full success response dict."""
    suggestion_dicts = [
        {
            "id": str(s.id),
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

    if dry_run:
        return await _preview_generation(campaign_id, campaign_uuid, batch_size)

    # --- Database session scope ---
    # Re-use the caller's session when provided; otherwise open and
    # commit our own. Used by ``apply_idempotency``'s session-aware
    # path so the new suggestion rows and the cache finalize commit
    # together.
    from bo_mcp_server.metrics import observe_suggestion_latency  # noqa: PLC0415

    started = time.perf_counter()
    try:
        async with session_scope(session) as db:
            repos = _init_repositories(db)
            response = await _generate_within_session(
                campaign_id,
                campaign_uuid,
                batch_size,
                verbosity_level,
                repos,
                db,
                progress_callback=progress_callback,
            )
            # Observe success-path latency only; the exception branches
            # below record their own envelopes without a backend handle.
            observe_suggestion_latency(
                response.get("_metadata", {}).get("backend"),
                time.perf_counter() - started,
            )
            return response
    except (
        ConcurrentModificationError,
        SearchSpaceExhaustedError,
        BackendOutputError,
    ) as err:
        return await _handle_generation_failure(err, campaign_id, session)


async def _handle_generation_failure(
    err: ConcurrentModificationError | SearchSpaceExhaustedError | BackendOutputError,
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
        return _make_suggestions_error(
            ErrorCode.SEARCH_SPACE_EXHAUSTED,
            message=str(err),
            details={
                "campaign_id": campaign_id,
                "n_requested": err.n_requested,
                "n_available": err.n_available,
                "n_total_combinations": err.n_total_combinations,
                "next_action_recommendation": "terminate_campaign",
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
    pending count, stale-pending count, observation-budget clamp,
    next iteration / planned batch — without writing any
    ``EXPIRED`` rows or running the BO algorithm.
    """

    next_iteration: int
    planned_batch_size: int
    n_results: int
    valid_pending: list[Suggestion]
    stale_pending: list[Suggestion]
    budget_remaining: int | None
    batch_clamped: bool


def _compute_preflight(
    campaign: Any,
    spec: CampaignSpec,
    results: list[Any],
    pending: list[Suggestion],
    batch_size: int | None,
) -> _GenerationPreflight | dict[str, Any]:
    """Run the budget / stopping / pending preflight in read-only form.

    Returns either a fully-populated ``_GenerationPreflight`` describing
    what *would* happen on the real path, or — when a stopping criterion
    or budget exhaustion would short-circuit the generation — the same
    structured envelope the real path emits via ``_build_stopping_response``.

    The function is intentionally pure (no DB writes, no model fit) so
    both ``_preview_generation`` (dry-run) and ``_generate_within_session``
    can call it before doing anything irreversible.
    """
    opt_spec = campaign_spec_to_optimization_spec(spec)
    valid_pending, stale_pending = _classify_pending_suggestions(pending)
    observations = results_to_observations(results)
    next_iteration = campaign.iteration + 1

    stopping = evaluate_stopping_decision(opt_spec, observations, next_iteration)
    if stopping.should_stop:
        return _build_stopping_response(stopping, campaign.iteration, str(campaign.id))

    planned = batch_size or spec.batch_size
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
            clamped = True
            planned = budget_remaining

    return _GenerationPreflight(
        next_iteration=next_iteration,
        planned_batch_size=planned,
        n_results=len(results),
        valid_pending=valid_pending,
        stale_pending=stale_pending,
        budget_remaining=budget_remaining,
        batch_clamped=clamped,
    )


async def _load_preflight_inputs(
    campaign_id: str,
    campaign_uuid: UUID,
) -> tuple[Any, CampaignSpec, list[Any], list[Suggestion]] | dict[str, Any]:
    """Fetch the read-only inputs the preflight needs in a single session.

    Returns ``(campaign, spec, results, pending)`` on success, or the
    structured error envelope the real path would emit for missing
    campaign / spec or an invalid state transition.
    """
    async with session_scope(None) as db:
        repos = _init_repositories(db)
        campaign = await repos.campaign.get(campaign_uuid)
        if campaign is None:
            return make_error_response(
                ErrorCode.CAMPAIGN_NOT_FOUND,
                message=f"Campaign {campaign_id} not found",
                details={"campaign_id": campaign_id},
            )
        if not campaign.can_generate_suggestions:
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
    return campaign, spec, results, pending


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
    campaign, spec, results, pending = loaded

    preflight = _compute_preflight(campaign, spec, results, pending, batch_size)
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
    campaign: Any,
    campaign_uuid: UUID,
    new_backend_state: Any,
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


async def _generate_within_session(
    campaign_id: str,
    campaign_uuid: UUID,
    batch_size: int | None,
    verbosity_level: VerbosityLevel,
    repos: _Repositories,
    db: AsyncSession,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run the suggestion generation within an active DB session.

    ``db`` is threaded through so post-commit notifications can attach
    to its ``after_commit`` event without having to plumb the session
    into every helper that needs to fire one. Handles fetching,
    validation, generation, and persistence.
    """
    campaign_repo = repos.campaign
    spec_repo = repos.spec
    result_repo = repos.result
    suggestion_repo = repos.suggestion

    # Fetch and validate campaign
    campaign = await campaign_repo.get(campaign_uuid)
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
            message=(f"Campaign status is {campaign.status.value}, cannot generate suggestions"),
            details={
                "current_status": campaign.status.value,
            },
        )
        response["iteration"] = campaign.iteration
        return response

    # Fetch campaign spec
    spec = await spec_repo.get(campaign.spec_id)
    if spec is None:
        response = make_error_response(
            ErrorCode.DATABASE_ERROR,
            message="Campaign spec not found",
            details={"spec_id": str(campaign.spec_id)},
        )
        response["iteration"] = campaign.iteration
        return response

    # Fetch existing results
    results = await result_repo.list_by_campaign(campaign_uuid)

    # Handle actionable suggestions. PENDING and ACCEPTED both reserve
    # experiment slots (see ``Suggestion.is_actionable``): PENDING means
    # "generated, awaiting execution"; ACCEPTED means "user approved, awaiting
    # result". Both must be considered for X_pending diversification *and* for
    # the observation budget so accepted-but-unsubmitted experiments cannot
    # have their slot stolen by free-floating submissions or new generations.
    pending = await suggestion_repo.list_actionable_by_campaign(campaign_uuid)
    pending_info, valid_pending = await _handle_pending_suggestions(pending, suggestion_repo)

    # Prepare generation inputs
    actual_batch_size = batch_size or spec.batch_size
    opt_spec = campaign_spec_to_optimization_spec(spec)
    new_iteration = campaign.iteration + 1
    backend = get_backend(spec.backend)

    # Budget / convergence-based automatic stopping. Runs before any BO work
    # so we never spend a model fit when the campaign has already exhausted
    # its iteration or observation budget, or improvement has plateaued
    # below ``convergence_tolerance``.
    observations = results_to_observations(results)
    stopping = evaluate_stopping_decision(opt_spec, observations, new_iteration)
    if stopping.should_stop:
        return _build_stopping_response(stopping, campaign.iteration, campaign_id)

    # Clamp the requested batch to the remaining observation budget so a
    # campaign with ``max_observations=3``, ``batch_size=2`` and two
    # existing observations does not finish with four observations. Valid
    # pending suggestions count as already-reserved budget: a generated-but-
    # unsubmitted batch is an in-flight experiment we have committed to.
    if opt_spec.max_observations is not None:
        remaining_budget = int(opt_spec.max_observations) - len(observations) - len(valid_pending)
        if remaining_budget <= 0:
            actionable_breakdown = _status_breakdown(valid_pending)
            stopping = StoppingDecision(
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
                    # ``n_pending`` is preserved for back-compat -- it has
                    # always meant "actionable count" since ACCEPTED was
                    # added to the budget calculation. The explicit
                    # ``actionable_breakdown`` makes the split visible.
                    "n_pending": len(valid_pending),
                    "actionable_breakdown": actionable_breakdown,
                    "max_observations": int(opt_spec.max_observations),
                    "next_action_recommendation": "terminate_campaign",
                },
            )
            return _build_stopping_response(stopping, campaign.iteration, campaign_id)
        if remaining_budget < actual_batch_size:
            logger.info(
                "Clamping batch_size %d -> %d to respect max_observations=%d "
                "(n_observations=%d, n_pending=%d)",
                actual_batch_size,
                remaining_budget,
                opt_spec.max_observations,
                len(observations),
                len(valid_pending),
            )
            actual_batch_size = remaining_budget

    logger.debug(
        "Generation context: n_results=%d, batch_size=%d, iteration=%d, n_pending=%d",
        len(results),
        actual_batch_size,
        new_iteration,
        len(valid_pending),
    )

    # Pending suggestions condition the acquisition so the new batch does
    # not cluster around in-flight experiments.
    pending_parameter_values = [p.parameter_values for p in valid_pending]

    # Generate suggestions via backend (heavy work offloaded to a thread)
    suggestion_data, new_backend_state, warnings = await _generate_via_backend(
        backend,
        opt_spec,
        observations,
        actual_batch_size,
        new_iteration,
        campaign.backend_state,
        pending_parameter_values,
        progress_callback=progress_callback,
    )

    # Create and save suggestion entities
    suggestions = await _create_and_save_suggestions(
        suggestion_data, campaign_uuid, suggestion_repo
    )

    # Compute batch diversity
    diversity_info = await _compute_diversity_info(backend, opt_spec, suggestions)

    # Update campaign state -- promotes CREATED→RUNNING when needed
    # and arms a post-commit hook so subscribers learn about that
    # transition only after the row is durable. Subscribers do not
    # care about per-iteration bumps so the hook is conditional.
    await _save_campaign_after_generation(
        campaign,
        campaign_uuid,
        new_backend_state,
        campaign_repo,
        db,
    )

    logger.info(
        "Generated %d suggestions for campaign %s, iteration=%d",
        len(suggestions),
        campaign_id,
        new_iteration,
    )

    # Format and return
    method_selection = backend.select_methods(opt_spec, n_observations=len(results))

    full_response = _build_success_response(
        suggestions,
        new_iteration,
        method_selection,
        warnings,
        pending_info,
        diversity_info,
    )
    return format_suggestions_response(full_response, verbosity_level)


async def _generate_via_backend(
    backend: BOBackend,
    opt_spec: OptimizationSpec,
    observations: list[ObservationData],
    batch_size: int,
    iteration: int,
    prior_backend_state: dict[str, Any] | None,
    pending_parameter_values: list[dict[str, Any]] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> tuple[
    SuggestionDataList,
    dict[str, Any] | None,
    list[str],
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
    truth for the initial-design threshold, the Sobol continuation
    (``n_drawn=len(observations)``), and the exhaustion check for finite
    categorical spaces.

    ``pending_parameter_values`` carries the parameter dicts of suggestions
    that are PENDING but not yet observed; the backend is expected to
    forward them to its acquisition optimizer as ``X_pending`` so
    parallel / batch BO does not cluster new candidates around the
    in-flight batch.

    Both paths are CPU-bound (Sobol sampling, GP fitting, acquisition
    optimization, MCMC) and are offloaded via ``asyncio.to_thread`` so
    concurrent requests do not stall the event loop.

    Returns (suggestion_data, backend_state, warnings).
    """
    batch = await asyncio.to_thread(
        backend.generate_suggestions,
        spec=opt_spec,
        observations=observations,
        batch_size=batch_size,
        iteration=iteration,
        backend_state=prior_backend_state,
        pending_points=pending_parameter_values,
        progress_callback=progress_callback,
    )
    # Re-validate the batch shape against the documented contract.
    # ``bo-engine`` is Pydantic-free for third-party backend plugins, so
    # a misbehaving backend can hand us a partial dict that would otherwise
    # surface downstream as an opaque ``KeyError`` or as a malformed
    # provenance row in storage. Convert the contract violation to a
    # typed :class:`BackendOutputError` here.
    batch = validate_backend_batch(batch)
    suggestion_data = [(item["parameter_values"], item["provenance"]) for item in batch.suggestions]
    return (
        suggestion_data,
        batch.backend_state,
        batch.warnings,
    )
