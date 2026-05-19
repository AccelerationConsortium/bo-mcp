"""Submit results operation — protocol-neutral business logic.

Backend calls that touch the BO engine (hypervolume, TuRBO state updates)
are offloaded via ``asyncio.to_thread`` so they cannot block the FastAPI /
MCP event loop during concurrent result submissions.

The previously-monolithic submit_results module has been split into
three companion files (TODO 8.54) so each file owns one concern and
stays well below the 1k LOC cognitive-load ceiling:

* :mod:`.submit_results_validation` — per-row shape validation
  (parameter presence, spec-bound enforcement, finite measurement
  uncertainty) plus the mutable bookkeeping (``_RowError``,
  ``_SubmitTracking``, ``_record_row_error``).
* :mod:`.submit_results_pipeline` — duplicate detection, suggestion-id
  classification / atomic resolution, and the phase-1 / phase-2 batch
  orchestration that materializes the in-flight ``Result`` entities.
* This module — input parsing, error envelope helpers, post-save
  campaign-state update, and the public
  :func:`submit_results_operation` entry point.
"""

import asyncio
import logging
from typing import Any, Literal
from uuid import UUID

from bo_engine.backend import BOBackend
from sqlalchemy.ext.asyncio import AsyncSession

from bo_mcp_server.backend import get_backend
from bo_mcp_server.converters import campaign_spec_to_optimization_spec
from bo_mcp_server.domain import (
    Campaign,
    CampaignSpec,
    Result,
    ResultSource,
    ResultSubmissionInput,
)
from bo_mcp_server.errors import (
    ErrorCode,
    make_concurrent_modification_response,
    make_error_response,
)
from bo_mcp_server.idempotency import session_scope
from bo_mcp_server.operations.helpers import (
    parse_campaign_id,
    parse_verbosity,
    results_to_observations,
)
from bo_mcp_server.operations.submit_results_pipeline import (
    _validate_and_create_results,
)
from bo_mcp_server.operations.submit_results_validation import (
    _SubmitTracking,
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
    tracking: _SubmitTracking,
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
    tracking: _SubmitTracking,
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
        session: Optional AsyncSession to reuse (e.g. from a test or an
            outer transaction). When ``None``, a fresh session scope is
            opened via :func:`session_scope`.
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
