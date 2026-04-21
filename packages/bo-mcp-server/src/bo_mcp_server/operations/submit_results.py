"""Submit results operation — protocol-neutral business logic."""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import UUID

from bo_engine.backend import BOBackend
from bo_engine.constants import DUPLICATE_DETECTION_TOLERANCE

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
from bo_mcp_server.operations.helpers import (
    parse_campaign_id,
    parse_verbosity,
    results_to_observations,
)
from bo_mcp_server.response_formatter import (
    VerbosityLevel,
    format_submit_results_response,
)
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    ConcurrentModificationError,
    ResultRepository,
    SuggestionRepository,
    get_session,
)

logger = logging.getLogger(__name__)


def _make_submit_error(
    code: ErrorCode,
    message: str | None = None,
    details: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    duplicates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a standardized error response for submit_results."""
    response = make_error_response(code, message=message, details=details)
    response.update(
        {
            "result_ids": [],
            "warnings": warnings or [],
            "duplicates_detected": duplicates or [],
        }
    )
    return response


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
        )

    try:
        result_source = ResultSource(source)
    except ValueError:
        return _make_submit_error(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid source '{source}', must be gui, file_upload, or api",
            details={"source": source},
        )

    if not results:
        return _make_submit_error(
            ErrorCode.VALIDATION_FAILED,
            message="At least one result is required",
        )

    return verbosity_level, campaign_uuid, submitter_uuid, result_source


def _check_duplicates_for_result(
    index: int,
    params: dict[str, Any],
    existing_params: list[dict[str, Any]],
    backend: BOBackend,
    atomic: bool,
    warnings: list[str],
    duplicates_detected: list[dict[str, Any]],
) -> str | None:
    """Run duplicate detection for a single result. Returns error string or None."""
    if not existing_params:
        return None
    duplicates = backend.detect_duplicates(
        new_params=params,
        existing_params=existing_params,
        tolerance=DUPLICATE_DETECTION_TOLERANCE,
    )
    if not duplicates:
        return None

    result_error = None
    for dup in duplicates:
        duplicates_detected.append(
            {
                "result_index": index,
                "duplicate_of_index": dup.index,
                "is_exact": dup.is_exact,
                "parameter_distance": dup.parameter_distance,
            }
        )
        if dup.is_exact:
            warnings.append(
                f"Result {index} appears to be an exact duplicate of "
                f"existing result at index {dup.index}. Use force=True to submit anyway."
            )
            if atomic:
                result_error = f"Result {index} is exact duplicate. Use force=True to override."
        else:
            warnings.append(
                f"Result {index} is very close to existing result at "
                f"index {dup.index} (distance={dup.parameter_distance:.6f}). "
                "This may indicate a duplicate measurement."
            )
    return result_error


async def _resolve_suggestion_id(
    suggestion_id_str: str | None,
    index: int,
    campaign_uuid: UUID,
    suggestion_repo: SuggestionRepository,
    warnings: list[str],
) -> UUID | None:
    """Resolve and validate a suggestion_id, marking it completed if valid."""
    if not suggestion_id_str:
        return None
    try:
        suggestion_id = UUID(suggestion_id_str)
    except ValueError:
        warnings.append(f"Result {index}: invalid suggestion_id format")
        return None

    suggestion = await suggestion_repo.get(suggestion_id)
    if suggestion is None:
        warnings.append(f"Result {index}: suggestion {suggestion_id_str} not found")
        return None
    if suggestion.campaign_id != campaign_uuid:
        warnings.append(f"Result {index}: suggestion belongs to different campaign")
        return None
    await suggestion_repo.save(suggestion.with_status(SuggestionStatus.COMPLETED))
    return suggestion_id


@dataclass
class _SubmitTracking:
    """Mutable state accumulated during result validation."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duplicates_detected: list[dict[str, Any]] = field(default_factory=list)
    partial_results: dict[int, str | dict[str, str]] = field(default_factory=dict)


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

    # Phase 1 — pure validation; no writes hit the session.
    valid_submissions: list[tuple[int, ResultSubmissionInput]] = []
    for i, r in enumerate(results):
        result_error = _validate_single_result(
            i,
            r,
            param_names,
            objective_names,
            existing_params,
            backend,
            force,
            atomic,
            tracking.warnings,
            tracking.duplicates_detected,
            parameters=parameters,
        )

        if result_error is not None:
            tracking.errors.append(result_error)
            if not atomic and continue_on_error:
                tracking.partial_results[i] = {"error": result_error}
            continue

        valid_submissions.append((i, r))

    # Atomic contract: any validation error aborts the write path entirely so
    # no suggestion-status updates or result rows leak out of the transaction.
    if atomic and tracking.errors:
        return [], {}

    # Phase 2 — mutate suggestion status and build result entities for the
    # submissions that cleared validation. All writes happen in the same
    # session and share a single commit/rollback boundary with save_batch
    # and any downstream campaign-state update.
    result_entities: list[Result] = []
    entity_to_input_index: dict[int, int] = {}
    for i, r in valid_submissions:
        suggestion_id = await _resolve_suggestion_id(
            r.suggestion_id,
            i,
            campaign_uuid,
            suggestion_repo,
            tracking.warnings,
        )

        result = Result(
            campaign_id=campaign_uuid,
            suggestion_id=suggestion_id,
            parameter_values=r.parameter_values,
            objective_values=r.objective_values,
            source=result_source,
            submitted_by=submitter_uuid,
            measurement_uncertainty=r.measurement_uncertainty,
            metadata=r.metadata,
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
) -> None:
    """Validate measurement uncertainty keys and values."""
    invalid_keys = set(uncertainty.keys()) - objective_names
    if invalid_keys:
        warnings.append(
            f"Result {index}: measurement_uncertainty has unknown "
            f"objective keys: {sorted(invalid_keys)}"
        )
    for obj_name, unc_val in uncertainty.items():
        if not isinstance(unc_val, (int, float)) or math.isnan(unc_val) or math.isinf(unc_val):
            warnings.append(
                f"Result {index}: measurement_uncertainty['{obj_name}'] "
                f"is not a finite number: {unc_val}"
            )
        elif unc_val < 0:
            warnings.append(
                f"Result {index}: measurement_uncertainty['{obj_name}'] "
                f"is negative ({unc_val}); expected non-negative std"
            )


def _validate_single_result(
    index: int,
    r: ResultSubmissionInput,
    param_names: set[str],
    objective_names: set[str],
    existing_params: list[dict[str, Any]],
    backend: BOBackend,
    force: bool,
    atomic: bool,
    warnings: list[str],
    duplicates_detected: list[dict[str, Any]],
    parameters: list[InputParameter] | None = None,
) -> str | None:
    """Validate a single result's parameters, duplicates, and objectives. Returns error or None."""
    if missing_params := (param_names - set(r.parameter_values.keys())):
        return f"Result {index} missing parameters: {missing_params}"

    # Validate parameter values against spec bounds/categories
    if parameters is not None:
        param_by_name = {p.name: p for p in parameters}
        for pname, pvalue in r.parameter_values.items():
            if pname in param_by_name:
                _validate_parameter_value(param_by_name[pname], pvalue, index, warnings)

    if not force:
        dup_error = _check_duplicates_for_result(
            index,
            r.parameter_values,
            existing_params,
            backend,
            atomic,
            warnings,
            duplicates_detected,
        )
        if dup_error is not None:
            return dup_error

    if missing_objectives := (objective_names - set(r.objective_values.keys())):
        return f"Result {index} missing objectives: {missing_objectives}"

    if r.measurement_uncertainty is not None:
        _validate_measurement_uncertainty(
            r.measurement_uncertainty, objective_names, index, warnings
        )

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
            hv = backend.compute_hypervolume(opt_spec, all_observations)
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
            new_state = backend.update_state_after_results(
                opt_spec, new_obs, campaign.backend_state
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
    campaign_id: str,
    tracking: _SubmitTracking,
) -> dict[str, Any] | None:
    """Check for atomic-mode failures (validation errors, exact duplicates).

    Returns error response dict or None.
    """
    if atomic and tracking.errors:
        logger.warning("Result submission validation failed (atomic mode): %s", tracking.errors)
        return {
            "success": False,
            "result_ids": [],
            "errors": tracking.errors,
            "warnings": tracking.warnings,
            "duplicates_detected": tracking.duplicates_detected,
        }

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
        )
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
        "warnings": tracking.warnings,
        "duplicates_detected": tracking.duplicates_detected,
    }
    if not atomic and continue_on_error:
        response_data["partial_results"] = tracking.partial_results
    return response_data


async def submit_results_operation(
    campaign_id: str,
    results: list[ResultSubmissionInput],
    submitted_by: str,
    source: str = "api",
    force: bool = False,
    atomic: bool = True,
    continue_on_error: bool = False,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
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
        async with get_session() as session:
            campaign_repo = CampaignRepository(session)
            spec_repo = CampaignSpecRepository(session)
            result_repo = ResultRepository(session)
            suggestion_repo = SuggestionRepository(session)

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
            )

            atomic_error = _check_atomic_failures(atomic, force, campaign_id, tracking)
            if atomic_error is not None:
                return atomic_error

            if not result_entities:
                response_data: dict[str, Any] = {
                    "success": False,
                    "result_ids": [],
                    "errors": (
                        tracking.errors if tracking.errors else ["No valid results to submit"]
                    ),
                    "warnings": tracking.warnings,
                    "duplicates_detected": tracking.duplicates_detected,
                }
                if not atomic and continue_on_error:
                    response_data["partial_results"] = tracking.partial_results
                return response_data

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
            }
        )
        return response
