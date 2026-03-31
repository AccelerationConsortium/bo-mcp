"""Submit results operation — protocol-neutral business logic."""

import logging
from typing import Any, Literal
from uuid import UUID

from bo_mcp_server.backend import get_backend
from bo_mcp_server.converters import campaign_spec_to_optimization_spec
from bo_mcp_server.domain import (
    Result,
    ResultSource,
    ResultSubmissionInput,
    SuggestionStatus,
)
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.operations.helpers import results_to_observations
from bo_mcp_server.response_formatter import (
    VerbosityLevel,
    format_submit_results_response,
)
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    ResultRepository,
    SuggestionRepository,
    get_session,
)

logger = logging.getLogger(__name__)

# Constants for duplicate detection
DUPLICATE_DETECTION_TOLERANCE = 1e-6


async def submit_results_operation(  # noqa: C901
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

    Args:
        campaign_id: UUID of the campaign
        results: List of result payloads, each containing:
            - parameter_values: Dict of parameter name -> value
            - objective_values: Dict of objective name -> observed value
            - suggestion_id: Optional UUID of the suggestion
            - metadata: Optional additional metadata
        submitted_by: UUID of the user submitting results
        source: Result source ("gui", "file_upload", or "api")
        force: If True, skip duplicate detection and submit anyway
        atomic: If True (default), rollback all results if any fail
            validation. Either all results are saved or none are.
        continue_on_error: If True, continue processing remaining
            results after errors. Ignored if atomic=True.
        verbosity: Response verbosity level. Options:
            - "minimal": ~40 tokens - n_submitted only
            - "standard": ~100 tokens - result_ids, warnings count
            - "detailed": ~300+ tokens - all fields

    Returns:
        Dictionary with:
            - success: Boolean indicating if submission succeeded
            - result_ids: List of created result UUIDs
            - errors: List of error messages (if any)
            - warnings: List of warning messages
            - duplicates_detected: List of detected duplicate info
            - partial_results: (if continue_on_error=True) Dict
              mapping index to result_id or error message
    """
    logger.info(
        "Submitting %d results for campaign_id=%s by user=%s, verbosity=%s",
        len(results),
        campaign_id,
        submitted_by,
        verbosity,
    )

    # Validate verbosity parameter
    try:
        verbosity_level = VerbosityLevel(verbosity)
    except ValueError:
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=(
                f"Invalid verbosity '{verbosity}'. Must be one of: minimal, standard, detailed"
            ),
        )

    errors: list[str] = []
    warnings: list[str] = []

    # Validate campaign_id
    try:
        campaign_uuid = UUID(campaign_id)
    except ValueError:
        logger.warning("Invalid campaign_id format: %s", campaign_id)
        response = make_error_response(
            ErrorCode.INVALID_CAMPAIGN_ID,
            details={"campaign_id": campaign_id},
        )
        response.update(
            {
                "result_ids": [],
                "warnings": [],
                "duplicates_detected": [],
            }
        )
        return response

    # Validate submitted_by
    try:
        submitter_uuid = UUID(submitted_by)
    except ValueError:
        response = make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message="Invalid submitted_by format",
            details={"submitted_by": submitted_by},
        )
        response.update(
            {
                "result_ids": [],
                "warnings": [],
                "duplicates_detected": [],
            }
        )
        return response

    # Validate source
    try:
        result_source = ResultSource(source)
    except ValueError:
        response = make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=(f"Invalid source '{source}', must be gui, file_upload, or api"),
            details={"source": source},
        )
        response.update(
            {
                "result_ids": [],
                "warnings": [],
                "duplicates_detected": [],
            }
        )
        return response

    if not results:
        response = make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message="At least one result is required",
        )
        response.update(
            {
                "result_ids": [],
                "warnings": [],
                "duplicates_detected": [],
            }
        )
        return response

    # Obtain backend once before entering the DB session
    backend = get_backend()

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        spec_repo = CampaignSpecRepository(session)
        result_repo = ResultRepository(session)
        suggestion_repo = SuggestionRepository(session)

        # Get campaign
        campaign = await campaign_repo.get(campaign_uuid)
        if campaign is None:
            response = make_error_response(
                ErrorCode.CAMPAIGN_NOT_FOUND,
                message=f"Campaign {campaign_id} not found",
                details={"campaign_id": campaign_id},
            )
            response.update(
                {
                    "result_ids": [],
                    "warnings": [],
                    "duplicates_detected": [],
                }
            )
            return response

        # Check campaign status
        if not campaign.can_submit_results:
            response = make_error_response(
                ErrorCode.INVALID_STATE_TRANSITION,
                message=(f"Campaign status is {campaign.status.value}, cannot submit results"),
                details={
                    "current_status": campaign.status.value,
                },
            )
            response.update(
                {
                    "result_ids": [],
                    "warnings": [],
                    "duplicates_detected": [],
                }
            )
            return response

        # Get spec for validation
        spec = await spec_repo.get(campaign.spec_id)
        if spec is None:
            response = make_error_response(
                ErrorCode.DATABASE_ERROR,
                message="Campaign spec not found",
                details={"spec_id": str(campaign.spec_id)},
            )
            response.update(
                {
                    "result_ids": [],
                    "warnings": [],
                    "duplicates_detected": [],
                }
            )
            return response

        # Fetch existing results for duplicate detection
        existing_results = await result_repo.list_by_campaign(campaign_uuid)
        existing_params = [r.parameter_values for r in existing_results]

        # Track duplicates detected across all submitted results
        duplicates_detected: list[dict[str, Any]] = []

        # Track partial results when continue_on_error=True
        partial_results: dict[int, str | dict[str, str]] = {}

        # Validate and create result entities
        result_entities: list[Result] = []
        # Map from result entity index to original input index
        entity_to_input_index: dict[int, int] = {}
        param_names = {p.name for p in spec.parameters}
        objective_names = {o.name for o in spec.objectives}

        for i, r in enumerate(results):
            result_error: str | None = None

            # Validate parameter_values
            if missing_params := (param_names - set(r.parameter_values.keys())):
                result_error = f"Result {i} missing parameters: {missing_params}"

            # Duplicate detection (Section 1.2)
            if result_error is None and not force and existing_params:
                duplicates = backend.detect_duplicates(
                    new_params=r.parameter_values,
                    existing_params=existing_params,
                    tolerance=DUPLICATE_DETECTION_TOLERANCE,
                )
                if duplicates:
                    for dup in duplicates:
                        dup_info = {
                            "result_index": i,
                            "duplicate_of_index": dup.index,
                            "is_exact": dup.is_exact,
                            "parameter_distance": (dup.parameter_distance),
                        }
                        duplicates_detected.append(dup_info)
                        if dup.is_exact:
                            warnings.append(
                                f"Result {i} appears to be "
                                "an exact duplicate of "
                                "existing result at index "
                                f"{dup.index}. Use "
                                "force=True to submit "
                                "anyway."
                            )
                            # Treat exact duplicate as error
                            # in atomic mode
                            if atomic:
                                result_error = (
                                    f"Result {i} is exact duplicate. Use force=True to override."
                                )
                        else:
                            dist = dup.parameter_distance
                            warnings.append(
                                f"Result {i} is very close "
                                "to existing result at "
                                f"index {dup.index} "
                                f"(distance={dist:.6f}). "
                                "This may indicate a "
                                "duplicate measurement."
                            )

            # Validate objective_values
            if result_error is None:
                if missing_objectives := (objective_names - set(r.objective_values.keys())):
                    result_error = f"Result {i} missing objectives: {missing_objectives}"

            # Handle validation error based on mode
            if result_error is not None:
                errors.append(result_error)
                if not atomic and continue_on_error:
                    partial_results[i] = {
                        "error": result_error,
                    }
                    continue
                elif atomic:
                    # In atomic mode, continue collecting errors
                    # but don't create entity
                    continue
                else:
                    # Non-atomic, non-continue: stop on first
                    # error
                    continue

            # Parse optional suggestion_id
            suggestion_id = None
            if r.suggestion_id:
                try:
                    suggestion_id = UUID(r.suggestion_id)
                    # Verify suggestion exists and belongs
                    # to campaign
                    suggestion = await suggestion_repo.get(suggestion_id)
                    if suggestion is None:
                        warnings.append(f"Result {i}: suggestion {r.suggestion_id} not found")
                        suggestion_id = None
                    elif suggestion.campaign_id != campaign_uuid:
                        warnings.append(f"Result {i}: suggestion belongs to different campaign")
                        suggestion_id = None
                    else:
                        # Mark suggestion as completed
                        updated_suggestion = suggestion.with_status(SuggestionStatus.COMPLETED)
                        await suggestion_repo.save(updated_suggestion)
                except ValueError:
                    warnings.append(f"Result {i}: invalid suggestion_id format")

            # Create result entity
            result = Result(
                campaign_id=campaign_uuid,
                suggestion_id=suggestion_id,
                parameter_values=r.parameter_values,
                objective_values=r.objective_values,
                source=result_source,
                submitted_by=submitter_uuid,
                metadata=r.metadata,
            )
            entity_to_input_index[len(result_entities)] = i
            result_entities.append(result)

        # In atomic mode, if any errors occurred, fail the batch
        if atomic and errors:
            logger.warning(
                "Result submission validation failed (atomic mode): %s",
                errors,
            )
            return {
                "success": False,
                "result_ids": [],
                "errors": errors,
                "warnings": warnings,
                "duplicates_detected": duplicates_detected,
            }

        # If exact duplicates detected and not forcing, reject
        # submission (atomic mode only)
        exact_duplicates = [d for d in duplicates_detected if d.get("is_exact")]
        if atomic and exact_duplicates and not force:
            logger.warning(
                "Exact duplicates detected for campaign %s: %s",
                campaign_id,
                exact_duplicates,
            )
            response = make_error_response(
                ErrorCode.DUPLICATE_RESULT,
                message=("Exact duplicate results detected. Use force=True to override."),
                details={
                    "duplicate_count": len(exact_duplicates),
                },
            )
            response.update(
                {
                    "result_ids": [],
                    "warnings": warnings,
                    "duplicates_detected": duplicates_detected,
                }
            )
            return response

        # Save results based on mode
        if not result_entities:
            # No valid results to save
            response_data: dict[str, Any] = {
                "success": False,
                "result_ids": [],
                "errors": (errors if errors else ["No valid results to submit"]),
                "warnings": warnings,
                "duplicates_detected": duplicates_detected,
            }
            if not atomic and continue_on_error:
                response_data["partial_results"] = partial_results
            return response_data

        # Save all valid results
        saved_results = await result_repo.save_batch(result_entities)
        result_ids = [str(r.id) for r in saved_results]

        # Populate partial_results for successful saves
        if not atomic and continue_on_error:
            for entity_idx, saved in enumerate(saved_results):
                input_idx = entity_to_input_index[entity_idx]
                partial_results[input_idx] = str(saved.id)

        # Get all results for campaign (including newly submitted)
        all_results = await result_repo.list_by_campaign(campaign_uuid)

        # Track if campaign needs updating
        updated_campaign = campaign

        # Convert spec once for backend calls
        opt_spec = campaign_spec_to_optimization_spec(spec)

        # Section 4.2: Update hypervolume history for
        # multi-objective campaigns
        if len(spec.objectives) >= 2:
            try:
                all_observations = results_to_observations(all_results)
                hv = backend.compute_hypervolume(opt_spec, all_observations)
                if hv is not None:
                    updated_campaign = updated_campaign.with_hypervolume(hv)
                    logger.debug(
                        "Updated hypervolume history for campaign %s: hv=%.4f",
                        campaign_id,
                        hv,
                    )
            except Exception as e:
                logger.debug("Failed to compute hypervolume: %s", e)
                warnings.append(f"Could not compute hypervolume: {e}")

        # Section 4.3: Update backend state for
        # single-objective campaigns
        if len(spec.objectives) == 1 and campaign.turbo_state is not None:
            try:
                new_obs = results_to_observations(result_entities)
                new_state = backend.update_state_after_results(
                    opt_spec, new_obs, campaign.turbo_state
                )
                if new_state is not None:
                    updated_campaign = updated_campaign.with_turbo_state(new_state)
            except Exception as e:
                logger.debug("Failed to update backend state: %s", e)
                warnings.append(f"Could not update backend state: {e}")

        # Save campaign if updated
        if updated_campaign.version != campaign.version:
            await campaign_repo.save(
                updated_campaign,
                expected_version=campaign.version,
            )

        logger.info(
            "Successfully submitted %d results for campaign %s",
            len(result_ids),
            campaign_id,
        )

        # Determine overall success based on mode
        # In continue_on_error mode, partial success is still
        # "success" but we include errors that occurred
        has_errors = len(errors) > 0
        partial_success_allowed = not atomic and continue_on_error
        overall_success = len(result_ids) > 0 and (not has_errors or partial_success_allowed)

        response_data: dict[str, Any] = {
            "success": overall_success,
            "result_ids": result_ids,
            "errors": errors,
            "warnings": warnings,
            "duplicates_detected": duplicates_detected,
        }

        # Include partial_results in non-atomic
        # continue_on_error mode
        if not atomic and continue_on_error:
            response_data["partial_results"] = partial_results

        return format_submit_results_response(response_data, verbosity_level)
