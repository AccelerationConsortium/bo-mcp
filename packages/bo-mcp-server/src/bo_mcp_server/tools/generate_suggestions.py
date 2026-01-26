"""Generate suggestions tool for MCP."""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from bo_engine.batch_diversity import compute_batch_diversity
from bo_engine.method_selector import select_methods
from bo_engine.pending_points import filter_pending_points
from bo_engine.suggestions import generate_initial_design, generate_next_batch
from bo_engine.turbo import TurboState, should_use_turbo
from bo_engine.types import ObservationData, OptimizationSpec

from bo_mcp_server.cache import diagnostics_cache
from bo_mcp_server.converters import campaign_spec_to_optimization_spec
from bo_mcp_server.domain import (
    Campaign,
    CampaignSpec,
    CampaignStatus,
    Result,
    Suggestion,
    SuggestionProvenance,
    SuggestionStatus,
)
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.response_formatter import VerbosityLevel, format_suggestions_response
from bo_mcp_server.server import mcp
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    ResultRepository,
    SuggestionRepository,
    get_session,
)

logger = logging.getLogger(__name__)

# Constants for pending point handling
PENDING_SUGGESTION_MAX_AGE_HOURS = 24


@dataclass
class GenerationContext:
    """Context for suggestion generation containing all necessary data."""

    campaign: Campaign
    spec: CampaignSpec
    opt_spec: OptimizationSpec
    results: list[Result]
    turbo_state: TurboState | None
    batch_size: int
    new_iteration: int
    pending_suggestions: list[Suggestion] | None = None  # Section 1.6: Pending point handling


def _results_to_observations(results: list[Result]) -> list[ObservationData]:
    """Convert domain Result objects to ObservationData for bo-engine."""
    return [
        ObservationData(
            parameter_values=r.parameter_values,
            objective_values=r.objective_values,
            cost=r.metadata.get("cost") if r.metadata else None,
        )
        for r in results
    ]


def _turbo_state_to_dict(state: TurboState) -> dict[str, Any]:
    """Serialize TurboState to dictionary for JSON storage."""
    return {
        "dim": state.dim,
        "batch_size": state.batch_size,
        "length": state.length,
        "length_min": state.length_min,
        "length_max": state.length_max,
        "failure_counter": state.failure_counter,
        "failure_tolerance": state.failure_tolerance,
        "success_counter": state.success_counter,
        "success_tolerance": state.success_tolerance,
        "best_value": state.best_value,
        "restart_triggered": state.restart_triggered,
    }


def _dict_to_turbo_state(data: dict[str, Any]) -> TurboState:
    """Deserialize dictionary to TurboState."""
    return TurboState(
        dim=data["dim"],
        batch_size=data["batch_size"],
        length=data["length"],
        length_min=data["length_min"],
        length_max=data["length_max"],
        failure_counter=data["failure_counter"],
        failure_tolerance=data["failure_tolerance"],
        success_counter=data["success_counter"],
        success_tolerance=data["success_tolerance"],
        best_value=data["best_value"],
        restart_triggered=data["restart_triggered"],
    )


def _make_error_response(errors: list[str], iteration: int | None = None) -> dict[str, Any]:
    """Create a standardized error response."""
    return {
        "success": False,
        "suggestions": [],
        "iteration": iteration,
        "errors": errors,
    }


def _generate_initial_design_data(
    opt_spec: OptimizationSpec,
    batch_size: int,
    iteration: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Generate initial design suggestions using Sobol sequence.

    Returns list of (parameter_values, provenance_dict) tuples.
    """
    designs = generate_initial_design(opt_spec, batch_size)
    return [
        (
            design,
            {
                "iteration": iteration,
                "batch_index": i,
                "generation_method": "initial_design",
                "explanation": (
                    f"Initial design point {i + 1}/{batch_size} using "
                    "Sobol sequence. Initial designs explore the parameter "
                    "space before model-guided optimization."
                ),
                "confidence_level": "medium",
            },
        )
        for i, design in enumerate(designs)
    ]


def _generate_bo_suggestions_data(
    ctx: GenerationContext,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], TurboState | None]:
    """Generate BO-based suggestions using Gaussian Process model.

    Returns tuple of (suggestion_data_list, new_turbo_state).
    """
    observations = _results_to_observations(ctx.results)
    suggestion_results, new_turbo_state = generate_next_batch(
        spec=ctx.opt_spec,
        observations=observations,
        batch_size=ctx.batch_size,
        iteration=ctx.new_iteration,
        turbo_state=ctx.turbo_state,
    )
    suggestion_data = [
        (
            sr.parameter_values,
            {
                "iteration": sr.iteration,
                "batch_index": sr.batch_index,
                "generation_method": sr.generation_method,
                "acquisition_value": sr.acquisition_value,
                "model_uncertainty": sr.model_uncertainty,
                "acquisition_function": sr.acquisition_function,
                "model_type": sr.model_type,
                "random_seed": sr.random_seed,
                "model_version": sr.model_version,
                "confidence_level": sr.confidence_level,
                "explanation": sr.explanation,
            },
        )
        for sr in suggestion_results
    ]
    return suggestion_data, new_turbo_state


async def _create_and_save_suggestions(
    suggestion_data: list[tuple[dict[str, Any], dict[str, Any]]],
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


def _format_suggestions_response(
    suggestions: list[Suggestion],
    iteration: int,
    opt_spec: OptimizationSpec,
    n_observations: int,
    pending_info: dict[str, Any] | None = None,
    diversity_info: dict[str, Any] | None = None,
    model_health_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Format the final success response with suggestions and method info.

    Args:
        suggestions: List of generated suggestions
        iteration: Current iteration number
        opt_spec: Optimization specification
        n_observations: Number of observations used
        pending_info: Info about pending points that were considered
        diversity_info: Info about batch diversity
        model_health_warnings: Warnings from model validation
    """
    suggestion_dicts = [
        {
            "id": str(s.id),
            "parameter_values": s.parameter_values,
            "provenance": s.provenance.model_dump(),
            "created_at": s.created_at.isoformat(),
        }
        for s in suggestions
    ]

    method_selection = select_methods(opt_spec, n_observations=n_observations)

    response: dict[str, Any] = {
        "success": True,
        "suggestions": suggestion_dicts,
        "iteration": iteration,
        "errors": [],
        "warnings": model_health_warnings or [],
        "method_selection": {
            "model_type": method_selection.model_type,
            "acquisition_function": method_selection.acquisition_function,
            "optimization_strategy": method_selection.optimization_strategy,
            "input_transforms": method_selection.input_transforms,
            "explanation": method_selection.explanation,
            "confidence": method_selection.confidence,
            "alternatives": method_selection.alternatives,
            "warnings": method_selection.warnings,
        },
    }

    # Add pending point info (Section 1.6)
    if pending_info:
        response["pending_points"] = pending_info

    # Add batch diversity info (Section 1.5)
    if diversity_info:
        response["batch_diversity"] = diversity_info

    return response


@mcp.tool()
async def generate_suggestions(
    campaign_id: str,
    batch_size: int | None = None,
    verbosity: str = "standard",
) -> dict[str, Any]:
    """Generate next batch of experiment suggestions for a campaign.

    Args:
        campaign_id: UUID of the campaign
        batch_size: Number of suggestions (default: campaign's batch_size)
        verbosity: Response verbosity level. Options:
            - "minimal": ~50 tokens - success + suggestion_ids + method only
            - "standard": ~200 tokens - excludes pending_points details
            - "detailed": ~500+ tokens - all fields including pending_points

    Returns:
        Dictionary with:
            - success: Boolean indicating if generation succeeded
            - suggestions: List of suggestion dictionaries
            - iteration: Current iteration number
            - errors: List of error messages (if failed)
    """
    logger.info(
        "Generating suggestions for campaign_id=%s, batch_size=%s, verbosity=%s",
        campaign_id,
        batch_size,
        verbosity,
    )

    # Validate verbosity parameter
    try:
        verbosity_level = VerbosityLevel(verbosity)
    except ValueError:
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid verbosity '{verbosity}'. Must be one of: minimal, standard, detailed",
        )

    # Validate campaign_id format
    try:
        campaign_uuid = UUID(campaign_id)
    except ValueError:
        logger.warning("Invalid campaign_id format: %s", campaign_id)
        return make_error_response(
            ErrorCode.INVALID_CAMPAIGN_ID,
            details={"campaign_id": campaign_id},
        )

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        spec_repo = CampaignSpecRepository(session)
        result_repo = ResultRepository(session)
        suggestion_repo = SuggestionRepository(session)

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
                message=f"Campaign status is {campaign.status.value}, cannot generate suggestions",
                details={"current_status": campaign.status.value},
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

        # Handle pending suggestions (Section 1.6)
        pending = await suggestion_repo.list_by_campaign(
            campaign_uuid, status=SuggestionStatus.PENDING
        )

        # Filter pending by age - only consider recent ones
        pending_info: dict[str, Any] | None = None
        valid_pending: list[Suggestion] = []
        if pending:
            pending_params = [p.parameter_values for p in pending]
            pending_times = [p.created_at for p in pending]
            now = datetime.now(UTC)

            valid_params, point_info = filter_pending_points(
                pending_params,
                pending_times,
                max_age_hours=PENDING_SUGGESTION_MAX_AGE_HOURS,
                now=now,
            )

            # Track which suggestions are valid vs stale
            stale_count = 0
            for sugg, info in zip(pending, point_info, strict=False):
                if info.is_stale:
                    # Expire stale suggestions
                    await suggestion_repo.save(sugg.with_status(SuggestionStatus.EXPIRED))
                    stale_count += 1
                else:
                    valid_pending.append(sugg)

            pending_info = {
                "total_pending": len(pending),
                "valid_pending": len(valid_pending),
                "stale_expired": stale_count,
                "note": (
                    f"{len(valid_pending)} pending experiments considered for diversity. "
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

        # Build generation context
        actual_batch_size = batch_size or spec.batch_size
        opt_spec = campaign_spec_to_optimization_spec(spec)
        new_iteration = campaign.iteration + 1

        turbo_state: TurboState | None = None
        if (opt_spec.use_turbo or should_use_turbo(spec.n_parameters)) and spec.n_objectives == 1:
            if campaign.turbo_state is not None:
                turbo_state = _dict_to_turbo_state(campaign.turbo_state)

        logger.debug(
            "Generation context: n_results=%d, batch_size=%d, iteration=%d, turbo=%s",
            len(results),
            actual_batch_size,
            new_iteration,
            turbo_state is not None,
        )

        # Generate suggestions (initial design or BO-based)
        model_health_warnings: list[str] = []
        if len(results) == 0:
            suggestion_data = _generate_initial_design_data(
                opt_spec, actual_batch_size, new_iteration
            )
            new_turbo_state = None
        else:
            ctx = GenerationContext(
                campaign=campaign,
                spec=spec,
                opt_spec=opt_spec,
                results=results,
                turbo_state=turbo_state,
                batch_size=actual_batch_size,
                new_iteration=new_iteration,
                pending_suggestions=valid_pending if valid_pending else None,
            )
            suggestion_data, new_turbo_state = _generate_bo_suggestions_data(ctx)

        # Create and save suggestion entities
        suggestions = await _create_and_save_suggestions(
            suggestion_data, campaign_uuid, suggestion_repo
        )

        # Compute batch diversity metrics (Section 1.5)
        diversity_info: dict[str, Any] | None = None
        if len(suggestions) > 1:
            import torch
            from bo_engine.transforms import get_bounds_tensor

            try:
                bounds = get_bounds_tensor(opt_spec)
                param_names = [p.name for p in spec.parameters]

                # Build tensor of suggestion parameter values
                suggestion_values = []
                for s in suggestions:
                    values = [float(s.parameter_values.get(name, 0.0)) for name in param_names]
                    suggestion_values.append(values)
                candidates = torch.tensor(suggestion_values)

                metrics = compute_batch_diversity(candidates, bounds)
                diversity_info = {
                    "min_pairwise_distance": round(metrics.min_pairwise_distance, 4),
                    "mean_pairwise_distance": round(metrics.mean_pairwise_distance, 4),
                    "diversity_score": round(metrics.diversity_score, 4),
                    "is_diverse": metrics.is_diverse,
                }
            except Exception as e:
                logger.debug("Could not compute batch diversity: %s", e)

        # Update campaign state
        updated_campaign = campaign.advance_iteration()
        if campaign.status == CampaignStatus.CREATED:
            updated_campaign = updated_campaign.with_status(CampaignStatus.RUNNING)
        if new_turbo_state is not None:
            turbo_dict = _turbo_state_to_dict(new_turbo_state)
            updated_campaign = updated_campaign.with_turbo_state(turbo_dict)
        await campaign_repo.save(updated_campaign, expected_version=campaign.version)

        # Invalidate diagnostics cache since iteration/suggestions have changed
        diagnostics_cache.invalidate(campaign_id)

        logger.info(
            "Generated %d suggestions for campaign %s, iteration=%d",
            len(suggestions),
            campaign_id,
            new_iteration,
        )
        full_response = _format_suggestions_response(
            suggestions,
            new_iteration,
            opt_spec,
            len(results),
            pending_info=pending_info,
            diversity_info=diversity_info,
            model_health_warnings=model_health_warnings,
        )
        return format_suggestions_response(full_response, verbosity_level)
