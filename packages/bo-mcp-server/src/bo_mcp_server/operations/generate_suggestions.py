"""Generate suggestions operation — protocol-neutral business logic.

Extracted from the MCP tool so it can be reused by any transport
(MCP, REST, CLI, tests) without importing the MCP server object.

Uses the BOBackend protocol for all BO-engine interactions,
keeping the server decoupled from specific backend implementations.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from bo_engine.backend import BOBackend
from bo_engine.constants import PENDING_SUGGESTION_MAX_AGE_HOURS
from bo_engine.pending_points import filter_pending_points
from bo_engine.types import OptimizationSpec
from sqlalchemy.ext.asyncio import AsyncSession

from bo_mcp_server.backend import get_backend
from bo_mcp_server.converters import campaign_spec_to_optimization_spec
from bo_mcp_server.domain import (
    CampaignStatus,
    Result,
    Suggestion,
    SuggestionProvenance,
    SuggestionStatus,
)
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
    format_suggestions_response,
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


async def _handle_pending_suggestions(
    pending: list[Suggestion],
    suggestion_repo: SuggestionRepository,
) -> tuple[dict[str, Any] | None, list[Suggestion]]:
    """Filter pending suggestions, expire stale ones.

    Returns (pending_info dict or None, valid pending list).
    """
    if not pending:
        return None, []

    pending_params = [p.parameter_values for p in pending]
    pending_times = [p.created_at for p in pending]
    now = datetime.now(UTC)

    _, point_info = filter_pending_points(
        pending_params,
        pending_times,
        max_age_hours=PENDING_SUGGESTION_MAX_AGE_HOURS,
        now=now,
    )

    valid_pending: list[Suggestion] = []
    stale_count = 0
    for sugg, info in zip(pending, point_info, strict=True):
        if info.is_stale:
            await suggestion_repo.save(sugg.with_status(SuggestionStatus.EXPIRED))
            stale_count += 1
        else:
            valid_pending.append(sugg)

    pending_info = {
        "total_pending": len(pending),
        "valid_pending": len(valid_pending),
        "stale_expired": stale_count,
        "note": (
            f"{len(valid_pending)} pending experiments"
            " considered for diversity. "
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


def _build_initial_design_data(
    designs: list[dict[str, Any]],
    iteration: int,
    batch_size: int,
    random_seed: int | None = None,
) -> SuggestionDataList:
    """Build suggestion data tuples from initial design points."""
    return [
        (
            design,
            {
                "iteration": iteration,
                "batch_index": i,
                "generation_method": "initial_design",
                "random_seed": random_seed,
                "explanation": (
                    f"Initial design point "
                    f"{i + 1}/{batch_size}"
                    " using Sobol sequence. Initial"
                    " designs explore the parameter"
                    " space before model-guided"
                    " optimization."
                ),
                "confidence_level": "medium",
            },
        )
        for i, design in enumerate(designs)
    ]


def _compute_diversity_info(
    backend: BOBackend,
    opt_spec: OptimizationSpec,
    suggestions: list[Suggestion],
) -> dict[str, Any] | None:
    """Compute batch diversity metrics via the backend.

    Returns diversity info dict, or None if fewer than 2
    suggestions or on failure.
    """
    if len(suggestions) <= 1:
        return None

    try:
        candidates = [s.parameter_values for s in suggestions]
        metrics = backend.compute_batch_diversity(opt_spec, candidates)
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


async def generate_suggestions_operation(
    campaign_id: str,
    batch_size: int | None = None,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
) -> dict[str, Any]:
    """Generate next batch of experiment suggestions.

    Protocol-neutral entry point — contains all business logic
    formerly housed in the MCP tool handler. Uses the BOBackend
    protocol for all BO-engine interactions.

    Args:
        campaign_id: UUID of the campaign
        batch_size: Number of suggestions (default: campaign's)
        verbosity: Response detail level

    Returns:
        Dictionary with success, suggestions, iteration, errors.
    """
    logger.info(
        "Generating suggestions for campaign_id=%s, batch_size=%s, verbosity=%s",
        campaign_id,
        batch_size,
        verbosity,
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

    # --- Database session scope ---
    try:
        async with get_session() as session:
            repos = _init_repositories(session)
            return await _generate_within_session(
                campaign_id,
                campaign_uuid,
                batch_size,
                verbosity_level,
                repos,
            )
    except ConcurrentModificationError as err:
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


def _init_repositories(session: AsyncSession) -> _Repositories:
    """Initialize all repository instances for a session."""
    return _Repositories(
        campaign=CampaignRepository(session),
        spec=CampaignSpecRepository(session),
        result=ResultRepository(session),
        suggestion=SuggestionRepository(session),
    )


async def _generate_within_session(
    campaign_id: str,
    campaign_uuid: UUID,
    batch_size: int | None,
    verbosity_level: VerbosityLevel,
    repos: _Repositories,
) -> dict[str, Any]:
    """Run the suggestion generation within an active DB session.

    Handles fetching, validation, generation, and persistence.
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

    # Handle pending suggestions
    pending = await suggestion_repo.list_by_campaign(campaign_uuid, status=SuggestionStatus.PENDING)
    pending_info, _ = await _handle_pending_suggestions(pending, suggestion_repo)

    # Prepare generation inputs
    actual_batch_size = batch_size or spec.batch_size
    opt_spec = campaign_spec_to_optimization_spec(spec)
    new_iteration = campaign.iteration + 1
    backend = get_backend(spec.backend)

    logger.debug(
        "Generation context: n_results=%d, batch_size=%d, iteration=%d",
        len(results),
        actual_batch_size,
        new_iteration,
    )

    # Generate suggestions via backend
    suggestion_data, new_backend_state, warnings = _generate_via_backend(
        backend,
        opt_spec,
        results,
        actual_batch_size,
        new_iteration,
        campaign.backend_state,
        random_seed=spec.random_seed,
    )

    # Create and save suggestion entities
    suggestions = await _create_and_save_suggestions(
        suggestion_data, campaign_uuid, suggestion_repo
    )

    # Compute batch diversity
    diversity_info = _compute_diversity_info(backend, opt_spec, suggestions)

    # Update campaign state
    updated_campaign = campaign.advance_iteration()
    if campaign.status == CampaignStatus.CREATED:
        updated_campaign = updated_campaign.with_status(CampaignStatus.RUNNING)
    if new_backend_state is not None:
        updated_campaign = updated_campaign.with_backend_state(new_backend_state)
    await campaign_repo.save(
        updated_campaign,
        expected_version=campaign.version,
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


def _generate_via_backend(
    backend: BOBackend,
    opt_spec: OptimizationSpec,
    results: list[Result],
    batch_size: int,
    iteration: int,
    prior_backend_state: dict[str, Any] | None,
    random_seed: int | None = None,
) -> tuple[
    SuggestionDataList,
    dict[str, Any] | None,
    list[str],
]:
    """Dispatch to initial design or BO suggestions.

    Returns (suggestion_data, backend_state, warnings).
    """
    if len(results) == 0:
        designs = backend.generate_initial_design(opt_spec, batch_size)
        suggestion_data = _build_initial_design_data(
            designs, iteration, batch_size, random_seed=random_seed
        )
        return suggestion_data, None, []

    observations = results_to_observations(results)
    batch = backend.generate_suggestions(
        spec=opt_spec,
        observations=observations,
        batch_size=batch_size,
        iteration=iteration,
        backend_state=prior_backend_state,
    )
    suggestion_data = [(item["parameter_values"], item["provenance"]) for item in batch.suggestions]
    return (
        suggestion_data,
        batch.backend_state,
        batch.warnings,
    )
