"""Get diagnostics operation — protocol-neutral diagnostics computation.

Orchestrates diagnostic sections by delegating to focused submodules in
``operations.diagnostics.*``. Model-based computation (GP fitting, LOO-CV,
feature importance, outlier detection, Pareto/hypervolume) is delegated to the
BOBackend protocol. Server-side concerns (campaign state, suggestion provenance,
caching, formatting) are handled here and in the submodules.
"""

import logging
from typing import Any
from uuid import UUID

from bo_engine.backend import DiagnosticSection

from bo_mcp_server.backend import get_backend
from bo_mcp_server.cache import diagnostics_cache
from bo_mcp_server.converters import campaign_spec_to_optimization_spec
from bo_mcp_server.domain import Campaign, CampaignSpec, Result, Suggestion, SuggestionStatus
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.operations.diagnostics import (
    compute_constraint_satisfaction_metrics,
    compute_convergence_diagnostics,
    compute_exploration_exploitation,
    compute_health_and_progress,
    compute_next_action_recommendation,
    compute_suggestion_diversity_metrics,
    compute_uncertainty_trends,
    enrich_diagnostics,
    enrich_outlier_results,
)
from bo_mcp_server.operations.helpers import (
    parse_campaign_id,
    parse_verbosity,
    results_to_observations,
)
from bo_mcp_server.response_formatter import VerbosityLevel, format_diagnostics_response
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    ResultRepository,
    SuggestionRepository,
    get_session,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Section constants
# =============================================================================

ALL_SECTIONS = frozenset(
    ["health", "objectives", "model", "convergence", "suggestions", "outliers", "constraints"]
)


# =============================================================================
# Input validation
# =============================================================================


def _validate_diagnostics_inputs(
    campaign_id: str,
    verbosity: str,
    sections: list[str] | None,
) -> tuple[VerbosityLevel, UUID, frozenset[str]] | dict[str, Any]:
    """Validate diagnostics inputs. Returns (level, uuid, sections) or error dict."""
    verbosity_result = parse_verbosity(verbosity)
    if isinstance(verbosity_result, dict):
        return verbosity_result
    verbosity_level = verbosity_result

    requested = ALL_SECTIONS if sections is None else frozenset(sections)
    invalid_sections = requested - ALL_SECTIONS
    if invalid_sections:
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=(
                f"Invalid sections: {sorted(invalid_sections)}. Valid: {sorted(ALL_SECTIONS)}"
            ),
        )

    campaign_id_result = parse_campaign_id(campaign_id)
    if isinstance(campaign_id_result, dict):
        return campaign_id_result
    campaign_uuid = campaign_id_result

    return verbosity_level, campaign_uuid, requested


# =============================================================================
# Backend section mapping
# =============================================================================


def _map_backend_sections(requested: frozenset[str]) -> frozenset[str]:
    """Map server-side section names to backend DiagnosticSection values."""
    mapping: dict[str, str] = {
        "objectives": DiagnosticSection.OBJECTIVES,
        "health": DiagnosticSection.OBJECTIVES,  # health needs objective data
        "model": DiagnosticSection.MODEL,
        "outliers": DiagnosticSection.OUTLIERS,
        "suggestions": DiagnosticSection.SUGGESTIONS_TENSOR,
    }
    return frozenset(mapping[s] for s in requested if s in mapping)


# =============================================================================
# Section orchestration
# =============================================================================


def _compute_sections(
    requested: frozenset[str],
    spec: CampaignSpec,
    results: list[Result],
    all_suggestions: list[Suggestion],
    pending_suggestions: list[Suggestion],
    campaign: Campaign,
) -> dict[str, Any]:
    """Compute only the requested diagnostic sections."""
    is_single_objective = len(spec.objectives) == 1
    opt_spec = campaign_spec_to_optimization_spec(spec)

    diagnostics: dict[str, Any] = {
        "success": True,
        "campaign_status": campaign.status.value,
        "iteration": campaign.iteration,
        "n_results": len(results),
        "n_pending_suggestions": len(pending_suggestions),
        "errors": [],
    }

    # Delegate model-based computation to the backend
    backend_sections = _map_backend_sections(requested)
    if backend_sections:
        backend = get_backend(spec.backend)
        observations = results_to_observations(results)
        diagnostics.update(backend.compute_diagnostics(opt_spec, observations, backend_sections))

    # Enrich with server-side model info
    if "objectives" in requested or "health" in requested:
        enrich_diagnostics(diagnostics, spec, results, is_single_objective)

    # Health (plain-Python functions + backend correlation)
    model_correlation = diagnostics.get("model_correlation") or 0.5
    if "health" in requested:
        compute_health_and_progress(
            spec,
            results,
            campaign.iteration,
            is_single_objective,
            model_correlation,
            campaign.hypervolume_history,
            diagnostics,
        )
        compute_next_action_recommendation(
            diagnostics,
            len(pending_suggestions),
            campaign.status.value,
        )

    # Suggestions — server-side parts (provenance-based)
    if "suggestions" in requested:
        compute_uncertainty_trends(all_suggestions, diagnostics)
        compute_exploration_exploitation(all_suggestions, results, spec, opt_spec, diagnostics)
        compute_suggestion_diversity_metrics(all_suggestions, opt_spec, diagnostics)

    if "constraints" in requested:
        compute_constraint_satisfaction_metrics(results, spec, diagnostics)

    if "convergence" in requested:
        compute_convergence_diagnostics(
            spec,
            is_single_objective,
            campaign.hypervolume_history,
            diagnostics,
        )

    if "outliers" in requested:
        diagnostics["outliers"] = enrich_outlier_results(
            diagnostics.get("outliers"),
            results,
        )

    return diagnostics


# =============================================================================
# Public entry point
# =============================================================================


async def get_diagnostics_operation(
    campaign_id: str,
    use_cache: bool = True,
    verbosity: str = "standard",
    sections: list[str] | None = None,
) -> dict[str, Any]:
    """Compute diagnostic information for a campaign.

    Protocol-neutral operation used by MCP tools and REST routes.
    Delegates model-based computation to the BOBackend protocol.

    Args:
        campaign_id: UUID of the campaign
        use_cache: Whether to use cached results (default True).
        verbosity: Response verbosity level (minimal, standard, detailed).
        sections: Optional list of sections to compute. When omitted, all
            sections are computed. Valid: health, objectives, model,
            convergence, suggestions, outliers, constraints.

    Returns:
        Formatted diagnostics dictionary.
    """
    logger.info(
        "Getting diagnostics for campaign_id=%s, use_cache=%s, verbosity=%s, sections=%s",
        campaign_id,
        use_cache,
        verbosity,
        sections,
    )

    validated = _validate_diagnostics_inputs(campaign_id, verbosity, sections)
    if isinstance(validated, dict):
        return validated
    verbosity_level, campaign_uuid, requested = validated

    is_full = requested == ALL_SECTIONS

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        spec_repo = CampaignSpecRepository(session)
        result_repo = ResultRepository(session)
        suggestion_repo = SuggestionRepository(session)

        campaign = await campaign_repo.get(campaign_uuid)
        if campaign is None:
            return make_error_response(
                ErrorCode.CAMPAIGN_NOT_FOUND,
                message=f"Campaign {campaign_id} not found",
                details={"campaign_id": campaign_id},
            )

        cache_key = f"diagnostics:{campaign_id}:{campaign.version}"
        if use_cache and is_full:
            cached = await diagnostics_cache.get(cache_key)
            if cached is not None:
                logger.debug("Returning cached diagnostics for campaign %s", campaign_id)
                return format_diagnostics_response(cached, verbosity_level)

        spec = await spec_repo.get(campaign.spec_id)
        if spec is None:
            return make_error_response(
                ErrorCode.DATABASE_ERROR,
                message="Campaign spec not found",
                details={"spec_id": str(campaign.spec_id)},
            )

        results = await result_repo.list_by_campaign(campaign_uuid)
        all_suggestions = await suggestion_repo.list_by_campaign(campaign_uuid)
        pending_suggestions = [s for s in all_suggestions if s.status == SuggestionStatus.PENDING]

        diagnostics = _compute_sections(
            requested,
            spec,
            results,
            all_suggestions,
            pending_suggestions,
            campaign,
        )

        logger.info(
            "Diagnostics computed for campaign %s: results=%d, health=%s",
            campaign_id,
            len(results),
            diagnostics.get("health_status", "unknown"),
        )

        if is_full:
            await diagnostics_cache.set(cache_key, diagnostics)

        return format_diagnostics_response(diagnostics, verbosity_level)
