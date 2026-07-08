"""Get diagnostics operation — protocol-neutral diagnostics computation.

Orchestrates diagnostic sections by delegating to focused submodules in
``operations.diagnostics.*``. Model-based computation (GP fitting, LOO-CV,
feature importance, outlier detection, Pareto/hypervolume) is delegated to the
BOBackend protocol. Server-side concerns (campaign state, suggestion provenance,
caching, formatting) are handled here and in the submodules.

The backend diagnostics call is offloaded via ``asyncio.to_thread`` so it
cannot stall the event loop while other requests are served.
"""

import asyncio
import logging
from typing import Any
from uuid import UUID

from bo_engine.backend import DiagnosticSection
from bo_engine.backend_base import BackendError
from bo_engine.progress import ProgressCallback
from bo_mcp_server.backend import get_backend_async
from bo_mcp_server.cache import diagnostics_cache
from bo_mcp_server.converters import campaign_spec_to_optimization_spec
from bo_mcp_server.domain import Campaign, CampaignSpec, Result, Suggestion, SuggestionStatus
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.metrics import record_diagnostics_cache
from bo_mcp_server.operations.diagnostics import (
    compute_constraint_satisfaction_metrics,
    compute_convergence_diagnostics,
    compute_exploration_exploitation,
    compute_health_and_progress,
    compute_next_action_recommendation,
    compute_outcome_constraint_calibration_metrics,
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


def health_model_correlation(raw_correlation: float | None) -> float:
    """Map the backend's ``model_correlation`` into the health functions' domain.

    ``None`` means the backend could not measure the correlation; it maps
    to NaN so the health functions treat it as "unknown" (no warning, no
    status downgrade). A measured value — including a genuine ``0.0``,
    which is exactly the "model is uninformative" signal — flows through
    unchanged. A falsy-zero rewrite to a neutral prior would suppress the
    low-correlation warning precisely when the model is worst.
    """
    return float("nan") if raw_correlation is None else float(raw_correlation)


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


async def _compute_sections(
    requested: frozenset[str],
    spec: CampaignSpec,
    results: list[Result],
    all_suggestions: list[Suggestion],
    pending_suggestions: list[Suggestion],
    campaign: Campaign,
    progress_callback: ProgressCallback | None = None,
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

    # Delegate model-based computation to the backend — offloaded to a
    # worker thread so GP fitting / LOO-CV do not block the event loop.
    # The backend is loaded at point of need: sections that are pure
    # server-side computation (convergence, constraints) must not pay a
    # multi-second cold import for a backend they never call.
    backend_sections = _map_backend_sections(requested)
    wants_model_info = bool({"objectives", "health"} & requested)
    if backend_sections or wants_model_info:
        backend = await get_backend_async(spec.backend)
        if backend_sections:
            observations = results_to_observations(results)
            diagnostics.update(
                await asyncio.to_thread(
                    backend.compute_diagnostics,
                    opt_spec,
                    observations,
                    backend_sections,
                    progress_callback,
                )
            )

        # Enrich with server-side model info, sourced from the campaign's own
        # backend (issue #57: the previous static text always described BoTorch).
        if wants_model_info:
            try:
                method_info = backend.select_methods(opt_spec, len(results))
            except BackendError:
                logger.warning("select_methods failed for backend %s", spec.backend, exc_info=True)
                method_info = None
            enrich_diagnostics(diagnostics, spec, results, method_info)

    # Health (plain-Python functions + backend correlation)
    model_correlation = health_model_correlation(diagnostics.get("model_correlation"))
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
        # Calibration fits one feasibility GP per outcome constraint —
        # offloaded like the backend sections so concurrent sessions
        # are not frozen for the fit duration.
        await asyncio.to_thread(compute_constraint_satisfaction_metrics, results, spec, diagnostics)
        await asyncio.to_thread(
            compute_outcome_constraint_calibration_metrics, results, spec, diagnostics
        )

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
    progress_callback: ProgressCallback | None = None,
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
        progress_callback: Optional progress hook (see
            :class:`bo_engine.progress.ProgressEvent`). Backend
            diagnostic sections emit start/done milestones through this.

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

        # Keyed by section set so partial-section reads (e.g. only
        # "constraints", which re-fits feasibility GPs) are cached too,
        # not just the full envelope.
        section_key = ",".join(sorted(requested))
        cache_key = f"diagnostics:{campaign_id}:{campaign.version}:{section_key}"
        if use_cache:
            cached = await diagnostics_cache.get(cache_key)
            if cached is not None:
                record_diagnostics_cache("hit")
                logger.debug("Returning cached diagnostics for campaign %s", campaign_id)
                return format_diagnostics_response(cached, verbosity_level)
            record_diagnostics_cache("miss")

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

        diagnostics = await _compute_sections(
            requested,
            spec,
            results,
            all_suggestions,
            pending_suggestions,
            campaign,
            progress_callback=progress_callback,
        )

        logger.info(
            "Diagnostics computed for campaign %s: results=%d, health=%s",
            campaign_id,
            len(results),
            diagnostics.get("health_status", "unknown"),
        )

        await diagnostics_cache.set(cache_key, diagnostics)

        return format_diagnostics_response(diagnostics, verbosity_level)
