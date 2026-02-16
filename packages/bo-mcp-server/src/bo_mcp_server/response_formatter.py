"""Response formatting utilities for verbosity control.

This module provides verbosity-based response formatting to reduce token
consumption for AI agents. Three verbosity levels are supported:

- MINIMAL (~50 tokens): Success status + key metrics only. Use for tight loops.
- STANDARD (~200 tokens): Default. Excludes debugging fields like hyperparameters.
- DETAILED (~500+ tokens): All fields including LOO-CV metrics and hyperparameters.

Usage:
    from bo_mcp_server.response_formatter import (
        VerbosityLevel,
        format_diagnostics_response,
        format_suggestions_response,
    )

    # In tool function:
    full_response = compute_diagnostics(...)
    return format_diagnostics_response(full_response, VerbosityLevel(verbosity))
"""

from enum import StrEnum
from typing import Any


class VerbosityLevel(StrEnum):
    """Verbosity levels for MCP tool responses.

    Attributes:
        MINIMAL: ~50 tokens - success + key metric only
        STANDARD: ~200 tokens - current default (excludes debug fields)
        DETAILED: ~500+ tokens - all fields including debug info
    """

    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"


def format_diagnostics_response(
    full_response: dict[str, Any],
    verbosity: VerbosityLevel = VerbosityLevel.STANDARD,
) -> dict[str, Any]:
    """Format diagnostics response based on verbosity level.

    Args:
        full_response: Complete diagnostics response dictionary from get_diagnostics.
        verbosity: Desired verbosity level.

    Returns:
        Filtered response dictionary based on verbosity.

    Example minimal response (~50 tokens):
        {
            "success": true,
            "status": "running",
            "iteration": 5,
            "n_results": 10,
            "health": "healthy",
            "progress": "improving",
            "key_metric": {"best_value": 0.123},
            "converged": false,
            "errors": []
        }
    """
    if verbosity == VerbosityLevel.MINIMAL:
        # Determine if multi-objective based on pareto_front presence
        is_multi = full_response.get("pareto_front") is not None
        key_metric = (
            {"hypervolume": full_response.get("hypervolume")}
            if is_multi
            else {"best_value": full_response.get("best_value")}
        )

        return {
            "success": full_response.get("success"),
            "status": full_response.get("campaign_status"),
            "iteration": full_response.get("iteration"),
            "n_results": full_response.get("n_results"),
            "health": full_response.get("health_status"),
            "progress": full_response.get("progress_status"),
            "key_metric": key_metric,
            "converged": full_response.get("convergence", {}).get("converged", False),
            "next_action": full_response.get("next_action_recommendation"),
            "errors": full_response.get("errors", []),
        }

    elif verbosity == VerbosityLevel.STANDARD:
        # Remove debugging/advanced fields for standard verbosity
        exclude_fields = {
            "hyperparameters",
            "loo_cv_metrics",
            "hypervolume_history",
            "outliers",
        }
        return {k: v for k, v in full_response.items() if k not in exclude_fields}

    # DETAILED returns everything
    return full_response


def format_suggestions_response(
    full_response: dict[str, Any],
    verbosity: VerbosityLevel = VerbosityLevel.STANDARD,
) -> dict[str, Any]:
    """Format suggestions response based on verbosity level.

    Args:
        full_response: Complete suggestions response dictionary from generate_suggestions.
        verbosity: Desired verbosity level.

    Returns:
        Filtered response dictionary based on verbosity.

    Example minimal response (~50 tokens):
        {
            "success": true,
            "iteration": 3,
            "suggestion_ids": ["uuid1", "uuid2"],
            "method": "qLogNEHVI",
            "errors": []
        }
    """
    if verbosity == VerbosityLevel.MINIMAL:
        suggestions = full_response.get("suggestions", [])
        method_selection = full_response.get("method_selection", {})

        return {
            "success": full_response.get("success"),
            "iteration": full_response.get("iteration"),
            "suggestion_ids": [s.get("id") for s in suggestions],
            "method": method_selection.get("acquisition_function"),
            "errors": full_response.get("errors", []),
        }

    elif verbosity == VerbosityLevel.STANDARD:
        # Remove pending_points for standard (still shows batch_diversity)
        exclude_fields = {"pending_points"}
        return {k: v for k, v in full_response.items() if k not in exclude_fields}

    # DETAILED returns everything
    return full_response


def format_compare_campaigns_response(
    full_response: dict[str, Any],
    verbosity: VerbosityLevel = VerbosityLevel.STANDARD,
) -> dict[str, Any]:
    """Format compare_campaigns response based on verbosity level.

    Args:
        full_response: Complete compare_campaigns response dictionary.
        verbosity: Desired verbosity level.

    Returns:
        Filtered response dictionary based on verbosity.

    Example minimal response:
        {
            "success": true,
            "n_campaigns": 3,
            "best_performer": "Campaign A",
            "recommendation": "...",
            "errors": []
        }
    """
    if verbosity == VerbosityLevel.MINIMAL:
        comparison = full_response.get("comparison", {}) or {}
        campaigns = full_response.get("campaigns", [])

        return {
            "success": full_response.get("success"),
            "n_campaigns": len(campaigns),
            "best_performer": (
                comparison.get("best_sample_efficiency")
                or comparison.get("best_single_objective")
                or comparison.get("best_multi_objective")
            ),
            "recommendation": comparison.get("recommendation"),
            "errors": full_response.get("errors", []),
        }

    elif verbosity == VerbosityLevel.STANDARD:
        # Keep most fields but simplify campaign metrics
        campaigns = full_response.get("campaigns", [])
        simplified_campaigns = []
        for c in campaigns:
            simplified_campaigns.append(
                {
                    "campaign_id": c.get("campaign_id"),
                    "campaign_name": c.get("campaign_name"),
                    "status": c.get("status"),
                    "n_results": c.get("n_results"),
                    "best_value": c.get("best_value"),
                    "hypervolume": c.get("hypervolume"),
                    "sample_efficiency": c.get("sample_efficiency"),
                }
            )
        return {
            "success": full_response.get("success"),
            "campaigns": simplified_campaigns,
            "comparison": full_response.get("comparison"),
            "errors": full_response.get("errors", []),
        }

    # DETAILED returns everything
    return full_response


def format_transfer_candidates_response(
    full_response: dict[str, Any],
    verbosity: VerbosityLevel = VerbosityLevel.STANDARD,
) -> dict[str, Any]:
    """Format discover_transfer_candidates response based on verbosity level.

    Args:
        full_response: Complete discover_transfer_candidates response dictionary.
        verbosity: Desired verbosity level.

    Returns:
        Filtered response dictionary based on verbosity.

    Example minimal response:
        {
            "success": true,
            "n_candidates": 3,
            "top_candidate_id": "uuid",
            "top_similarity": 0.85,
            "recommendation": "...",
            "errors": []
        }
    """
    if verbosity == VerbosityLevel.MINIMAL:
        candidates = full_response.get("candidates", [])
        top_candidate = candidates[0] if candidates else None

        return {
            "success": full_response.get("success"),
            "n_candidates": len(candidates),
            "top_candidate_id": top_candidate.get("campaign_id") if top_candidate else None,
            "top_similarity": top_candidate.get("similarity_score") if top_candidate else None,
            "recommendation": full_response.get("overall_recommendation"),
            "errors": full_response.get("errors", []),
        }

    elif verbosity == VerbosityLevel.STANDARD:
        # Simplify candidate info
        candidates = full_response.get("candidates", [])
        simplified_candidates = []
        for c in candidates:
            simplified_candidates.append(
                {
                    "campaign_id": c.get("campaign_id"),
                    "name": c.get("name"),
                    "n_results": c.get("n_results"),
                    "similarity_score": c.get("similarity_score"),
                    "recommendation": c.get("recommendation"),
                }
            )
        return {
            "success": full_response.get("success"),
            "target_campaign": full_response.get("target_campaign"),
            "candidates": simplified_candidates,
            "overall_recommendation": full_response.get("overall_recommendation"),
            "errors": full_response.get("errors", []),
        }

    # DETAILED returns everything
    return full_response


def format_create_campaign_response(
    full_response: dict[str, Any],
    verbosity: VerbosityLevel = VerbosityLevel.STANDARD,
) -> dict[str, Any]:
    """Format create_campaign response based on verbosity level.

    Args:
        full_response: Complete create_campaign response dictionary.
        verbosity: Desired verbosity level.

    Returns:
        Filtered response dictionary based on verbosity.

    Example minimal response (~30 tokens):
        {
            "success": true,
            "campaign_id": "uuid",
            "errors": []
        }
    """
    if verbosity == VerbosityLevel.MINIMAL:
        return {
            "success": full_response.get("success"),
            "campaign_id": full_response.get("campaign_id"),
            "errors": full_response.get("errors", []),
        }

    elif verbosity == VerbosityLevel.STANDARD:
        # Standard includes spec_id and campaign_name
        return {
            "success": full_response.get("success"),
            "campaign_id": full_response.get("campaign_id"),
            "spec_id": full_response.get("spec_id"),
            "campaign_name": full_response.get("campaign_name"),
            "errors": full_response.get("errors", []),
        }

    # DETAILED returns everything
    return full_response


def format_submit_results_response(
    full_response: dict[str, Any],
    verbosity: VerbosityLevel = VerbosityLevel.STANDARD,
) -> dict[str, Any]:
    """Format submit_results response based on verbosity level.

    Args:
        full_response: Complete submit_results response dictionary.
        verbosity: Desired verbosity level.

    Returns:
        Filtered response dictionary based on verbosity.

    Example minimal response (~40 tokens):
        {
            "success": true,
            "n_submitted": 3,
            "errors": []
        }
    """
    if verbosity == VerbosityLevel.MINIMAL:
        result_ids = full_response.get("result_ids", [])
        return {
            "success": full_response.get("success"),
            "n_submitted": len(result_ids),
            "errors": full_response.get("errors", []),
        }

    elif verbosity == VerbosityLevel.STANDARD:
        # Standard includes result_ids and warnings, but not duplicate details
        return {
            "success": full_response.get("success"),
            "result_ids": full_response.get("result_ids", []),
            "errors": full_response.get("errors", []),
            "warnings": full_response.get("warnings", []),
            "n_duplicates_detected": len(full_response.get("duplicates_detected", [])),
        }

    # DETAILED returns everything
    return full_response


def format_validate_intake_response(
    full_response: dict[str, Any],
    verbosity: VerbosityLevel = VerbosityLevel.STANDARD,
) -> dict[str, Any]:
    """Format validate_intake response based on verbosity level.

    Args:
        full_response: Complete validate_intake response dictionary.
        verbosity: Desired verbosity level.

    Returns:
        Filtered response dictionary based on verbosity.

    Example minimal response (~20 tokens):
        {
            "valid": true,
            "errors": []
        }
    """
    if verbosity == VerbosityLevel.MINIMAL:
        return {
            "valid": full_response.get("valid"),
            "errors": full_response.get("errors", []),
        }

    elif verbosity == VerbosityLevel.STANDARD:
        # Standard includes warnings but simplified spec
        spec = full_response.get("spec")
        spec_summary = None
        if spec:
            spec_summary = {
                "name": spec.get("name"),
                "n_parameters": len(spec.get("parameters", [])),
                "n_objectives": len(spec.get("objectives", [])),
                "n_constraints": len(spec.get("constraints", [])),
                "batch_size": spec.get("batch_size"),
            }
        return {
            "valid": full_response.get("valid"),
            "errors": full_response.get("errors", []),
            "warnings": full_response.get("warnings", []),
            "spec_summary": spec_summary,
        }

    # DETAILED returns everything
    return full_response
