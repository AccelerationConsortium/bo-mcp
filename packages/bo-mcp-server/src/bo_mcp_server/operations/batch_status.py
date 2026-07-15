"""Shared batch status operations."""

import logging
import math
from typing import Any
from uuid import UUID

from bo_mcp_server.constants import HYPERVOLUME_STABILITY_THRESHOLD
from bo_mcp_server.domain import Campaign, CampaignSpec, CampaignStatus
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.operations.helpers import parse_verbosity
from bo_mcp_server.response_formatter import VerbosityLevel, with_response_metadata
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    ResultRepository,
    SuggestionRepository,
    get_session,
)

logger = logging.getLogger(__name__)

MAX_BATCH_SIZE = 20


def _determine_health(status: CampaignStatus, n_results: int, iteration: int) -> str:
    if status == CampaignStatus.FAILED:
        return "critical"
    if status == CampaignStatus.PAUSED:
        return "paused"
    if n_results == 0 and iteration > 1:
        return "warning"
    return "healthy"


def _compute_key_metric(
    spec: CampaignSpec | None, hypervolume_history: list[float]
) -> dict[str, Any]:
    if spec and len(spec.objectives) >= 2 and hypervolume_history:
        return {"hypervolume": hypervolume_history[-1]}
    return {}


def _compute_convergence(hypervolume_history: list[float]) -> dict[str, Any]:
    convergence_info: dict[str, Any] = {"converged": False}
    if hypervolume_history and len(hypervolume_history) >= 5:
        recent = hypervolume_history[-5:]
        if (
            all(math.isclose(r, recent[0], rel_tol=1e-9) for r in recent)
            or (max(recent) - min(recent)) < HYPERVOLUME_STABILITY_THRESHOLD
        ):
            convergence_info["converged"] = True
            convergence_info["reason"] = "Hypervolume stable"
    return convergence_info


def _minimal_next_action(
    status: CampaignStatus,
    n_results: int,
    n_pending: int,
    iteration: int,
    max_iterations: int | None,
) -> dict[str, str]:
    """Lightweight next-action hint for the minimal-verbosity batch envelope.

    Mirrors the decision tree (including the iteration-budget guard) in
    :func:`bo_mcp_server.operations.diagnostics.actions.compute_next_action_recommendation`
    but consumes only the fields available at minimal verbosity (no
    convergence / outlier / health-status counters).  The richer
    recommendation lives on the ``bo_get_diagnostics`` surface.
    """
    # max_iterations lives in the immutable intake; neither resume nor reopen
    # resets the iteration counter, so once the budget is spent there is no
    # continuation path from any lifecycle state. Emit the same routable
    # terminate_campaign action the stopping decision uses
    # (bo_engine.convergence.evaluate_stopping_decision) instead of pointing
    # agents at resume/reopen/generate calls the server will reject.
    # Exception: suggestions still awaiting results (n_pending counts
    # PENDING and ACCEPTED here) need submission, which requires
    # CREATED/RUNNING (Campaign.can_submit_results) while terminate is
    # irreversible — so with outstanding work the resume/reopen hint must win.
    budget_exhausted = max_iterations is not None and iteration >= max_iterations
    if status in (CampaignStatus.PAUSED, CampaignStatus.COMPLETED, CampaignStatus.FAILED):
        if budget_exhausted and n_pending == 0 and status != CampaignStatus.FAILED:
            # COMPLETED is already terminate's target state — the lifecycle
            # layer would accept the call only as an idempotent no-op — so
            # there the hint reviews the finished campaign instead.
            if status == CampaignStatus.COMPLETED:
                action = "review_campaign_status"
                advice = "review the final results — the campaign is finished"
            else:
                action = "terminate_campaign"
                advice = "review results and terminate it"
            return {
                "action": action,
                "reason": (
                    f"Campaign is {status.value} and has reached "
                    f"max_iterations={max_iterations}; the budget cannot be "
                    f"extended — {advice}."
                ),
                "urgency": "low",
            }
        continuation = {
            CampaignStatus.PAUSED: "resume it to continue, or terminate it",
            CampaignStatus.COMPLETED: "reopen it to continue optimization",
            CampaignStatus.FAILED: "inspect errors before retrying",
        }[status]
        return {
            "action": "review_campaign_status",
            "reason": f"Campaign is {status.value}; {continuation}.",
            "urgency": "low",
        }
    if n_pending > 0:
        return {
            "action": "bo_submit_results",
            "reason": f"{n_pending} suggestion(s) awaiting results.",
            "urgency": "normal",
        }
    # Status never auto-transitions to COMPLETED on budget exhaustion, so a
    # campaign can sit in RUNNING forever with its budget already spent.
    if budget_exhausted:
        return {
            "action": "terminate_campaign",
            "reason": (
                f"Campaign has reached max_iterations={max_iterations}; the "
                "budget cannot be extended — review results and terminate it."
            ),
            "urgency": "low",
        }
    if n_results == 0:
        return {
            "action": "bo_generate_suggestions",
            "reason": "No results yet — generate initial suggestions to start optimization.",
            "urgency": "normal",
        }
    return {
        "action": "bo_generate_suggestions",
        "reason": (
            f"Campaign healthy with {n_results} result(s); request the next batch. "
            "Use bo_get_diagnostics or verbosity='detailed' for convergence/outlier checks."
        ),
        "urgency": "normal",
    }


def _build_minimal_info(
    name: str,
    campaign: Campaign,
    n_results: int,
    n_pending: int,
    spec: CampaignSpec | None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": campaign.status.value,
        "iteration": campaign.iteration,
        "n_results": n_results,
        "next_action_recommendation": _minimal_next_action(
            campaign.status,
            n_results,
            n_pending,
            campaign.iteration,
            spec.max_iterations if spec else None,
        ),
    }


def _build_standard_info(
    name: str,
    campaign: Campaign,
    n_results: int,
    n_pending: int,
    spec: CampaignSpec | None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": campaign.status.value,
        "iteration": campaign.iteration,
        "n_results": n_results,
        "n_pending_suggestions": n_pending,
        "health": _determine_health(campaign.status, n_results, campaign.iteration),
        "key_metric": _compute_key_metric(spec, campaign.hypervolume_history),
    }


def _build_detailed_info(
    name: str,
    campaign: Campaign,
    n_results: int,
    n_pending: int,
    spec: CampaignSpec | None,
) -> dict[str, Any]:
    info = _build_standard_info(name, campaign, n_results, n_pending, spec)
    info.update(
        {
            "convergence": _compute_convergence(campaign.hypervolume_history),
            "created_at": campaign.created_at.isoformat(),
            "owner_id": str(campaign.owner_id),
        }
    )
    return info


def _parse_campaign_ids(
    campaign_ids: list[str],
) -> tuple[list[tuple[str, UUID]], list[str]]:
    """Parse and validate campaign ID strings into UUIDs."""
    valid: list[tuple[str, UUID]] = []
    invalid: list[str] = []
    for cid in campaign_ids:
        try:
            valid.append((cid, UUID(cid)))
        except ValueError:
            invalid.append(cid)
    return valid, invalid


def _validate_batch_request(
    campaign_ids: list[str], verbosity: str
) -> dict[str, Any] | VerbosityLevel:
    """Validate batch request parameters. Returns error dict or VerbosityLevel."""
    verbosity_result = parse_verbosity(verbosity)
    if isinstance(verbosity_result, dict):
        return verbosity_result
    verbosity_level = verbosity_result

    if not campaign_ids:
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message="At least one campaign_id is required",
        )

    if len(campaign_ids) > MAX_BATCH_SIZE:
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Too many campaign_ids ({len(campaign_ids)}). Maximum is {MAX_BATCH_SIZE}.",
            details={"max_batch_size": MAX_BATCH_SIZE, "requested": len(campaign_ids)},
        )

    return verbosity_level


def _build_campaign_info(
    verbosity_level: VerbosityLevel,
    name: str,
    campaign: Campaign,
    n_results: int,
    n_pending: int,
    spec: CampaignSpec | None,
) -> dict[str, Any]:
    """Build campaign info dict based on verbosity."""
    if verbosity_level == VerbosityLevel.MINIMAL:
        return _build_minimal_info(name, campaign, n_results, n_pending, spec)
    if verbosity_level == VerbosityLevel.STANDARD:
        return _build_standard_info(name, campaign, n_results, n_pending, spec)
    return _build_detailed_info(name, campaign, n_results, n_pending, spec)


@with_response_metadata
async def batch_get_status_operation(
    campaign_ids: list[str],
    verbosity: str = "minimal",
) -> dict[str, Any]:
    """Get status for multiple campaigns in one call.

    Decorated with ``with_response_metadata`` so every return path
    carries the ``_metadata`` envelope and ``schema_version`` — the same
    contract the other envelope-shaped operations emit, and the one the
    REST ``ResponseEnvelope`` advertises for batch status. Reaches both
    transports because the MCP tool and REST route call this operation
    directly.
    """
    logger.info(
        "Batch getting status for %d campaigns, verbosity=%s",
        len(campaign_ids),
        verbosity,
    )

    validated = _validate_batch_request(campaign_ids, verbosity)
    if isinstance(validated, dict):
        return validated
    verbosity_level = validated

    valid_uuids, invalid_ids = _parse_campaign_ids(campaign_ids)
    errors: list[str] = []
    if invalid_ids:
        errors.append(f"Invalid UUID format: {invalid_ids}")

    failed_ids: list[str] = list(invalid_ids)
    campaigns_info: dict[str, dict[str, Any]] = {}

    if not valid_uuids:
        return {
            "success": False,
            "campaigns": campaigns_info,
            "failed_ids": failed_ids,
            "errors": errors,
        }

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        spec_repo = CampaignSpecRepository(session)
        result_repo = ResultRepository(session)
        suggestion_repo = SuggestionRepository(session)

        # Batch-fetch all data
        id_str_map = {uuid: id_str for id_str, uuid in valid_uuids}
        uuid_list = list(id_str_map.keys())
        campaigns = await campaign_repo.get_by_ids(uuid_list)

        failed_ids.extend(id_str_map[uuid] for uuid in uuid_list if uuid not in campaigns)

        found_uuids = list(campaigns.keys())
        spec_ids = list({c.spec_id for c in campaigns.values()})
        specs = await spec_repo.get_by_ids(spec_ids)
        result_counts = await result_repo.count_by_campaigns(found_uuids)

        # The standard/detailed envelopes report ``n_pending_suggestions``
        # with its historical strict meaning (PENDING only). The minimal
        # envelope carries no such field — its next-action hint instead needs
        # the actionable count (PENDING + ACCEPTED, the submit pipeline's
        # gate) to know whether results can still be submitted before hinting
        # at the irreversible terminate action (see _minimal_next_action).
        if verbosity_level == VerbosityLevel.MINIMAL:
            suggestion_counts = await suggestion_repo.count_actionable_by_campaigns(found_uuids)
        else:
            suggestion_counts = await suggestion_repo.count_pending_by_campaigns(found_uuids)

        for campaign_uuid, campaign in campaigns.items():
            cid = id_str_map[campaign_uuid]
            spec = specs.get(campaign.spec_id)
            name = spec.name if spec else "Unknown"
            campaigns_info[cid] = _build_campaign_info(
                verbosity_level,
                name,
                campaign,
                result_counts.get(campaign_uuid, 0),
                suggestion_counts.get(campaign_uuid, 0),
                spec,
            )

    if failed_ids:
        errors.append(f"Could not retrieve campaigns: {failed_ids}")

    logger.info(
        "Batch status complete: %d succeeded, %d failed",
        len(campaigns_info),
        len(failed_ids),
    )

    return {
        "success": len(campaigns_info) > 0 or len(failed_ids) == 0,
        "campaigns": campaigns_info,
        "failed_ids": failed_ids,
        "errors": errors,
    }
