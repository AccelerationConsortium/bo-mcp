"""Next action recommendation for agents."""

from typing import Any


def compute_next_action_recommendation(
    diagnostics: dict[str, Any],
    n_actionable_suggestions: int,
    campaign_status: str,
    iteration: int,
    max_iterations: int | None,
) -> None:
    """Compute proactive next action recommendation for agents.

    ``n_actionable_suggestions`` counts PENDING and ACCEPTED suggestions —
    the submit pipeline's valid result-submission targets — not just the
    strictly-PENDING set behind the envelope's ``n_pending_suggestions``
    field.
    """
    n_results = diagnostics.get("n_results", 0)
    health_status = diagnostics.get("health_status", "unknown")
    convergence = diagnostics.get("convergence", {})
    converged = convergence.get("converged", False)
    outliers = diagnostics.get("outliers", {})
    outlier_count = outliers.get("count", 0) if outliers else 0

    # Budget guard, mirroring batch_status._minimal_next_action:
    # max_iterations is immutable and never reset by resume/reopen, so once
    # iteration reaches it, generation is permanently blocked
    # (bo_engine.convergence.evaluate_stopping_decision) and the normal
    # tree's generate/resume/reopen advice points at calls the server will
    # reject. Suggestions still awaiting results outrank the guard —
    # submission stays valid after exhaustion — as does FAILED's
    # inspect-errors advice.
    budget_exhausted = max_iterations is not None and iteration >= max_iterations
    if budget_exhausted and n_actionable_suggestions == 0 and campaign_status != "failed":
        action, reason, urgency = _budget_exhausted_action(campaign_status, max_iterations)
    else:
        action, reason, urgency = _determine_next_action(
            campaign_status,
            converged,
            convergence,
            outlier_count,
            n_results,
            health_status,
            diagnostics,
            n_actionable_suggestions,
        )

    diagnostics["next_action_recommendation"] = {
        "action": action,
        "reason": reason,
        "urgency": urgency,
    }


def _budget_exhausted_action(
    campaign_status: str,
    max_iterations: int | None,
) -> tuple[str, str, str]:
    """Return action tuple for a campaign whose iteration budget is spent."""
    # COMPLETED is already terminate's target state — the lifecycle layer
    # would accept the call only as an idempotent no-op — so there the hint
    # reviews the finished campaign instead of a routable action.
    if campaign_status == "completed":
        action = "review_campaign_status"
        advice = "review the final results — the campaign is finished"
    else:
        action = "terminate_campaign"
        advice = "review results and terminate it"
    return (
        action,
        (
            f"Campaign is {campaign_status} and has reached "
            f"max_iterations={max_iterations}; the budget cannot be "
            f"extended — {advice}."
        ),
        "low",
    )


def _health_status_action(
    health_status: str,
    diagnostics: dict[str, Any],
) -> tuple[str, str, str]:
    """Return action tuple for critical/warning health status."""
    if health_status == "critical":
        warnings = diagnostics.get("warnings", [])
        return (
            "investigate_issues",
            f"Campaign health is critical. Issues: {warnings[:2] if warnings else 'Unknown'}",
            "high",
        )
    return (
        "monitor_progress",
        "Campaign health has warnings. Continue but monitor closely.",
        "normal",
    )


def _determine_next_action(
    campaign_status: str,
    converged: bool,
    convergence: dict[str, Any],
    outlier_count: int,
    n_results: int,
    health_status: str,
    diagnostics: dict[str, Any],
    n_actionable_suggestions: int,
) -> tuple[str, str, str]:
    """Determine the next action, reason, and urgency."""
    if campaign_status in ("paused", "completed", "failed"):
        continuation = {
            "paused": "Resume it to continue",
            "completed": "Reopen it to continue optimization",
            "failed": "Inspect errors before retrying",
        }[campaign_status]
        return (
            "review_campaign_status",
            f"Campaign is {campaign_status}. {continuation}.",
            "low",
        )
    if converged:
        reason = convergence.get("reason", "Optimization has converged")
        return "consider_stopping", f"{reason}. Consider terminating the campaign.", "normal"
    if outlier_count > 0 and n_results > 5:
        return (
            "review_outliers",
            f"Detected {outlier_count} potential outlier(s). Verify measurements for errors.",
            "normal",
        )
    if health_status in ("critical", "warning"):
        return _health_status_action(health_status, diagnostics)
    if n_actionable_suggestions > 0:
        return (
            "bo_submit_results",
            f"Campaign has {n_actionable_suggestions} suggestion(s) awaiting results.",
            "normal",
        )
    reason = (
        "No results yet. Generate initial suggestions to start optimization."
        if n_results == 0
        else f"Campaign healthy with {n_results} results. Ready for next batch of suggestions."
    )
    return "bo_generate_suggestions", reason, "normal"
