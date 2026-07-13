"""Next-action recommendations must direct agents to the reopen path.

A completed campaign is continuable via the ``reopen`` lifecycle action;
recommending "create a new one" steers agents toward rebuilding campaigns
and replaying results as seeds.
"""

from bo_mcp_server.domain import CampaignStatus
from bo_mcp_server.operations.batch_status import _minimal_next_action
from bo_mcp_server.operations.diagnostics.actions import _determine_next_action


def test_minimal_next_action_points_completed_campaigns_at_reopen() -> None:
    hint = _minimal_next_action(
        CampaignStatus.COMPLETED, n_results=5, n_pending=0, iteration=5, max_iterations=None
    )

    assert hint["action"] == "review_campaign_status"
    assert "reopen" in hint["reason"]
    assert "create a new one" not in hint["reason"]


def test_minimal_next_action_points_paused_campaigns_at_resume() -> None:
    hint = _minimal_next_action(
        CampaignStatus.PAUSED, n_results=5, n_pending=0, iteration=5, max_iterations=None
    )

    assert "resume" in hint["reason"]


def test_minimal_next_action_flags_exhausted_iteration_budget_on_running_campaign() -> None:
    """A running campaign that already hit max_iterations must not be told to
    generate more suggestions — the server would reject that call anyway
    (see bo_engine.convergence.evaluate_stopping_decision), and status never
    auto-transitions to COMPLETED on budget exhaustion. The cap is immutable
    and reopen only accepts COMPLETED campaigns, so the hint must point at
    termination, not at a continuation path that does not exist.
    """
    hint = _minimal_next_action(
        CampaignStatus.RUNNING, n_results=20, n_pending=0, iteration=10, max_iterations=10
    )

    assert hint["action"] != "bo_generate_suggestions"
    assert "max_iterations" in hint["reason"]
    assert "terminate" in hint["reason"]
    assert "reopen" not in hint["reason"]


def test_minimal_next_action_ignores_budget_when_pending_results_await_submission() -> None:
    """Pending suggestions are still actionable via bo_submit_results even
    once the iteration budget is exhausted — only new generation is blocked.
    """
    hint = _minimal_next_action(
        CampaignStatus.RUNNING, n_results=20, n_pending=2, iteration=10, max_iterations=10
    )

    assert hint["action"] == "bo_submit_results"


def test_minimal_next_action_generates_suggestions_under_budget() -> None:
    hint = _minimal_next_action(
        CampaignStatus.RUNNING, n_results=5, n_pending=0, iteration=3, max_iterations=10
    )

    assert hint["action"] == "bo_generate_suggestions"


def test_determine_next_action_points_completed_campaigns_at_reopen() -> None:
    action, reason, urgency = _determine_next_action(
        campaign_status="completed",
        converged=False,
        convergence={},
        outlier_count=0,
        n_results=5,
        health_status="healthy",
        diagnostics={},
        n_pending_suggestions=0,
    )

    assert action == "review_campaign_status"
    assert "reopen" in reason.lower()
    assert urgency == "low"
