"""Next-action recommendations must direct agents to the reopen path.

A completed campaign is continuable via the ``reopen`` lifecycle action;
recommending "create a new one" steers agents toward rebuilding campaigns
and replaying results as seeds.
"""

from bo_mcp_server.domain import CampaignStatus
from bo_mcp_server.operations.batch_status import _minimal_next_action
from bo_mcp_server.operations.diagnostics.actions import _determine_next_action


def test_minimal_next_action_points_completed_campaigns_at_reopen() -> None:
    hint = _minimal_next_action(CampaignStatus.COMPLETED, n_results=5, n_pending=0)

    assert hint["action"] == "review_campaign_status"
    assert "reopen" in hint["reason"]
    assert "create a new one" not in hint["reason"]


def test_minimal_next_action_points_paused_campaigns_at_resume() -> None:
    hint = _minimal_next_action(CampaignStatus.PAUSED, n_results=5, n_pending=0)

    assert "resume" in hint["reason"]


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
