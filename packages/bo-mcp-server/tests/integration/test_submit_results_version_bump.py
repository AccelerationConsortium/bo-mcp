"""Single-objective ``submit_results`` must always bump ``campaign.version``.

The generate-vs-submit TOCTOU window is closed by making every successful
submit advance the campaign row's version. A concurrent
``generate_suggestions`` phase 3 guards its write with
``CampaignRepository.save(expected_version=...)`` (an atomic
``UPDATE … WHERE version = expected``). Multi-objective submits already
advanced the version through the hypervolume bump, but the single-objective
path left ``campaign.version`` untouched when there was no hypervolume or
backend state to persist — so a batch computed against ``N`` observations
could commit *after* observation ``N+1`` landed, because nothing the
generate path locks on had changed. Forcing the bump puts both writers on
the same version-checked row, so the stale generate is rejected (and
retried) instead of committing.

The OCC ``UPDATE … WHERE version = expected`` mechanism itself is exercised
by ``tests/unit/test_diagnostics_cache_rollback_safety.py`` and
``tests/unit/test_generate_suggestions_three_phase.py``; this module pins
the load-bearing behavioural change those guards now rely on — that a
single-objective submit advances the version at all.

Reference: read/write skew under READ COMMITTED isolation (Berenson et al.,
1995, "A Critique of ANSI SQL Isolation Levels"). Routing the contending
writers through one version-checked row is the standard remediation.
"""

from __future__ import annotations

from itertools import pairwise
from uuid import UUID

import pytest

from bo_mcp_server.domain import ResultSubmissionInput
from bo_mcp_server.operations.submit_results import submit_results_operation
from bo_mcp_server.storage import CampaignRepository, get_session
from bo_mcp_server.tools.create_campaign import create_campaign
from tests.factories import seed_owner


async def _campaign_version(campaign_id: str) -> int:
    """Read the committed version of a campaign by id."""
    async with get_session() as session:
        campaign = await CampaignRepository(session).get(UUID(campaign_id))
    assert campaign is not None
    return campaign.version


async def _create_single_objective_campaign() -> tuple[str, str]:
    """Create a single-objective continuous campaign; return (id, owner_id)."""
    owner_id = await seed_owner()
    intake = {
        "name": "Submit version bump",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "batch_size": 1,
    }
    created = await create_campaign(intake, owner_id)
    assert created["success"] is True, created
    return created["campaign_id"], owner_id


@pytest.mark.asyncio
async def test_single_objective_submit_bumps_campaign_version(setup_database) -> None:
    """A single-objective submit (no hypervolume/backend state) advances version.

    This is the exact path the pre-fix code skipped: ``_update_campaign_state``
    only saved when the version had already changed, so a single-objective
    submit left the row at its snapshot version and slipped past the OCC.
    """
    _ = setup_database
    campaign_id, owner_id = await _create_single_objective_campaign()
    before = await _campaign_version(campaign_id)

    response = await submit_results_operation(
        campaign_id=campaign_id,
        results=[ResultSubmissionInput(parameter_values={"x": 0.42}, objective_values={"y": 0.99})],
        submitted_by=owner_id,
        source="api",
    )
    assert response["success"] is True, response

    after = await _campaign_version(campaign_id)
    assert after > before


@pytest.mark.asyncio
async def test_each_submit_advances_version_strictly(setup_database) -> None:
    """Every successful single-objective submit advances the version by at least one.

    Pins the contract the generate-path OCC depends on: a stale snapshot's
    ``expected_version`` can never coincide with a later state after one or
    more intervening submits.
    """
    _ = setup_database
    campaign_id, owner_id = await _create_single_objective_campaign()

    versions = [await _campaign_version(campaign_id)]
    for i in range(3):
        response = await submit_results_operation(
            campaign_id=campaign_id,
            results=[
                ResultSubmissionInput(
                    parameter_values={"x": 0.1 * (i + 1)},
                    objective_values={"y": float(i)},
                )
            ],
            submitted_by=owner_id,
            source="api",
        )
        assert response["success"] is True, response
        versions.append(await _campaign_version(campaign_id))

    assert all(b > a for a, b in pairwise(versions))


@pytest.mark.asyncio
async def test_dry_run_submit_does_not_bump_version(setup_database) -> None:
    """A ``dry_run`` submit validates but persists nothing — version is unchanged.

    Guards against the bump leaking into the preview path: ``dry_run`` returns
    before any campaign-state write, so the OCC contention only kicks in for
    submits that actually commit observations.
    """
    _ = setup_database
    campaign_id, owner_id = await _create_single_objective_campaign()
    before = await _campaign_version(campaign_id)

    response = await submit_results_operation(
        campaign_id=campaign_id,
        results=[ResultSubmissionInput(parameter_values={"x": 0.3}, objective_values={"y": 0.7})],
        submitted_by=owner_id,
        source="api",
        dry_run=True,
    )
    assert response["success"] is True, response
    assert response.get("dry_run") is True

    assert await _campaign_version(campaign_id) == before
