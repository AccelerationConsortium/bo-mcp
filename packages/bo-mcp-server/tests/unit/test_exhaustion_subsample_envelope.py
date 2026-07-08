"""Subsample-aware SEARCH_SPACE_EXHAUSTED (E011) envelope.

On the BayBE large-categorical safeguard path the exhausted candidate set
is a bounded subsample, not the full space — recommending termination
("no unseen combinations left") would close a viable campaign whose real
space holds orders of magnitude more combinations. The envelope must then
recommend raising ``backend_options['baybe'].max_candidates`` instead,
while the non-subsampled envelope keeps its historical termination
guidance.
"""

from __future__ import annotations

import pytest

from bo_engine.initial_design import SearchSpaceExhaustedError
from bo_mcp_server.errors import ErrorCode
from bo_mcp_server.operations.generate_suggestions import _handle_generation_failure

_CAMPAIGN_ID = "00000000-0000-0000-0000-000000000000"


class TestSubsampledExhaustionEnvelope:
    @pytest.mark.asyncio
    async def test_subsampled_exhaustion_recommends_raising_the_budget(self) -> None:
        err = SearchSpaceExhaustedError(
            n_requested=1,
            n_available=0,
            n_total_combinations=10_000,
            subsampled=True,
            n_full_combinations=125_000_000,
            max_candidates=10_000,
        )
        response = await _handle_generation_failure(err, _CAMPAIGN_ID, None)
        envelope = response["error"]
        assert envelope["code"] == ErrorCode.SEARCH_SPACE_EXHAUSTED.value
        details = envelope["details"]
        assert details["searchspace_subsampled"] is True
        assert details["n_full_combinations"] == 125_000_000
        assert details["max_candidates"] == 10_000
        assert details["next_action_recommendation"] == "increase_max_candidates"
        assert "max_candidates" in envelope["recovery_action"]
        assert "terminate" not in envelope["recovery_action"].split("before")[0].lower()

    @pytest.mark.asyncio
    async def test_full_space_exhaustion_keeps_termination_guidance(self) -> None:
        err = SearchSpaceExhaustedError(
            n_requested=1,
            n_available=0,
            n_total_combinations=27,
        )
        response = await _handle_generation_failure(err, _CAMPAIGN_ID, None)
        envelope = response["error"]
        assert envelope["code"] == ErrorCode.SEARCH_SPACE_EXHAUSTED.value
        details = envelope["details"]
        assert details["next_action_recommendation"] == "terminate_campaign"
        assert "searchspace_subsampled" not in details
        assert "bo_terminate_campaign" in envelope["recovery_action"]
