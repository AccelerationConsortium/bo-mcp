"""Live ``SuggestionBatch.method_info`` survives the MCP boundary (TODO 8.51).

The audit flagged that ``generate_suggestions_operation`` was throwing
away the live ``method_info`` populated by ``backend.generate_suggestions``
and re-invoking ``backend.select_methods`` after the fact. For BayBE the
recomputed result is the static fallback path that explicitly tagged
every label with ``(fallback)`` — users saw fallback labels in
responses even though the backend had known exactly which strategy /
recommender it used.

This module locks in:

1. A live BoTorch campaign carries the live method labels through the
   response — no ``(fallback)`` suffix anywhere.
2. A live BayBE campaign (≥5 observations, BO recommender active)
   carries the live recommender label (``BotorchRecommender``,
   ``RandomRecommender``, …) rather than the static fallback.
3. The structured ``is_fallback`` field is ``False`` on live paths and
   ``True`` only when the backend hands back an empty ``method_info``.
4. ``acquisition_function_inferred`` distinguishes a guessed acq label
   from a live one even when the broader method_info is live (BayBE
   may know the recommender but not the acquisition attribute).

The original "fallback distinguishability" test
(``test_select_methods_tags_labels_as_fallback`` in
``bo-engine-baybe/tests/test_backend_method_info.py``) was updated
to assert the structured ``is_fallback`` flag rather than the legacy
``"(fallback)"`` suffix; the logic the original test guarded is
unchanged — callers can still tell a guess from live metadata.

References
==========

* BoTorch reproducibility (live acquisition function names):
  https://botorch.org/docs/reproducibility
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from bo_mcp_server.domain import ResultSubmissionInput


def _to_result_inputs(results: list[dict[str, Any]]) -> list[ResultSubmissionInput]:
    return [ResultSubmissionInput.model_validate(r) for r in results]


def _no_fallback_suffix(method_info: dict[str, Any]) -> None:
    """Assert no string field carries the legacy ``(fallback)`` suffix."""
    for key, value in method_info.items():
        if isinstance(value, str):
            assert "(fallback)" not in value, (
                f"method_selection['{key}'] still carries (fallback) suffix: {value!r}"
            )


class TestLiveMethodInfoSurvivesMCPBoundary:
    @pytest.mark.asyncio
    async def test_botorch_live_method_info_is_used(self, setup_database) -> None:
        """BoTorch live method_info reaches the response unchanged.

        BoTorch always populates ``SuggestionBatch.method_info`` via
        ``select_methods``, so the live path and the static path agree —
        but the structured ``is_fallback`` flag must be False because
        the operation took the live branch (the backend returned a
        non-empty method_info).
        """
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())
        intake_data = {
            "name": "BoTorch Live Method Info",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "f", "direction": "minimize"}],
            "backend": "botorch",
        }
        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        # Initial design first, then enough observations to flip to BO.
        await generate_suggestions(campaign_id)
        await submit_results(
            campaign_id,
            _to_result_inputs(
                [
                    {"parameter_values": {"x": 0.1}, "objective_values": {"f": 0.9}},
                    {"parameter_values": {"x": 0.4}, "objective_values": {"f": 0.5}},
                    {"parameter_values": {"x": 0.6}, "objective_values": {"f": 0.3}},
                    {"parameter_values": {"x": 0.85}, "objective_values": {"f": 0.7}},
                    {"parameter_values": {"x": 0.95}, "objective_values": {"f": 0.85}},
                ]
            ),
            owner_id,
        )

        gen = await generate_suggestions(campaign_id)
        assert gen["success"] is True
        method = gen["method_selection"]
        assert method["is_fallback"] is False
        _no_fallback_suffix(method)

    @pytest.mark.asyncio
    async def test_baybe_live_recommender_label_reaches_response(self, setup_database) -> None:
        """BayBE BO-phase suggestions report the live recommender, not the fallback.

        Reproducer for the audit's flagged regression: before the fix,
        ``generate_suggestions_operation`` recomputed ``select_methods``
        which on BayBE explicitly tagged every label with ``(fallback)``.
        Now the live method_info from ``SuggestionBatch.method_info``
        threads through, so the response carries the live
        ``BotorchRecommender (GP-based)`` strategy and ``is_fallback=False``.
        """
        try:
            import bo_engine_baybe  # noqa: F401 — import is the availability probe
        except ImportError:
            pytest.skip("bo-engine-baybe not installed")

        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())
        intake_data = {
            "name": "BayBE Live Recommender",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "f", "direction": "minimize"}],
            "backend": "baybe",
        }
        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        # Initial design first, then ≥5 observations to clear BayBE's
        # ``switch_after`` threshold and activate the BO recommender.
        await generate_suggestions(campaign_id)
        await submit_results(
            campaign_id,
            _to_result_inputs(
                [
                    {"parameter_values": {"x": 0.1}, "objective_values": {"f": 0.9}},
                    {"parameter_values": {"x": 0.3}, "objective_values": {"f": 0.6}},
                    {"parameter_values": {"x": 0.5}, "objective_values": {"f": 0.4}},
                    {"parameter_values": {"x": 0.7}, "objective_values": {"f": 0.5}},
                    {"parameter_values": {"x": 0.9}, "objective_values": {"f": 0.8}},
                ]
            ),
            owner_id,
        )

        gen = await generate_suggestions(campaign_id)
        assert gen["success"] is True, gen
        method = gen["method_selection"]
        # Live BayBE metadata carries the live recommender + structured
        # is_fallback flag set to False.
        assert method["is_fallback"] is False
        # Post-switch BO phase: the recommender must report the
        # specific class BayBE flipped to (``BotorchRecommender``)
        # rather than a generic / random / fallback label. Asserting
        # the exact class name here is the load-bearing check the
        # reviewer asked for — without it, the test silently passed
        # against ``recommender is not None`` even when the run was
        # still in the random warmup phase.
        assert method["recommender"] == "BotorchRecommender", method
        # ``optimization_strategy`` is the free-form label rendered
        # from the live recommender; it must now mention the post-
        # switch surrogate path.
        assert "Botorch" in method["optimization_strategy"]
        assert "GP-based" in method["optimization_strategy"]
        assert method["is_nonpredictive"] is False
        # The acquisition function name comes from the live BayBE
        # recommender attribute (``qLogNoisyExpectedImprovement`` for
        # single-objective). BayBE does not always expose the
        # ``acquisition_function`` attribute on every recommender
        # configuration — when it does not, the label is the static
        # default and ``acquisition_function_inferred`` is True
        # (structured signal). Either way the label must mention the
        # expected-improvement family the BO phase actually uses.
        assert "ExpectedImprovement" in method["acquisition_function"]
        assert isinstance(method["acquisition_function_inferred"], bool)
        # And no legacy ``(fallback)`` suffix anywhere in free-form strings.
        _no_fallback_suffix(method)

    @pytest.mark.asyncio
    async def test_method_info_explanation_survives(self, setup_database) -> None:
        """Live method_info still carries the explanation field for transparency."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())
        intake_data = {
            "name": "Method Explanation Live Path",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "f", "direction": "minimize"}],
            "backend": "botorch",
        }
        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        gen = await generate_suggestions(campaign_id)
        assert gen["success"] is True
        method = gen["method_selection"]
        assert "explanation" in method
        assert method["explanation"]
        assert method["is_fallback"] is False
