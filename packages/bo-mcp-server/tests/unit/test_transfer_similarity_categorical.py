"""Transfer-candidate similarity for categorical-only campaigns.

The bounds-overlap component previously scored 0.0 whenever no common
parameter carried numeric bounds, so two *identical* categorical-only
specs capped at 0.8 — exactly at (never above) the "Highly recommended"
threshold — a structural bias against the chemistry campaigns this
service targets. Categorical parameters now score category-set Jaccard as
their search-space overlap, and when no common parameter is scorable at
all the remaining component weights are renormalized.
"""

from __future__ import annotations

import pytest

from bo_mcp_server.domain import CampaignSpec
from bo_mcp_server.domain.campaign_spec import InputParameter, Objective, ParameterType
from bo_mcp_server.operations.transfer_candidates import (
    _compute_bounds_overlap,
    _compute_overall_similarity,
)


def _categorical_spec(name: str, categories: tuple[str, ...]) -> CampaignSpec:
    return CampaignSpec(
        name=name,
        parameters=(
            InputParameter(name="solvent", type=ParameterType.CATEGORICAL, categories=categories),
            InputParameter(
                name="base", type=ParameterType.CATEGORICAL, categories=("k2co3", "dbu")
            ),
        ),
        objectives=(Objective(name="yield", direction="maximize"),),
    )


class TestCategoricalOnlySimilarity:
    def test_identical_categorical_specs_score_at_least_point_nine(self) -> None:
        spec = _categorical_spec("A", ("etoh", "meoh", "h2o"))
        twin = _categorical_spec("B", ("etoh", "meoh", "h2o"))

        overall, breakdown = _compute_overall_similarity(spec, twin, n_results=100, alias_index={})

        assert overall >= 0.9, (
            f"Two identical categorical-only specs scored {overall:.3f} — the "
            "bounds-overlap component must not act as a hidden zero."
        )
        assert breakdown["bounds_overlap"] == pytest.approx(1.0)

    def test_category_overlap_scores_jaccard(self) -> None:
        source = _categorical_spec("A", ("etoh", "meoh", "h2o"))
        target = _categorical_spec("B", ("etoh", "meoh", "dmso"))

        overlap = _compute_bounds_overlap(source, target, alias_index={})

        # solvent: |{etoh, meoh}| / |{etoh, meoh, h2o, dmso}| = 0.5; base: 1.0.
        assert overlap == pytest.approx((0.5 + 1.0) / 2)

    def test_no_common_parameters_renormalizes_instead_of_zeroing(self) -> None:
        source = _categorical_spec("A", ("etoh", "meoh"))
        target = CampaignSpec(
            name="C",
            parameters=(
                InputParameter(
                    name="temperature",
                    type=ParameterType.CONTINUOUS,
                    bounds=(0.0, 100.0),  # ty: ignore[invalid-argument-type]
                ),
            ),
            objectives=(Objective(name="yield", direction="maximize"),),
        )

        overlap = _compute_bounds_overlap(source, target, alias_index={})
        assert overlap is None

        overall, breakdown = _compute_overall_similarity(
            source, target, n_results=100, alias_index={}
        )
        assert breakdown["bounds_overlap"] is None
        # Nothing in common except the objective: parameter similarity 0,
        # objective similarity 1, data richness 1 — renormalized over the
        # three available weights (0.4 + 0.3 + 0.1).
        assert overall == pytest.approx((0.3 + 0.1) / 0.8)
