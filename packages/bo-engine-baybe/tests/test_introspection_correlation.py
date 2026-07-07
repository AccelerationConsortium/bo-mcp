"""``model_correlation`` sentinel semantics for BayBE diagnostics.

``None`` means "metric unavailable" and must stay distinguishable from a
measured ``0.0`` ("model fitted but uninformative"): the server health
scoring warns on the latter and skips the former, so a sentinel 0.0 on an
extraction failure would manufacture "model not matching results" warnings
from what is actually a missing metric.

References:
    - scipy.stats.spearmanr — undefined (NaN) for constant inputs; the
      NaN case is likewise "not measurable", not zero correlation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pandas as pd

from bo_engine.types import ObjectiveSpec, OptimizationSpec, ParameterSpec, ParameterType
from bo_engine_baybe.introspection import _baybe_model_correlation

if TYPE_CHECKING:
    from baybe import Campaign


def _spec() -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
    )


def _obs_df() -> pd.DataFrame:
    return pd.DataFrame({"x": [0.1, 0.5, 0.9], "y": [1.0, 2.0, 3.0]})


class _RaisingCampaign:
    """Stub whose posterior extraction fails like a BayBE-side error."""

    def posterior_stats(self, candidates: pd.DataFrame, stats: tuple) -> pd.DataFrame:
        _ = candidates, stats
        message = "posterior unavailable"
        raise RuntimeError(message)


class _ConstantMeanCampaign:
    """Stub returning constant posterior means (Spearman undefined)."""

    def posterior_stats(self, candidates: pd.DataFrame, stats: tuple) -> pd.DataFrame:
        _ = stats
        return pd.DataFrame({"y_mean": [1.0] * len(candidates)})


class TestModelCorrelationUnavailableIsNone:
    def test_extraction_failure_returns_none(self) -> None:
        corr = _baybe_model_correlation(cast("Campaign", _RaisingCampaign()), _obs_df(), _spec())
        assert corr is None, (
            "An extraction failure must read 'metric unavailable' (None), not "
            "a fabricated 0.0 that looks like a fitted-but-useless model."
        )

    def test_undefined_spearman_returns_none(self) -> None:
        corr = _baybe_model_correlation(
            cast("Campaign", _ConstantMeanCampaign()), _obs_df(), _spec()
        )
        assert corr is None
