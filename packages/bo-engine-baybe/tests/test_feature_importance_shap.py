"""SHAP feature-importance extraction against the BayBE 0.15 insights API.

``SHAPInsight`` exposes explanations only through **calling**
``explain()`` / ``explain_target()`` (there is no ``explanation``
attribute — the pre-fix code read one and silently returned ``None`` on
every call). These tests pin the corrected call path and the mean-|SHAP|
aggregation math with a stubbed insights module so the heavy optional
``shap`` dependency is not required, plus one real fitted campaign
(``slow``, shap-gated) as the end-to-end positive assertion.

References: BayBE insights userguide
(https://emdgroup.github.io/baybe/stable/userguide/insights.html) and the
SHAP global-importance convention (mean absolute SHAP value per feature,
Lundberg & Lee, NeurIPS 2017 — the quantity plotted by ``shap.plots.bar``).
"""

from __future__ import annotations

import logging
import sys
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

if TYPE_CHECKING:
    from baybe import Campaign

from bo_engine.types import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)
from bo_engine_baybe.introspection import (
    _aggregate_shap_importance,
    _extract_feature_importance,
    _mean_abs_shap_per_feature,
)

_SHAP_MODULE = "baybe.insights.shap"


def _dummy_campaign() -> Campaign:
    """Type-cast placeholder: the stubbed insight never touches the campaign."""
    return cast("Campaign", object())


def _stub_explanation(values: list[list[float]], names: list[str]) -> SimpleNamespace:
    """Duck-typed stand-in for ``shap.Explanation`` (values + feature_names)."""
    return SimpleNamespace(values=np.asarray(values, dtype=float), feature_names=names)


def _install_stub_insight(
    monkeypatch: pytest.MonkeyPatch,
    explanations: tuple[SimpleNamespace, ...] | None = None,
    from_campaign_error: Exception | None = None,
    explain_error: Exception | None = None,
) -> None:
    """Install a stub ``baybe.insights.shap`` module with a fake SHAPInsight."""

    class _StubInsight:
        @classmethod
        def from_campaign(
            cls,
            campaign: Any,
            explainer_cls: str = "KernelExplainer",
            *,
            use_comp_rep: bool = False,
        ) -> _StubInsight:
            # Mirrors the real SHAPInsight.from_campaign signature so the
            # production call path (explainer/use_comp_rep kwargs) is pinned.
            _ = campaign, explainer_cls, use_comp_rep
            if from_campaign_error is not None:
                raise from_campaign_error
            return cls()

        def explain(self) -> tuple[SimpleNamespace, ...]:
            if explain_error is not None:
                raise explain_error
            assert explanations is not None
            return explanations

    module = ModuleType(_SHAP_MODULE)
    vars(module)["SHAPInsight"] = _StubInsight
    monkeypatch.setitem(sys.modules, _SHAP_MODULE, module)


class TestAggregationMath:
    def test_mean_abs_shap_per_feature(self) -> None:
        """Mean-|SHAP| across rows, keyed by feature name (bar-plot convention)."""
        explanation = _stub_explanation([[1.0, -2.0], [3.0, -4.0]], ["a", "b"])
        assert _mean_abs_shap_per_feature(explanation) == {"a": 2.0, "b": 3.0}

    def test_single_target_aggregation(self) -> None:
        result = _aggregate_shap_importance(
            [_stub_explanation([[1.0, -2.0], [3.0, -4.0]], ["a", "b"])]
        )
        assert result == {"a": 2.0, "b": 3.0}

    def test_multi_target_averages_across_targets(self) -> None:
        """Pareto campaigns average per-target importances (equal target weight)."""
        result = _aggregate_shap_importance(
            [
                _stub_explanation([[2.0, 0.0]], ["a", "b"]),
                _stub_explanation([[0.0, 4.0]], ["a", "b"]),
            ]
        )
        assert result == {"a": 1.0, "b": 2.0}

    def test_empty_explanations_return_none(self) -> None:
        assert _aggregate_shap_importance([]) is None

    def test_disagreeing_feature_names_raise(self) -> None:
        with pytest.raises(ValueError, match="disagree on feature names"):
            _aggregate_shap_importance(
                [
                    _stub_explanation([[1.0]], ["a"]),
                    _stub_explanation([[1.0]], ["b"]),
                ]
            )


class TestExtractionPaths:
    def test_returns_importance_via_explain_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The extraction calls ``explain()`` — the positive-path regression pin."""
        _install_stub_insight(
            monkeypatch,
            explanations=(_stub_explanation([[1.0, -2.0], [3.0, -4.0]], ["x1", "x2"]),),
        )
        campaign = _dummy_campaign()
        assert _extract_feature_importance(campaign) == {"x1": 2.0, "x2": 3.0}

    def test_missing_shap_returns_none_without_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Optional dependency absent → None, and no spurious WARNING is logged."""
        monkeypatch.setitem(sys.modules, _SHAP_MODULE, None)  # forces ImportError
        with caplog.at_level(logging.DEBUG, logger="bo_engine_baybe.introspection"):
            assert _extract_feature_importance(_dummy_campaign()) is None
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_wrong_api_usage_logs_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """AttributeError (our usage drifted from BayBE) surfaces at WARNING."""
        _install_stub_insight(
            monkeypatch,
            explain_error=AttributeError("'SHAPInsight' object has no attribute 'explanation'"),
        )
        with caplog.at_level(logging.DEBUG, logger="bo_engine_baybe.introspection"):
            assert _extract_feature_importance(_dummy_campaign()) is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "API-drift AttributeError must be logged at WARNING"

    def test_phase_unavailability_stays_quiet(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Expected BayBE-side unavailability (no measurements) stays at DEBUG.

        Raises the *real* :class:`baybe.exceptions.NoMeasurementsError` —
        BayBE's phase exceptions subclass ``Exception`` directly, so a
        stand-in ``ValueError`` carrying the same message would pass while
        the production swallow misses the actual class (the pre-fix gap).
        """
        from baybe.exceptions import NoMeasurementsError

        _install_stub_insight(
            monkeypatch,
            from_campaign_error=NoMeasurementsError(
                "The campaign does not contain any measurements."
            ),
        )
        with caplog.at_level(logging.DEBUG, logger="bo_engine_baybe.introspection"):
            assert _extract_feature_importance(_dummy_campaign()) is None
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_model_not_trained_stays_quiet(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The surrogate-not-trained phase exception is swallowed at DEBUG too."""
        from baybe.exceptions import ModelNotTrainedError

        _install_stub_insight(
            monkeypatch,
            explain_error=ModelNotTrainedError("The surrogate has not been trained yet."),
        )
        with caplog.at_level(logging.DEBUG, logger="bo_engine_baybe.introspection"):
            assert _extract_feature_importance(_dummy_campaign()) is None
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_missing_shap_with_configured_insights_logs_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Explicitly configured insights + missing shap is a WARNING, not silence."""
        from bo_engine_baybe.introspection import _extract_feature_importance_report

        monkeypatch.setitem(sys.modules, _SHAP_MODULE, None)  # forces ImportError
        spec = OptimizationSpec(
            parameters=[],
            objectives=[ObjectiveSpec(name="y")],
            backend_options={"baybe": {"insights": {"explainer": "KernelExplainer"}}},
        )
        with caplog.at_level(logging.DEBUG, logger="bo_engine_baybe.introspection"):
            assert _extract_feature_importance_report(_dummy_campaign(), spec) is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "configured insights with missing shap must log a WARNING"


@pytest.mark.slow
class TestRealCampaignImportance:
    def test_fitted_single_objective_campaign_yields_importance(self) -> None:
        """A real fitted campaign returns a non-None, correctly-keyed dict.

        This is the positive assertion the old blanket swallow could never
        fail — it catches any future drift in our SHAPInsight usage. Mirrors
        the basic SHAP insight example of the BayBE insights userguide
        (https://emdgroup.github.io/baybe/stable/userguide/insights.html).
        """
        pytest.importorskip("shap")
        from bo_engine_baybe.introspection import _build_fitted_campaign

        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
            random_seed=42,
        )
        rng = np.random.default_rng(7)
        observations = [
            ObservationData(
                parameter_values={"x1": float(x1), "x2": float(x2)},
                objective_values={"y": float(x1**2 + 0.1 * x2 + rng.normal(0.0, 0.01))},
            )
            for x1, x2 in rng.random((8, 2))
        ]
        campaign = _build_fitted_campaign(spec, observations)
        importance = _extract_feature_importance(campaign)
        assert importance is not None
        assert set(importance) == {"x1", "x2"}
        assert all(v >= 0.0 for v in importance.values())


class TestInsightsOptions:
    def test_explainer_names_exist_in_installed_baybe(self) -> None:
        """Pin-guard: every curated explainer name is a valid BayBE explainer.

        Requires the optional ``shap`` extra (the insights module imports it
        at module scope) — skipped when the extra is not installed, same as
        the chem-availability gating.
        """
        pytest.importorskip("shap")
        from baybe.insights.shap import EXPLAINERS

        from bo_engine_baybe.options import BayBEExplainerKind

        for member in BayBEExplainerKind:
            assert member.value in EXPLAINERS, (
                f"BayBEExplainerKind.{member.name}={member.value!r} is not a "
                "member of baybe.insights.shap.EXPLAINERS"
            )

    def test_per_target_breakdown_for_multi_target(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from bo_engine.types import ObjectiveSpec, OptimizationSpec
        from bo_engine_baybe.introspection import _extract_feature_importance_report

        _install_stub_insight(
            monkeypatch,
            explanations=(
                _stub_explanation([[2.0, 0.0]], ["a", "b"]),
                _stub_explanation([[0.0, 4.0]], ["a", "b"]),
            ),
        )
        spec = OptimizationSpec(
            parameters=[],
            objectives=[ObjectiveSpec(name="t1"), ObjectiveSpec(name="t2")],
        )
        report = _extract_feature_importance_report(_dummy_campaign(), spec)
        assert report is not None
        assert report["feature_importance"] == {"a": 1.0, "b": 2.0}
        assert report["feature_importance_per_target"] == {
            "t1": {"a": 2.0, "b": 0.0},
            "t2": {"a": 0.0, "b": 4.0},
        }

    def test_row_level_rows_are_bounded_and_opt_in(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from bo_engine.types import ObjectiveSpec, OptimizationSpec
        from bo_engine_baybe.constants import DEFAULT_MAX_ROW_LEVEL_SHAP_ROWS
        from bo_engine_baybe.introspection import _extract_feature_importance_report

        n_rows = DEFAULT_MAX_ROW_LEVEL_SHAP_ROWS + 8
        _install_stub_insight(
            monkeypatch,
            explanations=(_stub_explanation([[1.0, 2.0]] * n_rows, ["a", "b"]),),
        )
        spec = OptimizationSpec(
            parameters=[],
            objectives=[ObjectiveSpec(name="t1")],
            backend_options={"baybe": {"insights": {"include_row_level": True}}},
        )
        report = _extract_feature_importance_report(_dummy_campaign(), spec)
        assert report is not None
        rows = report["feature_importance_rows"]
        assert len(rows) == DEFAULT_MAX_ROW_LEVEL_SHAP_ROWS
        assert rows[0] == {"a": 1.0, "b": 2.0}

        # Default (opt-out) payload carries no row-level block.
        spec_default = OptimizationSpec(parameters=[], objectives=[ObjectiveSpec(name="t1")])
        report_default = _extract_feature_importance_report(_dummy_campaign(), spec_default)
        assert report_default is not None
        assert "feature_importance_rows" not in report_default

    def test_row_level_max_rows_option_overrides_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`row_level_max_rows` bounds the payload per campaign."""
        from bo_engine.types import ObjectiveSpec, OptimizationSpec
        from bo_engine_baybe.introspection import _extract_feature_importance_report

        _install_stub_insight(
            monkeypatch,
            explanations=(_stub_explanation([[1.0, 2.0]] * 10, ["a", "b"]),),
        )
        spec = OptimizationSpec(
            parameters=[],
            objectives=[ObjectiveSpec(name="t1")],
            backend_options={
                "baybe": {"insights": {"include_row_level": True, "row_level_max_rows": 3}}
            },
        )
        report = _extract_feature_importance_report(_dummy_campaign(), spec)
        assert report is not None
        assert len(report["feature_importance_rows"]) == 3
