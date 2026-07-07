"""Tests for BO engine diagnostics."""

import math

import pytest
import torch

from bo_engine.backend_base import ObservationData
from bo_engine.botorch_backend import BoTorchBackend
from bo_engine.constants import (
    MIN_DATA_ABSOLUTE,
    MIN_DATA_PARAM_MULTIPLIER,
)
from bo_engine.diagnostics import (
    compute_convergence_metric,
    compute_hypervolume,
    compute_hypervolume_improvement,
    compute_observed_hypervolume,
    compute_pareto_front,
    compute_rank_correlation,
    determine_health_status,
    determine_progress_status,
    summarize_pareto_front,
)
from bo_engine.diagnostics_single import compute_single_objective_improvement_rate
from bo_engine.types import (
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)


class TestDiagnostics:
    """Tests for BO diagnostics functions."""

    def test_compute_pareto_front_simple(self):
        """compute_pareto_front finds non-dominated points."""
        # 3 points, 2 objectives (minimization)
        y = torch.tensor(
            [
                [1.0, 3.0],  # Pareto optimal
                [2.0, 2.0],  # Pareto optimal
                [3.0, 1.0],  # Pareto optimal
                [2.5, 2.5],  # Dominated by [2, 2]
            ]
        )

        pareto_y, pareto_mask = compute_pareto_front(y, minimize=True)

        # 3 points should be Pareto optimal
        assert pareto_y.shape[0] == 3
        assert pareto_mask.sum().item() == 3

    def test_compute_hypervolume(self):
        """compute_hypervolume computes correct value."""
        # Simple case: single point
        pareto_y = torch.tensor([[1.0, 1.0]])
        ref_point = torch.tensor([2.0, 2.0])

        hv = compute_hypervolume(pareto_y, ref_point)
        # Area = (2-1) * (2-1) = 1
        assert hv == 1.0

    def test_compute_hypervolume_empty(self):
        """compute_hypervolume returns 0 for empty Pareto front."""
        pareto_y = torch.zeros((0, 2))
        ref_point = torch.tensor([1.0, 1.0])

        hv = compute_hypervolume(pareto_y, ref_point)
        assert hv == 0.0

    def test_summarize_pareto_front(self):
        """summarize_pareto_front formats points correctly."""
        pareto_y = torch.tensor(
            [
                [1.0, 2.0],
                [2.0, 1.0],
            ]
        )
        objective_names = ["cost", "time"]

        summary = summarize_pareto_front(pareto_y, objective_names)

        assert len(summary) == 2
        assert summary[0]["cost"] == 1.0
        assert summary[0]["time"] == 2.0
        assert summary[1]["cost"] == 2.0
        assert summary[1]["time"] == 1.0


class TestNumericalSafety:
    """Pin numerical-safety branches against degenerate inputs.

    The diagnostics helpers run on user data that may include zero or
    near-zero baselines (a flat initial trajectory, a degenerate
    hypervolume history, an exactly-zero initial sample). Without the
    NUMERICAL_EPSILON clamps swept into these helpers, any of these
    inputs would produce ``inf`` / ``nan`` instead of a finite result.
    These tests pin the contract.

    Numerical-safety rule: no bare zero comparisons on computed floats;
    clamp to ``NUMERICAL_EPSILON`` before division.
    """

    def test_hypervolume_improvement_zero_baseline_returns_zero(self) -> None:
        """A previous hypervolume of zero must not divide by zero."""
        assert compute_hypervolume_improvement(current_hv=1.0, previous_hv=0.0) == pytest.approx(
            0.0
        )

    def test_convergence_metric_zero_baseline_returns_not_converged(self) -> None:
        """A zero previous average must surface as ``not converged`` (1.0)."""
        history = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert compute_convergence_metric(history, window=5) == pytest.approx(1.0)

    def test_determine_progress_status_zero_baseline_branch(self) -> None:
        """An exactly-zero previous value must not raise ``ZeroDivisionError``."""
        # Two-element history takes the short-circuit branch; the second
        # value is non-zero so the rate calculation must use the
        # ``is_zero`` helper instead of bare ``!= 0``.
        assert determine_progress_status([0.0, 0.5]) in {"improving", "stagnant", "regressing"}

    def test_single_objective_improvement_rate_zero_initial_uses_history_scale(self) -> None:
        """A zero initial best must not divide by zero — nor hide real progress.

        The rate falls back to the trajectory's robust scale as denominator
        (the shared ``_step_denominator``/``_history_scale`` convention from
        ``bo_engine.convergence``), so a genuinely improving run that starts
        at 0.0 reports a finite positive rate instead of the old hardcoded
        0.0 that made zero-crossing objectives read as "no progress".
        """
        history = [0.0, 0.1, 0.2, 0.3, 0.4]
        rate = compute_single_objective_improvement_rate(history, window=10)
        assert math.isfinite(rate)
        assert rate > 0.0


class TestOutlierDiagnosticsUserScale:
    """Outlier reports must surface values on the user's raw scale.

    The LOO-based outlier detector (BoTorch batch-mode CV pattern:
    https://botorch.org/docs/tutorials/batch_mode_cross_validation/) feeds
    a user-facing report, so ``actual_value`` / ``predicted_value`` must
    equal the observed values as submitted — for maximize objectives just
    as for minimize ones. Standardized residuals are sign-invariant, so
    the detector consumes raw targets with no direction negation.
    """

    def test_maximize_objective_outlier_reports_raw_values(self) -> None:
        from bo_engine.backend_base import ObservationData
        from bo_engine.botorch_backend import BoTorchBackend
        from bo_engine.types import (
            ObjectiveSpec,
            OptimizationSpec,
            ParameterSpec,
            ParameterType,
        )

        spec = OptimizationSpec(
            parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
            objectives=[ObjectiveSpec(name="yield", minimize=False)],
        )
        # Smooth linear trend with one grossly corrupted observation.
        corrupted_value = -50.0
        observations = [
            ObservationData(parameter_values={"x": x}, objective_values={"yield": 10.0 * x})
            for x in (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0)
        ]
        observations.append(
            ObservationData(
                parameter_values={"x": 0.5}, objective_values={"yield": corrupted_value}
            )
        )

        result = BoTorchBackend()._compute_outlier_diagnostics(spec, observations)

        outliers = result["outliers"]
        assert outliers is not None
        assert outliers["count"] >= 1, "The corrupted observation must be flagged as an outlier"
        flagged = {o["actual_value"] for o in outliers["outlier_results"]}
        assert corrupted_value in flagged, (
            f"Outlier report must carry the raw observed value "
            f"{corrupted_value} for a maximize objective; got {flagged} "
            "(a sign flip here means the report is on the internal scale)."
        )


class TestHypervolumeReferencePointConsistency:
    """``compute_hypervolume`` and the diagnostics path must agree.

    Both surfaces must derive the hypervolume reference point from the same
    ``get_reference_point`` helper. The standalone ``compute_hypervolume``
    previously inlined ``worst + 0.1 * range``, which diverges from the
    diagnostics path's relative-floor padding whenever an objective's spread
    is small relative to its magnitude — so the same campaign could report
    two different hypervolumes. The near-zero-range objective below sits
    exactly in that regime (range 0.5 around a worst of ~100), where the old
    inline formula and the relative-floor helper produced different
    reference points.
    """

    def test_standalone_matches_diagnostics_hypervolume(self) -> None:
        from bo_engine.backend_base import ObservationData
        from bo_engine.botorch_backend import BoTorchBackend
        from bo_engine.types import (
            ObjectiveSpec,
            OptimizationSpec,
            ParameterSpec,
            ParameterType,
        )

        spec = OptimizationSpec(
            parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
            objectives=[
                ObjectiveSpec(name="a", minimize=True),
                # Large magnitude, small spread → relative-floor regime, where
                # the old inline ``worst + 0.1 * range`` and the diagnostics
                # path's relative-floor padding produced different ref points.
                ObjectiveSpec(name="b", minimize=True),
            ],
        )
        observations = [
            ObservationData(parameter_values={"x": 0.1}, objective_values={"a": 1.0, "b": 100.0}),
            ObservationData(parameter_values={"x": 0.5}, objective_values={"a": 2.0, "b": 99.5}),
            ObservationData(parameter_values={"x": 0.9}, objective_values={"a": 1.5, "b": 99.8}),
        ]

        backend = BoTorchBackend()
        standalone = backend.compute_hypervolume(spec, observations)
        diagnostics = backend.compute_diagnostics(
            spec, observations, sections=frozenset({"objectives"})
        )["hypervolume"]

        assert standalone is not None
        assert standalone > 0.0
        assert standalone == pytest.approx(diagnostics)


def _continuous_spec(n_params: int) -> OptimizationSpec:
    """A single-objective spec with ``n_params`` continuous parameters on [0, 1]."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name=f"x{i}", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))
            for i in range(n_params)
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
    )


def _spread_observations(n_params: int, n: int) -> list[ObservationData]:
    """``n`` observations whose coordinates spread across [0, 1] per dimension."""
    denom = max(n - 1, 1)
    return [
        ObservationData(
            parameter_values={f"x{i}": ((k + i) % n) / denom for i in range(n_params)},
            objective_values={"y": float(k)},
        )
        for k in range(n)
    ]


def _multi_objective_spec(n_params: int, obj_names: tuple[str, ...]) -> OptimizationSpec:
    """A multi-objective spec with ``n_params`` continuous parameters on [0, 1]."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name=f"x{i}", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))
            for i in range(n_params)
        ],
        objectives=[ObjectiveSpec(name=name, minimize=True) for name in obj_names],
    )


def _multi_objective_observations(
    n_params: int, obj_names: tuple[str, ...], n: int
) -> list[ObservationData]:
    """``n`` observations spread across [0, 1] with one value per objective."""
    denom = max(n - 1, 1)
    return [
        ObservationData(
            parameter_values={f"x{i}": ((k + i) % n) / denom for i in range(n_params)},
            objective_values={name: float(k + 2 * j) for j, name in enumerate(obj_names)},
        )
        for k in range(n)
    ]


class TestDiagnosticDataSufficiencyGate:
    """Model/hyperparameter sections gate on the shared data-sufficiency constants.

    The minimum number of observations before a diagnostic GP is fit is
    ``max(MIN_DATA_ABSOLUTE, MIN_DATA_PARAM_MULTIPLIER * n_params)`` — the same
    constants the rest of the engine uses — so a value tuned in ``constants.py``
    reaches the backend rather than a re-hardcoded literal. Parameterizing on
    the constants pins the gate to them.
    """

    @pytest.mark.parametrize("n_params", [1, 2, 3])
    def test_below_minimum_yields_empty_model_sections(self, n_params: int) -> None:
        # ``n_params == 1`` exercises the MIN_DATA_ABSOLUTE floor; 2/3 exercise
        # the MIN_DATA_PARAM_MULTIPLIER branch.
        min_data = max(MIN_DATA_ABSOLUTE, MIN_DATA_PARAM_MULTIPLIER * n_params)
        spec = _continuous_spec(n_params)
        observations = _spread_observations(n_params, min_data - 1)

        result = BoTorchBackend().compute_diagnostics(
            spec, observations, sections=frozenset({"model", "suggestions_tensor"})
        )

        assert result["model_correlation"] is None
        assert result["feature_importance"] is None
        assert result["loo_cv_metrics"] is None
        assert result["hyperparameters"] is None

    @pytest.mark.parametrize("n_params", [2, 3])
    def test_at_minimum_populates_model_sections(self, n_params: int) -> None:
        min_data = max(MIN_DATA_ABSOLUTE, MIN_DATA_PARAM_MULTIPLIER * n_params)
        spec = _continuous_spec(n_params)
        observations = _spread_observations(n_params, min_data)

        result = BoTorchBackend().compute_diagnostics(
            spec, observations, sections=frozenset({"model", "suggestions_tensor"})
        )

        assert result["model_correlation"] is not None
        assert result["feature_importance"] is not None
        assert result["hyperparameters"] is not None


class TestDiagnosticModelFitSharing:
    """``compute_diagnostics`` fits the diagnostic GP at most once per call.

    The ``model`` and ``suggestions_tensor`` sections both consume a GP fit on
    the same data; they share a single fit instead of each refitting (which
    doubled GP-fit cost per diagnostics call).
    """

    def test_model_and_hyperparameter_sections_share_one_fit(self, monkeypatch) -> None:
        import bo_engine.botorch_backend as backend_mod

        real_factory = backend_mod.create_and_fit_single_task_model
        calls = {"n": 0}

        def counting_factory(*args, **kwargs):
            calls["n"] += 1
            return real_factory(*args, **kwargs)

        monkeypatch.setattr(backend_mod, "create_and_fit_single_task_model", counting_factory)

        spec = _continuous_spec(2)
        observations = _spread_observations(2, 6)

        result = BoTorchBackend().compute_diagnostics(
            spec, observations, sections=frozenset({"model", "suggestions_tensor"})
        )

        assert calls["n"] == 1, (
            "model and suggestions_tensor sections must share a single GP fit "
            f"per compute_diagnostics call; got {calls['n']}"
        )
        # Both sections still produced their outputs from the shared fit.
        assert result["model_correlation"] is not None
        assert result["hyperparameters"] is not None

    def test_multi_objective_sections_share_one_fit(self, monkeypatch) -> None:
        # The multi-objective path fits through ``create_and_fit_model`` (a
        # different factory than the single-objective ``SingleTaskGP`` one), so
        # it needs its own guard against reintroduced double-fitting.
        import bo_engine.botorch_backend as backend_mod

        real_factory = backend_mod.create_and_fit_model
        calls = {"n": 0}

        def counting_factory(*args, **kwargs):
            calls["n"] += 1
            return real_factory(*args, **kwargs)

        monkeypatch.setattr(backend_mod, "create_and_fit_model", counting_factory)

        spec = _multi_objective_spec(2, ("a", "b"))
        observations = _multi_objective_observations(2, ("a", "b"), 6)

        result = BoTorchBackend().compute_diagnostics(
            spec, observations, sections=frozenset({"model", "suggestions_tensor"})
        )

        assert calls["n"] == 1, (
            "multi-objective model and suggestions_tensor sections must share a "
            f"single GP fit per compute_diagnostics call; got {calls['n']}"
        )
        # Both sections still produced their outputs from the shared fit.
        assert result["model_correlation"] is not None
        assert result["hyperparameters"] is not None


class TestDiagnosticGatesReadConstants:
    """The LOO-CV and outlier gates move with their ``constants.py`` values.

    Monkeypatching the constant proves the backend reads the shared name rather
    than a re-hardcoded literal — a regression back to ``< 5`` would no longer
    track the constant and would fail these tests.
    """

    def test_loo_cv_gate_reads_constant(self, monkeypatch) -> None:
        import bo_engine.botorch_backend as backend_mod

        backend = BoTorchBackend()
        spec = _continuous_spec(2)
        # Six observations clear both the fit-data gate and the default LOO gate.
        fit = backend._fit_diagnostic_model(spec, _spread_observations(2, 6), True)
        assert fit is not None

        # Default constant (5): LOO-CV is computed for six observations.
        default = backend._compute_model_diagnostics(fit)
        assert default["model_correlation"] is not None
        assert default["loo_cv_metrics"] is not None

        # Raise the threshold above the data size: only the LOO gate must close;
        # correlation/feature importance (which do not read this constant) stay.
        monkeypatch.setattr(backend_mod, "MIN_OBSERVATIONS_FOR_LOO_CV", 100)
        gated = backend._compute_model_diagnostics(fit)
        assert gated["model_correlation"] is not None
        assert gated["loo_cv_metrics"] is None

    def test_outlier_gate_reads_constant(self, monkeypatch) -> None:
        import bo_engine.botorch_backend as backend_mod

        detect_calls = {"n": 0}

        def fake_detect_outliers(**_kwargs):
            detect_calls["n"] += 1
            return []

        monkeypatch.setattr(backend_mod, "detect_outliers", fake_detect_outliers)
        monkeypatch.setattr(backend_mod, "OUTLIER_DETECTION_MIN_OBSERVATIONS", 8)

        backend = BoTorchBackend()
        spec = _continuous_spec(2)

        # Seven observations sit below the patched threshold: the detector must
        # not run and the section reports no outlier result.
        below = backend._compute_outlier_diagnostics(spec, _spread_observations(2, 7))
        assert below == {"outliers": None}
        assert detect_calls["n"] == 0

        # Eight observations reach the patched threshold: the detector runs.
        backend._compute_outlier_diagnostics(spec, _spread_observations(2, 8))
        assert detect_calls["n"] == 1


class TestRankCorrelationUnknownIsNaN:
    """Unmeasurable correlation is NaN ("unknown"), never a fabricated 0.0.

    A numerical hiccup (constant predictions → undefined Spearman) must not
    manufacture a "model not matching results" warning: 0.0 is a *measured*
    "model is uninformative" signal, while NaN means "could not measure".

    References:
        - scipy.stats.spearmanr — returns NaN for constant input by design
          (correlation undefined), the exact case a sentinel 0.0 conflates.
    """

    def test_constant_predictions_yield_nan(self) -> None:
        corr = compute_rank_correlation(
            torch.tensor([1.0, 1.0, 1.0]), torch.tensor([1.0, 2.0, 3.0])
        )
        assert math.isnan(corr)

    def test_too_few_samples_yield_nan(self) -> None:
        corr = compute_rank_correlation(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))
        assert math.isnan(corr)

    def test_nan_correlation_emits_no_warning_or_downgrade(self) -> None:
        status, warnings = determine_health_status(
            n_results=20,
            hypervolume_improvement=0.1,
            model_correlation=float("nan"),
            iterations_without_improvement=0,
        )
        assert status == "healthy"
        assert not any("correlation" in w.lower() for w in warnings)

    def test_measured_zero_correlation_still_downgrades(self) -> None:
        """A genuine 0.0 is the worst-model signal and must keep firing."""
        status, warnings = determine_health_status(
            n_results=20,
            hypervolume_improvement=0.1,
            model_correlation=0.0,
            iterations_without_improvement=0,
        )
        assert status == "critical"
        assert any("correlation" in w.lower() for w in warnings)


class TestPinnedHypervolumeReference:
    """History hypervolume uses a pinned reference point — no phantom progress.

    ``compute_observed_hypervolume`` derives its reference point from the
    first ``MIN_OBSERVATIONS_FOR_HYPERVOLUME`` observations of the
    (append-only) list, so submitting a strictly dominated result that is
    *worse in one objective* cannot expand the reference box and inflate
    the hypervolume of an unchanged Pareto front.

    References:
        - Ishibuchi et al., "How to Specify a Reference Point in
          Hypervolume Calculation for Fair Performance Comparison" (2018) —
          hypervolume comparisons are only meaningful against a fixed
          reference point.
    """

    @staticmethod
    def _spec() -> OptimizationSpec:
        return OptimizationSpec(
            parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
            objectives=[
                ObjectiveSpec(name="a", minimize=True),
                ObjectiveSpec(name="b", minimize=True),
            ],
        )

    def test_dominated_worse_point_does_not_increase_hv(self) -> None:
        spec = self._spec()
        observations = [
            ObservationData(
                parameter_values={"x": 0.1 * i},
                objective_values={"a": 1.0 + 0.1 * i, "b": 2.0 - 0.1 * i},
            )
            for i in range(4)
        ]
        hv_before = compute_observed_hypervolume(spec, observations)

        dominated_worse = ObservationData(
            parameter_values={"x": 0.9},
            objective_values={"a": 50.0, "b": 50.0},
        )
        hv_after = compute_observed_hypervolume(spec, [*observations, dominated_worse])

        assert hv_after is not None
        assert hv_before is not None
        assert hv_after <= hv_before + 1e-12, (
            "A dominated, worse-in-one-objective observation expanded the "
            "reference box and inflated the hypervolume — phantom progress."
        )

    def test_reference_point_is_stable_across_appends(self) -> None:
        """The same prefix pins the same reference: HV is append-consistent."""
        spec = self._spec()
        observations = [
            ObservationData(
                parameter_values={"x": 0.1 * i},
                objective_values={"a": 2.0 - 0.2 * i, "b": 1.0 + 0.1 * i},
            )
            for i in range(3)
        ]
        improving = ObservationData(
            parameter_values={"x": 0.8}, objective_values={"a": 0.5, "b": 0.5}
        )

        hv_before = compute_observed_hypervolume(spec, observations)
        hv_after = compute_observed_hypervolume(spec, [*observations, improving])

        assert hv_after is not None
        assert hv_before is not None
        assert hv_after >= hv_before  # genuine improvement still registers

    def test_backend_diagnostics_objectives_hypervolume_uses_pinned_reference(self) -> None:
        """The live ``objectives.hypervolume`` diagnostics field, not just the
        shared helper, must be pinned.

        ``_compute_multi_objective`` used to reimplement the Pareto+HV
        computation inline with a reference point recomputed from *all*
        current observations, so it could disagree with (and rise faster
        than) ``campaign.hypervolume_history`` — which is fed by the pinned
        ``compute_hypervolume`` delegate. Both must now agree exactly.
        """
        spec = self._spec()
        observations = [
            ObservationData(
                parameter_values={"x": 0.1 * i},
                objective_values={"a": 1.0 + 0.1 * i, "b": 2.0 - 0.1 * i},
            )
            for i in range(4)
        ]
        backend = BoTorchBackend()
        result_before = backend._compute_multi_objective(spec, observations)

        dominated_worse = ObservationData(
            parameter_values={"x": 0.9},
            objective_values={"a": 50.0, "b": 50.0},
        )
        augmented = [*observations, dominated_worse]
        result_after = backend._compute_multi_objective(spec, augmented)

        assert result_after["hypervolume"] <= result_before["hypervolume"] + 1e-12, (
            "The diagnostics 'objectives' section's live hypervolume must not "
            "rise when a dominated, worse-in-one-objective point is appended."
        )
        assert result_before["hypervolume"] == pytest.approx(
            compute_observed_hypervolume(spec, observations)
        )
        assert result_after["hypervolume"] == pytest.approx(
            compute_observed_hypervolume(spec, augmented)
        )
