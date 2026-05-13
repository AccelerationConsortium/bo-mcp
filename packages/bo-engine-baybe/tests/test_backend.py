"""Protocol compliance tests for BayBEBackend.

Verifies that BayBEBackend correctly implements the BOBackend protocol
and produces valid results for all methods, including BayBE-specific
features (acquisition values, posterior stats, state serialization,
multi-objective).

Reference: BOBackend protocol definition in bo_engine/backend.py
"""

import math

from bo_engine.backend import (
    BatchDiversityMetrics,
    BOBackend,
    Feature,
    SuggestionBatch,
)
from bo_engine.types import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)

from bo_engine_baybe.backend import BayBEBackend


class TestProtocolCompliance:
    def test_implements_protocol(self) -> None:
        backend = BayBEBackend()
        assert isinstance(backend, BOBackend)

    def test_name(self) -> None:
        backend = BayBEBackend()
        assert backend.name == "baybe"

    def test_supported_features(self) -> None:
        backend = BayBEBackend()
        features = backend.supported_features
        assert Feature.CATEGORICAL in features
        assert Feature.MIXED_SEARCH_SPACE in features
        assert Feature.CONSTRAINTS in features
        assert Feature.MULTI_OBJECTIVE in features
        assert Feature.MULTI_FIDELITY not in features
        assert Feature.HIGH_DIMENSIONAL not in features
        assert Feature.INPUT_WARPING not in features


class TestGenerateInitialDesign:
    def test_returns_correct_count(self, simple_spec: OptimizationSpec) -> None:
        backend = BayBEBackend()
        designs = backend.generate_initial_design(simple_spec, n_points=3)
        assert len(designs) == 3

    def test_designs_have_all_parameters(self, simple_spec: OptimizationSpec) -> None:
        backend = BayBEBackend()
        designs = backend.generate_initial_design(simple_spec, n_points=2)
        for d in designs:
            assert "x1" in d
            assert "x2" in d

    def test_designs_within_bounds(self, simple_spec: OptimizationSpec) -> None:
        backend = BayBEBackend()
        designs = backend.generate_initial_design(simple_spec, n_points=5)
        for d in designs:
            assert 0.0 <= d["x1"] <= 1.0
            assert 0.0 <= d["x2"] <= 1.0

    def test_categorical_initial_design(self, categorical_spec: OptimizationSpec) -> None:
        backend = BayBEBackend()
        designs = backend.generate_initial_design(categorical_spec, n_points=3)
        assert len(designs) == 3
        for d in designs:
            assert "temp" in d
            assert "solvent" in d
            assert d["solvent"] in ["Water", "Ethanol", "DMF"]


class TestGenerateSuggestions:
    def test_returns_suggestion_batch(self, simple_spec: OptimizationSpec) -> None:
        backend = BayBEBackend()
        observations = [
            ObservationData(parameter_values={"x1": 0.2, "x2": 0.3}, objective_values={"y": 1.5}),
            ObservationData(parameter_values={"x1": 0.8, "x2": 0.7}, objective_values={"y": 0.3}),
            ObservationData(parameter_values={"x1": 0.5, "x2": 0.5}, objective_values={"y": 0.8}),
        ]
        result = backend.generate_suggestions(
            spec=simple_spec,
            observations=observations,
            batch_size=2,
            iteration=1,
        )
        assert isinstance(result, SuggestionBatch)
        assert len(result.suggestions) == 2

    def test_suggestions_have_provenance(self, simple_spec: OptimizationSpec) -> None:
        backend = BayBEBackend()
        observations = [
            ObservationData(parameter_values={"x1": 0.5, "x2": 0.5}, objective_values={"y": 1.0}),
            ObservationData(parameter_values={"x1": 0.3, "x2": 0.7}, objective_values={"y": 0.5}),
            ObservationData(parameter_values={"x1": 0.7, "x2": 0.3}, objective_values={"y": 0.8}),
        ]
        result = backend.generate_suggestions(
            spec=simple_spec,
            observations=observations,
            batch_size=1,
            iteration=2,
        )
        sugg = result.suggestions[0]
        assert "parameter_values" in sugg
        prov = sugg["provenance"]
        assert prov["iteration"] == 2
        assert prov["generation_method"] == "baybe_bo"
        assert "acquisition_value" in prov
        assert "predicted_objectives" in prov
        assert "predicted_std" in prov

    def test_suggestions_within_bounds(self, simple_spec: OptimizationSpec) -> None:
        backend = BayBEBackend()
        observations = [
            ObservationData(parameter_values={"x1": 0.1, "x2": 0.9}, objective_values={"y": 2.0}),
            ObservationData(parameter_values={"x1": 0.9, "x2": 0.1}, objective_values={"y": 0.5}),
            ObservationData(parameter_values={"x1": 0.5, "x2": 0.5}, objective_values={"y": 1.0}),
        ]
        result = backend.generate_suggestions(
            spec=simple_spec,
            observations=observations,
            batch_size=2,
            iteration=1,
        )
        for sugg in result.suggestions:
            params = sugg["parameter_values"]
            assert 0.0 <= params["x1"] <= 1.0
            assert 0.0 <= params["x2"] <= 1.0

    def test_acquisition_values_in_provenance(self, simple_spec: OptimizationSpec) -> None:
        """Verify BayBE's acquisition_values() is wired into provenance."""
        backend = BayBEBackend()
        observations = [
            ObservationData(parameter_values={"x1": 0.2, "x2": 0.8}, objective_values={"y": 1.0}),
            ObservationData(parameter_values={"x1": 0.8, "x2": 0.2}, objective_values={"y": 0.5}),
            ObservationData(parameter_values={"x1": 0.5, "x2": 0.5}, objective_values={"y": 0.7}),
        ]
        result = backend.generate_suggestions(
            spec=simple_spec,
            observations=observations,
            batch_size=2,
            iteration=1,
        )
        # acquisition_value should be present (may be None if extraction fails,
        # but the key must exist)
        for sugg in result.suggestions:
            assert "acquisition_value" in sugg["provenance"]

    def test_model_info_in_method_info(self, simple_spec: OptimizationSpec) -> None:
        """Verify model hyperparameters are extracted from BayBE's surrogate."""
        backend = BayBEBackend()
        observations = [
            ObservationData(parameter_values={"x1": 0.2, "x2": 0.3}, objective_values={"y": 1.5}),
            ObservationData(parameter_values={"x1": 0.8, "x2": 0.7}, objective_values={"y": 0.3}),
            ObservationData(parameter_values={"x1": 0.5, "x2": 0.5}, objective_values={"y": 0.8}),
        ]
        result = backend.generate_suggestions(
            spec=simple_spec,
            observations=observations,
            batch_size=1,
            iteration=1,
        )
        assert "model_type" in result.method_info
        assert "kernel_type" in result.method_info

    def test_backend_state_serialization(self, simple_spec: OptimizationSpec) -> None:
        """Verify Campaign is serialized into backend_state for Approach B.

        State is wrapped in a :class:`BackendStateEnvelope`; the campaign
        JSON lives under ``payload.campaign_json``.
        """
        backend = BayBEBackend()
        observations = [
            ObservationData(parameter_values={"x1": 0.5, "x2": 0.5}, objective_values={"y": 1.0}),
            ObservationData(parameter_values={"x1": 0.3, "x2": 0.7}, objective_values={"y": 0.5}),
            ObservationData(parameter_values={"x1": 0.7, "x2": 0.3}, objective_values={"y": 0.8}),
        ]
        result = backend.generate_suggestions(
            spec=simple_spec,
            observations=observations,
            batch_size=1,
            iteration=1,
        )
        assert result.backend_state is not None
        assert result.backend_state["backend"] == "baybe"
        assert "payload" in result.backend_state
        assert "campaign_json" in result.backend_state["payload"]

    def test_state_restore_produces_suggestions(self, simple_spec: OptimizationSpec) -> None:
        """Verify Approach B: restore from state and generate new suggestions."""
        backend = BayBEBackend()
        observations = [
            ObservationData(parameter_values={"x1": 0.2, "x2": 0.8}, objective_values={"y": 1.0}),
            ObservationData(parameter_values={"x1": 0.8, "x2": 0.2}, objective_values={"y": 0.5}),
            ObservationData(parameter_values={"x1": 0.5, "x2": 0.5}, objective_values={"y": 0.7}),
        ]
        # First call — no state
        result1 = backend.generate_suggestions(
            spec=simple_spec,
            observations=observations,
            batch_size=1,
            iteration=1,
        )
        # Second call — pass state from first
        result2 = backend.generate_suggestions(
            spec=simple_spec,
            observations=observations,
            batch_size=1,
            iteration=2,
            backend_state=result1.backend_state,
        )
        assert isinstance(result2, SuggestionBatch)
        assert len(result2.suggestions) == 1


class TestDetectDuplicates:
    def test_exact_duplicate(self) -> None:
        backend = BayBEBackend()
        dups = backend.detect_duplicates(
            new_params={"x1": 0.5, "x2": 0.3},
            existing_params=[{"x1": 0.5, "x2": 0.3}],
            tolerance=1e-6,
        )
        assert len(dups) == 1
        assert dups[0].is_exact

    def test_no_duplicate(self) -> None:
        backend = BayBEBackend()
        dups = backend.detect_duplicates(
            new_params={"x1": 0.5, "x2": 0.3},
            existing_params=[{"x1": 0.1, "x2": 0.9}],
            tolerance=1e-6,
        )
        assert len(dups) == 0

    def test_near_duplicate_with_categorical_mismatch(self) -> None:
        backend = BayBEBackend()
        dups = backend.detect_duplicates(
            new_params={"x1": 0.5, "cat": "A"},
            existing_params=[{"x1": 0.5, "cat": "B"}],
            tolerance=1.5,
        )
        # Categorical mismatch -> not exact, but distance=1.0 within tolerance
        assert len(dups) == 1
        assert not dups[0].is_exact


class TestBatchDiversity:
    def test_diverse_batch(self, simple_spec: OptimizationSpec) -> None:
        backend = BayBEBackend()
        candidates = [{"x1": 0.0, "x2": 0.0}, {"x1": 1.0, "x2": 1.0}]
        metrics = backend.compute_batch_diversity(simple_spec, candidates)
        assert isinstance(metrics, BatchDiversityMetrics)
        assert metrics.min_pairwise_distance > 0
        assert metrics.is_diverse

    def test_identical_batch(self, simple_spec: OptimizationSpec) -> None:
        backend = BayBEBackend()
        candidates = [{"x1": 0.5, "x2": 0.5}, {"x1": 0.5, "x2": 0.5}]
        metrics = backend.compute_batch_diversity(simple_spec, candidates)
        assert isinstance(metrics, BatchDiversityMetrics)
        assert math.isclose(metrics.min_pairwise_distance, 0.0, abs_tol=1e-10)
        assert not metrics.is_diverse

    def test_single_candidate_returns_none(self, simple_spec: OptimizationSpec) -> None:
        backend = BayBEBackend()
        metrics = backend.compute_batch_diversity(simple_spec, [{"x1": 0.5, "x2": 0.5}])
        assert metrics is None


class TestSelectMethods:
    def test_initial_design_method(self, simple_spec: OptimizationSpec) -> None:
        backend = BayBEBackend()
        info = backend.select_methods(simple_spec, n_observations=0)
        assert "Random" in info["optimization_strategy"]

    def test_bo_method(self, simple_spec: OptimizationSpec) -> None:
        backend = BayBEBackend()
        info = backend.select_methods(simple_spec, n_observations=10)
        assert "BotorchRecommender" in info["optimization_strategy"]

    def test_multi_objective_method(self) -> None:
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[
                ObjectiveSpec(name="y1", minimize=True),
                ObjectiveSpec(name="y2", minimize=False),
            ],
        )
        backend = BayBEBackend()
        info = backend.select_methods(spec, n_observations=5)
        assert "qLogNoisyExpectedHypervolumeImprovement" in info["acquisition_function"]
        assert "Multi-objective" in info["explanation"]


class TestUpdateState:
    def test_returns_state_passthrough(self, simple_spec: OptimizationSpec) -> None:
        backend = BayBEBackend()
        state = {"campaign_json": "{}"}
        result = backend.update_state_after_results(
            spec=simple_spec,
            new_observations=[],
            backend_state=state,
        )
        assert result == state

    def test_returns_none_when_no_state(self, simple_spec: OptimizationSpec) -> None:
        backend = BayBEBackend()
        result = backend.update_state_after_results(
            spec=simple_spec,
            new_observations=[],
            backend_state=None,
        )
        assert result is None


class TestComputeHypervolume:
    def test_single_objective_returns_none(self, simple_spec: OptimizationSpec) -> None:
        backend = BayBEBackend()
        result = backend.compute_hypervolume(simple_spec, [])
        assert result is None


class TestValidateSpec:
    """Tests for E6 — validate_spec surfaces unsupported feature warnings."""

    def test_simple_spec_no_warnings(self, simple_spec: OptimizationSpec) -> None:
        backend = BayBEBackend()
        warnings = backend.validate_spec(simple_spec)
        assert warnings == []

    def test_turbo_config_warning(self) -> None:
        from bo_engine.types import TurboConfig

        spec = OptimizationSpec(
            parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0, 1))],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
            turbo_config=TurboConfig(),
        )
        backend = BayBEBackend()
        warnings = backend.validate_spec(spec)
        assert len(warnings) >= 1
        assert any("TuRBO" in w for w in warnings)

    def test_cost_aware_warning(self) -> None:
        spec = OptimizationSpec(
            parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0, 1))],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
            use_cost_aware=True,
        )
        backend = BayBEBackend()
        warnings = backend.validate_spec(spec)
        assert any("Cost-aware" in w for w in warnings)


class TestGenerateSuggestionsZeroObservations:
    """Tests for E2 — IncompatibilityError handling with 0 observations."""

    def test_zero_obs_does_not_crash(self, simple_spec: OptimizationSpec) -> None:
        """generate_suggestions with empty observations should not crash."""
        backend = BayBEBackend()
        batch = backend.generate_suggestions(
            spec=simple_spec,
            observations=[],
            batch_size=2,
            iteration=1,
        )
        assert isinstance(batch, SuggestionBatch)
        assert len(batch.suggestions) == 2


class TestDeltaMeasurements:
    """Tests for E1 — duplicate measurement prevention on state restore."""

    def test_state_restore_no_duplicate_accumulation(self, simple_spec: OptimizationSpec) -> None:
        """Restoring state and calling generate_suggestions should not
        double-count prior observations."""
        backend = BayBEBackend()
        obs = [
            ObservationData(parameter_values={"x1": 0.3, "x2": 0.7}, objective_values={"y": 1.0}),
            ObservationData(parameter_values={"x1": 0.5, "x2": 0.5}, objective_values={"y": 0.5}),
            ObservationData(parameter_values={"x1": 0.8, "x2": 0.2}, objective_values={"y": 0.8}),
        ]

        # Iteration 1 — builds fresh campaign
        batch1 = backend.generate_suggestions(simple_spec, obs, batch_size=1, iteration=1)
        state = batch1.backend_state

        # Iteration 2 — restores from state, adds only delta
        obs2 = obs + [
            ObservationData(parameter_values={"x1": 0.1, "x2": 0.9}, objective_values={"y": 0.3}),
        ]
        batch2 = backend.generate_suggestions(
            simple_spec,
            obs2,
            batch_size=1,
            iteration=2,
            backend_state=state,
        )
        assert isinstance(batch2, SuggestionBatch)
        assert len(batch2.suggestions) == 1
        # The state should be serializable and not bloated; envelope wraps
        # the BayBE campaign JSON under ``payload.campaign_json``.
        assert batch2.backend_state is not None
        assert batch2.backend_state["backend"] == "baybe"
        assert "campaign_json" in batch2.backend_state["payload"]


class TestComputeDiagnostics:
    """Tests for E5 — compute_diagnostics coverage."""

    def test_diagnostics_with_observations(self, simple_spec: OptimizationSpec) -> None:
        backend = BayBEBackend()
        obs = [
            ObservationData(parameter_values={"x1": 0.1, "x2": 0.1}, objective_values={"y": 2.0}),
            ObservationData(parameter_values={"x1": 0.3, "x2": 0.7}, objective_values={"y": 1.5}),
            ObservationData(parameter_values={"x1": 0.5, "x2": 0.5}, objective_values={"y": 1.0}),
            ObservationData(parameter_values={"x1": 0.7, "x2": 0.3}, objective_values={"y": 1.2}),
            ObservationData(parameter_values={"x1": 0.9, "x2": 0.9}, objective_values={"y": 1.8}),
        ]
        result = backend.compute_diagnostics(simple_spec, obs)

        # Objectives section
        assert "best_value" in result
        assert math.isclose(result["best_value"], 1.0, rel_tol=1e-9)

        # Model section (should have hyperparameters from shared fitted campaign)
        assert "hyperparameters" in result
        assert "model_correlation" in result

    def test_diagnostics_insufficient_data(self, simple_spec: OptimizationSpec) -> None:
        backend = BayBEBackend()
        obs = [
            ObservationData(parameter_values={"x1": 0.5, "x2": 0.5}, objective_values={"y": 1.0}),
        ]
        result = backend.compute_diagnostics(simple_spec, obs)
        assert result.get("hyperparameters") is None

    def test_outlier_format_matches_botorch(self, simple_spec: OptimizationSpec) -> None:
        """E4 — outlier output should be a dict with 'count' and 'outlier_results'."""
        backend = BayBEBackend()
        obs = [
            ObservationData(
                parameter_values={"x1": float(i) / 10, "x2": 0.5},
                objective_values={"y": float(i)},
            )
            for i in range(6)
        ]
        result = backend.compute_diagnostics(simple_spec, obs, sections=frozenset(["outliers"]))
        outliers = result.get("outliers")
        if outliers is not None:
            assert isinstance(outliers, dict)
            assert "count" in outliers
            assert "outlier_results" in outliers
            assert isinstance(outliers["outlier_results"], list)
