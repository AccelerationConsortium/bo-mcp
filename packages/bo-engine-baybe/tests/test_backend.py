"""Protocol compliance tests for BayBEBackend.

Verifies that BayBEBackend correctly implements the BOBackend protocol
and produces valid results for all methods, including BayBE-specific
features (acquisition values, posterior stats, state serialization,
multi-objective).

Reference: BOBackend protocol definition in bo_engine/backend.py
"""

import math
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from baybe.searchspace import SearchSpaceType

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
    TargetMode,
)
from bo_engine_baybe.backend import (
    BayBEBackend,
    _named_lengthscales,
    _replace_duplicate_continuous_recommendation,
)


class _SumAcquisition(torch.nn.Module):
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return points.sum(dim=(-1, -2))


class TestContinuousDuplicateRecommendation:
    def test_normal_recommendation_is_unchanged(self, simple_spec: OptimizationSpec) -> None:
        campaign = SimpleNamespace(
            searchspace=SimpleNamespace(type=SearchSpaceType.CONTINUOUS),
        )
        recommendation = pd.DataFrame([{"x1": 0.8, "x2": 0.8}])
        observations = [
            ObservationData(
                parameter_values={"x1": 0.2, "x2": 0.2},
                objective_values={"y": 1.0},
            )
        ]

        actual, warning = _replace_duplicate_continuous_recommendation(
            campaign, simple_spec, recommendation, observations, None
        )

        assert actual is recommendation
        assert warning is None

    def test_duplicate_uses_best_unseen_fallback(
        self,
        simple_spec: OptimizationSpec,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sample_pool = pd.DataFrame(
            [
                {"x1": 0.2, "x2": 0.2},
                {"x1": 0.3, "x2": 0.4},
                {"x1": 0.8, "x2": 0.9},
            ]
        )
        continuous = SimpleNamespace(sample_uniform=lambda _count: sample_pool)
        campaign = SimpleNamespace(
            searchspace=SimpleNamespace(
                type=SearchSpaceType.CONTINUOUS,
                continuous=continuous,
            ),
            clear_cache=lambda: None,
        )
        recommender = SimpleNamespace(n_raw_samples=3, _botorch_acqf=_SumAcquisition())
        monkeypatch.setattr(
            "bo_engine_baybe.backend._active_recommender",
            lambda _campaign: recommender,
        )
        observations = [
            ObservationData(
                parameter_values={"x1": 0.2, "x2": 0.2},
                objective_values={"y": 1.0},
            )
        ]

        actual, warning = _replace_duplicate_continuous_recommendation(
            campaign,
            simple_spec,
            pd.DataFrame([{"x1": 0.2, "x2": 0.2}]),
            observations,
            None,
        )

        assert actual.to_dict(orient="records") == [{"x1": 0.8, "x2": 0.9}]
        assert warning is not None


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
        assert Feature.MULTI_OBJECTIVE in features
        # CONSTRAINTS is conditional (BayBE cannot honor hybrid /
        # categorical-arithmetic / discrete-LINEAR shapes), so it must not
        # be advertised unconditionally — see the conditional_features
        # surface and TestConstraintsAreConditional.
        assert Feature.CONSTRAINTS not in features
        assert Feature.CONSTRAINTS in backend.conditional_features
        assert Feature.MULTI_FIDELITY not in features
        assert Feature.HIGH_DIMENSIONAL not in features
        assert Feature.INPUT_WARPING not in features


class TestRandomSeedReproducibility:
    """``spec.random_seed`` makes BayBE campaigns reproducible.

    The neutral spec documents that a set ``random_seed`` makes
    suggestion generation reproducible across calls
    (:class:`bo_engine.types.OptimizationSpec`). BayBE exposes
    ``baybe.utils.random.set_random_seed`` for exactly this purpose:
    https://emdgroup.github.io/baybe/stable/userguide/utils.html
    The tests use a discrete-only spec so the random-warmup phase is
    exercised without a GP fit (fast path).
    """

    @staticmethod
    def _seeded_spec(seed: int | None) -> OptimizationSpec:
        return OptimizationSpec(
            parameters=[
                ParameterSpec(
                    name="x",
                    type=ParameterType.DISCRETE,
                    values=[float(i) for i in range(30)],
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
            random_seed=seed,
        )

    def test_same_seed_reproduces_suggestions(self) -> None:
        backend = BayBEBackend()
        spec = self._seeded_spec(seed=42)
        first = backend.generate_suggestions(spec=spec, observations=[], batch_size=3, iteration=1)
        second = backend.generate_suggestions(spec=spec, observations=[], batch_size=3, iteration=1)
        params_first = [s["parameter_values"] for s in first.suggestions]
        params_second = [s["parameter_values"] for s in second.suggestions]
        assert params_first == params_second

    def test_same_seed_reproduces_initial_design(self) -> None:
        backend = BayBEBackend()
        spec = self._seeded_spec(seed=7)
        first = backend.generate_initial_design(spec, n_points=4)
        second = backend.generate_initial_design(spec, n_points=4)
        assert first == second

    def test_seed_stamped_into_provenance(self) -> None:
        """Provenance records the derived seed so runs can be replayed/audited."""
        backend = BayBEBackend()
        batch = backend.generate_suggestions(
            spec=self._seeded_spec(seed=42), observations=[], batch_size=2, iteration=1
        )
        seeds = {s["provenance"]["random_seed"] for s in batch.suggestions}
        assert len(seeds) == 1
        assert next(iter(seeds)) is not None

    def test_unseeded_provenance_records_applied_seed(self) -> None:
        """No spec seed → a fresh seed is drawn, applied, and recorded.

        Mirrors the BoTorch backend's ``_resolve_acquisition_seed`` fallback:
        the actually-used seed lands in provenance for auditability. The
        fallback path is deliberately non-reproducible — no public spec
        field accepts a raw applied seed, so the recorded value documents
        the run rather than promising a replay.
        """
        backend = BayBEBackend()
        batch = backend.generate_suggestions(
            spec=self._seeded_spec(seed=None), observations=[], batch_size=1, iteration=1
        )
        recorded = batch.suggestions[0]["provenance"]["random_seed"]
        assert isinstance(recorded, int)

    def test_unseeded_call_leaves_global_python_stream_untouched(self) -> None:
        """The unseeded fallback seed must not consume the global ``random`` stream.

        The fallback draw comes from OS entropy
        (:func:`bo_engine.reproducibility.draw_fallback_seed`), never from
        the process-global Mersenne-Twister stream: a global draw would
        happen *before* ``GLOBAL_RNG_LOCK`` is acquired, so it could
        interleave with — and silently advance — a concurrent seeded
        scope's stream on another worker thread. Deterministic pin: after
        an unseeded call, the global stream must produce exactly the value
        it would have produced without the call.
        """
        import random

        backend = BayBEBackend()
        random.seed(123)
        expected_next = random.random()  # noqa: S311

        random.seed(123)
        backend.generate_suggestions(
            spec=self._seeded_spec(seed=None), observations=[], batch_size=2, iteration=1
        )
        assert random.random() == expected_next  # noqa: S311

    def test_seeded_calls_do_not_perturb_global_rng(self) -> None:
        """Seeded BayBE calls must not leak into process-wide RNG state.

        ``baybe.utils.random.temporary_seed`` snapshots and restores the
        Python/NumPy/Torch RNG states, mirroring the
        ``torch.random.fork_rng`` isolation used by the BoTorch backend
        (``bo_engine.suggestions``). Without it, one seeded campaign
        would silently re-seed every co-resident unseeded campaign,
        diagnostic, or test in the same worker process.
        """
        import random

        import numpy as np
        import torch

        # The legacy global RNG APIs are load-bearing here: temporary_seed
        # snapshots/restores exactly these process-wide streams, so the
        # test must sample them (not a local Generator) to detect leaks.
        def _reseed_globals() -> None:
            random.seed(123)
            np.random.seed(123)  # noqa: NPY002
            torch.manual_seed(123)

        def _sample_globals() -> tuple[float, float, float]:
            return (
                random.random(),  # noqa: S311
                float(np.random.rand()),  # noqa: NPY002
                float(torch.rand(1)),
            )

        _reseed_globals()
        baseline = _sample_globals()

        _reseed_globals()
        backend = BayBEBackend()
        spec = self._seeded_spec(seed=42)
        backend.generate_suggestions(spec=spec, observations=[], batch_size=2, iteration=1)
        backend.generate_initial_design(spec, n_points=2)
        after_calls = _sample_globals()

        assert after_calls == baseline

    def test_concurrent_seeded_calls_stay_reproducible(self) -> None:
        """Overlapping seeded calls must not clobber each other's RNG streams.

        The server offloads suggestion generation to worker threads via
        ``asyncio.to_thread``, so two seeded BayBE campaigns can overlap
        in one process. ``temporary_seed`` snapshots/restores the
        process-global RNG state, which is only safe when seeded scopes
        are serialized (``GLOBAL_RNG_LOCK``). Each thread's result must
        match its sequential baseline, and the ambient RNG state must be
        restored once both finish.
        """
        import random
        import threading
        from concurrent.futures import ThreadPoolExecutor

        import numpy as np
        import torch

        backend = BayBEBackend()
        specs = {seed: self._seeded_spec(seed=seed) for seed in (11, 97)}

        def _params(seed: int) -> list[dict[str, float]]:
            batch = backend.generate_suggestions(
                spec=specs[seed], observations=[], batch_size=3, iteration=1
            )
            return [s["parameter_values"] for s in batch.suggestions]

        baselines = {seed: _params(seed) for seed in specs}

        # Sample the ambient streams that temporary_seed must restore.
        # Legacy global APIs are load-bearing here (see the isolation test).
        random.seed(123)
        np.random.seed(123)  # noqa: NPY002
        torch.manual_seed(123)
        ambient_baseline = (
            random.random(),  # noqa: S311
            float(np.random.rand()),  # noqa: NPY002
            float(torch.rand(1)),
        )

        random.seed(123)
        np.random.seed(123)  # noqa: NPY002
        torch.manual_seed(123)
        barrier = threading.Barrier(len(specs))

        def _run(seed: int) -> list[dict[str, float]]:
            barrier.wait()
            return _params(seed)

        with ThreadPoolExecutor(max_workers=len(specs)) as pool:
            futures = {seed: pool.submit(_run, seed) for seed in specs}
            results = {seed: f.result() for seed, f in futures.items()}

        assert results == baselines
        ambient_after = (
            random.random(),  # noqa: S311
            float(np.random.rand()),  # noqa: NPY002
            float(torch.rand(1)),
        )
        assert ambient_after == ambient_baseline

    def test_concurrent_unseeded_call_does_not_perturb_seeded_call(self) -> None:
        """An overlapping unseeded BayBE call must not consume a seeded stream.

        Unseeded BayBE generation draws from the same process-global RNGs
        a seeded ``temporary_seed`` window temporarily owns, so both
        paths must hold ``GLOBAL_RNG_LOCK`` — serializing only seeded
        scopes would leave a seeded campaign's reproducibility at the
        mercy of co-resident unseeded traffic. The barrier forces the
        overlap the lock must resolve; several rounds widen the race
        window.
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor

        backend = BayBEBackend()
        seeded_spec = self._seeded_spec(seed=11)
        unseeded_spec = self._seeded_spec(seed=None)

        def _seeded_params() -> list[dict[str, float]]:
            batch = backend.generate_suggestions(
                spec=seeded_spec, observations=[], batch_size=3, iteration=1
            )
            return [s["parameter_values"] for s in batch.suggestions]

        baseline = _seeded_params()

        def _run_seeded(barrier: threading.Barrier) -> list[dict[str, float]]:
            barrier.wait()
            return _seeded_params()

        def _run_unseeded(barrier: threading.Barrier) -> None:
            barrier.wait()
            backend.generate_suggestions(
                spec=unseeded_spec, observations=[], batch_size=3, iteration=1
            )

        for _ in range(5):
            barrier = threading.Barrier(2)
            with ThreadPoolExecutor(max_workers=2) as pool:
                seeded_future = pool.submit(_run_seeded, barrier)
                unseeded_future = pool.submit(_run_unseeded, barrier)
                unseeded_future.result()
                assert seeded_future.result() == baseline

    def test_concurrent_botorch_call_does_not_perturb_seeded_call(self) -> None:
        """A concurrent BoTorch generation must not perturb a seeded BayBE call.

        Both backends snapshot/restore the process-global Torch RNG —
        BoTorch via ``torch.random.fork_rng`` in its suggestion pipeline,
        BayBE via ``temporary_seed`` — and an overlapping ``fork_rng``
        exit restores a stale snapshot into the seeded window, rolling
        the BayBE stream back. Both paths therefore hold the shared
        ``bo_engine.reproducibility.GLOBAL_RNG_LOCK``; this pins the
        cross-backend serialization with a real BoTorch suggestion call
        (GP fit + acquisition) overlapping a seeded BayBE call.
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from bo_engine.botorch_backend import BoTorchBackend

        baybe_backend = BayBEBackend()
        seeded_spec = self._seeded_spec(seed=11)

        botorch_backend = BoTorchBackend()
        botorch_spec = OptimizationSpec(
            parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
            random_seed=7,
        )
        botorch_observations = [
            ObservationData(parameter_values={"x": v}, objective_values={"y": (v - 0.3) ** 2})
            for v in (0.1, 0.5, 0.9)
        ]

        def _seeded_baybe_params() -> list[dict[str, float]]:
            batch = baybe_backend.generate_suggestions(
                spec=seeded_spec, observations=[], batch_size=3, iteration=1
            )
            return [s["parameter_values"] for s in batch.suggestions]

        baseline = _seeded_baybe_params()

        def _run_baybe(barrier: threading.Barrier) -> list[dict[str, float]]:
            barrier.wait()
            return _seeded_baybe_params()

        def _run_botorch(barrier: threading.Barrier) -> None:
            barrier.wait()
            botorch_backend.generate_suggestions(
                spec=botorch_spec,
                observations=botorch_observations,
                batch_size=1,
                iteration=1,
            )

        for _ in range(2):
            barrier = threading.Barrier(2)
            with ThreadPoolExecutor(max_workers=2) as pool:
                baybe_future = pool.submit(_run_baybe, barrier)
                botorch_future = pool.submit(_run_botorch, barrier)
                botorch_future.result()
                assert baybe_future.result() == baseline

    def test_different_iterations_derive_different_seeds(self) -> None:
        """Per-iteration derivation: replays are identical, iterations are not.

        Mirrors the BoTorch backend's ``derive_seed(master, context)``
        scheme so the same master seed cannot reuse one RNG stream across
        iterations (which would bias the random warmup toward repeats).
        """
        backend = BayBEBackend()
        spec = self._seeded_spec(seed=42)
        iter_one = backend.generate_suggestions(
            spec=spec, observations=[], batch_size=1, iteration=1
        )
        iter_two = backend.generate_suggestions(
            spec=spec, observations=[], batch_size=1, iteration=2
        )
        seed_one = iter_one.suggestions[0]["provenance"]["random_seed"]
        seed_two = iter_two.suggestions[0]["provenance"]["random_seed"]
        assert seed_one != seed_two


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
        assert prov["generation_method"] in ("initial_design", "bo")
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

    def test_categorical_mismatch_is_not_a_duplicate(self) -> None:
        backend = BayBEBackend()
        dups = backend.detect_duplicates(
            new_params={"x1": 0.5, "cat": "A"},
            existing_params=[{"x1": 0.5, "cat": "B"}],
            tolerance=1.5,
        )
        # A different category is a different experiment: identical numerics
        # with a different categorical value are the normal shape of
        # categorical DOE, never a near-duplicate.
        assert len(dups) == 0


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


class TestHypervolumeCrossBackendParity:
    """The hypervolume contract must be identical across backends.

    ``campaign.hypervolume_history`` drives convergence detection, so the two
    backends must agree on (a) the ``None``/``0.0``/value return contract and
    (b) the dominated hypervolume itself — both now delegate to the single
    ``bo_engine.diagnostics.compute_observed_hypervolume`` helper using one
    shared reference point. Without this both backends could report different
    HV for the same Pareto front (BayBE previously returned ``None`` below two
    observations where BoTorch returned ``0.0``, and used a different
    degenerate-range reference-point fallback).
    """

    @staticmethod
    def _multi_objective_spec() -> OptimizationSpec:
        return OptimizationSpec(
            parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
            objectives=[
                ObjectiveSpec(name="y1", minimize=True),
                ObjectiveSpec(name="y2", minimize=False),
            ],
        )

    @staticmethod
    def _observations() -> list[ObservationData]:
        # A fixed 2-objective trade-off front (mixed directions).
        return [
            ObservationData(parameter_values={"x": 0.1}, objective_values={"y1": 1.0, "y2": 4.0}),
            ObservationData(parameter_values={"x": 0.4}, objective_values={"y1": 2.0, "y2": 3.0}),
            ObservationData(parameter_values={"x": 0.7}, objective_values={"y1": 3.0, "y2": 2.0}),
            ObservationData(parameter_values={"x": 0.9}, objective_values={"y1": 4.0, "y2": 1.0}),
        ]

    def test_backends_agree_on_hypervolume(self) -> None:
        from bo_engine.botorch_backend import BoTorchBackend

        spec = self._multi_objective_spec()
        observations = self._observations()

        baybe_hv = BayBEBackend().compute_hypervolume(spec, observations)
        botorch_hv = BoTorchBackend().compute_hypervolume(spec, observations)

        assert baybe_hv is not None
        assert botorch_hv is not None
        assert baybe_hv == botorch_hv

    def test_hypervolume_matches_hand_computed_value(self) -> None:
        """Independent numeric oracle (not just routing parity).

        Two minimize objectives with ``A=(1,1)`` dominating ``B=(2,2)`` give a
        Pareto front of ``{(1,1)}``. The static reference point is
        ``worst + 0.1·window`` with ``worst=(2,2)`` and ``window=range=(1,1)``
        (the ``|worst|·0.01`` relative floor is below the range), i.e.
        ``(2.1, 2.1)``, so the dominated hypervolume is
        ``(2.1-1)·(2.1-1) = 1.21`` — independent of the backend delegation.
        """
        spec = OptimizationSpec(
            parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
            objectives=[
                ObjectiveSpec(name="y1", minimize=True),
                ObjectiveSpec(name="y2", minimize=True),
            ],
        )
        observations = [
            ObservationData(parameter_values={"x": 0.2}, objective_values={"y1": 1.0, "y2": 1.0}),
            ObservationData(parameter_values={"x": 0.6}, objective_values={"y1": 2.0, "y2": 2.0}),
        ]

        hv = BayBEBackend().compute_hypervolume(spec, observations)
        assert hv == pytest.approx(1.21)

    def test_backends_agree_below_observation_threshold(self) -> None:
        """Fewer than two observations ⇒ both return ``0.0`` (front not formed)."""
        from bo_engine.botorch_backend import BoTorchBackend

        spec = self._multi_objective_spec()
        one_obs = self._observations()[:1]

        assert BayBEBackend().compute_hypervolume(spec, one_obs) == 0.0
        assert BoTorchBackend().compute_hypervolume(spec, one_obs) == 0.0

    def test_backends_agree_for_single_objective(self) -> None:
        """A single-objective spec ⇒ both return ``None`` (HV undefined)."""
        from bo_engine.botorch_backend import BoTorchBackend

        spec = OptimizationSpec(
            parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        observations = [
            ObservationData(parameter_values={"x": 0.2}, objective_values={"y": 1.0}),
            ObservationData(parameter_values={"x": 0.6}, objective_values={"y": 0.5}),
        ]

        assert BayBEBackend().compute_hypervolume(spec, observations) is None
        assert BoTorchBackend().compute_hypervolume(spec, observations) is None


class TestMultiObjectiveDiagnosticsHypervolumePinned:
    """The live ``objectives.hypervolume`` diagnostics field must use the
    same pinned reference point as ``compute_hypervolume`` / history.

    ``_multi_objective_diagnostics`` used to reimplement the Pareto+HV
    computation inline with a reference point recomputed from *all* current
    observations, so it could disagree with (and rise faster than)
    ``campaign.hypervolume_history``. Both now delegate to
    ``compute_observed_hypervolume`` and must agree exactly.
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

    def test_dominated_worse_point_does_not_increase_diagnostics_hv(self) -> None:
        from bo_engine_baybe.backend import BayBEBackend

        spec = self._spec()
        observations = [
            ObservationData(
                parameter_values={"x": 0.1 * i},
                objective_values={"a": 1.0 + 0.1 * i, "b": 2.0 - 0.1 * i},
            )
            for i in range(4)
        ]
        backend = BayBEBackend()
        result_before = backend._multi_objective_diagnostics(spec, observations)

        dominated_worse = ObservationData(
            parameter_values={"x": 0.9},
            objective_values={"a": 50.0, "b": 50.0},
        )
        augmented = [*observations, dominated_worse]
        result_after = backend._multi_objective_diagnostics(spec, augmented)

        assert result_after["hypervolume"] <= result_before["hypervolume"] + 1e-12, (
            "The diagnostics 'objectives' section's live hypervolume must not "
            "rise when a dominated, worse-in-one-objective point is appended."
        )
        assert result_before["hypervolume"] == pytest.approx(
            backend.compute_hypervolume(spec, observations)
        )
        assert result_after["hypervolume"] == pytest.approx(
            backend.compute_hypervolume(spec, augmented)
        )


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
        obs2 = [
            *obs,
            ObservationData(
                parameter_values={"x1": 0.1, "x2": 0.9},
                objective_values={"y": 0.3},
            ),
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

    def test_named_lengthscales_use_parameter_names_for_matching_dimensions(self) -> None:
        assert _named_lengthscales([0.5, 1.25], ["x1", "x2"]) == {
            "x1": 0.5,
            "x2": 1.25,
        }

    def test_named_lengthscales_use_encoded_labels_for_expanded_features(self) -> None:
        assert _named_lengthscales([0.5, 1.25, 2.0], ["solvent"]) == {
            "encoded_dim_0": 0.5,
            "encoded_dim_1": 1.25,
            "encoded_dim_2": 2.0,
        }

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
        hyperparameters = result["hyperparameters"]
        assert hyperparameters is not None
        assert hyperparameters["lengthscales"].keys() == {"x1", "x2"}
        assert all(isinstance(value, float) for value in hyperparameters["lengthscales"].values())
        assert "model_correlation" in result

    def test_single_objective_direction_follows_target_mode(self) -> None:
        """A ``target_mode`` override drives "best", not the stale boolean default.

        The engine dataclass leaves ``minimize=True`` when only
        ``target_mode='maximize'`` is set; diagnostics must report the
        maximum as best — the same direction the optimization itself
        resolves through ``effective_mode``.
        """
        backend = BayBEBackend()
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="y", target_mode=TargetMode.MAXIMIZE)],
        )
        obs = [
            ObservationData(parameter_values={"x1": 0.1}, objective_values={"y": 1.0}),
            ObservationData(parameter_values={"x1": 0.5}, objective_values={"y": 3.0}),
            ObservationData(parameter_values={"x1": 0.9}, objective_values={"y": 2.0}),
        ]
        result = backend._single_objective_diagnostics(spec, obs)
        assert math.isclose(result["best_value"], 3.0, rel_tol=1e-9)
        assert result["best_parameters"] == {"x1": 0.5}

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
