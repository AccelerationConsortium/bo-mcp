"""TODO 1.67 — method/provenance metadata is sourced from the active campaign.

Previously the backend hard-coded labels like ``qLogNoisyExpectedImprovement``
and ``BotorchRecommender`` based on the spec, ignoring whether the active
recommender phase actually used them. The new code reads
``Campaign.recommender`` / ``Campaign.acquisition_function`` and only falls
back to the static labels when no campaign is available (i.e. inside
``select_methods`` for the pre-iteration diagnostics path).

The tests assert:

* During the random/warmup phase the strategy reflects ``RandomRecommender``
  (or the active non-meta recommender) — not the post-switch BO label.
* After enough observations, the strategy includes ``BotorchRecommender``.
* The static ``select_methods`` fallback is explicitly tagged with
  ``"(fallback)"`` so callers can distinguish live metadata from a guess.
* Version metadata for BayBE and bo-engine-baybe is present.
"""

from __future__ import annotations

from bo_engine.types import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)

from bo_engine_baybe.backend import BayBEBackend


def _spec() -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
    )


class TestLiveMethodMetadata:
    def test_zero_observations_random_phase_strategy(self) -> None:
        """No observations → BayBE picks the initial recommender (Random).

        Random-warmup phase has no GP surrogate and no acquisition function,
        so both ``model_type`` and ``acquisition_function`` must report the
        space-filling sentinel rather than the qLogNEI/BayBE-GP labels that
        only apply post-switch.
        """
        backend = BayBEBackend()
        batch = backend.generate_suggestions(
            spec=_spec(),
            observations=[],
            batch_size=1,
            iteration=1,
        )
        assert "Random" in batch.method_info["optimization_strategy"]
        assert batch.method_info["is_nonpredictive"] is True
        assert batch.method_info["acquisition_function"] == "none (space-filling)"
        assert batch.method_info["model_type"] == "none (space-filling)"

    def test_post_switch_bo_phase_strategy(self) -> None:
        """After observations, BayBE has switched to the BO recommender."""
        backend = BayBEBackend()
        observations = [
            ObservationData(parameter_values={"x1": 0.1, "x2": 0.9}, objective_values={"y": 1.0}),
            ObservationData(parameter_values={"x1": 0.5, "x2": 0.5}, objective_values={"y": 0.5}),
            ObservationData(parameter_values={"x1": 0.8, "x2": 0.2}, objective_values={"y": 0.3}),
        ]
        batch = backend.generate_suggestions(
            spec=_spec(),
            observations=observations,
            batch_size=1,
            iteration=1,
        )
        assert "Botorch" in batch.method_info["optimization_strategy"]
        assert batch.method_info["is_nonpredictive"] is False
        # The recommender name and search-space type should be reported.
        assert batch.method_info.get("recommender") is not None
        assert batch.method_info.get("searchspace_type")

    def test_version_metadata_present(self) -> None:
        """Version info for benchmarking provenance is included."""
        backend = BayBEBackend()
        batch = backend.generate_suggestions(
            spec=_spec(),
            observations=[],
            batch_size=1,
            iteration=1,
        )
        assert batch.method_info.get("baybe_version")
        assert batch.method_info.get("bo_engine_baybe_version")


class TestFallbackSelectMethods:
    def test_select_methods_tags_labels_as_fallback(self) -> None:
        """``select_methods`` is the no-campaign path → its labels are tagged."""
        backend = BayBEBackend()
        info = backend.select_methods(_spec(), n_observations=0)
        assert "(fallback)" in info["optimization_strategy"]
        assert "(fallback)" in info["acquisition_function"]

    def test_select_methods_multi_objective_label(self) -> None:
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[
                ObjectiveSpec(name="y1", minimize=True),
                ObjectiveSpec(name="y2", minimize=False),
            ],
        )
        info = BayBEBackend().select_methods(spec, n_observations=5)
        assert "qLogNoisyExpectedHypervolumeImprovement" in info["acquisition_function"]
        assert "Multi-objective" in info["explanation"]
