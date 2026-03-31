"""BoTorch backend — wraps existing bo-engine functions behind BOBackend.

This is the default (and currently only) backend. It delegates to the
existing bo-engine modules without duplicating logic.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from bo_engine.backend import (
    BatchDiversityMetrics,
    DuplicateInfo,
    Feature,
    SuggestionBatch,
)
from bo_engine.batch_diversity import compute_batch_diversity
from bo_engine.device import get_device, get_dtype
from bo_engine.diagnostics import (
    compute_hypervolume,
    compute_pareto_front,
)
from bo_engine.method_selector import select_methods
from bo_engine.result_validation import detect_duplicates
from bo_engine.suggestions import (
    generate_initial_design,
    generate_next_batch,
    update_turbo_after_evaluation,
)
from bo_engine.transforms import get_bounds_tensor
from bo_engine.turbo import TurboState, should_use_turbo
from bo_engine.types import ObservationData, OptimizationSpec

logger = logging.getLogger(__name__)


def _dict_to_turbo_state(data: dict[str, Any]) -> TurboState:
    """Deserialize dict to TurboState."""
    return TurboState(
        dim=data["dim"],
        batch_size=data["batch_size"],
        length=data["length"],
        length_min=data["length_min"],
        length_max=data["length_max"],
        failure_counter=data["failure_counter"],
        failure_tolerance=data["failure_tolerance"],
        success_counter=data["success_counter"],
        success_tolerance=data["success_tolerance"],
        best_value=data["best_value"],
        restart_triggered=data["restart_triggered"],
    )


def _turbo_state_to_dict(state: TurboState) -> dict[str, Any]:
    """Serialize TurboState to dict."""
    return {
        "dim": state.dim,
        "batch_size": state.batch_size,
        "length": state.length,
        "length_min": state.length_min,
        "length_max": state.length_max,
        "failure_counter": state.failure_counter,
        "failure_tolerance": state.failure_tolerance,
        "success_counter": state.success_counter,
        "success_tolerance": state.success_tolerance,
        "best_value": state.best_value,
        "restart_triggered": state.restart_triggered,
    }


class BoTorchBackend:
    """BoTorch-based Bayesian Optimization backend.

    Wraps existing bo-engine functions behind the BOBackend protocol.
    """

    @property
    def name(self) -> str:
        return "botorch"

    @property
    def supported_features(self) -> frozenset[Feature]:
        return frozenset(Feature)  # BoTorch supports all features

    # ----- Suggestion Generation -----

    def generate_initial_design(
        self,
        spec: OptimizationSpec,
        n_points: int,
    ) -> list[dict[str, Any]]:
        return generate_initial_design(spec, n_points)

    def generate_suggestions(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
        batch_size: int,
        iteration: int,
        backend_state: dict[str, Any] | None = None,
    ) -> SuggestionBatch:
        turbo_state = None
        use_turbo = spec.use_turbo or should_use_turbo(spec.n_parameters)
        if use_turbo and spec.n_objectives == 1 and backend_state is not None:
            turbo_state = _dict_to_turbo_state(backend_state)

        results, new_turbo = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=batch_size,
            iteration=iteration,
            turbo_state=turbo_state,
        )

        suggestions = [
            {
                "parameter_values": sr.parameter_values,
                "provenance": {
                    "iteration": sr.iteration,
                    "batch_index": sr.batch_index,
                    "generation_method": sr.generation_method,
                    "acquisition_value": sr.acquisition_value,
                    "model_uncertainty": sr.model_uncertainty,
                    "acquisition_function": sr.acquisition_function,
                    "model_type": sr.model_type,
                    "random_seed": sr.random_seed,
                    "model_version": sr.model_version,
                    "confidence_level": sr.confidence_level,
                    "explanation": sr.explanation,
                    "predicted_objectives": sr.predicted_objectives,
                    "predicted_std": sr.predicted_std,
                },
            }
            for sr in results
        ]

        new_state = _turbo_state_to_dict(new_turbo) if new_turbo else None
        method_info = self.select_methods(spec, len(observations))

        return SuggestionBatch(
            suggestions=suggestions,
            method_info=method_info,
            backend_state=new_state,
        )

    # ----- Metrics -----

    def compute_hypervolume(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
    ) -> float | None:
        if spec.n_objectives < 2:
            return None
        if len(observations) < 2:
            return 0.0

        obj_names = [o.name for o in spec.objectives]
        minimize_mask = torch.tensor([o.minimize for o in spec.objectives], dtype=torch.bool)

        y_list = [
            torch.tensor(
                [obs.objective_values[n] for n in obj_names],
                dtype=torch.double,
            )
            for obs in observations
        ]
        y_tensor = torch.stack(y_list)

        y_bo = y_tensor.clone()
        y_bo[:, ~minimize_mask] = -y_bo[:, ~minimize_mask]

        pareto_y, _ = compute_pareto_front(y_bo)

        worst = y_bo.max(dim=0).values
        ranges = y_bo.max(dim=0).values - y_bo.min(dim=0).values
        ranges = torch.where(ranges < 1e-6, torch.ones_like(ranges), ranges)
        ref_point = worst + 0.1 * ranges

        return compute_hypervolume(pareto_y, ref_point)

    def detect_duplicates(
        self,
        new_params: dict[str, Any],
        existing_params: list[dict[str, Any]],
        tolerance: float,
    ) -> list[DuplicateInfo]:
        raw = detect_duplicates(new_params, existing_params, tolerance)
        return [
            DuplicateInfo(
                index=d.index,
                is_exact=d.is_exact,
                parameter_distance=d.parameter_distance,
            )
            for d in raw
        ]

    def update_state_after_results(
        self,
        spec: OptimizationSpec,
        new_observations: list[ObservationData],
        backend_state: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if spec.n_objectives != 1 or backend_state is None:
            return None

        turbo_state = _dict_to_turbo_state(backend_state)
        new_turbo = update_turbo_after_evaluation(
            turbo_state=turbo_state,
            new_observations=new_observations,
            spec=spec,
        )
        return _turbo_state_to_dict(new_turbo)

    def compute_batch_diversity(
        self,
        spec: OptimizationSpec,
        candidates: list[dict[str, Any]],
    ) -> BatchDiversityMetrics | None:
        if len(candidates) < 2:
            return None

        try:
            bounds = get_bounds_tensor(spec)
            param_names = [p.name for p in spec.parameters]

            values = [[float(c.get(name, 0.0)) for name in param_names] for c in candidates]
            tensor = torch.tensor(values, device=get_device(), dtype=get_dtype())

            m = compute_batch_diversity(tensor, bounds)
            return BatchDiversityMetrics(
                min_pairwise_distance=m.min_pairwise_distance,
                mean_pairwise_distance=m.mean_pairwise_distance,
                diversity_score=m.diversity_score,
                is_diverse=m.is_diverse,
            )
        except Exception as e:
            logger.debug("Batch diversity computation failed: %s", e)
            return None

    def select_methods(
        self,
        spec: OptimizationSpec,
        n_observations: int,
    ) -> dict[str, Any]:
        ms = select_methods(spec, n_observations)
        return {
            "model_type": ms.model_type,
            "acquisition_function": ms.acquisition_function,
            "optimization_strategy": ms.optimization_strategy,
            "input_transforms": ms.input_transforms,
            "explanation": ms.explanation,
            "confidence": ms.confidence,
            "alternatives": ms.alternatives,
            "warnings": ms.warnings,
        }
