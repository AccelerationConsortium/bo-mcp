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
    LOOCVMetrics,
    compute_best_value,
    compute_hypervolume,
    compute_improvement_history,
    compute_loo_cv_for_model,
    compute_pareto_front,
    compute_rank_correlation,
    compute_single_objective_improvement_rate,
    extract_hyperparameters,
    summarize_pareto_front,
)
from bo_engine.feature_importance import compute_feature_importance
from bo_engine.method_selector import select_methods
from bo_engine.models import create_and_fit_model, create_and_fit_single_task_model
from bo_engine.reference_point import get_reference_point
from bo_engine.result_validation import detect_duplicates, detect_outliers
from bo_engine.suggestions import (
    generate_initial_design,
    generate_next_batch,
    update_turbo_after_evaluation,
)
from bo_engine.transforms import encode_categorical, get_bounds_tensor
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

    def validate_spec(self, spec: OptimizationSpec) -> list[str]:
        return []  # BoTorch supports all features

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

        batch_warnings: list[str] = []
        if spec.use_cost_aware and not any(o.cost is not None for o in observations):
            batch_warnings.append(
                "Cost-aware optimization was requested but no observations have "
                "cost data in metadata. Falling back to standard optimization. "
                "Add 'cost' to result metadata to enable cost weighting."
            )

        return SuggestionBatch(
            suggestions=suggestions,
            method_info=method_info,
            backend_state=new_state,
            warnings=batch_warnings,
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
        except (RuntimeError, ValueError, TypeError) as e:
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

    # ----- Diagnostics -----

    def compute_diagnostics(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
        sections: frozenset[str] | None = None,
    ) -> dict[str, Any]:
        """Compute model-based diagnostics using BoTorch/GPyTorch."""
        all_sections = frozenset(["objectives", "model", "outliers", "suggestions_tensor"])
        requested = all_sections if sections is None else sections
        result: dict[str, Any] = {}
        is_single = spec.n_objectives == 1

        if "objectives" in requested:
            result.update(self._compute_objective_diagnostics(spec, observations))

        if "model" in requested:
            result.update(self._compute_model_diagnostics(spec, observations, is_single))

        if "outliers" in requested:
            result.update(self._compute_outlier_diagnostics(spec, observations))

        if "suggestions_tensor" in requested:
            result.update(self._compute_hyperparameters(spec, observations, is_single))

        return result

    def _compute_objective_diagnostics(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
    ) -> dict[str, Any]:
        """Compute single/multi-objective metrics (best value, Pareto, HV)."""
        if spec.n_objectives == 1:
            return self._compute_single_objective(spec, observations)
        return self._compute_multi_objective(spec, observations)

    def _compute_single_objective(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
    ) -> dict[str, Any]:
        """Compute best value and improvement history for single-objective."""
        obj = spec.objectives[0]
        values = [obs.objective_values[obj.name] for obs in observations]
        result: dict[str, Any] = {
            "pareto_front": None,
            "hypervolume": None,
            "n_pareto_points": None,
        }
        if values:
            best_val, best_idx = compute_best_value(values, minimize=obj.minimize)
            imp_hist = compute_improvement_history(values, minimize=obj.minimize)
            result["best_value"] = best_val
            result["best_parameters"] = observations[best_idx].parameter_values
            result["improvement_history"] = imp_hist
            result["improvement_rate"] = compute_single_objective_improvement_rate(imp_hist)
        else:
            result["best_value"] = None
            result["best_parameters"] = None
            result["improvement_history"] = []
            result["improvement_rate"] = 0.0
        return result

    def _compute_multi_objective(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
    ) -> dict[str, Any]:
        """Compute Pareto front and hypervolume for multi-objective."""
        obj_names = [o.name for o in spec.objectives]
        result: dict[str, Any] = {
            "best_value": None,
            "best_parameters": None,
            "improvement_history": None,
            "improvement_rate": None,
        }
        if len(observations) < 2:
            result["pareto_front"] = []
            result["hypervolume"] = 0.0
            result["n_pareto_points"] = 0
            return result

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
        pareto_display = pareto_y.clone()
        pareto_display[:, ~minimize_mask] = -pareto_display[:, ~minimize_mask]

        ref_point = get_reference_point(y_bo, minimize_mask)
        hv = compute_hypervolume(pareto_y, ref_point)

        result["pareto_front"] = summarize_pareto_front(pareto_display, obj_names)
        result["hypervolume"] = hv
        result["n_pareto_points"] = len(result["pareto_front"])
        return result

    def _compute_model_diagnostics(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
        is_single: bool,
    ) -> dict[str, Any]:
        """Fit GP model and compute correlation, feature importance, LOO-CV."""
        n_params = len(spec.parameters)
        min_data = max(3, 2 * n_params)
        empty: dict[str, Any] = {
            "feature_importance": None,
            "loo_cv_metrics": None,
            "model_correlation": None,
        }

        if len(observations) < min_data:
            return empty

        try:
            train_x, train_y, param_names, obj_names = self._prepare_training_data(
                spec, observations
            )
            bounds = get_bounds_tensor(spec)

            if is_single:
                model = create_and_fit_single_task_model(
                    train_x,
                    train_y,
                    bounds,
                    use_input_warping=spec.use_input_warping,
                )
            else:
                model = create_and_fit_model(
                    train_x,
                    train_y,
                    bounds,
                    use_input_warping=spec.use_input_warping,
                )

            corr = self._model_correlation(model, train_x, train_y, is_single)
            fi = compute_feature_importance(
                model,
                train_x,
                param_names,
                include_shap=False,
            )
            loo = self._loo_cv(model, train_x, train_y, obj_names, len(observations))

            return {
                "model_correlation": corr,
                "feature_importance": fi,
                "loo_cv_metrics": loo,
            }
        except (RuntimeError, ValueError, TypeError) as e:
            logger.debug("Model diagnostics failed: %s", e)
            return empty

    def _model_correlation(
        self,
        model: Any,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        is_single: bool,
    ) -> float:
        """Compute rank correlation between model predictions and actuals."""
        model.eval()
        with torch.no_grad():
            predictions = model.posterior(train_x).mean
        if is_single:
            return compute_rank_correlation(predictions.squeeze(), train_y.squeeze())
        corrs = [
            compute_rank_correlation(predictions[:, i], train_y[:, i])
            for i in range(train_y.shape[1])
        ]
        return sum(corrs) / len(corrs)

    def _loo_cv(
        self,
        model: Any,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        obj_names: list[str],
        n_obs: int,
    ) -> dict[str, dict[str, float]] | None:
        """Compute LOO-CV metrics per objective."""
        if n_obs < 5:
            return None
        try:
            loo = compute_loo_cv_for_model(model, train_x, train_y)
            loo_by_obj: dict[str, dict[str, float]] = {}
            if isinstance(loo, dict):
                for idx, name in enumerate(obj_names):
                    if idx in loo:
                        loo_by_obj[name] = self._loo_to_dict(loo[idx])
            elif obj_names:
                loo_by_obj[obj_names[0]] = self._loo_to_dict(loo)
            return loo_by_obj
        except (RuntimeError, ValueError, TypeError) as e:
            logger.debug("LOO-CV failed: %s", e)
            return None

    def _compute_outlier_diagnostics(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
    ) -> dict[str, Any]:
        """Detect outliers via LOO residuals."""
        if len(observations) < 5:
            return {"outliers": None}

        try:
            train_x, train_y, _, obj_names = self._prepare_training_data(spec, observations)
            minimize_mask = torch.tensor([o.minimize for o in spec.objectives], dtype=torch.bool)
            train_y_bo = train_y.clone()
            train_y_bo[:, ~minimize_mask] = -train_y_bo[:, ~minimize_mask]
            bounds = get_bounds_tensor(spec)

            outliers = detect_outliers(
                train_x=train_x,
                train_y=train_y_bo,
                bounds=bounds,
                objective_names=obj_names,
            )
            if outliers:
                info = [
                    {
                        "result_index": o.index,
                        "standardized_error": round(o.standardized_error, 2),
                        "actual_value": round(o.actual_value, 4),
                        "predicted_value": round(o.predicted_value, 4),
                        "objective": o.objective_name,
                        "parameter_values": observations[o.index].parameter_values,
                    }
                    for o in outliers
                ]
                return {
                    "outliers": {
                        "count": len(outliers),
                        "outlier_results": info,
                    }
                }
            return {"outliers": {"count": 0, "outlier_results": []}}
        except (RuntimeError, ValueError, TypeError) as e:
            logger.debug("Outlier detection failed: %s", e)
            return {"outliers": None}

    def _compute_hyperparameters(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
        is_single: bool,
    ) -> dict[str, Any]:
        """Extract GP hyperparameters and compute suggestion-tensor metrics."""
        n_params = len(spec.parameters)
        min_data = max(3, 2 * n_params)
        if len(observations) < min_data:
            return {"hyperparameters": None}

        try:
            train_x, train_y, param_names, _ = self._prepare_training_data(spec, observations)
            bounds = get_bounds_tensor(spec)
            if is_single:
                model = create_and_fit_single_task_model(
                    train_x,
                    train_y,
                    bounds,
                    use_input_warping=spec.use_input_warping,
                )
            else:
                model = create_and_fit_model(
                    train_x,
                    train_y,
                    bounds,
                    use_input_warping=spec.use_input_warping,
                )
            hp = extract_hyperparameters(model, param_names)
            return {
                "hyperparameters": {
                    "lengthscales": hp.lengthscales,
                    "noise_variance": hp.noise_variance,
                    "output_scale": hp.output_scale,
                    "kernel_type": hp.kernel_type,
                    "model_type": hp.model_type,
                }
            }
        except (RuntimeError, ValueError, TypeError) as e:
            logger.debug("Hyperparameter extraction failed: %s", e)
            return {"hyperparameters": None}

    def _prepare_training_data(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
    ) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
        """Build train_x and train_y tensors from observations."""
        param_names = [p.name for p in spec.parameters]
        obj_names = [o.name for o in spec.objectives]

        x_list = [encode_categorical(obs.parameter_values, spec) for obs in observations]
        train_x = torch.stack(x_list)

        y_list = [
            torch.tensor([obs.objective_values[n] for n in obj_names], dtype=torch.double)
            for obs in observations
        ]
        train_y = torch.stack(y_list)

        minimize_mask = torch.tensor([o.minimize for o in spec.objectives], dtype=torch.bool)
        train_y[:, ~minimize_mask] = -train_y[:, ~minimize_mask]

        return train_x, train_y, param_names, obj_names

    @staticmethod
    def _loo_to_dict(m: LOOCVMetrics) -> dict[str, float]:
        return {"rmse": m.rmse, "mae": m.mae, "r_squared": m.r_squared}
