"""BayBE backend implementing the BOBackend protocol.

Maximizes use of BayBE-native functionality:
- Campaign.recommend() for suggestions
- Campaign.posterior_stats() for model predictions
- Campaign.acquisition_values() for per-suggestion acquisition values
- Campaign.get_surrogate().to_botorch() for model hyperparameter extraction
- Campaign.to_json()/from_json() for state persistence (Approach B)
- ParetoObjective for multi-objective optimization
- allow_recommending_already_measured=False to prevent re-suggestions

Delegates to bo_engine only for what BayBE doesn't provide:
hypervolume computation, near-duplicate detection, batch diversity metrics.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import torch
from baybe import Campaign
from baybe.recommenders import (
    BotorchRecommender,
    RandomRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.searchspace import SearchSpaceType
from bo_engine.backend import (
    BatchDiversityMetrics,
    DuplicateInfo,
    Feature,
    SuggestionBatch,
)
from bo_engine.batch_diversity import compute_batch_diversity
from bo_engine.device import get_device, get_dtype
from bo_engine.diagnostics import (
    compute_best_value,
    compute_improvement_history,
    compute_single_objective_improvement_rate,
    summarize_pareto_front,
)
from bo_engine.diagnostics import (
    compute_hypervolume as engine_compute_hypervolume,
)
from bo_engine.diagnostics import (
    compute_pareto_front as engine_compute_pareto_front,
)
from bo_engine.reference_point import get_reference_point
from bo_engine.result_validation import (
    detect_duplicates as engine_detect_duplicates,
)
from bo_engine.result_validation import (
    detect_outliers,
)
from bo_engine.transforms import encode_categorical, get_bounds_tensor
from bo_engine.types import ObservationData, OptimizationSpec
from scipy import stats as scipy_stats

from bo_engine_baybe.converters import (
    dataframe_to_suggestions,
    observations_to_dataframe,
    spec_to_objective,
    spec_to_searchspace,
)

logger = logging.getLogger(__name__)

_SUPPORTED_FEATURES = frozenset(
    {
        Feature.MULTI_OBJECTIVE,
        Feature.CONSTRAINTS,
        Feature.CATEGORICAL,
        Feature.MIXED_SEARCH_SPACE,
    }
)

_MIN_OBSERVATIONS_FOR_CONFIDENCE = 5
_MODEL_TYPE_SINGLE = "BayBE GP"
_MODEL_TYPE_MULTI = "BayBE GP (CompositeSurrogate)"

# Minimum observations before fitting a model for diagnostics
_MIN_DATA_ABSOLUTE = 3
_MIN_DATA_PARAM_MULTIPLIER = 2

# BayBE-specific exceptions that must be caught alongside standard ones.
# IncompatibilityError is raised when the recommender phase (random vs BO)
# doesn't provide the requested API (e.g. posterior_stats during random phase).
_BAYBE_SAFE_EXCEPTIONS: tuple[type[Exception], ...] = (
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
)

try:
    from baybe.exceptions import IncompatibilityError

    _BAYBE_SAFE_EXCEPTIONS = (*_BAYBE_SAFE_EXCEPTIONS, IncompatibilityError)
except ImportError:
    pass  # Older BayBE versions may not expose this


# ---------------------------------------------------------------------------
# Campaign construction & state management
# ---------------------------------------------------------------------------


def _build_campaign(spec: OptimizationSpec) -> Campaign:
    """Create a fresh BayBE Campaign from an OptimizationSpec."""
    searchspace = spec_to_searchspace(spec)
    is_purely_discrete = searchspace.type == SearchSpaceType.DISCRETE

    kwargs: dict[str, object] = {
        "searchspace": searchspace,
        "objective": spec_to_objective(spec),
        "recommender": TwoPhaseMetaRecommender(
            initial_recommender=RandomRecommender(),
            recommender=BotorchRecommender(),
        ),
    }

    if is_purely_discrete:
        kwargs["allow_recommending_already_measured"] = False
        kwargs["allow_recommending_already_recommended"] = False

    return Campaign(**kwargs)  # ty: ignore[invalid-argument-type]


def _restore_or_build_campaign(
    spec: OptimizationSpec,
    backend_state: dict[str, Any] | None,
) -> Campaign:
    """Restore a Campaign from serialized state, or build a fresh one."""
    if backend_state and "campaign_json" in backend_state:
        try:
            return Campaign.from_json(backend_state["campaign_json"])
        except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
            logger.debug("Failed to restore Campaign from state: %s, rebuilding fresh", e)
    return _build_campaign(spec)


def _serialize_campaign(campaign: Campaign) -> dict[str, str]:
    """Serialize a Campaign to a dict for storage as backend_state."""
    return {"campaign_json": campaign.to_json()}


def _count_existing_measurements(campaign: Campaign) -> int:
    """Return the number of measurements already in a restored campaign."""
    try:
        return len(campaign.measurements)
    except (AttributeError, TypeError):
        return 0


def _add_delta_measurements(
    campaign: Campaign,
    observations: list[ObservationData],
    spec: OptimizationSpec,
) -> None:
    """Add only *new* observations not already present in the campaign.

    A restored campaign already contains measurements from prior iterations.
    Adding the full observation list again would create duplicates, corrupting
    the GP and bloating the serialized state.
    """
    n_existing = _count_existing_measurements(campaign)
    if n_existing >= len(observations):
        return  # Campaign already has all observations

    new_observations = observations[n_existing:]
    obs_df = observations_to_dataframe(new_observations, spec)
    campaign.add_measurements(obs_df)


def _observations_to_minimization_tensor(
    spec: OptimizationSpec,
    observations: list[ObservationData],
) -> torch.Tensor:
    """Convert observations to a tensor in BoTorch minimization convention."""
    obj_names = [o.name for o in spec.objectives]
    minimize_mask = torch.tensor([o.minimize for o in spec.objectives], dtype=torch.bool)

    y_list = [
        torch.tensor([obs.objective_values[n] for n in obj_names], dtype=torch.double)
        for obs in observations
    ]
    y_tensor = torch.stack(y_list)
    y_bo = y_tensor.clone()
    y_bo[:, ~minimize_mask] = -y_bo[:, ~minimize_mask]
    return y_bo


def _compute_reference_point(y_bo: torch.Tensor) -> torch.Tensor:
    """Compute a reference point from minimization-convention objective values."""
    worst = y_bo.max(dim=0).values
    ranges = worst - y_bo.min(dim=0).values
    abs_scale = worst.abs().clamp(min=1e-6)
    ranges = torch.where(ranges < 1e-6, abs_scale, ranges)
    return worst + 0.1 * ranges


# ---------------------------------------------------------------------------
# BayBE-native extraction helpers
# ---------------------------------------------------------------------------


def _extract_posterior_stats(
    campaign: Campaign,
    rec_df: pd.DataFrame,
    spec: OptimizationSpec,
) -> list[dict[str, dict[str, float] | None]]:
    """Extract BayBE posterior mean and std per objective for each recommendation."""
    obj_names = [o.name for o in spec.objectives]
    empty: dict[str, dict[str, float] | None] = {
        "predicted_objectives": None,
        "predicted_std": None,
    }
    try:
        stats_df = campaign.posterior_stats(candidates=rec_df, stats=("mean", "std"))
        return [_parse_prediction_row(stats_df.iloc[i], obj_names) for i in range(len(rec_df))]
    except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
        logger.debug("Posterior stats extraction failed: %s", e)
        return [empty] * len(rec_df)


def _parse_prediction_row(
    row: pd.Series,  # type: ignore[type-arg]
    obj_names: list[str],
) -> dict[str, dict[str, float] | None]:
    """Parse a single row of posterior stats into predicted mean/std dicts."""
    pred_obj: dict[str, float] = {}
    pred_std: dict[str, float] = {}
    for name in obj_names:
        mean_col = f"{name}_mean"
        std_col = f"{name}_std"
        if mean_col in row.index:
            pred_obj[name] = float(row[mean_col])
        if std_col in row.index:
            pred_std[name] = float(row[std_col])
    return {
        "predicted_objectives": pred_obj or None,
        "predicted_std": pred_std or None,
    }


def _extract_acquisition_values(
    campaign: Campaign,
    rec_df: pd.DataFrame,
) -> list[float | None]:
    """Extract per-suggestion acquisition function values from BayBE."""
    try:
        acq_series = campaign.acquisition_values(candidates=rec_df)
        return [float(v) for v in acq_series.values]
    except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
        logger.debug("Acquisition value extraction failed: %s", e)
        return [None] * len(rec_df)


def _extract_model_info(campaign: Campaign) -> dict[str, str | list[float] | float | None]:
    """Extract model hyperparameters from BayBE's fitted surrogate."""
    info: dict[str, str | list[float] | float | None] = {
        "model_type": _MODEL_TYPE_SINGLE,
        "kernel_type": "unknown",
    }
    try:
        surrogate = campaign.get_surrogate()
        botorch_model = surrogate.to_botorch()
        covar = botorch_model.covar_module
        kernel = getattr(covar, "base_kernel", covar)
        info["kernel_type"] = type(kernel).__name__

        ls = kernel.lengthscale.detach().squeeze()  # ty: ignore[call-non-callable, unresolved-attribute]
        if ls.numel() == 1:
            info["lengthscales"] = [round(float(ls.item()), 4)]
        else:
            info["lengthscales"] = [round(float(v), 4) for v in ls.tolist()]

        if hasattr(botorch_model, "likelihood") and hasattr(botorch_model.likelihood, "noise"):
            noise = botorch_model.likelihood.noise.item()  # ty: ignore[unresolved-attribute]
            info["noise_variance"] = round(float(noise), 6)
        if hasattr(covar, "outputscale"):
            oscale = covar.outputscale.item()  # ty: ignore[unresolved-attribute]
            info["output_scale"] = round(float(oscale), 4)
    except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
        logger.debug("Model info extraction failed: %s", e)
    return info


def _extract_feature_importance(campaign: Campaign) -> dict[str, float] | None:
    """Extract SHAP-based feature importance from BayBE (optional)."""
    try:
        from baybe.insights.shap import SHAPInsight

        insight = SHAPInsight.from_campaign(campaign)
        values = insight.explanation.values  # ty: ignore[unresolved-attribute]
        feature_names = insight.explanation.feature_names  # ty: ignore[unresolved-attribute]
        if values is not None and feature_names is not None:
            mean_abs = [float(abs(v).mean()) for v in values.T]
            return dict(zip(feature_names, mean_abs, strict=False))
    except (ImportError, *_BAYBE_SAFE_EXCEPTIONS) as e:
        logger.debug("SHAP feature importance extraction failed: %s", e)
    return None


def _build_fitted_campaign(
    spec: OptimizationSpec,
    observations: list[ObservationData],
) -> Campaign:
    """Build a Campaign, add measurements, and trigger model fitting.

    Used by diagnostics methods that need a fitted surrogate. A single
    call replaces the duplicate build-add-recommend pattern that was
    previously in both _compute_model_diagnostics and
    _compute_hyperparameter_diagnostics.
    """
    campaign = _build_campaign(spec)
    obs_df = observations_to_dataframe(observations, spec)
    campaign.add_measurements(obs_df)
    campaign.recommend(batch_size=1)  # triggers model fitting
    return campaign


# ---------------------------------------------------------------------------
# BayBEBackend
# ---------------------------------------------------------------------------


class BayBEBackend:
    """BayBE-based Bayesian Optimization backend.

    Implements the BOBackend protocol, maximizing use of BayBE-native APIs.
    Delegates to bo_engine only for hypervolume, near-duplicate detection,
    and batch diversity (which BayBE does not provide).
    """

    @property
    def name(self) -> str:
        return "baybe"

    @property
    def supported_features(self) -> frozenset[Feature]:
        return _SUPPORTED_FEATURES

    # -- Spec features that BayBE does NOT support --------------------------
    _UNSUPPORTED_CHECKS: list[tuple[str, str]] = [
        ("turbo_config", "TuRBO trust-region optimization"),
        ("saasbo_config", "SAASBO high-dimensional optimization"),
        ("fidelity_parameter", "Multi-fidelity optimization"),
        ("transfer_learning", "Transfer learning (RGPE)"),
    ]

    _UNSUPPORTED_BOOL_CHECKS: list[tuple[str, str]] = [
        ("use_cost_aware", "Cost-aware optimization (EIpu)"),
        ("use_input_warping", "Input warping"),
    ]

    def validate_spec(self, spec: OptimizationSpec) -> list[str]:
        """Return warnings for spec features that BayBE will silently ignore."""
        warnings: list[str] = []
        for attr, label in self._UNSUPPORTED_CHECKS:
            if getattr(spec, attr, None) is not None:
                warnings.append(f"{label} is not supported by BayBE and will be ignored.")
        for attr, label in self._UNSUPPORTED_BOOL_CHECKS:
            if getattr(spec, attr, False):
                warnings.append(f"{label} is not supported by BayBE and will be ignored.")
        if spec.outcome_constraints:
            warnings.append("Outcome constraints are not supported by BayBE and will be ignored.")
        return warnings

    def generate_initial_design(
        self,
        spec: OptimizationSpec,
        n_points: int,
    ) -> list[dict[str, Any]]:
        campaign = _build_campaign(spec)
        rec_df = campaign.recommend(batch_size=n_points)
        return dataframe_to_suggestions(rec_df, spec)

    def generate_suggestions(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
        batch_size: int,
        iteration: int,
        backend_state: dict[str, Any] | None = None,
    ) -> SuggestionBatch:
        campaign = _restore_or_build_campaign(spec, backend_state)

        # Add only delta observations to avoid duplicate accumulation (E1)
        if observations:
            _add_delta_measurements(campaign, observations, spec)

        rec_df = campaign.recommend(batch_size=batch_size)
        param_dicts = dataframe_to_suggestions(rec_df, spec)

        predictions = _extract_posterior_stats(campaign, rec_df, spec)
        acq_values = _extract_acquisition_values(campaign, rec_df)
        model_info = _extract_model_info(campaign)

        suggestions = _build_suggestion_list(
            param_dicts,
            predictions,
            acq_values,
            model_info,
            spec,
            observations,
            iteration,
            batch_size,
        )

        method_info = self.select_methods(spec, len(observations))
        method_info.update(model_info)

        # Surface unsupported-feature warnings via validate_spec (E6)
        spec_warnings = self.validate_spec(spec)

        return SuggestionBatch(
            suggestions=suggestions,
            method_info=method_info,
            backend_state=_serialize_campaign(campaign),
            warnings=spec_warnings,
        )

    def compute_hypervolume(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
    ) -> float | None:
        if spec.n_objectives < 2 or len(observations) < 2:
            return None

        y_bo = _observations_to_minimization_tensor(spec, observations)
        pareto_y, _ = engine_compute_pareto_front(y_bo)
        ref_point = _compute_reference_point(y_bo)
        return engine_compute_hypervolume(pareto_y, ref_point)

    def detect_duplicates(
        self,
        new_params: dict[str, Any],
        existing_params: list[dict[str, Any]],
        tolerance: float,
    ) -> list[DuplicateInfo]:
        raw = engine_detect_duplicates(new_params, existing_params, tolerance)
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
        _ = spec, new_observations
        return backend_state

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
        except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
            logger.debug("Batch diversity computation failed: %s", e)
            return None

    def select_methods(
        self,
        spec: OptimizationSpec,
        n_observations: int,
    ) -> dict[str, Any]:
        is_multi = spec.n_objectives > 1

        if n_observations == 0:
            strategy = "RandomRecommender (space-filling initial design)"
        else:
            strategy = "BotorchRecommender (GP-based)"

        acq_fn = (
            "qLogNoisyExpectedHypervolumeImprovement"
            if is_multi
            else "qLogNoisyExpectedImprovement"
        )

        return {
            "model_type": _MODEL_TYPE_MULTI if is_multi else _MODEL_TYPE_SINGLE,
            "acquisition_function": acq_fn,
            "optimization_strategy": strategy,
            "input_transforms": ["BayBE internal encoding"],
            "explanation": (
                f"BayBE backend with {n_observations} observations. "
                f"Using {strategy}."
                + (f" Multi-objective with {spec.n_objectives} targets." if is_multi else "")
            ),
            "confidence": (
                "medium" if n_observations < _MIN_OBSERVATIONS_FOR_CONFIDENCE else "high"
            ),
            "alternatives": [],
            "warnings": [],
        }

    def compute_diagnostics(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
        sections: frozenset[str] | None = None,
    ) -> dict[str, Any]:
        all_sections = frozenset(["objectives", "model", "outliers", "suggestions_tensor"])
        requested = all_sections if sections is None else sections
        result: dict[str, Any] = {}

        if "objectives" in requested:
            result.update(self._compute_objective_diagnostics(spec, observations))

        # Model and hyperparameter diagnostics share a single fitted campaign (E5)
        need_model = "model" in requested or "suggestions_tensor" in requested
        if need_model:
            result.update(self._compute_model_and_hyperparameter_diagnostics(spec, observations))

        if "outliers" in requested:
            result.update(self._compute_outlier_diagnostics(spec, observations))

        return result

    def _compute_objective_diagnostics(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
    ) -> dict[str, Any]:
        """Compute objective metrics (best value, Pareto front, hypervolume)."""
        if spec.n_objectives == 1:
            return self._single_objective_diagnostics(spec, observations)
        return self._multi_objective_diagnostics(spec, observations)

    def _single_objective_diagnostics(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
    ) -> dict[str, Any]:
        obj = spec.objectives[0]
        values = [obs.objective_values[obj.name] for obs in observations]
        if not values:
            return {
                "best_value": None,
                "best_parameters": None,
                "improvement_history": [],
                "improvement_rate": 0.0,
                "pareto_front": None,
                "hypervolume": None,
                "n_pareto_points": None,
            }
        best_val, best_idx = compute_best_value(values, minimize=obj.minimize)
        imp_hist = compute_improvement_history(values, minimize=obj.minimize)
        return {
            "best_value": best_val,
            "best_parameters": observations[best_idx].parameter_values,
            "improvement_history": imp_hist,
            "improvement_rate": compute_single_objective_improvement_rate(imp_hist),
            "pareto_front": None,
            "hypervolume": None,
            "n_pareto_points": None,
        }

    def _multi_objective_diagnostics(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
    ) -> dict[str, Any]:
        if len(observations) < 2:
            return {
                "best_value": None,
                "best_parameters": None,
                "improvement_history": None,
                "improvement_rate": None,
                "pareto_front": [],
                "hypervolume": 0.0,
                "n_pareto_points": 0,
            }

        y_bo = _observations_to_minimization_tensor(spec, observations)
        minimize_mask = torch.tensor([o.minimize for o in spec.objectives], dtype=torch.bool)
        pareto_y, _ = engine_compute_pareto_front(y_bo)
        pareto_display = pareto_y.clone()
        pareto_display[:, ~minimize_mask] = -pareto_display[:, ~minimize_mask]

        obj_names = [o.name for o in spec.objectives]
        ref_point = get_reference_point(y_bo, minimize_mask)
        hv = engine_compute_hypervolume(pareto_y, ref_point)

        return {
            "best_value": None,
            "best_parameters": None,
            "improvement_history": None,
            "improvement_rate": None,
            "pareto_front": summarize_pareto_front(pareto_display, obj_names),
            "hypervolume": hv,
            "n_pareto_points": len(pareto_y),
        }

    def _compute_model_and_hyperparameter_diagnostics(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
    ) -> dict[str, Any]:
        """Compute model + hyperparameter diagnostics with a single fitted campaign (E5)."""
        empty: dict[str, Any] = {
            "feature_importance": None,
            "loo_cv_metrics": None,
            "model_correlation": None,
            "hyperparameters": None,
        }

        min_data = max(_MIN_DATA_ABSOLUTE, _MIN_DATA_PARAM_MULTIPLIER * len(spec.parameters))
        if len(observations) < min_data:
            return empty

        try:
            campaign = _build_fitted_campaign(spec, observations)
            obs_df = observations_to_dataframe(observations, spec)

            model_info = _extract_model_info(campaign)
            fi = _extract_feature_importance(campaign)
            corr = _baybe_model_correlation(campaign, obs_df, spec)

            hp = {
                "kernel_type": model_info.get("kernel_type"),
                "lengthscales": model_info.get("lengthscales"),
                "noise_variance": model_info.get("noise_variance"),
                "output_scale": model_info.get("output_scale"),
            }

            return {
                "model_correlation": corr,
                "feature_importance": fi,
                "loo_cv_metrics": None,
                "hyperparameters": hp,
            }
        except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
            logger.debug("BayBE model diagnostics failed: %s", e)
            return empty

    def _compute_outlier_diagnostics(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
    ) -> dict[str, Any]:
        """Detect outliers — normalized to BoTorch format (E4)."""
        if len(observations) < 5:
            return {"outliers": None}

        try:
            train_x, train_y, bounds = _prepare_tensors(spec, observations)
            minimize_mask = torch.tensor([o.minimize for o in spec.objectives], dtype=torch.bool)
            train_y_bo = train_y.clone()
            train_y_bo[:, ~minimize_mask] = -train_y_bo[:, ~minimize_mask]
            obj_names = [o.name for o in spec.objectives]

            outliers = detect_outliers(
                train_x=train_x,
                train_y=train_y_bo,
                bounds=bounds,
                objective_names=obj_names,
            )
            # Normalized to match BoTorch backend format (E4)
            outlier_results = (
                [
                    {
                        "result_index": o.index,
                        "standardized_error": round(o.standardized_error, 2),
                        "objective": o.objective_name,
                    }
                    for o in outliers
                ]
                if outliers
                else []
            )

            return {
                "outliers": {
                    "count": len(outlier_results),
                    "outlier_results": outlier_results,
                }
            }
        except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
            logger.debug("Outlier detection failed: %s", e)
            return {"outliers": None}


# ---------------------------------------------------------------------------
# Module-level helpers (no self)
# ---------------------------------------------------------------------------


def _baybe_model_correlation(
    campaign: Campaign,
    obs_df: pd.DataFrame,
    spec: OptimizationSpec,
) -> float:
    """Compute rank correlation between BayBE posterior mean and actuals."""
    try:
        stats_df = campaign.posterior_stats(candidates=obs_df, stats=("mean",))
        obj_name = spec.objectives[0].name
        mean_col = f"{obj_name}_mean"

        # Fall back to first mean column if exact name not found
        if mean_col not in stats_df.columns:
            mean_cols = [c for c in stats_df.columns if "mean" in str(c).lower()]
            if not mean_cols:
                return 0.0
            mean_col = mean_cols[0]

        predicted = stats_df[mean_col].values
        actual = obs_df[obj_name].values
        result = scipy_stats.spearmanr(predicted, actual)
        corr = float(result.statistic)
        return corr if corr == corr else 0.0  # Handle NaN  # noqa: PLR0124
    except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
        logger.debug("BayBE model correlation failed: %s", e)
        return 0.0


def _prepare_tensors(
    spec: OptimizationSpec,
    observations: list[ObservationData],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare train_x, train_y, bounds tensors from observations."""
    obj_names = [o.name for o in spec.objectives]

    x_rows = [encode_categorical(obs.parameter_values, spec) for obs in observations]
    train_x = torch.stack(x_rows)

    y_rows = [[obs.objective_values[n] for n in obj_names] for obs in observations]
    train_y = torch.tensor(y_rows, dtype=get_dtype(), device=get_device())

    bounds = get_bounds_tensor(spec)
    return train_x, train_y, bounds


# ---------------------------------------------------------------------------
# Provenance construction
# ---------------------------------------------------------------------------


def _build_suggestion_list(
    param_dicts: list[dict[str, Any]],
    predictions: list[dict[str, dict[str, float] | None]],
    acq_values: list[float | None],
    model_info: dict[str, str | list[float] | float | None],
    spec: OptimizationSpec,
    observations: list[ObservationData],
    iteration: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Build the suggestion list with full BayBE-sourced provenance."""
    is_multi = spec.n_objectives > 1
    acq_fn = (
        "qLogNoisyExpectedHypervolumeImprovement" if is_multi else "qLogNoisyExpectedImprovement"
    )

    suggestions: list[dict[str, Any]] = []
    for i, params in enumerate(param_dicts):
        pred = predictions[i] if i < len(predictions) else {}
        acq_val = acq_values[i] if i < len(acq_values) else None

        suggestions.append(
            {
                "parameter_values": params,
                "provenance": {
                    "iteration": iteration,
                    "batch_index": i,
                    "generation_method": "baybe_bo",
                    "acquisition_function": acq_fn,
                    "acquisition_value": acq_val,
                    "model_type": model_info.get("model_type", _MODEL_TYPE_SINGLE),
                    "confidence_level": (
                        "medium" if len(observations) < _MIN_OBSERVATIONS_FOR_CONFIDENCE else "high"
                    ),
                    "explanation": (
                        f"Suggestion {i + 1}/{batch_size} generated by BayBE "
                        f"with {len(observations)} prior observations."
                    ),
                    "predicted_objectives": pred.get("predicted_objectives"),
                    "predicted_std": pred.get("predicted_std"),
                },
            }
        )
    return suggestions
