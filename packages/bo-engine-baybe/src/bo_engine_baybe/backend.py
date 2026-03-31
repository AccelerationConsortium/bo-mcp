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
    compute_hypervolume as engine_compute_hypervolume,
)
from bo_engine.diagnostics import (
    compute_pareto_front as engine_compute_pareto_front,
)
from bo_engine.result_validation import detect_duplicates as engine_detect_duplicates
from bo_engine.transforms import get_bounds_tensor
from bo_engine.types import ObservationData, OptimizationSpec

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


# ---------------------------------------------------------------------------
# Campaign construction & state management
# ---------------------------------------------------------------------------


def _build_campaign(spec: OptimizationSpec) -> Campaign:
    """Create a fresh BayBE Campaign from an OptimizationSpec.

    Uses RandomRecommender for the initial design phase (works for all
    space types), then BotorchRecommender for model-guided BO.
    For purely discrete spaces, prevents re-suggesting already-measured points.
    (BayBE only supports this flag for SearchSpaceType.DISCRETE.)
    """
    searchspace = spec_to_searchspace(spec)
    is_purely_discrete = searchspace.type == SearchSpaceType.DISCRETE

    kwargs: dict[str, Any] = {
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

    return Campaign(**kwargs)


def _restore_or_build_campaign(
    spec: OptimizationSpec,
    backend_state: dict[str, Any] | None,
) -> Campaign:
    """Restore a Campaign from serialized state, or build a fresh one.

    Approach B: if backend_state contains a serialized Campaign JSON,
    deserialize it to avoid redundant model re-fitting. Falls back to
    Approach A (fresh Campaign) if state is missing or deserialization fails.
    """
    if backend_state and "campaign_json" in backend_state:
        try:
            return Campaign.from_json(backend_state["campaign_json"])
        except (RuntimeError, ValueError, TypeError) as e:
            logger.debug("Failed to restore Campaign from state: %s, rebuilding fresh", e)
    return _build_campaign(spec)


def _serialize_campaign(campaign: Campaign) -> dict[str, Any]:
    """Serialize a Campaign to a dict for storage as backend_state."""
    return {"campaign_json": campaign.to_json()}


def _observations_to_minimization_tensor(
    spec: OptimizationSpec,
    observations: list[ObservationData],
) -> torch.Tensor:
    """Convert observations to a tensor in BoTorch minimization convention.

    Maximization objectives are negated so all objectives are treated as
    minimization internally.
    """
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
    """Compute a reference point from minimization-convention objective values.

    Sets the reference point to worst observed + 10% of the range per objective.
    """
    worst = y_bo.max(dim=0).values
    ranges = worst - y_bo.min(dim=0).values
    ranges = torch.where(ranges < 1e-6, torch.ones_like(ranges), ranges)
    return worst + 0.1 * ranges


# ---------------------------------------------------------------------------
# BayBE-native extraction helpers
# ---------------------------------------------------------------------------


def _extract_posterior_stats(
    campaign: Campaign,
    rec_df: Any,
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
        mean_cols = [c for c in stats_df.columns if "mean" in str(c).lower()]
        std_cols = [c for c in stats_df.columns if "std" in str(c).lower()]
        predictions: list[dict[str, dict[str, float] | None]] = []
        for i in range(len(rec_df)):
            row = stats_df.iloc[i]
            pred_obj = {obj_name: float(row[mean_cols[0]]) for obj_name in obj_names if mean_cols}
            pred_std = {obj_name: float(row[std_cols[0]]) for obj_name in obj_names if std_cols}
            predictions.append(
                {
                    "predicted_objectives": pred_obj or None,
                    "predicted_std": pred_std or None,
                }
            )
        return predictions
    except (RuntimeError, ValueError, TypeError) as e:
        logger.debug("Posterior stats extraction failed: %s", e)
        return [empty] * len(rec_df)


def _extract_acquisition_values(
    campaign: Campaign,
    rec_df: Any,
) -> list[float | None]:
    """Extract per-suggestion acquisition function values from BayBE."""
    try:
        acq_series = campaign.acquisition_values(candidates=rec_df)
        return [float(v) for v in acq_series.values]
    except (RuntimeError, ValueError, TypeError) as e:
        logger.debug("Acquisition value extraction failed: %s", e)
        return [None] * len(rec_df)


def _extract_model_info(campaign: Campaign) -> dict[str, Any]:
    """Extract model hyperparameters from BayBE's fitted surrogate."""
    info: dict[str, Any] = {"model_type": _MODEL_TYPE_SINGLE, "kernel_type": "unknown"}
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
    except (RuntimeError, ValueError, TypeError, AttributeError) as e:
        logger.debug("Model info extraction failed: %s", e)
    return info


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

        if observations:
            obs_df = observations_to_dataframe(observations, spec)
            campaign.add_measurements(obs_df)

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

        return SuggestionBatch(
            suggestions=suggestions,
            method_info=method_info,
            backend_state=_serialize_campaign(campaign),
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
        # Protocol requires spec and new_observations; BayBE manages state
        # via Campaign serialization in generate_suggestions instead.
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
        except (RuntimeError, ValueError, TypeError) as e:
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


# ---------------------------------------------------------------------------
# Provenance construction
# ---------------------------------------------------------------------------


def _build_suggestion_list(
    param_dicts: list[dict[str, Any]],
    predictions: list[dict[str, dict[str, float] | None]],
    acq_values: list[float | None],
    model_info: dict[str, Any],
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
