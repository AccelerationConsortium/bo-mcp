"""BayBE-native introspection helpers.

Split from :mod:`bo_engine_baybe.backend` so that posterior-stats,
acquisition-value, model-info, and SHAP feature-importance extraction
plus the tensor utilities used by diagnostics and reference-point
computation live in one place. The :class:`~bo_engine_baybe.backend.BayBEBackend`
class composes these helpers when building :class:`SuggestionBatch`
payloads and diagnostic envelopes.
"""

from __future__ import annotations

import logging
from typing import cast

import pandas as pd
import torch
from baybe import Campaign
from botorch.models import ModelListGP, SingleTaskGP

from bo_engine.device import get_device, get_dtype
from bo_engine.diagnostics import observations_to_minimization_form
from bo_engine.transforms import encode_categorical, get_bounds_tensor
from bo_engine.types import ObservationData, OptimizationSpec
from bo_engine_baybe.converters import observations_to_dataframe
from bo_engine_baybe.state import _BAYBE_SAFE_EXCEPTIONS, _build_campaign

logger = logging.getLogger(__name__)


_MODEL_TYPE_SINGLE = "BayBE GP"
_MODEL_TYPE_MULTI = "BayBE GP (CompositeSurrogate)"
_FALLBACK_ACQ_SINGLE = "qLogNoisyExpectedImprovement"
_FALLBACK_ACQ_MULTI = "qLogNoisyExpectedHypervolumeImprovement"


def _observations_to_minimization_tensor(
    spec: OptimizationSpec,
    observations: list[ObservationData],
) -> torch.Tensor:
    """Convert observations to a tensor in BoTorch minimization convention.

    Thin wrapper over the shared
    :func:`bo_engine.diagnostics.observations_to_minimization_form` so the
    BayBE diagnostics surface and the engine backends build the
    minimization-form tensor identically (this helper drops the mask the
    shared function also returns).
    """
    y_bo, _ = observations_to_minimization_form(spec, observations)
    return y_bo


def _extract_posterior_stats(
    campaign: Campaign,
    rec_df: pd.DataFrame,
    spec: OptimizationSpec,
) -> tuple[list[dict[str, dict[str, float] | None]], str | None]:
    """Extract BayBE posterior mean and std per objective for each recommendation.

    Returns ``(predictions, warning)`` where ``warning`` is a short
    message explaining why posterior_stats was unavailable for the active
    recommender phase, or ``None`` when extraction succeeded. The warning
    is forwarded to :class:`SuggestionBatch.warnings` so users
    see "no posterior in random-warmup phase" instead of a silent ``None``.
    """
    obj_names = [o.name for o in spec.objectives]
    empty: dict[str, dict[str, float] | None] = {
        "predicted_objectives": None,
        "predicted_std": None,
    }
    try:
        stats_df = campaign.posterior_stats(candidates=rec_df, stats=("mean", "std"))
        return (
            [_parse_prediction_row(stats_df.iloc[i], obj_names) for i in range(len(rec_df))],
            None,
        )
    except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
        logger.debug("Posterior stats extraction failed: %s", e)
        return [empty] * len(rec_df), str(e)


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
    pending_df: pd.DataFrame | None,
) -> tuple[list[float | None], str | None]:
    """Extract per-suggestion acquisition function values from BayBE.

    Pending experiments are forwarded so the acquisition values reflect
    the same conditioning BayBE used during ``recommend``. A
    short warning message is returned alongside the values when
    extraction is impossible in the current recommender phase so the
    backend can surface it via :class:`SuggestionBatch.warnings`.
    """
    try:
        acq_series = campaign.acquisition_values(
            candidates=rec_df,
            pending_experiments=pending_df,
        )
        return [float(v) for v in acq_series.to_numpy()], None
    except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
        logger.debug("Acquisition value extraction failed: %s", e)
        return [None] * len(rec_df), str(e)


def _extract_model_info(
    campaign: Campaign,
) -> tuple[dict[str, str | list[float] | float | None], str | None]:
    """Extract model hyperparameters from BayBE's fitted surrogate.

    Returns ``(info, warning)`` mirroring :func:`_extract_posterior_stats`.
    Multi-target campaigns produce a ``ModelListGP``; in that case the
    first sub-model's hyperparameters are reported (BayBE composes one
    GP per target). The ``kernel_type`` key is always present so callers
    can distinguish "fitted" (string class name) from "not available"
    (``None``) — surfacing only the latter as a method warning.
    """
    info: dict[str, str | list[float] | float | None] = {
        "kernel_type": None,
    }
    try:
        surrogate = campaign.get_surrogate()
        botorch_model = cast("SingleTaskGP | ModelListGP", surrogate.to_botorch())
        model_for_kernel = _select_kernel_model(botorch_model)
        covar = getattr(model_for_kernel, "covar_module", None)
        if covar is None:
            return info, "BayBE surrogate did not expose a covariance module"
        kernel = getattr(covar, "base_kernel", covar)
        info["kernel_type"] = type(kernel).__name__

        ls = cast("torch.Tensor", kernel.lengthscale).detach().squeeze()
        if ls.numel() == 1:
            info["lengthscales"] = [round(float(ls.item()), 4)]
        else:
            info["lengthscales"] = [round(float(v), 4) for v in ls.tolist()]

        if hasattr(model_for_kernel, "likelihood") and hasattr(
            model_for_kernel.likelihood, "noise"
        ):
            noise = cast("torch.Tensor", model_for_kernel.likelihood.noise).item()
            info["noise_variance"] = round(float(noise), 6)
        if hasattr(covar, "outputscale"):
            oscale = cast("torch.Tensor", covar.outputscale).item()
            info["output_scale"] = round(float(oscale), 4)
    except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
        logger.debug("Model info extraction failed: %s", e)
        return info, str(e)
    return info, None


def _select_kernel_model(botorch_model: SingleTaskGP | ModelListGP) -> SingleTaskGP:
    """Pick the BoTorch model whose hyperparameters we report.

    ``ModelListGP`` (used by multi-target/multi-objective BayBE campaigns)
    wraps one sub-model per target. We report the first sub-model so the
    metadata is non-empty and consistent across calls; the per-target
    breakdown can be reconstructed from BayBE's posterior_stats output
    when finer granularity is needed.
    """
    if isinstance(botorch_model, ModelListGP) and len(botorch_model.models) > 0:
        return cast("SingleTaskGP", botorch_model.models[0])
    return cast("SingleTaskGP", botorch_model)


def _extract_feature_importance(campaign: Campaign) -> dict[str, float] | None:
    """Extract SHAP-based feature importance from BayBE (optional)."""
    try:
        from baybe.insights.shap import SHAPInsight

        insight = SHAPInsight.from_campaign(campaign)
        values = insight.explanation.values  # noqa: PD011  # ty: ignore[unresolved-attribute]
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


def _active_recommender(campaign: Campaign) -> object:
    """Return the active non-meta recommender of the campaign, if exposed.

    BayBE's meta-recommender flow advances state during ``recommend``; the
    method-info path needs the recommender after that switch happened
    rather than the meta-recommender wrapper. Falls back to the raw
    ``campaign.recommender`` attribute when the campaign exposes no
    helper.
    """
    helper = getattr(campaign, "_get_non_meta_recommender", None)
    if callable(helper):
        try:
            return helper()
        except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
            logger.debug("Could not resolve active BayBE recommender: %s", e)
    return getattr(campaign, "recommender", None)


def _active_acquisition_label(recommender: object) -> str | None:
    """Read the acquisition function class name from a BayBE recommender, if any."""
    acq_fn = getattr(recommender, "acquisition_function", None)
    if acq_fn is None:
        return None
    return type(acq_fn).__name__


def _campaign_searchspace_label(campaign: Campaign) -> str:
    """Return the BayBE searchspace type as a stable string label."""
    searchspace_type = getattr(getattr(campaign, "searchspace", None), "type", None)
    if hasattr(searchspace_type, "value"):
        return str(searchspace_type.value)
    return str(searchspace_type)


def _strategy_and_model(
    rec_name: str | None,
    is_nonpredictive: bool,
    is_multi: bool,
) -> tuple[str, str]:
    """Compose the live ``(strategy, model_type)`` labels for method-info.

    Nonpredictive recommenders (e.g. random warmup) explicitly report
    ``"none (space-filling)"`` for the model type so the audit trail
    cannot claim a GP surrogate when none is fitted.
    """
    if is_nonpredictive:
        strategy = (
            f"{rec_name} (space-filling, no surrogate)"
            if rec_name
            else "RandomRecommender (space-filling initial design)"
        )
        return strategy, "none (space-filling)"
    surrogate = _MODEL_TYPE_MULTI if is_multi else _MODEL_TYPE_SINGLE
    if rec_name is not None:
        return f"{rec_name} (GP-based)", surrogate
    return "BotorchRecommender (GP-based)", surrogate


def _acquisition_label(
    recommender: object,
    is_nonpredictive: bool,
    is_multi: bool,
) -> tuple[str, bool]:
    """Resolve the acquisition-function label and whether it is inferred.

    Returns ``(label, inferred)``. ``inferred=True`` signals that the
    label was guessed from the static fallback table because the live
    recommender did not expose an acquisition function attribute. The
    caller stamps the flag onto the structured ``method_info``
    so downstream consumers can distinguish "BayBE told us qLogNEI" from
    "we couldn't read the acq function and assumed qLogNEI"; the legacy
    ``(fallback)`` suffix on the label itself is removed because the
    structured field is the load-bearing signal.
    """
    if is_nonpredictive:
        # Nonpredictive recommenders do not have an acquisition function;
        # claiming qLogNEI here would mislead audit metadata.
        return "none (space-filling)", False
    live_acq = _active_acquisition_label(recommender)
    if live_acq is not None:
        return live_acq, False
    return (_FALLBACK_ACQ_MULTI if is_multi else _FALLBACK_ACQ_SINGLE), True


def _baybe_model_correlation(
    campaign: Campaign,
    obs_df: pd.DataFrame,
    spec: OptimizationSpec,
) -> float:
    """Compute rank correlation between BayBE posterior mean and actuals."""
    from scipy import stats as scipy_stats

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

        predicted = stats_df[mean_col].to_numpy()
        actual = obs_df[obj_name].to_numpy()
        result = scipy_stats.spearmanr(predicted, actual)
        corr = float(result.statistic)
    except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
        logger.debug("BayBE model correlation failed: %s", e)
        return 0.0
    return corr if corr == corr else 0.0  # Handle NaN


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
