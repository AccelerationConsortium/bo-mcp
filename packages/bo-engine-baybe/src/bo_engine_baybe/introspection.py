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
import math
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from baybe import Campaign
from botorch.models import ModelListGP, SingleTaskGP

from bo_engine.device import get_device, get_dtype
from bo_engine.transforms import encode_categorical, get_bounds_tensor
from bo_engine.types import ObjectiveSpec, ObservationData, OptimizationSpec
from bo_engine_baybe.constants import DEFAULT_MAX_ROW_LEVEL_SHAP_ROWS
from bo_engine_baybe.converters import observations_to_dataframe
from bo_engine_baybe.options import BayBESurrogateKind, extract_baybe_backend_options
from bo_engine_baybe.state import _BAYBE_SAFE_EXCEPTIONS, _add_measurements, _build_campaign

logger = logging.getLogger(__name__)


_MODEL_TYPE_SINGLE = "BayBE GP"
_MODEL_TYPE_MULTI = "BayBE GP (CompositeSurrogate)"
# Model-type and acquisition label while a space-filling (nonpredictive)
# recommender is active: no surrogate is fitted and no acquisition
# function exists, and the metadata must not claim otherwise.
_MODEL_TYPE_NONPREDICTIVE = "none (space-filling)"
_FALLBACK_ACQ_SINGLE = "qLogNoisyExpectedImprovement"
_FALLBACK_ACQ_MULTI = "qLogNoisyExpectedHypervolumeImprovement"


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


def _rounded_tensor_values(value: object, digits: int) -> list[float]:
    """Detach a tensor-like hyperparameter into a flat list of rounded floats."""
    tensor = cast("torch.Tensor", value).detach().squeeze()
    if tensor.numel() == 1:
        return [round(float(tensor.item()), digits)]
    return [round(float(v), digits) for v in tensor.tolist()]


def _extract_model_info(
    campaign: Campaign,
    non_gp_surrogate: BayBESurrogateKind | None = None,
) -> tuple[dict[str, str | list[float] | float | None], str | None]:
    """Extract model hyperparameters from BayBE's fitted surrogate.

    Returns ``(info, warning)`` mirroring :func:`_extract_posterior_stats`.
    Multi-target campaigns produce a ``ModelListGP``; in that case the
    first sub-model's hyperparameters are reported (BayBE composes one
    GP per target). The ``kernel_type`` key is always present so callers
    can distinguish "fitted" (string class name) from "not available"
    (``None``) — surfacing only the latter as a method warning.

    ``non_gp_surrogate`` short-circuits extraction: a configured non-GP
    surrogate has no GP covariance module *by construction*, so skipping
    it (instead of failing on it) keeps the warning channel reserved for
    genuine introspection failures.

    Each hyperparameter is read independently: kernels without a
    lengthscale (gpytorch ``has_lengthscale=False``, e.g. polynomial
    kernels) return ``None`` for it, and skipping the absent attribute
    must not abort the noise/output-scale extraction that still applies.
    Kernel-specific learned parameters are reported alongside — the
    linear/polynomial ``offset``, the periodic ``period_length``, and
    the rational-quadratic ``alpha``.
    """
    info: dict[str, str | list[float] | float | None] = {
        "kernel_type": None,
    }
    if non_gp_surrogate is not None:
        return info, None
    try:
        surrogate = campaign.get_surrogate()
        botorch_model = cast("SingleTaskGP | ModelListGP", surrogate.to_botorch())
        model_for_kernel = _select_kernel_model(botorch_model)
        covar = getattr(model_for_kernel, "covar_module", None)
        if covar is None:
            return info, "BayBE surrogate did not expose a covariance module"
        kernel = getattr(covar, "base_kernel", covar)
        info["kernel_type"] = type(kernel).__name__

        lengthscale = getattr(kernel, "lengthscale", None)
        if lengthscale is not None:
            info["lengthscales"] = _rounded_tensor_values(lengthscale, 4)
        offset = getattr(kernel, "offset", None)
        if offset is not None:
            info["kernel_offset"] = _rounded_tensor_values(offset, 4)[0]
        period_length = getattr(kernel, "period_length", None)
        if period_length is not None:
            info["kernel_period_length"] = _rounded_tensor_values(period_length, 4)
        alpha = getattr(kernel, "alpha", None)
        if alpha is not None:
            info["kernel_alpha"] = _rounded_tensor_values(alpha, 4)[0]

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


def _mean_abs_shap_per_feature(explanation: object) -> dict[str, float]:
    """Mean absolute SHAP value per feature for one target's explanation.

    Mean-|SHAP| over the background rows is the standard global feature
    importance aggregation (the quantity behind ``shap.plots.bar``; see
    Lundberg & Lee, NeurIPS 2017, and the BayBE insights userguide,
    https://emdgroup.github.io/baybe/stable/userguide/insights.html).
    The explanation is duck-typed (``values`` matrix of shape
    ``(n_rows, n_features)`` plus ``feature_names``) so the aggregation
    math stays testable without the heavy optional ``shap`` dependency.
    """
    # ``.values`` here is shap.Explanation's SHAP-value matrix, not a
    # pandas accessor — the pandas lint heuristic misfires on the name.
    matrix = np.abs(np.asarray(explanation.values, dtype=float))  # ty: ignore[unresolved-attribute]
    feature_names = [str(name) for name in explanation.feature_names]  # ty: ignore[unresolved-attribute]
    mean_abs = matrix.mean(axis=0)
    return {name: float(value) for name, value in zip(feature_names, mean_abs, strict=True)}


def _aggregate_shap_importance(explanations: Sequence[object]) -> dict[str, float] | None:
    """Aggregate mean-|SHAP| feature importance across per-target explanations.

    ``SHAPInsight.explain()`` returns one ``shap.Explanation`` per target:
    single-objective campaigns contribute exactly one, Pareto campaigns one
    per target. The diagnostics contract exposes a single scalar per
    feature, so multi-target importances are **averaged across targets**
    (equal weight per target) — per-target breakdowns can be added as a
    separate diagnostics field if a consumer needs them.
    """
    per_target = [_mean_abs_shap_per_feature(e) for e in explanations]
    if not per_target:
        return None
    names = list(per_target[0])
    for target in per_target[1:]:
        if set(target) != set(names):
            msg = "Per-target SHAP explanations disagree on feature names"
            raise ValueError(msg)
    return {name: float(np.mean([target[name] for target in per_target])) for name in names}


def _row_level_attributions(explanation: object, max_rows: int) -> list[dict[str, float]]:
    """Per-observation SHAP attributions, bounded to ``max_rows`` entries.

    Rows follow the explanation's background-data order (the campaign's
    measurement order) and carry no per-row identifiers; on multi-target
    campaigns the caller passes the *first* target's explanation only —
    both limitations are documented on
    :class:`~bo_engine_baybe.options.BayBEInsightsOptions.include_row_level`.
    """
    # ``.values`` is shap.Explanation's matrix, not a pandas accessor.
    matrix = np.asarray(explanation.values, dtype=float)  # ty: ignore[unresolved-attribute]
    names = [str(n) for n in explanation.feature_names]  # ty: ignore[unresolved-attribute]
    rows = matrix[:max_rows]
    return [{name: float(value) for name, value in zip(names, row, strict=True)} for row in rows]


def _extract_feature_importance_report(
    campaign: Campaign,
    spec: OptimizationSpec,
) -> dict[str, Any] | None:
    """Extract the SHAP feature-importance block for diagnostics (optional).

    Importance is computed by **calling** ``SHAPInsight.explain()`` (the
    class exposes no ``explanation`` attribute) and aggregating the
    returned ``shap.Explanation`` object(s) via
    :func:`_aggregate_shap_importance`. The explainer backend and
    representation are configurable via
    ``backend_options['baybe'].insights`` (``explainer`` /
    ``use_comp_rep`` / ``include_row_level``); multi-target campaigns
    additionally expose the per-target breakdown.

    The "optional, never crash diagnostics" contract distinguishes three
    failure modes instead of one blanket swallow:

    * ``shap`` (an optional heavy extra) is not installed → expected,
      returns ``None`` with a *debug* log only.
    * BayBE cannot produce an explanation for the campaign's current
      phase (no measurements, nonpredictive recommender, incompatible
      explainer, …) → expected, returns ``None`` with a *debug* log.
    * ``AttributeError`` → our own SHAPInsight usage drifted from the
      installed BayBE API (the failure mode that silently disabled this
      feature before). Logged at *warning* so the regression is visible
      on dashboards instead of degrading diagnostics invisibly.

    A missing ``shap`` is additionally promoted to a *warning* when the
    caller explicitly configured insights options — silence is then a
    misconfiguration signal, not the expected optional-extra default.
    """
    insights = extract_baybe_backend_options(spec.backend_options).insights
    try:
        from baybe.insights.shap import SHAPInsight
    except ImportError as e:
        if insights is not None:
            logger.warning(
                "backend_options['baybe'].insights is configured but the "
                "optional shap dependency is not installed; feature "
                "importance stays unavailable: %s",
                e,
            )
        else:
            logger.debug("SHAP feature importance unavailable (optional dependency): %s", e)
        return None
    explainer = insights.explainer if insights is not None else None
    use_comp_rep = insights.use_comp_rep if insights is not None else False
    include_rows = insights.include_row_level if insights is not None else False
    max_rows = (
        insights.row_level_max_rows
        if insights is not None and insights.row_level_max_rows is not None
        else DEFAULT_MAX_ROW_LEVEL_SHAP_ROWS
    )
    try:
        kwargs: dict[str, Any] = {"use_comp_rep": use_comp_rep}
        if explainer is not None:
            kwargs["explainer_cls"] = explainer.value
        insight = SHAPInsight.from_campaign(campaign, **kwargs)
        explanations = insight.explain()
        importance = _aggregate_shap_importance(explanations)
        if importance is None:
            return None
        report: dict[str, Any] = {"feature_importance": importance}
        obj_names = [o.name for o in spec.objectives]
        if len(explanations) > 1 and len(explanations) == len(obj_names):
            report["feature_importance_per_target"] = {
                name: _mean_abs_shap_per_feature(explanation)
                for name, explanation in zip(obj_names, explanations, strict=True)
            }
        if include_rows:
            report["feature_importance_rows"] = _row_level_attributions(explanations[0], max_rows)
    except AttributeError as e:
        logger.warning(
            "SHAP feature importance extraction used the BayBE insights API "
            "incorrectly (BayBE version drift?): %s",
            e,
        )
        return None
    except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
        logger.debug("SHAP feature importance extraction failed: %s", e)
        return None
    else:
        return report


def _extract_feature_importance(campaign: Campaign) -> dict[str, float] | None:
    """Aggregate mean-|SHAP| feature importance with default insights options.

    Compatibility wrapper over :func:`_extract_feature_importance_report`
    for callers without a spec in hand: objective names are synthesized
    from the campaign's own targets (only needed for the per-target
    breakdown, which this wrapper discards anyway).
    """
    targets = getattr(getattr(campaign, "objective", None), "targets", None) or []
    spec = OptimizationSpec(
        parameters=[],
        objectives=[ObjectiveSpec(name=str(t.name)) for t in targets],
    )
    report = _extract_feature_importance_report(campaign, spec)
    if report is None:
        return None
    return cast("dict[str, float]", report["feature_importance"])


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
    campaign = _build_campaign(spec, observations)
    obs_df = observations_to_dataframe(observations, spec)
    _add_measurements(campaign, obs_df, spec)
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


def _non_gp_model_type(kind: BayBESurrogateKind, is_multi: bool) -> str:
    """Model-type label for a configured non-GP surrogate.

    Mirrors the GP labels' ``(CompositeSurrogate)`` suffix convention for
    multi-output replication.
    """
    model_type = f"BayBE {kind.value}"
    if is_multi:
        model_type += " (CompositeSurrogate)"
    return model_type


def _strategy_and_model(
    rec_name: str | None,
    is_nonpredictive: bool,
    is_multi: bool,
    non_gp_surrogate: BayBESurrogateKind | None = None,
) -> tuple[str, str]:
    """Compose the live ``(strategy, model_type)`` labels for method-info.

    Nonpredictive recommenders (e.g. random warmup) explicitly report
    ``"none (space-filling)"`` for the model type so the audit trail
    cannot claim a GP surrogate when none is fitted; likewise a
    configured non-GP surrogate reports its own kind instead of the GP
    labels. The fallback path (``select_methods``) reuses this helper so
    live and pre-campaign metadata agree phase-for-phase.
    """
    if is_nonpredictive:
        name = rec_name if rec_name is not None else "RandomRecommender"
        return f"{name} (space-filling, no surrogate)", _MODEL_TYPE_NONPREDICTIVE
    name = rec_name if rec_name is not None else "BotorchRecommender"
    if non_gp_surrogate is not None:
        return (
            f"{name} ({non_gp_surrogate.value} surrogate)",
            _non_gp_model_type(non_gp_surrogate, is_multi),
        )
    surrogate = _MODEL_TYPE_MULTI if is_multi else _MODEL_TYPE_SINGLE
    return f"{name} (GP-based)", surrogate


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
        return _MODEL_TYPE_NONPREDICTIVE, False
    live_acq = _active_acquisition_label(recommender)
    if live_acq is not None:
        return live_acq, False
    return (_FALLBACK_ACQ_MULTI if is_multi else _FALLBACK_ACQ_SINGLE), True


def _baybe_model_correlation(
    campaign: Campaign,
    obs_df: pd.DataFrame,
    spec: OptimizationSpec,
) -> float | None:
    """Compute rank correlation between BayBE posterior mean and actuals.

    Returns ``None`` when the correlation cannot be extracted (posterior
    stats unavailable, undefined Spearman, or any BayBE-side failure).
    ``None`` means "metric unavailable" and must stay distinguishable from
    a measured ``0.0`` ("model is uninformative") — downstream health
    scoring treats only the latter as a warning signal.
    """
    from scipy import stats as scipy_stats

    try:
        stats_df = campaign.posterior_stats(candidates=obs_df, stats=("mean",))
        obj_name = spec.objectives[0].name
        mean_col = f"{obj_name}_mean"

        # Fall back to first mean column if exact name not found
        if mean_col not in stats_df.columns:
            mean_cols = [c for c in stats_df.columns if "mean" in str(c).lower()]
            if not mean_cols:
                return None
            mean_col = mean_cols[0]

        predicted = stats_df[mean_col].to_numpy()
        actual = obs_df[obj_name].to_numpy()
        result = scipy_stats.spearmanr(predicted, actual)
        corr = float(result.statistic)
    except (*_BAYBE_SAFE_EXCEPTIONS,) as e:
        logger.warning("BayBE model correlation unavailable: %s", e)
        return None
    return corr if not math.isnan(corr) else None


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
