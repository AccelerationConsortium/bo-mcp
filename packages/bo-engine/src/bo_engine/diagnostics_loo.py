"""Leave-One-Out cross-validation diagnostics for BO surrogate models.

Split from :mod:`bo_engine.diagnostics` to keep that module focused on
campaign-level health and progress reporting. LOO-CV estimates the
surrogate model's generalization error from held-out predictions: for a
fitted model this is the exact LOO posterior downdate at its fitted
hyperparameters (one O(n³) solve instead of ``n`` refits), otherwise each
training point is left out in turn and the GP refit on the remaining
``n-1`` points. The aggregate RMSE / MAE / R² and per-fold errors are the
model-quality signals consumed by
:func:`bo_engine.diagnostics.assess_model_health` and the diagnostics
tool surface.

Held-out targets are noisy measurements, so standardized errors and
coverage are computed against the posterior predictive (latent function
plus observation noise) rather than the latent-only posterior.

Reference: Rasmussen & Williams, *Gaussian Processes for Machine
Learning* (2006), §5.4.2 ("Leave-one-out cross-validation",
Eqs. 5.10-5.12 for the exact downdate).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, overload

import torch
from botorch.cross_validation import batch_cross_validation, gen_loo_cv_folds
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.nn import Module

from bo_engine.cross_validation import (
    CVMetrics,
    compute_cv_score_fields,
    compute_loo_cv_optimized,
)
from bo_engine.device import ensure_device

if TYPE_CHECKING:
    from bo_engine.diagnostics import LOOCVMetrics

logger = logging.getLogger(__name__)


def compute_loo_cv_metrics(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
) -> LOOCVMetrics:
    """Compute leave-one-out cross-validation metrics for model quality.

    LOO-CV provides an unbiased estimate of generalization error by training
    on n-1 points and predicting the held-out point, for all n points.

    Args:
        train_x: Training inputs of shape (n_samples, n_dims)
        train_y: Training outputs of shape (n_samples, 1) or (n_samples,)
        bounds: Parameter bounds of shape (2, n_dims)

    Returns:
        LOOCVMetrics with RMSE, MAE, R², and per-fold errors
    """
    # Import locally to break the cyclic import between this module and
    # :mod:`bo_engine.diagnostics` (which re-exports these functions).
    from bo_engine.diagnostics import LOOCVMetrics

    train_x, train_y, bounds = ensure_device(train_x, train_y, bounds)

    if train_y.dim() == 1:
        train_y = train_y.unsqueeze(-1)

    n_samples = train_x.shape[0]

    if n_samples < 3:
        return LOOCVMetrics(
            rmse=float("nan"),
            mae=float("nan"),
            r_squared=float("nan"),
            mean_standardized_error=float("nan"),
            per_fold_errors=[],
        )

    cv_folds = gen_loo_cv_folds(train_X=train_x, train_Y=train_y)

    predictions: list[float] = []
    actuals: list[float] = []
    variances: list[float] = []
    per_fold_errors: list[float] = []

    for fold_idx in range(n_samples):
        train_fold = cv_folds.train_X[fold_idx]
        train_y_fold = cv_folds.train_Y[fold_idx]
        test_fold = cv_folds.test_X[fold_idx]
        test_y_fold = cv_folds.test_Y[fold_idx]

        if train_fold.shape[0] < 2:
            continue

        try:
            model = SingleTaskGP(
                train_X=train_fold,
                train_Y=train_y_fold,
                input_transform=Normalize(d=train_x.shape[-1], bounds=bounds),
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            model.eval()
            with torch.no_grad():
                posterior = model.posterior(test_fold, observation_noise=True)
                pred_mean = posterior.mean
                pred_var = posterior.variance

            per_fold_errors.append((pred_mean - test_y_fold).abs().item())
            predictions.append(pred_mean.squeeze().item())
            actuals.append(test_y_fold.squeeze().item())
            variances.append(pred_var.squeeze().item())

        except (RuntimeError, ValueError, TypeError) as e:
            logger.debug("LOO-CV fold %d failed to fit: %s: %s", fold_idx, type(e).__name__, e)
            continue

    if len(predictions) < 2:
        return LOOCVMetrics(
            rmse=float("nan"),
            mae=float("nan"),
            r_squared=float("nan"),
            mean_standardized_error=float("nan"),
            per_fold_errors=per_fold_errors,
        )

    # Same metric block as the batched LOO path — including the measured 95%
    # coverage — so the per-fold-refit surface cannot drift from it.
    scores = compute_cv_score_fields(
        torch.tensor(predictions),
        torch.tensor(variances),
        torch.tensor(actuals),
    )

    return LOOCVMetrics(
        rmse=scores.rmse,
        mae=scores.mae,
        r_squared=scores.r_squared,
        mean_standardized_error=scores.mean_standardized_error,
        per_fold_errors=per_fold_errors,
        coverage_95=scores.coverage_95,
    )


@overload
def compute_loo_cv_for_model(
    model: SingleTaskGP,
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor | None = None,
) -> LOOCVMetrics: ...


@overload
def compute_loo_cv_for_model(
    model: ModelListGP,
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor | None = None,
) -> dict[int, LOOCVMetrics]: ...


def compute_loo_cv_for_model(
    model: SingleTaskGP | ModelListGP,
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor | None = None,
) -> LOOCVMetrics | dict[int, LOOCVMetrics]:
    """Compute LOO-CV metrics for a fitted model.

    Validates the passed model via its exact LOO posterior downdate at the
    fitted hyperparameters (GPML §5.4.2, Eqs. 5.10-5.12) — one O(n³) solve
    with the model's own transforms (input normalization, warping, outcome
    transform) intact, instead of ``n`` refits of a differently-configured
    default model. When the downdate cannot serve (the passed data is not
    the model's training set, a batched fully Bayesian model, a
    non-Gaussian likelihood), falls back to BoTorch's batched refit CV with
    ``Normalize``-equipped fold models.

    Args:
        model: Fitted GP model
        train_x: Training inputs
        train_y: Training outputs (n_samples, n_objectives)
        bounds: Optional parameter bounds of shape (2, n_dims), used to
            normalize the fold models' inputs on the refit fallback path.
            When None, the fallback normalizes to the fold data's range.

    Returns:
        LOOCVMetrics for single-objective models, or
        Dictionary mapping objective index to LOOCVMetrics for multi-objective
    """
    if isinstance(model, SingleTaskGP):
        return _model_loo_metrics(
            model, train_x, train_y, bounds=bounds, label="single-objective model"
        )

    results: dict[int, LOOCVMetrics] = {}
    for i, sub_model in enumerate(model.models):
        train_y_i = train_y[:, i : i + 1]
        results[i] = _model_loo_metrics(
            sub_model, train_x, train_y_i, bounds=bounds, label=f"objective {i}"
        )

    return results


def _model_loo_metrics(
    model: Module,
    train_x: Tensor,
    train_y: Tensor,
    *,
    bounds: Tensor | None,
    label: str,
) -> LOOCVMetrics:
    """LOO metrics for one fitted single-output model, downdate first.

    The exact downdate honors the fitted model's configuration; the batched
    refit fallback engages when the downdate cannot serve (see
    :func:`compute_loo_cv_for_model`), including for sub-models that are not
    ``SingleTaskGP`` instances — ``compute_loo_cv_optimized``'s polymorphic
    first argument dispatches on that exact type.
    """
    if isinstance(model, SingleTaskGP):
        try:
            return _to_loo_cv_metrics(compute_loo_cv_optimized(model, train_x, train_y))
        except (RuntimeError, TypeError, ValueError) as e:
            logger.debug(
                "Exact LOO downdate for %s unavailable, falling back to batched refit CV: %s: %s",
                label,
                type(e).__name__,
                e,
            )
    return _batched_loo_metrics(train_x, train_y, bounds=bounds, label=label)


def _to_loo_cv_metrics(metrics: CVMetrics) -> LOOCVMetrics:
    """Convert the CV surface's metrics record to the diagnostics record."""
    from bo_engine.diagnostics import LOOCVMetrics

    return LOOCVMetrics(
        rmse=metrics.rmse,
        mae=metrics.mae,
        r_squared=metrics.r_squared,
        mean_standardized_error=metrics.mean_standardized_error,
        per_fold_errors=metrics.per_fold_errors,
        coverage_95=metrics.coverage_95,
    )


def _batched_loo_metrics(
    train_x: Tensor,
    train_y: Tensor,
    *,
    bounds: Tensor | None,
    label: str,
) -> LOOCVMetrics:
    """Batched LOO-CV metrics for one output, scored via the shared helper.

    Refits ``SingleTaskGP`` folds with BoTorch's batch CV and routes the
    held-out predictions through :func:`compute_cv_score_fields`, the same
    metric block the optimized CV surface uses, so the diagnostics and CV
    surfaces report identical numbers for identical inputs. Fold models
    carry a ``Normalize`` input transform (BoTorch's dim-scaled lengthscale
    prior is unit-cube-calibrated; raw-scale fits misreport model quality
    on non-unit-cube problems) and BoTorch's default batch-shaped
    ``Standardize`` outcome transform.
    """
    from bo_engine.diagnostics import LOOCVMetrics

    cv_folds = gen_loo_cv_folds(train_X=train_x, train_Y=train_y)
    input_transform = Normalize(d=train_x.shape[-1], bounds=bounds)
    try:
        cv_results = batch_cross_validation(
            model_cls=SingleTaskGP,
            mll_cls=ExactMarginalLogLikelihood,
            cv_folds=cv_folds,
            observation_noise=True,
            model_init_kwargs={"input_transform": input_transform},
        )
    except (RuntimeError, ValueError, TypeError) as e:
        logger.debug("Cross-validation for %s failed: %s: %s", label, type(e).__name__, e)
        return LOOCVMetrics(
            rmse=float("nan"),
            mae=float("nan"),
            r_squared=float("nan"),
            mean_standardized_error=float("nan"),
            per_fold_errors=[],
            coverage_95=float("nan"),
        )

    pred_mean = cv_results.posterior.mean.squeeze()
    pred_var = cv_results.posterior.variance.squeeze()
    actuals = train_y.squeeze()
    per_fold_errors = (pred_mean - actuals).abs().tolist()

    scores = compute_cv_score_fields(pred_mean, pred_var, actuals)

    return LOOCVMetrics(
        rmse=scores.rmse,
        mae=scores.mae,
        r_squared=scores.r_squared,
        mean_standardized_error=scores.mean_standardized_error,
        per_fold_errors=per_fold_errors,
        coverage_95=scores.coverage_95,
    )
