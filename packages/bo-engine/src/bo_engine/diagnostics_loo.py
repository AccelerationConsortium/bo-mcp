"""Leave-One-Out cross-validation diagnostics for BO surrogate models.

Split from :mod:`bo_engine.diagnostics` to keep that module focused on
campaign-level health and progress reporting. LOO-CV computes an unbiased
estimate of the surrogate model's generalization error by leaving out
each training point in turn, refitting the GP on the remaining ``n-1``
points, and recording the predictive error on the held-out sample. The
aggregate RMSE / MAE / R² and per-fold errors are the model-quality
signals consumed by :func:`bo_engine.diagnostics.assess_model_health`
and the diagnostics tool surface.

Held-out targets are noisy measurements, so standardized errors and
coverage are computed against the posterior predictive (latent function
plus observation noise) rather than the latent-only posterior.

Reference: Rasmussen & Williams, *Gaussian Processes for Machine
Learning* (2006), §5.4.2 ("Leave-one-out cross-validation").
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

from bo_engine.constants import CI_95_Z_SCORE, NUMERICAL_EPSILON, SAFE_DIVISION_EPSILON
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
    standardized_errors: list[float] = []
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

            error = (pred_mean - test_y_fold).abs().item()
            per_fold_errors.append(error)
            predictions.append(pred_mean.squeeze().item())
            actuals.append(test_y_fold.squeeze().item())

            std_err = (pred_mean - test_y_fold).abs() / (pred_var.sqrt() + NUMERICAL_EPSILON)
            standardized_errors.append(std_err.item())

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

    predictions_t = torch.tensor(predictions)
    actuals_t = torch.tensor(actuals)
    errors = (predictions_t - actuals_t).abs()

    rmse = (errors**2).mean().sqrt().item()
    mae = errors.mean().item()

    ss_res = ((predictions_t - actuals_t) ** 2).sum()
    ss_tot = ((actuals_t - actuals_t.mean()) ** 2).sum()
    r_squared = (
        1 - (ss_res / (ss_tot + NUMERICAL_EPSILON)).item() if ss_tot > NUMERICAL_EPSILON else 0.0
    )

    mean_std_error = sum(standardized_errors) / len(standardized_errors)

    return LOOCVMetrics(
        rmse=rmse,
        mae=mae,
        r_squared=r_squared,
        mean_standardized_error=mean_std_error,
        per_fold_errors=per_fold_errors,
    )


@overload
def compute_loo_cv_for_model(
    model: SingleTaskGP,
    train_x: Tensor,
    train_y: Tensor,
) -> LOOCVMetrics: ...


@overload
def compute_loo_cv_for_model(
    model: ModelListGP,
    train_x: Tensor,
    train_y: Tensor,
) -> dict[int, LOOCVMetrics]: ...


def compute_loo_cv_for_model(
    model: SingleTaskGP | ModelListGP,
    train_x: Tensor,
    train_y: Tensor,
) -> LOOCVMetrics | dict[int, LOOCVMetrics]:
    """Compute LOO-CV metrics for a fitted model.

    Uses BoTorch's batch_cross_validation for efficiency.

    Args:
        model: Fitted GP model
        train_x: Training inputs
        train_y: Training outputs (n_samples, n_objectives)

    Returns:
        LOOCVMetrics for single-objective models, or
        Dictionary mapping objective index to LOOCVMetrics for multi-objective
    """
    from bo_engine.diagnostics import LOOCVMetrics

    if isinstance(model, SingleTaskGP):
        cv_folds = gen_loo_cv_folds(train_X=train_x, train_Y=train_y)
        try:
            cv_results = batch_cross_validation(
                model_cls=SingleTaskGP,
                mll_cls=ExactMarginalLogLikelihood,
                cv_folds=cv_folds,
                observation_noise=True,
            )
            pred_mean = cv_results.posterior.mean.squeeze()
            pred_var = cv_results.posterior.variance.squeeze()
            actuals = train_y.squeeze()
            errors = (pred_mean - actuals).abs()

            rmse = (errors**2).mean().sqrt().item()
            mae = errors.mean().item()

            ss_res = ((pred_mean - actuals) ** 2).sum()
            ss_tot = ((actuals - actuals.mean()) ** 2).sum()
            r_squared = (
                1 - (ss_res / (ss_tot + NUMERICAL_EPSILON)).item()
                if ss_tot > NUMERICAL_EPSILON
                else 0.0
            )

            std = pred_var.sqrt().clamp(min=SAFE_DIVISION_EPSILON)
            standardized_errors = errors / std
            mean_std_error = standardized_errors.mean().item()

            within_95ci = standardized_errors < CI_95_Z_SCORE
            coverage_95 = within_95ci.float().mean().item()

            return LOOCVMetrics(
                rmse=rmse,
                mae=mae,
                r_squared=r_squared,
                mean_standardized_error=mean_std_error,
                per_fold_errors=errors.tolist(),
                coverage_95=coverage_95,
            )
        except (RuntimeError, ValueError, TypeError) as e:
            logger.debug(
                "Cross-validation for single-objective model failed: %s: %s", type(e).__name__, e
            )
            return LOOCVMetrics(
                rmse=float("nan"),
                mae=float("nan"),
                r_squared=float("nan"),
                mean_standardized_error=float("nan"),
                per_fold_errors=[],
                coverage_95=float("nan"),
            )

    results: dict[int, LOOCVMetrics] = {}
    for i, _m in enumerate(model.models):
        train_y_i = train_y[:, i : i + 1]
        cv_folds = gen_loo_cv_folds(train_X=train_x, train_Y=train_y_i)
        try:
            cv_results = batch_cross_validation(
                model_cls=SingleTaskGP,
                mll_cls=ExactMarginalLogLikelihood,
                cv_folds=cv_folds,
                observation_noise=True,
            )
            pred_mean = cv_results.posterior.mean.squeeze()
            pred_var = cv_results.posterior.variance.squeeze()
            actuals = train_y_i.squeeze()
            errors = (pred_mean - actuals).abs()

            rmse = (errors**2).mean().sqrt().item()
            mae = errors.mean().item()

            ss_res = ((pred_mean - actuals) ** 2).sum()
            ss_tot = ((actuals - actuals.mean()) ** 2).sum()
            r_squared = (
                1 - (ss_res / (ss_tot + NUMERICAL_EPSILON)).item()
                if ss_tot > NUMERICAL_EPSILON
                else 0.0
            )

            std = pred_var.sqrt().clamp(min=SAFE_DIVISION_EPSILON)
            standardized_errors = errors / std
            mean_std_error = standardized_errors.mean().item()

            within_95ci = standardized_errors < CI_95_Z_SCORE
            coverage_95 = within_95ci.float().mean().item()

            results[i] = LOOCVMetrics(
                rmse=rmse,
                mae=mae,
                r_squared=r_squared,
                mean_standardized_error=mean_std_error,
                per_fold_errors=errors.tolist(),
                coverage_95=coverage_95,
            )
        except (RuntimeError, ValueError, TypeError) as e:
            logger.debug("Cross-validation for objective %d failed: %s: %s", i, type(e).__name__, e)
            results[i] = LOOCVMetrics(
                rmse=float("nan"),
                mae=float("nan"),
                r_squared=float("nan"),
                mean_standardized_error=float("nan"),
                per_fold_errors=[],
                coverage_95=float("nan"),
            )

    return results
