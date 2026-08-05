"""Result validation for duplicate and outlier detection.

Sections 1.2 and 1.3 of Implementation Plan:
- 1.2: Duplicate Detection - Prevent submitting same experimental result multiple times
- 1.3: Outlier Detection - Identify outliers that may bias the GP model

References:
- Leave-One-Out CV for outlier detection: https://botorch.org/docs/tutorials/batch_mode_cross_validation/
- Statistical outlier detection: https://en.wikipedia.org/wiki/Outlier#Detection
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch
from botorch.cross_validation import gen_loo_cv_folds
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor

from bo_engine.constants import (
    DUPLICATE_DETECTION_TOLERANCE,
    OUTLIER_DETECTION_MIN_OBSERVATIONS,
    OUTLIER_DETECTION_SIGMA_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class DuplicateResult:
    """Information about a potential duplicate result.

    Attributes:
        index: Index of the existing result that is a duplicate
        parameter_distance: Normalized distance between parameter values
        is_exact: Whether the duplicate is exact (within tolerance)
    """

    index: int
    parameter_distance: float
    is_exact: bool


@dataclass
class OutlierResult:
    """Information about a detected outlier.

    Attributes:
        index: Index of the outlier in the results
        standardized_error: The standardized prediction error (z-score)
        actual_value: The actual observed value
        predicted_value: The model's predicted value
        predicted_std: The model's predicted standard deviation
        objective_name: Name of the objective (if available)
    """

    index: int
    standardized_error: float
    actual_value: float
    predicted_value: float
    predicted_std: float
    objective_name: str | None = None


def detect_duplicates(
    new_params: dict[str, Any],
    existing_params: list[dict[str, Any]],
    tolerance: float = DUPLICATE_DETECTION_TOLERANCE,
) -> list[DuplicateResult]:
    """Detect near-duplicate results based on parameter values.

    Checks if the new parameter values are too close to existing results,
    which could corrupt the GP model by having near-identical inputs with
    potentially different outputs.

    Args:
        new_params: Parameter values for the new result
        existing_params: List of parameter value dicts from existing results
        tolerance: Tolerance for considering parameters as duplicates

    Returns:
        List of DuplicateResult objects for each detected duplicate

    Reference:
        Section 1.2 of Implementation Plan - Duplicate Detection
    """
    duplicates: list[DuplicateResult] = []

    if not existing_params:
        return duplicates

    # Get common parameter names
    param_names = sorted(new_params.keys())

    for idx, existing in enumerate(existing_params):
        # Check if same parameters are present
        existing_names = set(existing.keys())
        if not all(p in existing_names for p in param_names):
            continue

        comparison = _compare_parameter_rows(new_params, existing, param_names, tolerance)
        if comparison is None:
            continue
        is_exact, distance = comparison

        if is_exact or distance < tolerance * len(param_names) ** 0.5:
            duplicates.append(
                DuplicateResult(
                    index=idx,
                    parameter_distance=distance,
                    is_exact=is_exact,
                )
            )

    return duplicates


def _compare_parameter_rows(
    new_params: dict[str, Any],
    existing: dict[str, Any],
    param_names: list[str],
    tolerance: float,
) -> tuple[bool, float] | None:
    """Compare one candidate row against one existing row.

    Returns ``(is_exact, euclidean_distance)`` over the numeric parameters,
    or ``None`` when any categorical parameter differs: a different category
    is a different experiment, never a near-duplicate — identical numerics
    with a different solvent is the normal shape of categorical DOE, and the
    categorical mismatch contributes no numeric distance that could keep the
    row under the near-duplicate threshold.
    """
    is_exact = True
    total_distance_sq = 0.0

    for param_name in param_names:
        new_val = new_params[param_name]
        existing_val = existing.get(param_name)

        if existing_val is None:
            is_exact = False
            continue

        try:
            diff = abs(float(new_val) - float(existing_val))
        except (TypeError, ValueError):
            # Non-numeric (categorical) parameter.
            if new_val != existing_val:
                return None
            continue

        if diff > tolerance:
            is_exact = False
        total_distance_sq += diff**2

    return is_exact, total_distance_sq**0.5


def _pairwise_distances_below_tolerance(
    new_x: Tensor,
    existing_x: Tensor,
    tolerance: float,
) -> tuple[Tensor, Tensor]:
    """Return pairwise distances and the ``(n_new, n_existing)`` duplicate matrix.

    Distances are computed in ``new_x``'s dtype — casting to float32 would
    collapse distinct large-magnitude float64 coordinates onto each other.
    The tolerance is scaled by ``sqrt(n_dims)`` so the per-dimension meaning
    of ``DUPLICATE_DETECTION_TOLERANCE`` is preserved across dimensionality.
    """
    distances = torch.cdist(new_x, existing_x.to(dtype=new_x.dtype, device=new_x.device), p=2)
    scaled_tolerance = tolerance * (new_x.shape[1] ** 0.5)
    return distances, distances < scaled_tolerance


def duplicate_row_mask(
    new_x: Tensor,
    existing_x: Tensor,
    tolerance: float = DUPLICATE_DETECTION_TOLERANCE,
) -> Tensor:
    """Return one boolean per ``new_x`` row marking duplicates of ``existing_x`` rows.

    Fully vectorized (no per-pair Python loop or ``.item()`` synchronization)
    for callers that only need membership, e.g. acquisition-level exclusion
    of already-evaluated points.

    Args:
        new_x: New parameter values of shape (n_new, n_dims)
        existing_x: Existing parameter values of shape (n_existing, n_dims)
        tolerance: Tolerance for considering parameters as duplicates

    Returns:
        Boolean tensor of shape (n_new,); True where a row duplicates any
        existing row.
    """
    if existing_x.shape[0] == 0 or new_x.shape[0] == 0:
        return torch.zeros(new_x.shape[0], dtype=torch.bool, device=new_x.device)
    _, below = _pairwise_distances_below_tolerance(new_x, existing_x, tolerance)
    return below.any(dim=1)


def detect_duplicates_batch(
    new_x: Tensor,
    existing_x: Tensor,
    tolerance: float = DUPLICATE_DETECTION_TOLERANCE,
) -> list[tuple[int, int, float]]:
    """Detect duplicates using tensor operations for efficiency.

    Args:
        new_x: New parameter values of shape (n_new, n_dims)
        existing_x: Existing parameter values of shape (n_existing, n_dims)
        tolerance: Tolerance for considering parameters as duplicates

    Returns:
        List of (new_idx, existing_idx, distance) tuples for duplicates
    """
    if existing_x.shape[0] == 0 or new_x.shape[0] == 0:
        return []
    distances, below = _pairwise_distances_below_tolerance(new_x, existing_x, tolerance)
    return [
        (new_idx, existing_idx, float(distances[new_idx, existing_idx]))
        for new_idx, existing_idx in below.nonzero(as_tuple=False).tolist()
    ]


def detect_outliers(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    sigma_threshold: float = OUTLIER_DETECTION_SIGMA_THRESHOLD,
    min_observations: int = OUTLIER_DETECTION_MIN_OBSERVATIONS,
    objective_names: list[str] | None = None,
) -> list[OutlierResult]:
    """Detect outliers using Leave-One-Out cross-validation.

    For each observation, fits a GP model on all other observations and
    predicts the held-out point. Points with large standardized errors
    (actual - predicted) / std are flagged as outliers.

    Args:
        train_x: Training inputs of shape (n_samples, n_dims)
        train_y: Training outputs of shape (n_samples,) or (n_samples, n_objectives)
        bounds: Parameter bounds of shape (2, n_dims)
        sigma_threshold: Number of standard deviations for outlier cutoff
        min_observations: Minimum observations required before detection
        objective_names: Optional names for objectives

    Returns:
        List of OutlierResult objects for detected outliers

    Reference:
        Section 1.3 of Implementation Plan - Outlier Detection
        BoTorch LOO-CV: https://botorch.org/docs/tutorials/batch_mode_cross_validation/
    """
    outliers: list[OutlierResult] = []
    n_samples = train_x.shape[0]

    if n_samples < min_observations:
        return outliers

    # Ensure train_y is 2D
    if train_y.dim() == 1:
        train_y = train_y.unsqueeze(-1)

    n_objectives = train_y.shape[1]

    # Perform LOO-CV for each objective
    for obj_idx in range(n_objectives):
        obj_name = (
            objective_names[obj_idx] if objective_names and obj_idx < len(objective_names) else None
        )
        obj_y = train_y[:, obj_idx : obj_idx + 1]

        obj_outliers = _detect_outliers_single_objective(
            train_x,
            obj_y,
            bounds,
            sigma_threshold,
            obj_name,
        )
        outliers.extend(obj_outliers)

    return outliers


def _detect_outliers_single_objective(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    sigma_threshold: float,
    objective_name: str | None,
) -> list[OutlierResult]:
    """Detect outliers for a single objective using LOO-CV.

    Args:
        train_x: Training inputs
        train_y: Training outputs for single objective (n_samples, 1)
        bounds: Parameter bounds
        sigma_threshold: Sigma threshold for outlier detection
        objective_name: Name of the objective

    Returns:
        List of OutlierResult objects
    """
    outliers: list[OutlierResult] = []
    n_samples = train_x.shape[0]

    # Need at least 3 samples for meaningful CV
    if n_samples < 3:
        return outliers

    # Generate LOO CV folds
    cv_folds = gen_loo_cv_folds(train_X=train_x, train_Y=train_y)

    for fold_idx in range(n_samples):
        train_fold = cv_folds.train_X[fold_idx]
        train_y_fold = cv_folds.train_Y[fold_idx]
        test_fold = cv_folds.test_X[fold_idx]
        test_y_fold = cv_folds.test_Y[fold_idx]

        # Skip if not enough training data
        if train_fold.shape[0] < 2:
            continue

        try:
            # Create and fit model on training fold
            model = SingleTaskGP(
                train_X=train_fold,
                train_Y=train_y_fold,
                input_transform=Normalize(d=train_x.shape[-1], bounds=bounds),
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Predict the held-out observation; the comparison target is a
            # noisy measurement, so include observation noise in the std
            model.eval()
            with torch.no_grad():
                posterior = model.posterior(test_fold, observation_noise=True)
                pred_mean = posterior.mean.squeeze().item()
                pred_std = posterior.variance.sqrt().squeeze().item()

            actual = test_y_fold.squeeze().item()

            # Compute standardized error
            std_error = abs(actual - pred_mean) / pred_std if pred_std > 1e-10 else 0.0

            # Check if outlier
            if std_error > sigma_threshold:
                outliers.append(
                    OutlierResult(
                        index=fold_idx,
                        standardized_error=std_error,
                        actual_value=actual,
                        predicted_value=pred_mean,
                        predicted_std=pred_std,
                        objective_name=objective_name,
                    )
                )

        except (RuntimeError, ValueError, TypeError) as e:
            # Skip folds that fail to fit (singular matrices, numerical instability)
            logger.debug("LOO-CV fold failed: %s", e)
            continue

    return outliers


def compute_loo_standardized_errors(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
) -> list[float]:
    """Compute LOO-CV standardized errors for all observations.

    Useful for general model diagnostics and calibration checking.

    Args:
        train_x: Training inputs
        train_y: Training outputs (single objective)
        bounds: Parameter bounds

    Returns:
        List of standardized errors for each observation
    """
    # Ensure train_y is 2D
    if train_y.dim() == 1:
        train_y = train_y.unsqueeze(-1)

    n_samples = train_x.shape[0]
    errors: list[float] = []

    if n_samples < 3:
        return [0.0] * n_samples

    cv_folds = gen_loo_cv_folds(train_X=train_x, train_Y=train_y)

    for fold_idx in range(n_samples):
        train_fold = cv_folds.train_X[fold_idx]
        train_y_fold = cv_folds.train_Y[fold_idx]
        test_fold = cv_folds.test_X[fold_idx]
        test_y_fold = cv_folds.test_Y[fold_idx]

        if train_fold.shape[0] < 2:
            errors.append(0.0)
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
                pred_mean = posterior.mean.squeeze().item()
                pred_std = posterior.variance.sqrt().squeeze().item()

            actual = test_y_fold.squeeze().item()
            std_error = (actual - pred_mean) / pred_std if pred_std > 1e-10 else 0.0
            errors.append(std_error)
        except (RuntimeError, ValueError, TypeError):
            errors.append(0.0)

    return errors
