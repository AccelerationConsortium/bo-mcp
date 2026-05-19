"""Model validation for GP models before generating suggestions.

Section 1.1 of Implementation Plan: Validates GP model health before generating
suggestions to prevent garbage suggestions from poorly fitted models.

References:
- BoTorch GP models: https://botorch.org/docs/models/
- GPyTorch diagnostics: https://docs.gpytorch.ai/en/latest/examples/01_Exact_GPs/

This module provides validation that:
- Model fitting converged (lengthscales not at bounds)
- Noise variance is reasonable relative to data variance
- Overall model health assessment
"""

from dataclasses import dataclass
from typing import cast

import torch
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from torch import Tensor

from bo_engine.constants import (
    MODEL_VALIDATION_MAX_LENGTHSCALE_RATIO,
    MODEL_VALIDATION_MAX_NOISE_RATIO,
    MODEL_VALIDATION_MIN_LENGTHSCALE_RATIO,
)


@dataclass
class ModelHealthReport:
    """Report on GP model health before suggestion generation.

    Attributes:
        is_healthy: Whether model passes all validation checks
        issues: List of identified issues with the model
        warnings: List of non-critical warnings
        lengthscales: Dictionary mapping parameter index to lengthscale value
        noise_variance: Estimated observation noise variance
        data_variance: Variance of the training data
        param_ranges: Range of each parameter from bounds
    """

    is_healthy: bool
    issues: list[str]
    warnings: list[str]
    lengthscales: dict[int, float]
    noise_variance: float
    data_variance: float
    param_ranges: list[float]


def _compute_data_variance(train_y: Tensor) -> float:
    """Compute data variance, averaging across objectives for multi-output."""
    if train_y.dim() == 1:
        return train_y.var().item()
    return train_y.var(dim=0).mean().item()


def _validate_model_list_gp(
    model: ModelListGP,
    param_ranges: list[float],
    param_names: list[str] | None,
) -> tuple[dict[int, float], float, list[str], list[str]]:
    """Validate a ModelListGP (multi-objective) and aggregate results."""
    issues: list[str] = []
    warnings: list[str] = []
    noise_variances: list[float] = []
    all_lengthscales: list[Tensor] = []

    for i, gp in enumerate(model.models):
        ls, nv, ls_issues, ls_warnings = _validate_single_gp(
            gp, param_ranges, param_names, objective_idx=i
        )
        all_lengthscales.append(ls)
        noise_variances.append(nv)
        issues.extend(ls_issues)
        warnings.extend(ls_warnings)

    lengthscales_dict: dict[int, float] = {}
    if all_lengthscales:
        avg_ls = torch.stack(all_lengthscales).mean(dim=0)
        for j in range(avg_ls.numel()):
            lengthscales_dict[j] = avg_ls[j].item()

    noise_variance = sum(noise_variances) / len(noise_variances) if noise_variances else 0.0
    return lengthscales_dict, noise_variance, issues, warnings


def validate_model_health(
    model: SingleTaskGP | ModelListGP,
    train_y: Tensor,
    bounds: Tensor,
    param_names: list[str] | None = None,
) -> ModelHealthReport:
    """Validate GP model health before generating suggestions.

    Checks for common model fitting issues that indicate the model may not
    be suitable for generating reliable suggestions.

    Args:
        model: Fitted GP model (SingleTaskGP or ModelListGP)
        train_y: Training outputs of shape (n_samples, n_objectives)
        bounds: Parameter bounds of shape (2, n_dims)
        param_names: Optional parameter names for human-readable messages

    Returns:
        ModelHealthReport with validation results

    Reference:
        Section 1.1 of Implementation Plan - Model Validation Before Suggestions
    """
    param_ranges = (bounds[1] - bounds[0]).tolist()
    data_variance = _compute_data_variance(train_y)

    if isinstance(model, ModelListGP):
        lengthscales_dict, noise_variance, issues, warnings = _validate_model_list_gp(
            model, param_ranges, param_names
        )
    else:
        ls, noise_variance, issues, warnings = _validate_single_gp(model, param_ranges, param_names)
        lengthscales_dict = {j: ls[j].item() for j in range(ls.numel())}

    # Check noise-to-signal ratio
    if data_variance > 0 and noise_variance > data_variance * MODEL_VALIDATION_MAX_NOISE_RATIO:
        issues.append(
            f"Noise variance ({noise_variance:.4f}) exceeds data variance ({data_variance:.4f}). "
            "Model may be fitting noise rather than signal."
        )

    return ModelHealthReport(
        is_healthy=len(issues) == 0,
        issues=issues,
        warnings=warnings,
        lengthscales=lengthscales_dict,
        noise_variance=noise_variance,
        data_variance=data_variance,
        param_ranges=param_ranges,
    )


def _extract_noise_variance(gp: SingleTaskGP) -> float:
    """Extract noise variance from a GP model's likelihood."""
    if not (hasattr(gp, "likelihood") and hasattr(gp.likelihood, "noise")):
        return 0.0
    noise = gp.likelihood.noise
    if isinstance(noise, Tensor):
        return float(noise.squeeze().item())
    return 0.0


def _check_lengthscale(
    ls_val: float,
    param_range: float,
    param_name: str,
    obj_prefix: str,
) -> tuple[str | None, str | None]:
    """Check a single lengthscale against parameter range. Returns (issue, warning)."""
    if param_range <= 0:
        return None, None
    ratio = ls_val / param_range
    if ratio < MODEL_VALIDATION_MIN_LENGTHSCALE_RATIO:
        return (
            f"{obj_prefix}Lengthscale for '{param_name}' is very small "
            f"(ratio={ratio:.4f}). Model may be overfitting."
        ), None
    if ratio > MODEL_VALIDATION_MAX_LENGTHSCALE_RATIO:
        return None, (
            f"{obj_prefix}Lengthscale for '{param_name}' is very large "
            f"(ratio={ratio:.4f}). Parameter may have little effect."
        )
    return None, None


def _validate_single_gp(
    gp: SingleTaskGP,
    param_ranges: list[float],
    param_names: list[str] | None = None,
    objective_idx: int | None = None,
) -> tuple[Tensor, float, list[str], list[str]]:
    """Validate a single GP model.

    Args:
        gp: Single GP model (SingleTaskGP or similar)
        param_ranges: Range of each parameter
        param_names: Optional parameter names
        objective_idx: Optional objective index for multi-objective models

    Returns:
        Tuple of (lengthscales, noise_variance, issues, warnings)
    """
    issues: list[str] = []
    warnings: list[str] = []
    obj_prefix = f"Objective {objective_idx}: " if objective_idx is not None else ""

    # Extract lengthscales
    covar = gp.covar_module
    kernel = getattr(covar, "base_kernel", covar)
    lengthscales = cast("Tensor", kernel.lengthscale).detach().squeeze()
    if lengthscales.dim() == 0:
        lengthscales = lengthscales.unsqueeze(0)

    for i, (ls, pr) in enumerate(zip(lengthscales, param_ranges, strict=True)):
        param_name = param_names[i] if param_names and i < len(param_names) else f"param_{i}"
        issue, warning = _check_lengthscale(ls.item(), pr, param_name, obj_prefix)
        if issue:
            issues.append(issue)
        if warning:
            warnings.append(warning)

    noise_variance = _extract_noise_variance(gp)
    return lengthscales, noise_variance, issues, warnings


def compute_model_convergence_score(
    model: SingleTaskGP | ModelListGP,
    train_x: Tensor,
    train_y: Tensor,
) -> float:
    """Compute a convergence score indicating model fit quality.

    Higher scores indicate better model fit. Score of 0 indicates
    serious issues, 1.0 indicates excellent fit.

    Args:
        model: Fitted GP model
        train_x: Training inputs
        train_y: Training outputs

    Returns:
        Convergence score between 0 and 1
    """
    try:
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(train_x)
            pred_mean = posterior.mean

            # Compute R² as measure of fit
            if train_y.dim() == 1:
                train_y = train_y.unsqueeze(-1)

            ss_res = ((pred_mean - train_y) ** 2).sum()
            ss_tot = ((train_y - train_y.mean(dim=0)) ** 2).sum()

            if ss_tot > 0:
                r_squared = 1 - (ss_res / ss_tot).item()
                # Clamp to [0, 1]
                return max(0.0, min(1.0, r_squared))
            return 0.5  # Neutral if no variance
    except (RuntimeError, ValueError, TypeError):
        return 0.0
