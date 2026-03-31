"""Sensitivity analysis for Bayesian Optimization solutions.

This module provides functions to compute and visualize how robust optimal
solutions are to parameter perturbations. In real experiments, exact parameter
values often cannot be hit, so understanding which parameters are sensitive
helps experimentalists prioritize precision.

Section 3.1 - Missing Trust-Building Features

References:
    - Saltelli, A. et al. "Global Sensitivity Analysis: The Primer" (2008)
    - Sobol, I.M. "Sensitivity Estimates for Nonlinear Mathematical Models" (1993)
    - BoTorch GP Models: https://botorch.org/docs/models/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from botorch.models import ModelListGP, SingleTaskGP
from torch import Tensor

from bo_engine.constants import (
    SENSITIVITY_HIGH_THRESHOLD,
    SENSITIVITY_MEDIUM_THRESHOLD,
    SENSITIVITY_PERTURBATION_FRACTION,
)
from bo_engine.device import get_device, get_dtype

if TYPE_CHECKING:
    pass


@dataclass
class ParameterSensitivity:
    """Sensitivity information for a single parameter.

    Attributes:
        name: Parameter name.
        sensitivity: Normalized sensitivity value (higher = more sensitive).
        gradient: Raw gradient of objective w.r.t. parameter.
        sensitivity_level: Categorical level ("high", "medium", "low").
        effect_size: Expected change in objective for 1% perturbation.
    """

    name: str
    sensitivity: float
    gradient: float
    sensitivity_level: str
    effect_size: float


@dataclass
class SensitivityReport:
    """Complete sensitivity analysis report for a solution.

    Attributes:
        parameters: List of per-parameter sensitivity information.
        most_sensitive: Name of the most sensitive parameter.
        least_sensitive: Name of the least sensitive parameter.
        overall_sensitivity: Aggregate sensitivity score (0-1).
        robustness_score: Inverse of overall sensitivity (higher = more robust).
        recommendation: Actionable recommendation based on analysis.
        warnings: List of warnings about highly sensitive parameters.
    """

    parameters: list[ParameterSensitivity]
    most_sensitive: str
    least_sensitive: str
    overall_sensitivity: float
    robustness_score: float
    recommendation: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class LocalSensitivityResult:
    """Result from local sensitivity analysis at a single point.

    Attributes:
        x: The point where sensitivity was computed.
        gradients: Gradient vector at the point.
        normalized_sensitivities: Normalized by parameter ranges.
        hessian_diagonal: Diagonal of Hessian if computed.
    """

    x: Tensor
    gradients: Tensor
    normalized_sensitivities: Tensor
    hessian_diagonal: Tensor | None = None


def compute_sensitivity(
    model: SingleTaskGP | ModelListGP,
    x_best: Tensor,
    bounds: Tensor,
    parameter_names: list[str] | None = None,
    perturbation_fraction: float = SENSITIVITY_PERTURBATION_FRACTION,
    objective_index: int = 0,
) -> SensitivityReport:
    """Compute sensitivity of objective to parameter perturbations at best point.

    Uses gradient-based sensitivity analysis to determine how the predicted
    objective value changes with respect to each parameter. Gradients are
    normalized by parameter ranges to allow fair comparison.

    Args:
        model: Fitted GP model (SingleTaskGP or ModelListGP).
        x_best: Best observed parameter configuration (1 x d tensor).
        bounds: Parameter bounds (2 x d tensor, first row lower, second row upper).
        parameter_names: Names for each parameter. Defaults to param_0, param_1, etc.
        perturbation_fraction: Fraction of range for sensitivity computation.
        objective_index: Which objective to analyze (for multi-objective).

    Returns:
        SensitivityReport containing per-parameter sensitivities and recommendations.

    Example:
        >>> from bo_engine import compute_sensitivity
        >>> report = compute_sensitivity(model, x_best, bounds)
        >>> print(f"Most sensitive: {report.most_sensitive}")
        >>> for p in report.parameters:
        ...     print(f"{p.name}: {p.sensitivity_level}")

    References:
        - Saltelli, A. "Sensitivity Analysis in Practice" (2004)
    """
    device = get_device()
    dtype = get_dtype()

    # Ensure proper shape and device
    x_best = x_best.to(device=device, dtype=dtype)
    if x_best.dim() == 1:
        x_best = x_best.unsqueeze(0)
    bounds = bounds.to(device=device, dtype=dtype)

    n_params = x_best.shape[-1]
    if parameter_names is None:
        parameter_names = [f"param_{i}" for i in range(n_params)]

    # Compute gradients
    local_result = _compute_local_sensitivity(
        model=model,
        x=x_best,
        bounds=bounds,
        objective_index=objective_index,
    )

    # Build per-parameter sensitivity info
    param_sensitivities: list[ParameterSensitivity] = []
    normalized_sens = local_result.normalized_sensitivities.squeeze()

    for i, name in enumerate(parameter_names):
        sens_value = normalized_sens[i].item()
        grad_value = local_result.gradients.squeeze()[i].item()

        # Determine sensitivity level
        abs_sens = abs(sens_value)
        if abs_sens >= SENSITIVITY_HIGH_THRESHOLD:
            level = "high"
        elif abs_sens >= SENSITIVITY_MEDIUM_THRESHOLD:
            level = "medium"
        else:
            level = "low"

        # Effect size: expected objective change for 1% parameter perturbation
        param_range = (bounds[1, i] - bounds[0, i]).item()
        effect_size = abs(grad_value) * (0.01 * param_range)

        param_sensitivities.append(
            ParameterSensitivity(
                name=name,
                sensitivity=abs_sens,
                gradient=grad_value,
                sensitivity_level=level,
                effect_size=effect_size,
            )
        )

    # Sort by sensitivity for ranking
    sorted_params = sorted(param_sensitivities, key=lambda p: p.sensitivity, reverse=True)
    most_sensitive = sorted_params[0].name
    least_sensitive = sorted_params[-1].name

    # Overall sensitivity (mean of normalized sensitivities)
    overall_sensitivity = normalized_sens.abs().mean().item()
    robustness_score = 1.0 - min(overall_sensitivity, 1.0)

    # Generate warnings and recommendations
    warnings: list[str] = []
    high_sens_params = [p for p in param_sensitivities if p.sensitivity_level == "high"]
    for p in high_sens_params:
        warnings.append(
            f"Parameter '{p.name}' is highly sensitive (effect size: {p.effect_size:.4f}). "
            f"Precise control is important."
        )

    recommendation = _generate_sensitivity_recommendation(
        param_sensitivities, overall_sensitivity, high_sens_params
    )

    return SensitivityReport(
        parameters=param_sensitivities,
        most_sensitive=most_sensitive,
        least_sensitive=least_sensitive,
        overall_sensitivity=overall_sensitivity,
        robustness_score=robustness_score,
        recommendation=recommendation,
        warnings=warnings,
    )


def compute_pareto_sensitivity(
    model: ModelListGP,
    pareto_points: Tensor,
    bounds: Tensor,
    parameter_names: list[str] | None = None,
) -> list[SensitivityReport]:
    """Compute sensitivity for each point on the Pareto front.

    For multi-objective optimization, understanding sensitivity of each
    Pareto-optimal solution helps users choose solutions that are not
    only optimal but also robust.

    Args:
        model: Fitted ModelListGP for multi-objective problem.
        pareto_points: Tensor of Pareto-optimal points (n_points x d).
        bounds: Parameter bounds (2 x d tensor).
        parameter_names: Names for each parameter.

    Returns:
        List of SensitivityReport, one for each Pareto point.
    """
    reports: list[SensitivityReport] = []

    for i in range(pareto_points.shape[0]):
        x = pareto_points[i : i + 1]

        # Compute sensitivity for each objective and aggregate
        n_objectives = len(model.models)
        all_sensitivities: list[Tensor] = []

        for obj_idx in range(n_objectives):
            local_result = _compute_local_sensitivity(
                model=model,
                x=x,
                bounds=bounds,
                objective_index=obj_idx,
            )
            all_sensitivities.append(local_result.normalized_sensitivities)

        # Stack and take max sensitivity across objectives
        stacked = torch.stack(all_sensitivities, dim=0)
        max_sensitivities = stacked.abs().max(dim=0).values

        # Create report with aggregated sensitivities
        n_params = x.shape[-1]
        if parameter_names is None:
            parameter_names = [f"param_{j}" for j in range(n_params)]

        param_sens_list: list[ParameterSensitivity] = []
        for j, name in enumerate(parameter_names):
            sens_value = max_sensitivities.squeeze()[j].item()
            if sens_value >= SENSITIVITY_HIGH_THRESHOLD:
                level = "high"
            elif sens_value >= SENSITIVITY_MEDIUM_THRESHOLD:
                level = "medium"
            else:
                level = "low"

            param_sens_list.append(
                ParameterSensitivity(
                    name=name,
                    sensitivity=sens_value,
                    gradient=0.0,  # Aggregated, no single gradient
                    sensitivity_level=level,
                    effect_size=0.0,
                )
            )

        sorted_params = sorted(param_sens_list, key=lambda p: p.sensitivity, reverse=True)
        overall = max_sensitivities.abs().mean().item()

        reports.append(
            SensitivityReport(
                parameters=param_sens_list,
                most_sensitive=sorted_params[0].name,
                least_sensitive=sorted_params[-1].name,
                overall_sensitivity=overall,
                robustness_score=1.0 - min(overall, 1.0),
                recommendation=f"Pareto point {i + 1}: Consider trade-off between "
                f"optimality and robustness.",
                warnings=[],
            )
        )

    return reports


def compute_sensitivity_heatmap_data(
    model: SingleTaskGP | ModelListGP,
    center: Tensor,
    bounds: Tensor,
    param_indices: tuple[int, int],
    n_grid: int = 20,
    objective_index: int = 0,
) -> dict[str, Tensor]:
    """Compute sensitivity values over a 2D grid for visualization.

    Creates a grid around the center point and computes sensitivity at
    each grid point. Useful for generating heatmap visualizations.

    Args:
        model: Fitted GP model.
        center: Center point for the grid (1 x d tensor).
        bounds: Parameter bounds (2 x d tensor).
        param_indices: Tuple of two parameter indices to vary.
        n_grid: Number of grid points per dimension.
        objective_index: Which objective to analyze.

    Returns:
        Dictionary with:
            - 'x_grid': Grid values for first parameter (n_grid,)
            - 'y_grid': Grid values for second parameter (n_grid,)
            - 'sensitivity_x': Sensitivity w.r.t. first param (n_grid x n_grid)
            - 'sensitivity_y': Sensitivity w.r.t. second param (n_grid x n_grid)
            - 'total_sensitivity': Combined sensitivity magnitude (n_grid x n_grid)
    """
    device = get_device()
    dtype = get_dtype()

    center = center.to(device=device, dtype=dtype)
    if center.dim() == 1:
        center = center.unsqueeze(0)
    bounds = bounds.to(device=device, dtype=dtype)

    i, j = param_indices

    # Create grid
    x_range = torch.linspace(
        bounds[0, i].item(), bounds[1, i].item(), n_grid, device=device, dtype=dtype
    )
    y_range = torch.linspace(
        bounds[0, j].item(), bounds[1, j].item(), n_grid, device=device, dtype=dtype
    )

    sensitivity_x = torch.zeros(n_grid, n_grid, device=device, dtype=dtype)
    sensitivity_y = torch.zeros(n_grid, n_grid, device=device, dtype=dtype)

    for xi, x_val in enumerate(x_range):
        for yi, y_val in enumerate(y_range):
            # Create point by modifying center
            point = center.clone()
            point[0, i] = x_val
            point[0, j] = y_val

            local_result = _compute_local_sensitivity(
                model=model,
                x=point,
                bounds=bounds,
                objective_index=objective_index,
            )

            sens = local_result.normalized_sensitivities.squeeze()
            sensitivity_x[xi, yi] = sens[i].abs()
            sensitivity_y[xi, yi] = sens[j].abs()

    total_sensitivity = (sensitivity_x**2 + sensitivity_y**2).sqrt()

    return {
        "x_grid": x_range,
        "y_grid": y_range,
        "sensitivity_x": sensitivity_x,
        "sensitivity_y": sensitivity_y,
        "total_sensitivity": total_sensitivity,
    }


def rank_parameters_by_sensitivity(
    model: SingleTaskGP | ModelListGP,
    observations_x: Tensor,
    bounds: Tensor,
    parameter_names: list[str] | None = None,
    objective_index: int = 0,
) -> list[tuple[str, float]]:
    """Rank parameters by their average sensitivity across all observations.

    Computes sensitivity at each observed point and averages to get
    a global ranking of parameter importance.

    Args:
        model: Fitted GP model.
        observations_x: All observed parameter configurations (n x d tensor).
        bounds: Parameter bounds (2 x d tensor).
        parameter_names: Names for each parameter.
        objective_index: Which objective to analyze.

    Returns:
        List of (parameter_name, average_sensitivity) tuples, sorted by
        sensitivity in descending order.
    """
    device = get_device()
    dtype = get_dtype()

    observations_x = observations_x.to(device=device, dtype=dtype)
    bounds = bounds.to(device=device, dtype=dtype)

    n_points, n_params = observations_x.shape
    if parameter_names is None:
        parameter_names = [f"param_{i}" for i in range(n_params)]

    # Accumulate sensitivities across all points
    total_sensitivities = torch.zeros(n_params, device=device, dtype=dtype)

    for i in range(n_points):
        x = observations_x[i : i + 1]
        local_result = _compute_local_sensitivity(
            model=model,
            x=x,
            bounds=bounds,
            objective_index=objective_index,
        )
        total_sensitivities += local_result.normalized_sensitivities.squeeze().abs()

    avg_sensitivities = total_sensitivities / n_points

    # Create ranking
    ranking: list[tuple[str, float]] = [
        (name, avg_sensitivities[i].item()) for i, name in enumerate(parameter_names)
    ]
    ranking.sort(key=lambda x: x[1], reverse=True)

    return ranking


def _compute_local_sensitivity(
    model: SingleTaskGP | ModelListGP,
    x: Tensor,
    bounds: Tensor,
    objective_index: int = 0,
) -> LocalSensitivityResult:
    """Compute local sensitivity at a single point using gradients.

    Args:
        model: Fitted GP model.
        x: Point at which to compute sensitivity (1 x d tensor).
        bounds: Parameter bounds for normalization.
        objective_index: Which objective to analyze.

    Returns:
        LocalSensitivityResult with gradients and normalized sensitivities.
    """
    x = x.clone().detach().requires_grad_(True)

    # Get posterior mean
    if isinstance(model, ModelListGP):
        posterior = model.models[objective_index].posterior(x)  # ty: ignore[call-non-callable]
    else:
        posterior = model.posterior(x)

    mean = posterior.mean

    # Compute gradient
    mean.backward()
    grad = x.grad
    if grad is None:
        raise RuntimeError("Gradient computation failed - x.grad is None")
    gradients = grad.detach()

    # Normalize by parameter ranges
    param_ranges = bounds[1] - bounds[0]
    # Avoid division by zero
    param_ranges = torch.clamp(param_ranges, min=1e-8)
    normalized_sensitivities = gradients * param_ranges

    return LocalSensitivityResult(
        x=x.detach(),
        gradients=gradients,
        normalized_sensitivities=normalized_sensitivities,
        hessian_diagonal=None,
    )


def _generate_sensitivity_recommendation(
    param_sensitivities: list[ParameterSensitivity],
    overall_sensitivity: float,
    high_sens_params: list[ParameterSensitivity],
) -> str:
    """Generate actionable recommendation based on sensitivity analysis."""
    if overall_sensitivity < 0.1:
        return (
            "The solution is highly robust. Parameter variations within typical "
            "experimental tolerances should have minimal impact on the objective."
        )
    elif overall_sensitivity < 0.3:
        return (
            "The solution has moderate sensitivity. Focus precision efforts on "
            f"'{high_sens_params[0].name}' if any high-sensitivity parameters exist."
            if high_sens_params
            else "The solution has moderate sensitivity. No individual parameter "
            "dominates, so general precision is recommended."
        )
    else:
        high_names = ", ".join(p.name for p in high_sens_params[:3])
        return (
            f"The solution is sensitive to parameter variations. Prioritize precise "
            f"control of: {high_names}. Consider exploring nearby configurations "
            f"for more robust alternatives."
        )
