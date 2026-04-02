"""What-If Analysis for Bayesian Optimization.

This module provides functions to simulate the impact of hypothetical
results before committing experimental resources. Users can explore
"what would happen if I added this hypothetical result?" to plan
experiments more effectively.

Section 3.8 - Missing Trust-Building Features

References:
    - Rasmussen & Williams "GPML" Ch. 2 (GP Prediction)
    - BoTorch Models: https://botorch.org/docs/models/
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from botorch.models import SingleTaskGP
from torch import Tensor

from bo_engine.constants import (
    WHATIF_DEFAULT_NUM_SUGGESTIONS,
)
from bo_engine.device import get_device, get_dtype
from bo_engine.diagnostics import compute_hypervolume
from bo_engine.models import create_and_fit_single_task_model
from bo_engine.suggestions import generate_next_batch
from bo_engine.types import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)


@dataclass
class HypotheticalResult:
    """A hypothetical result to simulate.

    Attributes:
        parameters: Parameter values for the hypothetical experiment.
        objective_values: Hypothetical objective values.
        constraint_values: Hypothetical constraint values (if any).
    """

    parameters: dict[str, float]
    objective_values: dict[str, float]
    constraint_values: dict[str, float] | None = None


@dataclass
class ModelImpact:
    """Impact of hypothetical result on the model.

    Attributes:
        lengthscale_changes: Change in lengthscales per parameter.
        noise_variance_change: Change in noise variance.
        prediction_changes: Changes in predictions at key points.
        uncertainty_reduction: Average reduction in uncertainty.
    """

    lengthscale_changes: dict[str, float]
    noise_variance_change: float
    prediction_changes: dict[str, float]
    uncertainty_reduction: float


@dataclass
class SuggestionImpact:
    """Impact on next suggestions.

    Attributes:
        original_suggestions: Suggestions without the hypothetical.
        new_suggestions: Suggestions with the hypothetical.
        suggestion_shift: How much suggestions moved.
        new_acquisition_values: Acquisition values for new suggestions.
        diversity_change: Change in batch diversity.
    """

    original_suggestions: list[dict[str, float]]
    new_suggestions: list[dict[str, float]]
    suggestion_shift: float
    new_acquisition_values: list[float]
    diversity_change: float


@dataclass
class ParetoImpact:
    """Impact on Pareto front (multi-objective).

    Attributes:
        original_pareto_size: Number of Pareto points before.
        new_pareto_size: Number of Pareto points after.
        hypothetical_is_pareto: Whether hypothetical would be Pareto-optimal.
        dominated_points: Points that would be dominated.
        hypervolume_change: Change in hypervolume.
    """

    original_pareto_size: int
    new_pareto_size: int
    hypothetical_is_pareto: bool
    dominated_points: list[int]
    hypervolume_change: float


@dataclass
class WhatIfReport:
    """Complete what-if analysis report.

    Attributes:
        hypothetical: The hypothetical result analyzed.
        model_impact: Impact on model parameters.
        suggestion_impact: Impact on next suggestions.
        pareto_impact: Impact on Pareto front (if multi-objective).
        value_of_information: Estimated value of running this experiment.
        recommendation: Actionable recommendation.
        warnings: Any warnings about the analysis.
    """

    hypothetical: HypotheticalResult
    model_impact: ModelImpact | None
    suggestion_impact: SuggestionImpact | None
    pareto_impact: ParetoImpact | None
    value_of_information: float
    recommendation: str
    warnings: list[str] = field(default_factory=list)


def simulate_result(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    hypothetical: HypotheticalResult,
    parameter_names: list[str],
    objective_names: list[str] | None = None,
    use_input_warping: bool = False,
    compute_suggestions: bool = True,
    n_suggestions: int = WHATIF_DEFAULT_NUM_SUGGESTIONS,
) -> WhatIfReport:
    """Simulate adding a hypothetical result and analyze the impact.

    This function temporarily adds a hypothetical result to the training
    data and computes how it would affect:
    - Model predictions and uncertainty
    - Next batch of suggestions
    - Pareto front (for multi-objective)

    Args:
        train_x: Current training inputs (n x d tensor).
        train_y: Current training outputs (n tensor or n x m for multi-obj).
        bounds: Parameter bounds (2 x d tensor).
        hypothetical: The hypothetical result to simulate.
        parameter_names: Names of parameters.
        objective_names: Names of objectives.
        use_input_warping: Whether to use input warping.
        compute_suggestions: Whether to compute new suggestions.
        n_suggestions: Number of suggestions to generate.

    Returns:
        WhatIfReport with detailed impact analysis.

    Example:
        >>> hyp = HypotheticalResult(
        ...     parameters={"x1": 0.5, "x2": 0.3},
        ...     objective_values={"obj": 0.1}
        ... )
        >>> report = simulate_result(train_x, train_y, bounds, hyp, ["x1", "x2"])
        >>> print(f"Value of information: {report.value_of_information:.3f}")
    """
    device = get_device()
    dtype = get_dtype()

    train_x = train_x.to(device=device, dtype=dtype)
    train_y = train_y.to(device=device, dtype=dtype)
    bounds = bounds.to(device=device, dtype=dtype)

    if train_y.dim() == 1:
        train_y = train_y.unsqueeze(-1)

    n_objectives = train_y.shape[1]
    is_multi_objective = n_objectives > 1

    if objective_names is None:
        objective_names = [f"obj_{i}" for i in range(n_objectives)]

    warnings: list[str] = []

    # Convert hypothetical to tensor
    hyp_x = torch.zeros(1, len(parameter_names), device=device, dtype=dtype)
    for i, name in enumerate(parameter_names):
        if name in hypothetical.parameters:
            hyp_x[0, i] = hypothetical.parameters[name]
        else:
            warnings.append(f"Parameter '{name}' not in hypothetical, using 0")

    hyp_y = torch.zeros(1, n_objectives, device=device, dtype=dtype)
    for i, name in enumerate(objective_names):
        if name in hypothetical.objective_values:
            hyp_y[0, i] = hypothetical.objective_values[name]
        else:
            warnings.append(f"Objective '{name}' not in hypothetical, using 0")

    # Create augmented training data
    aug_x = torch.cat([train_x, hyp_x], dim=0)
    aug_y = torch.cat([train_y, hyp_y], dim=0)

    # Fit models before and after
    model_impact = _compute_model_impact(
        train_x=train_x,
        train_y=train_y,
        aug_x=aug_x,
        aug_y=aug_y,
        bounds=bounds,
        use_input_warping=use_input_warping,
    )

    # Compute suggestion impact
    suggestion_impact = None
    if compute_suggestions:
        suggestion_impact = _compute_suggestion_impact(
            train_x=train_x,
            train_y=train_y,
            aug_x=aug_x,
            aug_y=aug_y,
            bounds=bounds,
            n_suggestions=n_suggestions,
            parameter_names=parameter_names,
        )

    # Compute Pareto impact for multi-objective
    pareto_impact = None
    if is_multi_objective:
        pareto_impact = _compute_pareto_impact(
            train_y=train_y,
            aug_y=aug_y,
        )

    # Estimate value of information
    voi = _estimate_value_of_information(
        model_impact=model_impact,
        suggestion_impact=suggestion_impact,
        pareto_impact=pareto_impact,
    )

    # Generate recommendation
    recommendation = _generate_whatif_recommendation(voi=voi)

    return WhatIfReport(
        hypothetical=hypothetical,
        model_impact=model_impact,
        suggestion_impact=suggestion_impact,
        pareto_impact=pareto_impact,
        value_of_information=voi,
        recommendation=recommendation,
        warnings=warnings,
    )


def simulate_multiple_results(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    hypotheticals: list[HypotheticalResult],
    parameter_names: list[str],
    objective_names: list[str] | None = None,
) -> list[WhatIfReport]:
    """Simulate multiple hypothetical results and compare.

    Useful for deciding between several possible experiments.

    Args:
        train_x: Current training inputs.
        train_y: Current training outputs.
        bounds: Parameter bounds.
        hypotheticals: List of hypothetical results to compare.
        parameter_names: Names of parameters.
        objective_names: Names of objectives.

    Returns:
        List of WhatIfReport, one for each hypothetical.
    """
    reports: list[WhatIfReport] = []

    for hyp in hypotheticals:
        report = simulate_result(
            train_x=train_x,
            train_y=train_y,
            bounds=bounds,
            hypothetical=hyp,
            parameter_names=parameter_names,
            objective_names=objective_names,
        )
        reports.append(report)

    return reports


def find_most_informative_point(
    model: SingleTaskGP,
    bounds: Tensor,
    n_candidates: int = 100,
    parameter_names: list[str] | None = None,
) -> tuple[dict[str, float], float]:
    """Find the most informative point to evaluate next.

    Uses uncertainty maximization to identify where an experiment
    would provide the most information about the function.

    Args:
        model: Fitted GP model.
        bounds: Parameter bounds.
        n_candidates: Number of candidates to consider.
        parameter_names: Names of parameters.

    Returns:
        Tuple of (best_parameters, information_value).
    """
    device = get_device()
    dtype = get_dtype()

    bounds = bounds.to(device=device, dtype=dtype)
    n_dims = bounds.shape[1]

    if parameter_names is None:
        parameter_names = [f"param_{i}" for i in range(n_dims)]

    # Generate candidates
    sobol = torch.quasirandom.SobolEngine(dimension=n_dims, scramble=True)
    candidates = sobol.draw(n_candidates).to(device=device, dtype=dtype)
    candidates = bounds[0] + candidates * (bounds[1] - bounds[0])

    # Compute uncertainties
    with torch.no_grad():
        posterior = model.posterior(candidates)
        uncertainties = posterior.variance.squeeze()

    # Find maximum uncertainty point
    best_idx = uncertainties.argmax()
    best_point = candidates[best_idx]
    info_value = uncertainties[best_idx].item()

    # Convert to dict
    best_params = {name: best_point[i].item() for i, name in enumerate(parameter_names)}

    return best_params, info_value


def compare_hypotheticals(
    reports: list[WhatIfReport],
) -> dict[str, list[float]]:
    """Compare multiple what-if scenarios.

    Args:
        reports: List of WhatIfReport to compare.

    Returns:
        Dictionary with comparison metrics.
    """
    voi_values = [r.value_of_information for r in reports]
    uncertainty_reductions = [
        r.model_impact.uncertainty_reduction if r.model_impact else 0.0 for r in reports
    ]
    suggestion_shifts = [
        r.suggestion_impact.suggestion_shift if r.suggestion_impact else 0.0 for r in reports
    ]

    return {
        "value_of_information": voi_values,
        "uncertainty_reduction": uncertainty_reductions,
        "suggestion_shift": suggestion_shifts,
        "best_by_voi": [voi_values.index(max(voi_values))],
    }


def get_whatif_summary(report: WhatIfReport) -> str:
    """Generate human-readable summary of what-if analysis.

    Args:
        report: WhatIfReport to summarize.

    Returns:
        Formatted string summary.
    """
    lines = [
        "=== What-If Analysis Report ===",
        "",
        "Hypothetical Result:",
        f"  Parameters: {report.hypothetical.parameters}",
        f"  Objectives: {report.hypothetical.objective_values}",
        "",
        f"Value of Information: {report.value_of_information:.4f}",
    ]

    if report.model_impact:
        lines.append("")
        lines.append("Model Impact:")
        lines.append(f"  Uncertainty reduction: {report.model_impact.uncertainty_reduction:.4f}")
        lines.append(f"  Noise variance change: {report.model_impact.noise_variance_change:.4f}")

    if report.suggestion_impact:
        lines.append("")
        lines.append("Suggestion Impact:")
        lines.append(f"  Suggestion shift: {report.suggestion_impact.suggestion_shift:.4f}")
        lines.append(f"  Diversity change: {report.suggestion_impact.diversity_change:.4f}")

    if report.pareto_impact:
        lines.append("")
        lines.append("Pareto Impact:")
        lines.append(f"  Is Pareto-optimal: {report.pareto_impact.hypothetical_is_pareto}")
        lines.append(f"  Hypervolume change: {report.pareto_impact.hypervolume_change:.4f}")

    lines.append("")
    lines.append(f"Recommendation: {report.recommendation}")

    if report.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in report.warnings:
            lines.append(f"  - {w}")

    return "\n".join(lines)


def _compute_model_impact(
    train_x: Tensor,
    train_y: Tensor,
    aug_x: Tensor,
    aug_y: Tensor,
    bounds: Tensor,
    use_input_warping: bool,
) -> ModelImpact:
    """Compute impact on model parameters."""
    # Fit original model
    orig_model = create_and_fit_single_task_model(
        train_x=train_x,
        train_y=train_y.squeeze(-1) if train_y.shape[1] == 1 else train_y[:, 0],
        bounds=bounds,
        use_input_warping=use_input_warping,
    )

    # Fit augmented model
    aug_model = create_and_fit_single_task_model(
        train_x=aug_x,
        train_y=aug_y.squeeze(-1) if aug_y.shape[1] == 1 else aug_y[:, 0],
        bounds=bounds,
        use_input_warping=use_input_warping,
    )

    # Extract lengthscales
    orig_ls = orig_model.covar_module.base_kernel.lengthscale.detach().squeeze()  # ty: ignore[call-non-callable, unresolved-attribute]
    aug_ls = aug_model.covar_module.base_kernel.lengthscale.detach().squeeze()  # ty: ignore[call-non-callable, unresolved-attribute]

    if orig_ls.dim() == 0:
        orig_ls = orig_ls.unsqueeze(0)
        aug_ls = aug_ls.unsqueeze(0)

    ls_changes = {f"param_{i}": (aug_ls[i] - orig_ls[i]).item() for i in range(len(orig_ls))}

    # Noise variance
    orig_noise = orig_model.likelihood.noise.item()  # ty: ignore[call-non-callable]
    aug_noise = aug_model.likelihood.noise.item()  # ty: ignore[call-non-callable]
    noise_change = aug_noise - orig_noise

    # Prediction changes (sample at random points)
    n_test = 10
    sobol = torch.quasirandom.SobolEngine(dimension=train_x.shape[1], scramble=True)
    test_x = sobol.draw(n_test).to(device=train_x.device, dtype=train_x.dtype)
    test_x = bounds[0] + test_x * (bounds[1] - bounds[0])

    with torch.no_grad():
        orig_pred = orig_model.posterior(test_x).mean.squeeze()
        aug_pred = aug_model.posterior(test_x).mean.squeeze()
        orig_var = orig_model.posterior(test_x).variance.squeeze()
        aug_var = aug_model.posterior(test_x).variance.squeeze()

    pred_changes = {f"point_{i}": (aug_pred[i] - orig_pred[i]).item() for i in range(n_test)}

    # Average uncertainty reduction
    uncertainty_reduction = (orig_var.mean() - aug_var.mean()).item()

    return ModelImpact(
        lengthscale_changes=ls_changes,
        noise_variance_change=noise_change,
        prediction_changes=pred_changes,
        uncertainty_reduction=uncertainty_reduction,
    )


def _compute_suggestion_impact(
    train_x: Tensor,
    train_y: Tensor,
    aug_x: Tensor,
    aug_y: Tensor,
    bounds: Tensor,
    n_suggestions: int,
    parameter_names: list[str],
) -> SuggestionImpact:
    """Compute impact on next suggestions."""
    # Create specs
    param_specs = [
        ParameterSpec(
            name=name,
            type=ParameterType.CONTINUOUS,
            bounds=(bounds[0, i].item(), bounds[1, i].item()),
        )
        for i, name in enumerate(parameter_names)
    ]

    obj_specs = [ObjectiveSpec(name="obj", minimize=True)]

    spec = OptimizationSpec(
        parameters=param_specs,
        objectives=obj_specs,
    )

    # Convert to observations
    def to_observations(x: Tensor, y: Tensor) -> list[ObservationData]:
        obs = []
        for i in range(x.shape[0]):
            params = {name: x[i, j].item() for j, name in enumerate(parameter_names)}
            objs = {"obj": y[i, 0].item()}
            obs.append(ObservationData(parameter_values=params, objective_values=objs))
        return obs

    orig_obs = to_observations(train_x, train_y)
    aug_obs = to_observations(aug_x, aug_y)

    # Generate suggestions
    orig_sugg_result, _ = generate_next_batch(spec, orig_obs, n_suggestions)
    aug_sugg_result, _ = generate_next_batch(spec, aug_obs, n_suggestions)

    orig_params = [s.parameter_values for s in orig_sugg_result]
    aug_params = [s.parameter_values for s in aug_sugg_result]

    # Compute shift
    shift = 0.0
    for i in range(min(len(orig_params), len(aug_params))):
        for name in parameter_names:
            if name in orig_params[i] and name in aug_params[i]:
                shift += abs(orig_params[i][name] - aug_params[i][name])

    # Compute diversity change
    orig_div = _compute_diversity(orig_params, parameter_names, bounds)
    aug_div = _compute_diversity(aug_params, parameter_names, bounds)
    div_change = aug_div - orig_div

    # Acquisition values (if available)
    acq_values = [
        s.acquisition_value if s.acquisition_value is not None else 0.0 for s in aug_sugg_result
    ]

    return SuggestionImpact(
        original_suggestions=orig_params,
        new_suggestions=aug_params,
        suggestion_shift=shift,
        new_acquisition_values=acq_values,
        diversity_change=div_change,
    )


def _compute_pareto_impact(
    train_y: Tensor,
    aug_y: Tensor,
) -> ParetoImpact:
    """Compute impact on Pareto front."""
    # Get original Pareto front
    orig_pareto_mask = _is_pareto_efficient(train_y)
    orig_pareto_size = orig_pareto_mask.sum().item()

    # Get augmented Pareto front
    aug_pareto_mask = _is_pareto_efficient(aug_y)
    new_pareto_size = aug_pareto_mask.sum().item()

    # Check if hypothetical is Pareto-optimal
    hyp_is_pareto = aug_pareto_mask[-1].item()

    # Find dominated points
    dominated = []
    for i in range(train_y.shape[0]):
        if orig_pareto_mask[i] and not aug_pareto_mask[i]:
            dominated.append(i)

    # Compute hypervolume change
    ref_point = train_y.max(dim=0).values + 0.1 * (
        train_y.max(dim=0).values - train_y.min(dim=0).values
    )

    orig_hv = compute_hypervolume(train_y[orig_pareto_mask], ref_point)
    aug_hv = compute_hypervolume(aug_y[aug_pareto_mask], ref_point)
    hv_change = aug_hv - orig_hv

    return ParetoImpact(
        original_pareto_size=int(orig_pareto_size),
        new_pareto_size=int(new_pareto_size),
        hypothetical_is_pareto=bool(hyp_is_pareto),
        dominated_points=dominated,
        hypervolume_change=hv_change,
    )


def _is_pareto_efficient(objectives: Tensor) -> Tensor:
    """Check Pareto efficiency (assuming minimization)."""
    n = objectives.shape[0]
    is_efficient = torch.ones(n, dtype=torch.bool, device=objectives.device)

    for i in range(n):
        for j in range(n):
            if i != j:
                # j dominates i if j <= i in all objectives and j < i in at least one
                if torch.all(objectives[j] <= objectives[i]) and torch.any(
                    objectives[j] < objectives[i]
                ):
                    is_efficient[i] = False
                    break

    return is_efficient


def _compute_diversity(
    suggestions: list[dict[str, float]],
    parameter_names: list[str],
    bounds: Tensor,
) -> float:
    """Compute diversity score for suggestions."""
    if len(suggestions) < 2:
        return 1.0

    device = bounds.device
    dtype = bounds.dtype

    # Convert to tensor
    n = len(suggestions)
    d = len(parameter_names)
    X = torch.zeros(n, d, device=device, dtype=dtype)

    for i, s in enumerate(suggestions):
        for j, name in enumerate(parameter_names):
            X[i, j] = s.get(name, 0.0)

    # Normalize
    ranges = bounds[1] - bounds[0]
    ranges = torch.clamp(ranges, min=1e-8)
    x_norm = (X - bounds[0]) / ranges

    # Compute pairwise distances
    dists = torch.cdist(x_norm, x_norm)
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)

    return dists[mask].mean().item()


def _estimate_value_of_information(
    model_impact: ModelImpact | None,
    suggestion_impact: SuggestionImpact | None,
    pareto_impact: ParetoImpact | None,
) -> float:
    """Estimate the value of information from a hypothetical result."""
    voi = 0.0

    if model_impact:
        # Uncertainty reduction is valuable
        voi += model_impact.uncertainty_reduction * 0.5

    if suggestion_impact:
        # Significant suggestion shift indicates high information
        voi += suggestion_impact.suggestion_shift * 0.1

    if pareto_impact:
        # Hypervolume improvement is valuable
        voi += pareto_impact.hypervolume_change * 0.3
        # Being Pareto-optimal is valuable
        if pareto_impact.hypothetical_is_pareto:
            voi += 0.2

    return max(0.0, voi)


def _generate_whatif_recommendation(
    voi: float,
) -> str:
    """Generate actionable recommendation based on analysis."""
    if voi > 0.5:
        return (
            "This experiment would provide high information value. "
            "Strongly recommend running this experiment."
        )
    elif voi > 0.2:
        return (
            "This experiment would provide moderate information value. "
            "Consider running if resources permit."
        )
    elif voi > 0.1:
        return (
            "This experiment would provide low information value. "
            "May not significantly advance optimization."
        )
    else:
        return (
            "This experiment would provide minimal information value. "
            "Consider alternative experiments."
        )
