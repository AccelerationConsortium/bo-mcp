"""Automatic method selection for Bayesian Optimization.

This module provides transparent method selection based on problem characteristics.
It analyzes the optimization spec and returns the optimal BO methods with
human-readable explanations.
"""

from dataclasses import dataclass, field

from bo_engine.constants import (
    HIGH_DIMENSION_WARNING_THRESHOLD,
    MIN_DATA_PARAM_MULTIPLIER,
)
from bo_engine.types import AcquisitionMethod, OptimizationSpec, ParameterType


@dataclass
class MethodSelection:
    """Transparent method selection result.

    Contains the selected methods along with explanations, confidence levels,
    and alternative approaches for user transparency.
    """

    model_type: str
    acquisition_function: str
    optimization_strategy: str
    input_transforms: list[str] = field(default_factory=list)
    explanation: str = ""
    confidence: str = "high"  # "high", "medium", "low"
    alternatives: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _select_acquisition(
    spec: OptimizationSpec,
    n_objectives: int,
) -> tuple[str, list[dict[str, str]]]:
    """Select acquisition function and alternatives."""
    alternatives: list[dict[str, str]] = []
    if spec.acquisition_method != AcquisitionMethod.AUTO:
        return spec.acquisition_method.value, alternatives
    if n_objectives == 1:
        alternatives.append(
            {
                "acquisition": AcquisitionMethod.EXPECTED_IMPROVEMENT.value,
                "reason": "Use if observations are noiseless",
            }
        )
        return AcquisitionMethod.NOISY_EI.value, alternatives
    alternatives.append(
        {
            "acquisition": AcquisitionMethod.SCALARIZED_MULTI_OBJ.value,
            "reason": "Better diversity in Pareto front exploration",
        }
    )
    return AcquisitionMethod.HYPERVOLUME_IMPROVEMENT.value, alternatives


def _select_strategy(
    n_observations: int,
    n_objectives: int,
    is_high_dim: bool,
    warnings: list[str],
) -> str:
    """Select optimization strategy."""
    if n_observations == 0:
        return "Sobol sequence (initial design)"
    if is_high_dim:
        if n_objectives > 1:
            warnings.append(
                "TuRBO is designed for single-objective optimization. "
                "Using standard L-BFGS-B instead for multi-objective."
            )
            return "L-BFGS-B with random restarts"
        return "TuRBO (Trust Region)"
    return "L-BFGS-B with random restarts"


def _assess_confidence(
    n_observations: int,
    n_parameters: int,
    warnings: list[str],
) -> str:
    """Assess confidence level and add data-related warnings."""
    if n_observations == 0:
        return "high"
    if n_observations < MIN_DATA_PARAM_MULTIPLIER * n_parameters:
        warnings.append(
            f"Only {n_observations} observations with {n_parameters} parameters. "
            "Model predictions will improve with more data."
        )
        return "medium"
    return "high"


def select_methods(spec: OptimizationSpec, n_observations: int) -> MethodSelection:
    """Automatically select optimal BO methods based on problem structure.

    Analyzes the optimization specification and number of observations to
    determine the best model type, acquisition function, and optimization
    strategy. Returns a MethodSelection with transparent explanations.

    Args:
        spec: The optimization specification defining the problem.
        n_observations: Number of observations already collected.

    Returns:
        MethodSelection with chosen methods and explanations.
    """
    n_objectives = spec.n_objectives
    n_parameters = spec.n_parameters
    has_categorical = any(p.type == ParameterType.CATEGORICAL for p in spec.parameters)
    is_high_dim = n_parameters > HIGH_DIMENSION_WARNING_THRESHOLD

    warnings: list[str] = []
    model_type = "SingleTaskGP" if n_objectives == 1 else "ModelListGP"
    acquisition_function, alternatives = _select_acquisition(spec, n_objectives)
    optimization_strategy = _select_strategy(n_observations, n_objectives, is_high_dim, warnings)
    confidence = _assess_confidence(n_observations, n_parameters, warnings)

    input_transforms = ["Normalize (scale inputs to [0,1])"]
    if has_categorical:
        input_transforms.append("One-hot encoding (categorical parameters)")
    if spec.use_input_warping:
        input_transforms.append("Kumaraswamy CDF warping (non-stationary)")
    input_transforms.append("Standardize (normalize outputs)")

    explanation = _build_explanation(
        model_type=model_type,
        acquisition_function=acquisition_function,
        optimization_strategy=optimization_strategy,
        n_objectives=n_objectives,
        n_parameters=n_parameters,
        n_observations=n_observations,
        is_high_dim=is_high_dim,
    )

    return MethodSelection(
        model_type=model_type,
        acquisition_function=acquisition_function,
        optimization_strategy=optimization_strategy,
        input_transforms=input_transforms,
        explanation=explanation,
        confidence=confidence,
        alternatives=alternatives,
        warnings=warnings,
    )


def _build_explanation(
    model_type: str,
    acquisition_function: str,
    optimization_strategy: str,
    n_objectives: int,
    n_parameters: int,
    n_observations: int,
    is_high_dim: bool,
) -> str:
    """Build human-readable explanation for method selection."""
    parts = []

    # Objective type explanation
    if n_objectives == 1:
        parts.append(
            f"Your problem has **1 objective**, so we're using single-objective "
            f"optimization with {acquisition_function}."
        )
    else:
        parts.append(
            f"Your problem has **{n_objectives} objectives**, so we're using "
            f"multi-objective optimization to find the Pareto-optimal trade-offs."
        )

    # Model explanation
    model_explanations = {
        "SingleTaskGP": (
            "A Gaussian Process model that learns the relationship between "
            "your parameters and objective."
        ),
        "ModelListGP": (
            "Separate GP models for each objective, allowing independent "
            "learning of each response surface."
        ),
    }
    if model_type in model_explanations:
        parts.append(f"\n\n**Model**: {model_explanations[model_type]}")

    # Data status
    if n_observations == 0:
        parts.append(
            "\n\nSince you have no observations yet, we'll generate an initial "
            "space-filling design using Sobol sequences."
        )
    elif n_observations < MIN_DATA_PARAM_MULTIPLIER * n_parameters:
        parts.append(
            f"\n\nWith {n_observations} observations and {n_parameters} parameters, "
            "the model is still learning. Suggestions balance exploration and exploitation."
        )
    else:
        parts.append(
            f"\n\nWith {n_observations} observations, the model has good coverage. "
            "Suggestions focus on exploiting promising regions."
        )

    # High-dimensional note
    if is_high_dim:
        parts.append(
            f"\n\n**Note**: With {n_parameters} parameters, this is a high-dimensional "
            "problem. Consider using TuRBO (Trust Region BO) for better scaling."
        )

    return "".join(parts)
