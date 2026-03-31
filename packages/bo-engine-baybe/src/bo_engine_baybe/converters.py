"""Converters between bo-engine types and BayBE types.

Maps OptimizationSpec, ObservationData, and related types to the
BayBE equivalents (SearchSpace, Objective, Recommender, DataFrame).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from baybe.objectives import ParetoObjective, SingleTargetObjective
from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.recommenders import (
    BotorchRecommender,
    RandomRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from bo_engine.types import (
    ConstraintSpec,
    ConstraintType,
    ObservationData,
    OptimizationSpec,
    ParameterType,
)


def spec_to_parameters(
    spec: OptimizationSpec,
) -> list[NumericalContinuousParameter | NumericalDiscreteParameter | CategoricalParameter]:
    """Convert bo-engine ParameterSpecs to BayBE parameter objects."""
    params: list[Any] = []
    for p in spec.parameters:
        if p.type == ParameterType.CONTINUOUS:
            if p.bounds is None:
                msg = f"Continuous parameter '{p.name}' requires bounds"
                raise ValueError(msg)
            params.append(
                NumericalContinuousParameter(
                    name=p.name,
                    bounds=(p.bounds[0], p.bounds[1]),
                )
            )
        elif p.type == ParameterType.DISCRETE:
            if p.values is None:
                msg = f"Discrete parameter '{p.name}' requires values"
                raise ValueError(msg)
            params.append(NumericalDiscreteParameter(p.name, tuple(p.values)))
        elif p.type == ParameterType.CATEGORICAL:
            if p.categories is None:
                msg = f"Categorical parameter '{p.name}' requires categories"
                raise ValueError(msg)
            params.append(CategoricalParameter(p.name, tuple(p.categories)))
        else:
            msg = f"Unsupported parameter type: {p.type}"
            raise ValueError(msg)
    return params


def spec_to_searchspace(spec: OptimizationSpec) -> SearchSpace:
    """Convert OptimizationSpec to BayBE SearchSpace."""
    parameters = spec_to_parameters(spec)
    constraints = spec_to_constraints(spec.constraints) if spec.constraints else None
    return SearchSpace.from_product(parameters=parameters, constraints=constraints)


def spec_to_objective(
    spec: OptimizationSpec,
) -> SingleTargetObjective | ParetoObjective:
    """Convert OptimizationSpec to BayBE Objective.

    Single-objective specs produce SingleTargetObjective.
    Multi-objective specs produce ParetoObjective (uses qLogNEHVI internally).
    """
    targets = [NumericalTarget(name=o.name, minimize=o.minimize) for o in spec.objectives]

    if len(targets) == 1:
        return SingleTargetObjective(target=targets[0])
    return ParetoObjective(targets)  # ty: ignore[invalid-argument-type]


def spec_to_recommender() -> TwoPhaseMetaRecommender:
    """Create default BayBE Recommender.

    Uses TwoPhaseMetaRecommender: RandomRecommender for initial design
    (works for all space types), then BotorchRecommender for model-guided
    suggestions.
    """
    return TwoPhaseMetaRecommender(
        initial_recommender=RandomRecommender(),
        recommender=BotorchRecommender(),
    )


def spec_to_constraints(constraints: list[ConstraintSpec]) -> list[Any] | None:
    """Convert bo-engine ConstraintSpecs to BayBE constraints.

    Supports continuous linear constraints (SUM_EQUALS, SUM_LESS_THAN,
    SUM_GREATER_THAN, LINEAR).
    """
    from baybe.constraints import ContinuousLinearConstraint

    baybe_constraints: list[Any] = []
    for c in constraints:
        if c.type == ConstraintType.SUM_EQUALS:
            coefficients = c.coefficients or [1.0] * len(c.parameters)
            baybe_constraints.append(
                ContinuousLinearConstraint(
                    parameters=c.parameters,
                    coefficients=coefficients,
                    rhs=c.value,
                    operator="=",
                )
            )
        elif c.type == ConstraintType.SUM_LESS_THAN:
            coefficients = c.coefficients or [1.0] * len(c.parameters)
            baybe_constraints.append(
                ContinuousLinearConstraint(
                    parameters=c.parameters,
                    coefficients=coefficients,
                    rhs=c.value,
                    operator="<=",
                )
            )
        elif c.type == ConstraintType.SUM_GREATER_THAN:
            coefficients = c.coefficients or [1.0] * len(c.parameters)
            baybe_constraints.append(
                ContinuousLinearConstraint(
                    parameters=c.parameters,
                    coefficients=coefficients,
                    rhs=c.value,
                    operator=">=",
                )
            )
        elif c.type == ConstraintType.LINEAR:
            if c.coefficients is None:
                msg = "Linear constraint requires coefficients"
                raise ValueError(msg)
            baybe_constraints.append(
                ContinuousLinearConstraint(
                    parameters=c.parameters,
                    coefficients=c.coefficients,
                    rhs=c.value,
                    operator="<=",
                )
            )

    return baybe_constraints if baybe_constraints else None


def observations_to_dataframe(
    observations: list[ObservationData],
    spec: OptimizationSpec,
) -> pd.DataFrame:
    """Convert ObservationData list to a pandas DataFrame for BayBE.

    The DataFrame has columns for all parameters and all objectives.
    """
    param_names = [p.name for p in spec.parameters]
    obj_names = [o.name for o in spec.objectives]

    rows: list[dict[str, Any]] = []
    for obs in observations:
        row: dict[str, Any] = {}
        for name in param_names:
            row[name] = obs.parameter_values.get(name)
        for name in obj_names:
            row[name] = obs.objective_values.get(name)
        rows.append(row)

    return pd.DataFrame(rows)


def dataframe_to_suggestions(
    df: pd.DataFrame,
    spec: OptimizationSpec,
) -> list[dict[str, Any]]:
    """Convert a BayBE recommendation DataFrame to parameter-value dicts."""
    param_names = [p.name for p in spec.parameters]
    suggestions: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        params: dict[str, Any] = {}
        for name in param_names:
            val = row[name]
            if pd.notna(val):
                params[name] = float(val) if isinstance(val, (int, float)) else val
        suggestions.append(params)
    return suggestions
