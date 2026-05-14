"""Converters between bo-engine types and BayBE types.

Maps OptimizationSpec, ObservationData, and related types to the
BayBE equivalents (SearchSpace, Objective, Recommender, DataFrame).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from baybe.constraints import (
    ContinuousLinearConstraint,
    DiscreteProductConstraint,
    DiscreteSumConstraint,
)
from baybe.constraints.conditions import ThresholdCondition
from baybe.objectives import ParetoObjective, SingleTargetObjective
from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
    SubstanceParameter,
    TaskParameter,
)
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from bo_engine.types import (
    ConstraintSpec,
    ConstraintType,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)

from bo_engine_baybe.options import (
    BayBEParameterEncoding,
    BayBEParameterOptions,
    BayBEParameterRole,
    extract_baybe_parameter_options,
)

# Maps neutral ConstraintType to the BayBE continuous-linear operator string.
_CONTINUOUS_OPERATOR_MAP: dict[ConstraintType, str] = {
    ConstraintType.SUM_EQUALS: "=",
    ConstraintType.SUM_LESS_THAN: "<=",
    ConstraintType.SUM_GREATER_THAN: ">=",
    ConstraintType.LINEAR: "<=",
}

# Maps neutral ConstraintType to the BayBE ThresholdCondition operator used
# inside DiscreteSumConstraint / DiscreteProductConstraint.
_DISCRETE_OPERATOR_MAP: dict[ConstraintType, str] = {
    ConstraintType.SUM_EQUALS: "=",
    ConstraintType.SUM_LESS_THAN: "<=",
    ConstraintType.SUM_GREATER_THAN: ">=",
}


def _build_baybe_parameter(
    p: ParameterSpec,
) -> NumericalContinuousParameter | NumericalDiscreteParameter | CategoricalParameter:
    """Build a single BayBE parameter, honoring typed BayBE parameter options."""
    opts = extract_baybe_parameter_options(p.parameter_options)
    if p.type == ParameterType.CONTINUOUS:
        if p.bounds is None:
            msg = f"Continuous parameter '{p.name}' requires bounds"
            raise ValueError(msg)
        return NumericalContinuousParameter(
            name=p.name,
            bounds=(p.bounds[0], p.bounds[1]),
        )
    if p.type == ParameterType.DISCRETE:
        if p.values is None:
            msg = f"Discrete parameter '{p.name}' requires values"
            raise ValueError(msg)
        return NumericalDiscreteParameter(p.name, tuple(float(v) for v in p.values))
    if p.type == ParameterType.CATEGORICAL:
        if p.categories is None:
            msg = f"Categorical parameter '{p.name}' requires categories"
            raise ValueError(msg)
        return _build_categorical_parameter(p, opts)
    msg = f"Unsupported parameter type: {p.type}"
    raise ValueError(msg)


def _build_categorical_parameter(
    p: ParameterSpec,
    opts: BayBEParameterOptions,
) -> Any:
    """Build a BayBE categorical-family parameter, dispatching on the role.

    ``role=task`` produces a :class:`TaskParameter`; ``role=substance``
    produces a :class:`SubstanceParameter` with the user-provided SMILES
    map; otherwise a vanilla :class:`CategoricalParameter` is returned with
    the requested encoding. Constructor calls go through BayBE's attrs
    ``alias="values"`` convention, which the type checker cannot follow —
    the explicit ``ty: ignore`` keeps the cast localized.
    """
    assert p.categories is not None
    categories = tuple(p.categories)
    if opts.role == BayBEParameterRole.TASK:
        active = tuple(opts.active_values) if opts.active_values else categories
        return TaskParameter(p.name, categories, active_values=active)  # ty: ignore[unknown-argument]
    if opts.role == BayBEParameterRole.SUBSTANCE:
        if not opts.substance_data:
            msg = (
                f"BayBE substance parameter '{p.name}' requires "
                "parameter_options['baybe'].substance_data"
            )
            raise ValueError(msg)
        substance_encoding = (
            opts.substance_encoding.value if opts.substance_encoding is not None else "MORDRED"
        )
        return SubstanceParameter(
            name=p.name,
            data=dict(opts.substance_data),
            encoding=substance_encoding,  # ty: ignore[invalid-argument-type]
        )
    encoding = (
        opts.encoding.value if opts.encoding is not None else BayBEParameterEncoding.OHE.value
    )
    return CategoricalParameter(
        p.name,
        categories,
        encoding=encoding,  # ty: ignore[parameter-already-assigned, invalid-argument-type]
    )


def spec_to_parameters(
    spec: OptimizationSpec,
) -> list[NumericalContinuousParameter | NumericalDiscreteParameter | CategoricalParameter]:
    """Convert bo-engine ParameterSpecs to BayBE parameter objects.

    Honors ``parameter_options['baybe']`` to emit BayBE-native parameter
    classes (categorical encoding, ``TaskParameter`` for transfer learning,
    ``SubstanceParameter`` for cheminformatics descriptors). Parameters
    without BayBE options fall back to the previous BoTorch-shaped
    numeric/categorical mapping.
    """
    return [_build_baybe_parameter(p) for p in spec.parameters]


def spec_to_searchspace(spec: OptimizationSpec) -> SearchSpace:
    """Convert OptimizationSpec to BayBE SearchSpace."""
    parameters = spec_to_parameters(spec)
    constraints = (
        spec_to_constraints(spec.constraints, spec.parameters) if spec.constraints else None
    )
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


def classify_constraint_target(
    constraint: ConstraintSpec,
    parameters: list[ParameterSpec],
) -> str:
    """Classify a constraint as ``continuous`` / ``discrete`` / ``hybrid``.

    Used by both the BayBE converter and ``validate_capabilities`` so the
    selection logic and the construction logic agree on what BayBE can
    express. Returns ``"unknown"`` whenever **any** referenced parameter
    is missing from ``parameters`` — a mix of known and unknown names is
    just as fatal during ``SearchSpace.from_product`` as an entirely
    unknown reference, so both must surface the same way.
    """
    by_name = {p.name: p for p in parameters}
    missing = [name for name in constraint.parameters if name not in by_name]
    if missing:
        return "unknown"
    types = {by_name[name].type for name in constraint.parameters}
    if not types:
        return "unknown"
    if ParameterType.CATEGORICAL in types:
        return "categorical"
    if types == {ParameterType.CONTINUOUS}:
        return "continuous"
    if types == {ParameterType.DISCRETE}:
        return "discrete"
    return "hybrid"


def baybe_constraint_support(
    constraint: ConstraintSpec,
    parameters: list[ParameterSpec],
) -> tuple[bool, str | None]:
    """Return ``(is_supported, reason)`` for a BayBE constraint mapping.

    Used by both :func:`spec_to_constraints` and the BayBE backend's
    ``validate_capabilities`` so capability reporting and construction
    agree. The mapping rules:

    * Continuous-only: any supported neutral type (sum_*/linear) maps to
      :class:`ContinuousLinearConstraint`.
    * Discrete-only: sum_equals / sum_less_than / sum_greater_than map to
      :class:`DiscreteSumConstraint`. ``LINEAR`` is **not** supported on
      a discrete-only target because BayBE has no general
      ``DiscreteLinearConstraint`` — pretending the coefficients are
      uniform would silently drop user-supplied weights.
    * Hybrid: not supported by BayBE.
    * Categorical-targeted arithmetic: not supported by BayBE.
    """
    target_class = classify_constraint_target(constraint, parameters)
    if target_class == "continuous":
        return True, None
    if target_class == "discrete" and constraint.type in _DISCRETE_OPERATOR_MAP:
        return True, None
    return False, _constraint_unsupported_reason(target_class, constraint, parameters)


def _constraint_unsupported_reason(
    target_class: str,
    constraint: ConstraintSpec,
    parameters: list[ParameterSpec],
) -> str:
    """Build the user-facing rejection reason for an unsupported constraint."""
    if target_class == "discrete":
        return (
            f"BayBE has no native discrete equivalent for {constraint.type.value} over "
            f"numerical-discrete parameters {sorted(constraint.parameters)}"
        )
    if target_class == "hybrid":
        return (
            f"BayBE cannot express constraint over mixed continuous/discrete "
            f"parameters {sorted(constraint.parameters)}"
        )
    if target_class == "categorical":
        return (
            f"BayBE cannot express arithmetic constraint over categorical "
            f"parameters {sorted(constraint.parameters)}"
        )
    declared = {p.name for p in parameters}
    missing = sorted(set(constraint.parameters) - declared)
    if missing:
        return f"Constraint references unknown parameters {missing}"
    return f"Constraint references unknown parameters {sorted(constraint.parameters)}"


def spec_to_constraints(
    constraints: list[ConstraintSpec],
    parameters: list[ParameterSpec] | None = None,
) -> list[Any] | None:
    """Convert bo-engine ConstraintSpecs to BayBE constraints.

    Continuous linear constraints (over only continuous parameters) map to
    :class:`ContinuousLinearConstraint`. Numerical-discrete sum/product
    constraints map to BayBE :class:`DiscreteSumConstraint` /
    :class:`DiscreteProductConstraint` so finite mixture grids and
    integer-sum constraints are expressed natively instead of being
    forced through the continuous mapping. Hybrid, categorical, and
    discrete-LINEAR constraints raise ``ValueError`` — BayBE has no
    matching native construct and silently dropping the constraint
    would let an "auto" selection produce a SearchSpace BayBE cannot
    construct.
    """
    if parameters is None:
        parameters = []
    baybe_constraints: list[Any] = []
    for c in constraints:
        if not parameters:
            baybe_constraints.append(_build_continuous_constraint(c))
            continue
        ok, reason = baybe_constraint_support(c, parameters)
        if not ok:
            raise ValueError(reason or "BayBE cannot express constraint")
        target_class = classify_constraint_target(c, parameters)
        if target_class == "discrete":
            baybe_constraints.append(_build_discrete_constraint(c))
        else:
            baybe_constraints.append(_build_continuous_constraint(c))

    return baybe_constraints if baybe_constraints else None


def _build_continuous_constraint(c: ConstraintSpec) -> ContinuousLinearConstraint:
    """Build a BayBE ContinuousLinearConstraint from a neutral ConstraintSpec."""
    if c.type == ConstraintType.LINEAR:
        if c.coefficients is None:
            msg = "Linear constraint requires coefficients"
            raise ValueError(msg)
        coefficients = c.coefficients
    else:
        coefficients = c.coefficients or [1.0] * len(c.parameters)
    operator = _CONTINUOUS_OPERATOR_MAP[c.type]
    return ContinuousLinearConstraint(
        parameters=c.parameters,
        coefficients=coefficients,
        rhs=c.value,
        operator=operator,
    )


def _build_discrete_constraint(
    c: ConstraintSpec,
) -> DiscreteSumConstraint | DiscreteProductConstraint:
    """Build a BayBE discrete sum/product constraint with a ThresholdCondition.

    BayBE's ``ThresholdCondition`` requires an explicit ``tolerance`` for
    equality-style operators (``=``/``==``/``!=``); for ordering operators
    (``<``/``<=``/``>``/``>=``) the tolerance must remain ``None`` or BayBE
    raises during validation. The default ``1e-8`` matches BayBE's own
    convention for equality constraints over numerical-discrete grids.
    """
    operator = _DISCRETE_OPERATOR_MAP[c.type]
    tolerance: float | None = 1e-8 if operator in {"=", "==", "!="} else None
    threshold = ThresholdCondition(
        threshold=float(c.value),
        operator=operator,
        tolerance=tolerance,
    )
    if (
        c.type == ConstraintType.SUM_EQUALS
        or c.type == ConstraintType.SUM_LESS_THAN
        or c.type == ConstraintType.SUM_GREATER_THAN
    ):
        return DiscreteSumConstraint(parameters=list(c.parameters), condition=threshold)
    return DiscreteProductConstraint(parameters=list(c.parameters), condition=threshold)


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


def pending_points_to_dataframe(
    pending: list[dict[str, Any]],
    spec: OptimizationSpec,
) -> pd.DataFrame:
    """Convert pending-suggestion parameter dicts to a BayBE pending-experiments DataFrame.

    BayBE's ``Campaign.recommend(pending_experiments=...)`` expects a
    DataFrame in experimental representation containing exactly the
    parameter columns BayBE knows about. Stripping out objective columns
    (which may leak in if a caller reuses observation dicts) and missing
    parameter columns keeps the input shape consistent with
    ``validate_parameter_input`` in BayBE.
    """
    param_names = [p.name for p in spec.parameters]
    rows: list[dict[str, Any]] = []
    for entry in pending:
        if not isinstance(entry, dict):
            msg = f"Pending point entries must be dicts; got {type(entry).__name__}"
            raise TypeError(msg)
        missing = [name for name in param_names if name not in entry]
        if missing:
            msg = f"Pending point is missing parameter columns required by BayBE: {missing}"
            raise ValueError(msg)
        rows.append({name: entry[name] for name in param_names})
    return pd.DataFrame(rows, columns=param_names)


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
