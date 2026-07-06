"""Converters between bo-engine types and BayBE types.

Maps OptimizationSpec, ObservationData, and related types to the
BayBE equivalents (SearchSpace, Objective, Recommender, DataFrame).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
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
    CustomDiscreteParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
    SubstanceParameter,
    TaskParameter,
)
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget

from bo_engine.constants import DISCRETE_ENUMERATION_MAX_POINTS
from bo_engine.spec_ir import (
    ConstraintTargetClass,
    NormalizedConstraint,
    NormalizedSpec,
    normalize_spec,
)
from bo_engine.spec_ir import (
    classify_constraint_target as _classify_constraint_target,
)
from bo_engine.transforms import (
    count_discrete_combinations,
    discrete_enumeration_limit_error,
)
from bo_engine.types import (
    AcquisitionMethod,
    ConstraintSpec,
    ConstraintType,
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)
from bo_engine_baybe.options import (
    DEFAULT_SUBSTANCE_ENCODING,
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

# Acquisition methods that have no equivalent in BayBE's BotorchRecommender.
# The capability layer (BayBEBackend._option_reports) turns these into
# UNSUPPORTED/acknowledgeable reports; spec_to_acquisition_function falls
# back to BayBE's default so an acknowledged run still produces suggestions.
BAYBE_UNSUPPORTED_ACQUISITION: frozenset[AcquisitionMethod] = frozenset(
    {
        AcquisitionMethod.COST_WEIGHTED_EI,
        AcquisitionMethod.MULTI_FIDELITY_KG,
    }
)

# Maps the neutral AcquisitionMethod to the BayBE acquisition-function names
# accepted by ``BotorchRecommender(acquisition_function=...)``. The dispatch
# mirrors ``bo_engine.acquisition.create_acquisition``: the objective count
# decides the acquisition family, and the requested method is consulted
# within that family (a multi-objective-only method on a single-objective
# spec resolves to the noisy-EI default, exactly like the BoTorch backend).
_SINGLE_OBJECTIVE_ACQF_MAP: dict[AcquisitionMethod, str] = {
    AcquisitionMethod.EXPECTED_IMPROVEMENT: "qLogEI",
    AcquisitionMethod.NOISY_EI: "qLogNEI",
    AcquisitionMethod.HYPERVOLUME_IMPROVEMENT: "qLogNEI",
    AcquisitionMethod.SCALARIZED_MULTI_OBJ: "qLogNEI",
}
_MULTI_OBJECTIVE_ACQF_MAP: dict[AcquisitionMethod, str] = {
    AcquisitionMethod.EXPECTED_IMPROVEMENT: "qLogNEHVI",
    AcquisitionMethod.NOISY_EI: "qLogNEHVI",
    AcquisitionMethod.HYPERVOLUME_IMPROVEMENT: "qLogNEHVI",
    AcquisitionMethod.SCALARIZED_MULTI_OBJ: "qLogNParEGO",
}


def spec_to_acquisition_function(spec: OptimizationSpec) -> str | None:
    """Resolve ``spec.acquisition_method`` to a BayBE acquisition-function name.

    Returns ``None`` when BayBE should pick its own default: the user chose
    ``AUTO`` (no explicit preference) or a method BayBE cannot express
    (``BAYBE_UNSUPPORTED_ACQUISITION`` — surfaced separately through the
    capability reports). Explicit, mappable choices resolve through the
    objective-count-aware tables above so the user's selection is honored
    instead of silently overridden by ``BotorchRecommender()``'s default.
    """
    method = spec.acquisition_method
    if method == AcquisitionMethod.AUTO or method in BAYBE_UNSUPPORTED_ACQUISITION:
        return None
    table = _SINGLE_OBJECTIVE_ACQF_MAP if spec.n_objectives == 1 else _MULTI_OBJECTIVE_ACQF_MAP
    return table[method]


def _integer_grid_from_bounds(p: ParameterSpec) -> tuple[float, ...]:
    """Materialize the integer grid a bounds-only discrete parameter spans."""
    if p.bounds is None:  # pragma: no cover - caller guarantees bounds
        msg = f"Discrete parameter '{p.name}' has no bounds"
        raise ValueError(msg)
    lo, hi = math.ceil(p.bounds[0]), math.floor(p.bounds[1])
    n_points = hi - lo + 1
    if n_points < 2:
        msg = (
            f"Discrete parameter '{p.name}' bounds {p.bounds} contain fewer "
            "than 2 integer values; declare explicit values instead"
        )
        raise ValueError(msg)
    if n_points > DISCRETE_ENUMERATION_MAX_POINTS:
        msg = (
            f"Discrete parameter '{p.name}' bounds {p.bounds} span {n_points} "
            f"integer values, above the enumeration limit of "
            f"{DISCRETE_ENUMERATION_MAX_POINTS}; declare explicit values or "
            "use a continuous parameter"
        )
        raise ValueError(msg)
    return tuple(float(v) for v in range(lo, hi + 1))


def _build_baybe_parameter(
    p: ParameterSpec,
) -> (
    NumericalContinuousParameter
    | NumericalDiscreteParameter
    | CategoricalParameter
    | TaskParameter
    | SubstanceParameter
    | CustomDiscreteParameter
):
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
        if p.values is not None:  # noqa: PD011
            return NumericalDiscreteParameter(p.name, tuple(float(v) for v in p.values))  # noqa: PD011
        if p.bounds is not None:
            # Neutral-spec semantics (mirrors BoTorch): bounds-only discrete
            # means an integer grid over [lo, hi]. BayBE needs the grid
            # materialized, so cap it at the shared enumeration limit.
            return NumericalDiscreteParameter(p.name, _integer_grid_from_bounds(p))
        msg = f"Discrete parameter '{p.name}' requires values or bounds"
        raise ValueError(msg)
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
) -> CategoricalParameter | TaskParameter | SubstanceParameter | CustomDiscreteParameter:
    """Build a BayBE categorical-family parameter, dispatching on the role.

    ``role=task`` produces a :class:`TaskParameter`; ``role=substance``
    produces a :class:`SubstanceParameter` with the user-provided SMILES
    map; ``role=custom`` produces a :class:`CustomDiscreteParameter` from
    the user-supplied per-label descriptor table; otherwise a vanilla
    :class:`CategoricalParameter` is returned with the requested encoding.
    """
    if p.categories is None:
        msg = f"Categorical parameter '{p.name}' requires categories"
        raise ValueError(msg)
    categories = tuple(p.categories)
    if opts.role == BayBEParameterRole.TASK:
        active = tuple(opts.active_values) if opts.active_values else categories
        return TaskParameter(p.name, categories, active_values=active)
    if opts.role == BayBEParameterRole.CUSTOM:
        return _build_custom_parameter(p, opts)
    if opts.role == BayBEParameterRole.SUBSTANCE:
        if not opts.substance_data:
            msg = (
                f"BayBE substance parameter '{p.name}' requires "
                "parameter_options['baybe'].substance_data"
            )
            raise ValueError(msg)
        substance_encoding = (
            opts.substance_encoding
            if opts.substance_encoding is not None
            else DEFAULT_SUBSTANCE_ENCODING
        )
        return SubstanceParameter(
            name=p.name,
            data=dict(opts.substance_data),
            encoding=substance_encoding.value,
        )
    encoding = (
        opts.encoding.value if opts.encoding is not None else BayBEParameterEncoding.OHE.value
    )
    return CategoricalParameter(
        p.name,
        categories,
        encoding=encoding,  # ty: ignore[parameter-already-assigned]
    )


def _build_custom_parameter(
    p: ParameterSpec,
    opts: BayBEParameterOptions,
) -> CustomDiscreteParameter:
    """Build a BayBE ``CustomDiscreteParameter`` from the per-label descriptor table.

    ``custom_descriptors`` maps each category label to a dict of
    ``{descriptor name: value}``; ``DataFrame.from_dict(orient="index")``
    turns it into the labels × descriptors table BayBE consumes (its
    ``values`` are the DataFrame index). Capability validation has already
    checked category coverage and construction validity, so a malformed
    table cannot reach this point through the normal intake path.
    """
    if not opts.custom_descriptors:
        msg = (
            f"BayBE custom parameter '{p.name}' requires "
            "parameter_options['baybe'].custom_descriptors"
        )
        raise ValueError(msg)
    data = pd.DataFrame.from_dict(opts.custom_descriptors, orient="index")
    return CustomDiscreteParameter(name=p.name, data=data, decorrelate=opts.decorrelate)


def spec_to_parameters(
    spec: OptimizationSpec,
) -> list[
    NumericalContinuousParameter
    | NumericalDiscreteParameter
    | CategoricalParameter
    | TaskParameter
    | SubstanceParameter
    | CustomDiscreteParameter
]:
    """Convert bo-engine ParameterSpecs to BayBE parameter objects.

    Honors ``parameter_options['baybe']`` to emit BayBE-native parameter
    classes (categorical encoding, ``TaskParameter`` for transfer learning,
    ``SubstanceParameter`` for cheminformatics descriptors,
    ``CustomDiscreteParameter`` for user-supplied representations). Parameters
    without BayBE options fall back to the previous BoTorch-shaped
    numeric/categorical mapping.
    """
    return [_build_baybe_parameter(p) for p in spec.parameters]


def spec_to_searchspace(spec: OptimizationSpec) -> SearchSpace:
    """Convert OptimizationSpec to BayBE SearchSpace.

    The BayBE constraint dispatch (continuous vs. discrete vs. unsupported)
    runs once via :func:`bo_engine.spec_ir.normalize_spec`; the normalized
    bundle is then handed to :func:`spec_to_constraints` so capability
    reporting and construction share a single classification result.

    ``SearchSpace.from_product`` materializes the Cartesian product of all
    discrete/categorical parameters into an experimental-representation
    DataFrame, so the shared enumeration limit is enforced on the *product*
    before any parameter is built — per-parameter checks alone would let two
    just-under-limit grids multiply into an unbuildable frame.
    """
    n_combinations = count_discrete_combinations(spec)
    if n_combinations > DISCRETE_ENUMERATION_MAX_POINTS:
        raise discrete_enumeration_limit_error(n_combinations)
    parameters = spec_to_parameters(spec)
    if spec.constraints:
        normalized = normalize_spec(spec)
        constraints = spec_to_constraints(normalized)
    else:
        constraints = None
    return SearchSpace.from_product(parameters=parameters, constraints=constraints)


def log_transform_maximize_reason(objective_name: str) -> str:
    """Reason why ``log_transform=True`` + ``minimize=False`` is rejected.

    Shared between the runtime rejection in :func:`_build_baybe_target`
    and the backend's capability reporting so intake-time validation and
    suggestion-time construction can never drift apart on this contract.
    """
    return (
        f"log_transform=True requires minimize=True for objective "
        f"'{objective_name}'; the maximize combination is outside the "
        "supported contract. Disable log_transform or restate the "
        "objective in minimization form."
    )


def _build_baybe_target(o: ObjectiveSpec) -> NumericalTarget:
    """Build one BayBE target, honoring the per-objective ``log_transform`` flag.

    ``log_transform=True`` chains BayBE's logarithmic target transformation
    onto the target so acquisition improvements are measured on the log
    scale of multi-decade objectives. The neutral contract restricts the
    flag to ``minimize=True`` objectives (see
    :class:`bo_engine.types.ObjectiveSpec`); the maximize combination is
    rejected here with the same failure mode as the BoTorch model factory.
    Note that BayBE applies target transformations as part of the
    acquisition objective, not as a surrogate outcome transform — the GP
    itself still fits the raw target scale, which the backend surfaces as
    a ``DEGRADED`` capability report.
    """
    target = NumericalTarget(name=o.name, minimize=o.minimize)
    if not o.log_transform:
        return target
    if not o.minimize:
        raise ValueError(log_transform_maximize_reason(o.name))
    return target.log()


def spec_to_objective(
    spec: OptimizationSpec,
) -> SingleTargetObjective | ParetoObjective:
    """Convert OptimizationSpec to BayBE Objective.

    Single-objective specs produce SingleTargetObjective.
    Multi-objective specs produce ParetoObjective (uses qLogNEHVI internally).
    """
    targets = [_build_baybe_target(o) for o in spec.objectives]

    if len(targets) == 1:
        return SingleTargetObjective(target=targets[0])
    return ParetoObjective(targets)  # ty: ignore[invalid-argument-type]


def classify_constraint_target(
    constraint: ConstraintSpec,
    parameters: list[ParameterSpec],
) -> ConstraintTargetClass:
    """Classify a constraint as ``continuous`` / ``discrete`` / ``hybrid``.

    Thin wrapper around :func:`bo_engine.spec_ir.classify_constraint_target`
    so capability reporting and converter construction share a single
    dispatch implementation. The return value is a
    :class:`~bo_engine.spec_ir.ConstraintTargetClass` (a :class:`StrEnum`),
    so callers comparing against bare strings (``"continuous"`` etc.) keep
    working.
    """
    return _classify_constraint_target(constraint, parameters)


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
    if target_class == ConstraintTargetClass.CONTINUOUS:
        return True, None
    if target_class == ConstraintTargetClass.DISCRETE and constraint.type in _DISCRETE_OPERATOR_MAP:
        return True, None
    return False, _constraint_unsupported_reason(target_class, constraint, parameters)


def _constraint_unsupported_reason(
    target_class: ConstraintTargetClass,
    constraint: ConstraintSpec,
    parameters: list[ParameterSpec],
) -> str:
    """Build the user-facing rejection reason for an unsupported constraint."""
    if target_class == ConstraintTargetClass.DISCRETE:
        return (
            f"BayBE has no native discrete equivalent for {constraint.type.value} over "
            f"numerical-discrete parameters {sorted(constraint.parameters)}"
        )
    if target_class == ConstraintTargetClass.HYBRID:
        return (
            f"BayBE cannot express constraint over mixed continuous/discrete "
            f"parameters {sorted(constraint.parameters)}"
        )
    if target_class == ConstraintTargetClass.CATEGORICAL:
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
    constraints: list[ConstraintSpec] | NormalizedSpec,
    parameters: list[ParameterSpec] | None = None,
) -> list[Any] | None:
    """Convert bo-engine ConstraintSpecs to BayBE constraints.

    Two call shapes are accepted to keep the public surface stable while
    letting backend converters consume the shared IR directly:

    * ``spec_to_constraints(normalized_spec)`` — preferred. The
      :class:`~bo_engine.spec_ir.NormalizedSpec` carries the per-constraint
      classification already, so no dispatch re-runs.
    * ``spec_to_constraints(constraints, parameters)`` — legacy. The
      function normalizes internally; equivalent to the above and kept
      for direct callers.

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
    normalized_pairs, declared_params = _resolve_normalized(constraints, parameters)
    if not normalized_pairs:
        return None

    baybe_constraints: list[Any] = []
    for nc in normalized_pairs:
        c = nc.spec
        if not declared_params:
            # Parameter list unknown to the caller — fall back to the
            # continuous mapping, mirroring historical behavior.
            baybe_constraints.append(_build_continuous_constraint(c))
            continue
        ok, reason = _support_from_normalized(nc)
        if not ok:
            raise ValueError(
                reason or _constraint_unsupported_reason(nc.target_class, c, declared_params)
            )
        if nc.target_class == ConstraintTargetClass.DISCRETE:
            baybe_constraints.append(_build_discrete_constraint(c))
        else:
            baybe_constraints.append(_build_continuous_constraint(c))

    return baybe_constraints if baybe_constraints else None


def _resolve_normalized(
    constraints: list[ConstraintSpec] | NormalizedSpec,
    parameters: list[ParameterSpec] | None,
) -> tuple[tuple[NormalizedConstraint, ...], list[ParameterSpec]]:
    """Coerce either call shape into ``(normalized_constraints, parameters)``."""
    if isinstance(constraints, NormalizedSpec):
        return constraints.constraints, list(constraints.spec.parameters)
    declared = parameters if parameters is not None else []
    normalized = tuple(
        NormalizedConstraint(spec=c, target_class=_classify_constraint_target(c, declared))
        for c in constraints
    )
    return normalized, declared


def _support_from_normalized(nc: NormalizedConstraint) -> tuple[bool, str | None]:
    """``baybe_constraint_support`` answer derived from a normalized constraint.

    Avoids re-running classification when the caller already has a
    :class:`NormalizedConstraint`. Kept private — public callers should
    use :func:`baybe_constraint_support`.
    """
    if nc.target_class == ConstraintTargetClass.CONTINUOUS:
        return True, None
    if nc.target_class == ConstraintTargetClass.DISCRETE and nc.spec.type in _DISCRETE_OPERATOR_MAP:
        return True, None
    return False, None


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


def _assert_log_transform_targets_valid(
    df: pd.DataFrame,
    spec: OptimizationSpec,
) -> None:
    """Raise ``ValueError`` when a ``log_transform`` objective has invalid targets.

    The logarithmic target transformation operates on the raw observation
    values, so non-finite or non-positive targets would surface as an
    opaque NaN cascade inside the BayBE/BoTorch fit. Enforcing positivity
    at the conversion boundary mirrors the BoTorch model factory's
    construction-time check so both backends fail with the same clear
    envelope.
    """
    for obj in spec.objectives:
        if not obj.log_transform:
            continue
        values = pd.to_numeric(df[obj.name], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            msg = (
                f"log_transform=True requires finite targets; objective "
                f"'{obj.name}' has NaN or inf values. Drop or impute those "
                "rows before fitting."
            )
            raise ValueError(msg)
        if not (values > 0).all():
            min_value = float(values.min())
            msg = (
                f"log_transform=True requires strictly positive targets for "
                f"objective '{obj.name}'; got min={min_value}. Either drop "
                "non-positive observations, pre-shift the target, or disable "
                "log_transform for this objective."
            )
            raise ValueError(msg)


def observations_to_dataframe(
    observations: list[ObservationData],
    spec: OptimizationSpec,
) -> pd.DataFrame:
    """Convert ObservationData list to a pandas DataFrame for BayBE.

    The DataFrame has columns for all parameters and all objectives.
    Targets of ``log_transform`` objectives are validated for finiteness
    and strict positivity at this boundary (see
    :func:`_assert_log_transform_targets_valid`).
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

    df = pd.DataFrame(rows)
    if not df.empty:
        _assert_log_transform_targets_valid(df, spec)
    return df


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
