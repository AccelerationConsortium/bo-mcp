"""Converters between bo-engine types and BayBE types.

Maps OptimizationSpec, ObservationData, and related types to the
BayBE equivalents (SearchSpace, Objective, Recommender, DataFrame).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from baybe.acquisition import qNIPV, qUCB
from baybe.acquisition.base import AcquisitionFunction as BayBEAcquisitionFunction
from baybe.constraints import (
    ContinuousCardinalityConstraint,
    ContinuousLinearConstraint,
    DiscreteCardinalityConstraint,
    DiscreteLinkedParametersConstraint,
    DiscreteNoLabelDuplicatesConstraint,
    DiscretePermutationInvarianceConstraint,
    DiscreteProductConstraint,
    DiscreteSumConstraint,
)
from baybe.constraints.base import Constraint as BayBEConstraint
from baybe.constraints.base import ContinuousConstraint as BayBEContinuousConstraint
from baybe.constraints.conditions import ThresholdCondition
from baybe.objectives import DesirabilityObjective, ParetoObjective, SingleTargetObjective
from baybe.parameters import (
    CategoricalParameter,
    CustomDiscreteParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
    SubstanceParameter,
    TaskParameter,
)
from baybe.parameters.base import DiscreteParameter as BayBEDiscreteParameter
from baybe.searchspace import SearchSpace, SubspaceContinuous
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
from bo_engine.types import (
    PRODUCT_CONSTRAINT_TYPES,
    SET_BASED_CONSTRAINT_TYPES,
    UCB_FAMILY_ACQUISITION,
    AcquisitionMethod,
    ConstraintSpec,
    ConstraintType,
    MatchShape,
    ObjectiveSpec,
    ObjectiveTransformKind,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
    ScalarizationMode,
    ScalarizerKind,
    TargetMode,
)
from bo_engine_baybe.constants import DEFAULT_NIPV_SAMPLING_POINTS
from bo_engine_baybe.options import (
    DEFAULT_SUBSTANCE_ENCODING,
    BayBEParameterEncoding,
    BayBEParameterOptions,
    BayBEParameterRole,
    extract_baybe_parameter_options,
)
from bo_engine_baybe.searchspace_budget import (
    DiscreteSpaceBudget,
    build_subsampled_discrete_subspace,
    cheap_budget_prescreen,
    resolve_discrete_space_budget,
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
    ConstraintType.PRODUCT_EQUALS: "=",
    ConstraintType.PRODUCT_LESS_THAN: "<=",
    ConstraintType.PRODUCT_GREATER_THAN: ">=",
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
# (abbreviations) accepted by ``BotorchRecommender(acquisition_function=...)``.
# The dispatch mirrors ``bo_engine.acquisition.create_acquisition``: the
# objective count decides the acquisition family, and the requested method is
# consulted within that family (a multi-objective-only method on a
# single-objective spec resolves to the noisy-EI default, and a
# single-objective-only method on a multi-objective spec resolves to the
# hypervolume default, exactly like the BoTorch backend).
_SINGLE_OBJECTIVE_ACQF_MAP: dict[AcquisitionMethod, str] = {
    AcquisitionMethod.EXPECTED_IMPROVEMENT: "qLogEI",
    AcquisitionMethod.NOISY_EI: "qLogNEI",
    AcquisitionMethod.HYPERVOLUME_IMPROVEMENT: "qLogNEI",
    AcquisitionMethod.SCALARIZED_MULTI_OBJ: "qLogNEI",
    AcquisitionMethod.UPPER_CONFIDENCE_BOUND: "qUCB",
    AcquisitionMethod.PROBABILITY_OF_IMPROVEMENT: "qPI",
    AcquisitionMethod.SIMPLE_REGRET: "qSR",
    AcquisitionMethod.POSTERIOR_MEAN: "PM",
    AcquisitionMethod.POSTERIOR_STANDARD_DEVIATION: "qPSTD",
    AcquisitionMethod.THOMPSON_SAMPLING: "qTS",
    AcquisitionMethod.KNOWLEDGE_GRADIENT: "qKG",
    AcquisitionMethod.ACTIVE_LEARNING: "qNIPV",
    AcquisitionMethod.EXPECTED_IMPROVEMENT_NONLOG: "qEI",
    AcquisitionMethod.NOISY_EI_NONLOG: "qNEI",
    # A non-log hypervolume request on a single-objective spec keeps the
    # caller's non-log intent within the single-objective family.
    AcquisitionMethod.HYPERVOLUME_IMPROVEMENT_NONLOG: "qNEI",
}
_MULTI_OBJECTIVE_ACQF_MAP: dict[AcquisitionMethod, str] = {
    AcquisitionMethod.EXPECTED_IMPROVEMENT: "qLogNEHVI",
    AcquisitionMethod.NOISY_EI: "qLogNEHVI",
    AcquisitionMethod.HYPERVOLUME_IMPROVEMENT: "qLogNEHVI",
    AcquisitionMethod.SCALARIZED_MULTI_OBJ: "qLogNParEGO",
    # Single-objective-family members resolve to the hypervolume default —
    # BayBE's ParetoObjective supports only hypervolume / ParEGO acqfs, so
    # honoring e.g. UCB verbatim would fail inside the recommender. The
    # family-default fallback mirrors the pre-existing EI → qLogNEHVI rule.
    AcquisitionMethod.UPPER_CONFIDENCE_BOUND: "qLogNEHVI",
    AcquisitionMethod.PROBABILITY_OF_IMPROVEMENT: "qLogNEHVI",
    AcquisitionMethod.SIMPLE_REGRET: "qLogNEHVI",
    AcquisitionMethod.POSTERIOR_MEAN: "qLogNEHVI",
    AcquisitionMethod.POSTERIOR_STANDARD_DEVIATION: "qLogNEHVI",
    AcquisitionMethod.THOMPSON_SAMPLING: "qLogNEHVI",
    AcquisitionMethod.KNOWLEDGE_GRADIENT: "qLogNEHVI",
    AcquisitionMethod.ACTIVE_LEARNING: "qLogNEHVI",
    AcquisitionMethod.EXPECTED_IMPROVEMENT_NONLOG: "qNEHVI",
    AcquisitionMethod.NOISY_EI_NONLOG: "qNEHVI",
    AcquisitionMethod.HYPERVOLUME_IMPROVEMENT_NONLOG: "qNEHVI",
}


def acquisition_output_count(spec: OptimizationSpec) -> int:
    """Number of outputs the built BayBE objective exposes to the acquisition.

    Desirability scalarizes every declared objective into a single figure
    of merit (:class:`baybe.objectives.DesirabilityObjective` is a
    single-output objective), so acquisition dispatch must resolve
    through the *single*-objective family — BayBE rejects hypervolume /
    ParEGO acquisition functions for a scalarized objective. Shared
    between :func:`spec_to_acquisition_function` and the backend's
    acquisition capability reports so intake classification and
    construction can never disagree on the family.
    """
    if spec.n_objectives > 1 and spec.scalarization == ScalarizationMode.DESIRABILITY:
        return 1
    return spec.n_objectives


def spec_to_acquisition_function(
    spec: OptimizationSpec,
) -> str | BayBEAcquisitionFunction | None:
    """Resolve ``spec.acquisition_method`` to a BayBE acquisition function.

    Returns ``None`` when BayBE should pick its own default: the user chose
    ``AUTO`` (no explicit preference) or a method BayBE cannot express
    (``BAYBE_UNSUPPORTED_ACQUISITION`` — surfaced separately through the
    capability reports). Explicit, mappable choices resolve through the
    objective-count-aware tables above — with the objective count taken
    from :func:`acquisition_output_count`, so a desirability spec
    (single scalarized output) resolves through the single-objective
    family — and the user's selection is honored instead of silently
    overridden by ``BotorchRecommender()``'s default. Most methods
    resolve to a BayBE abbreviation string; a UCB request with an
    explicit ``acquisition_beta`` builds a :class:`baybe.acquisition.qUCB`
    instance so the exploration weight actually reaches BoTorch.
    """
    method = spec.acquisition_method
    if method == AcquisitionMethod.AUTO or method in BAYBE_UNSUPPORTED_ACQUISITION:
        return None
    if spec.acquisition_beta is not None and method not in UCB_FAMILY_ACQUISITION:
        # The capability layer reports this combination UNSUPPORTED at
        # intake; the converter enforces the same contract so the two
        # surfaces cannot drift.
        msg = (
            f"acquisition_beta={spec.acquisition_beta} is only valid for the "
            f"UCB acquisition family; got method={method.value!r}."
        )
        raise ValueError(msg)
    table = (
        _SINGLE_OBJECTIVE_ACQF_MAP
        if acquisition_output_count(spec) == 1
        else _MULTI_OBJECTIVE_ACQF_MAP
    )
    resolved = table[method]
    if resolved == "qUCB" and spec.acquisition_beta is not None:
        return qUCB(beta=spec.acquisition_beta)
    if resolved == "qNIPV" and _is_purely_continuous(spec):
        # BayBE's qNIPV integrates the posterior over sampled points; its
        # sampling_fraction default only applies to enumerable discrete
        # candidates, and purely continuous spaces require an explicit
        # integration-point budget.
        return qNIPV(sampling_n_points=DEFAULT_NIPV_SAMPLING_POINTS)  # ty: ignore[missing-argument]
    return resolved


def _is_purely_continuous(spec: OptimizationSpec) -> bool:
    """True when the spec declares only continuous parameters."""
    return all(p.type == ParameterType.CONTINUOUS for p in spec.parameters)


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
        tolerance = opts.tolerance if opts.tolerance is not None else 0.0
        if p.values is not None:  # noqa: PD011
            return NumericalDiscreteParameter(
                p.name,
                tuple(float(v) for v in p.values),  # noqa: PD011
                tolerance=tolerance,
            )
        if p.bounds is not None:
            # Neutral-spec semantics (mirrors BoTorch): bounds-only discrete
            # means an integer grid over [lo, hi]. BayBE needs the grid
            # materialized, so cap it at the shared enumeration limit.
            return NumericalDiscreteParameter(
                p.name, _integer_grid_from_bounds(p), tolerance=tolerance
            )
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
        substance_kwargs: dict[str, Any] = {}
        if opts.kwargs_fingerprint is not None:
            substance_kwargs["kwargs_fingerprint"] = dict(opts.kwargs_fingerprint)
        if opts.kwargs_conformer is not None:
            substance_kwargs["kwargs_conformer"] = dict(opts.kwargs_conformer)
        if opts.active_values:
            substance_kwargs["active_values"] = tuple(opts.active_values)
        return SubstanceParameter(
            name=p.name,
            data=dict(opts.substance_data),
            encoding=substance_encoding.value,
            **substance_kwargs,
        )
    encoding = (
        opts.encoding.value if opts.encoding is not None else BayBEParameterEncoding.OHE.value
    )
    return CategoricalParameter(
        p.name,
        categories,
        encoding=encoding,  # ty: ignore[parameter-already-assigned]
        active_values=tuple(opts.active_values) if opts.active_values else None,
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
    return CustomDiscreteParameter(
        name=p.name,
        data=data,
        decorrelate=opts.decorrelate,
        active_values=tuple(opts.active_values) if opts.active_values else None,
    )


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


def resolve_searchspace_budget(spec: OptimizationSpec) -> DiscreteSpaceBudget:
    """Resolve the discrete-subspace size budget for ``spec``.

    Consumed by the backend's warning surfacing and capability reports so
    they agree with the construction branch in :func:`spec_to_searchspace`.
    The construction-free pre-screen resolves ordinarily-sized specs
    without building BayBE parameter objects (substance descriptor tables
    are expensive and their construction errors belong to the
    parameter-role validators, not this probe); only specs the upper
    bounds cannot prove "below budget" fall through to the precise,
    parameter-built estimate.
    """
    prescreen = cheap_budget_prescreen(spec)
    if prescreen is not None:
        return prescreen
    parameters = spec_to_parameters(spec)
    discrete_parameters: list[BayBEDiscreteParameter] = [
        p for p in parameters if isinstance(p, BayBEDiscreteParameter)
    ]
    return resolve_discrete_space_budget(spec, discrete_parameters)


def spec_to_searchspace(
    spec: OptimizationSpec,
    observations: list[ObservationData] | None = None,
    pending_points: list[dict[str, Any]] | None = None,
) -> SearchSpace:
    """Convert OptimizationSpec to BayBE SearchSpace.

    The BayBE constraint dispatch (continuous vs. discrete vs. unsupported)
    runs once via :func:`bo_engine.spec_ir.normalize_spec`; the normalized
    bundle is then handed to :func:`spec_to_constraints` so capability
    reporting and construction share a single classification result.

    ``SearchSpace.from_product`` materializes the Cartesian product of all
    discrete/categorical parameters, so the discrete subspace is budgeted
    first (:func:`resolve_discrete_space_budget`): below budget the
    historical ``from_product`` path is unchanged byte-for-byte; above it,
    the discrete subspace is built from a deterministic, bounded candidate
    sample via direct ``SubspaceDiscrete`` construction over the sampled
    frame (the large-categorical safeguard; see
    :func:`bo_engine_baybe.searchspace_budget.build_subsampled_discrete_subspace`
    for why ``from_dataframe`` is bypassed). ``observations`` /
    ``pending_points`` are unioned into the sampled candidates so the
    per-candidate metadata ledger keeps the
    ``allow_recommending_already_measured/_recommended=False`` contracts
    working on the subsampled space.
    """
    parameters = spec_to_parameters(spec)
    if spec.constraints:
        normalized = normalize_spec(spec)
        constraints = spec_to_constraints(normalized)
    else:
        constraints = None

    discrete_parameters: list[BayBEDiscreteParameter] = [
        p for p in parameters if isinstance(p, BayBEDiscreteParameter)
    ]
    budget = resolve_discrete_space_budget(spec, discrete_parameters)
    if not budget.exceeds_budget:
        return SearchSpace.from_product(parameters=parameters, constraints=constraints)

    discrete_subspace = build_subsampled_discrete_subspace(
        spec,
        discrete_parameters,
        budget,
        baybe_constraints=constraints,
        observations=observations,
        pending_points=pending_points,
    )
    continuous_parameters = [p for p in parameters if not isinstance(p, BayBEDiscreteParameter)]
    continuous_constraints = [
        c for c in (constraints or []) if isinstance(c, BayBEContinuousConstraint)
    ]
    continuous_subspace = SubspaceContinuous.from_product(
        continuous_parameters,
        continuous_constraints or None,
    )
    return SearchSpace(discrete=discrete_subspace, continuous=continuous_subspace)


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


def log_transform_match_reason(objective_name: str) -> str:
    """Reason why ``log_transform=True`` + ``target_mode='match'`` is rejected.

    A match target is built from the ``NumericalTarget.match_*``
    constructors, which have no slot for a chained log transformation —
    accepting the flag would silently drop it. Shared between intake
    validation and the backend's legacy warning surface so the two can
    never drift apart on this contract.
    """
    return (
        f"log_transform cannot be combined with target_mode='match' on objective '{objective_name}'"
    )


def objective_support_issues(spec: OptimizationSpec) -> list[tuple[str, str]]:
    """Return ``(key, reason)`` pairs for objective configurations BayBE rejects.

    Shared between :func:`spec_to_objective` (construction) and the BayBE
    backend's ``validate_capabilities`` so intake-time validation and
    suggestion-time construction can never drift — the
    ``baybe_constraint_support`` precedent applied to objectives.
    """
    issues: list[tuple[str, str]] = []
    for idx, o in enumerate(spec.objectives):
        key = f"objectives[{idx}]"
        issues.extend((key, reason) for reason in _single_objective_issues(o))
    issues.extend(_desirability_issues(spec))
    return issues


def _single_objective_issues(o: ObjectiveSpec) -> list[str]:
    """Validation issues for one objective's match/transform configuration."""
    issues: list[str] = []
    mode = o.effective_mode
    if mode == TargetMode.MATCH:
        if o.target_value is None:
            issues.append(f"target_mode='match' requires target_value on objective '{o.name}'")
        shape = o.match_shape or MatchShape.ABSOLUTE
        if shape in (MatchShape.BELL, MatchShape.TRIANGULAR) and o.match_scale is None:
            issues.append(
                f"match_shape='{shape.value}' requires match_scale on objective "
                f"'{o.name}' (bell: sigma; triangular: total base width)"
            )
        if shape in (MatchShape.ABSOLUTE, MatchShape.QUADRATIC) and o.match_scale is not None:
            issues.append(
                f"match_shape='{shape.value}' takes no match_scale on objective '{o.name}'"
            )
    elif o.match_shape is not None or o.match_scale is not None or o.target_value is not None:
        issues.append(
            f"target_value/match_shape/match_scale require target_mode='match' "
            f"on objective '{o.name}'"
        )
    if o.log_transform and mode == TargetMode.MATCH:
        # Mirrors the typed-transform × match rule below.
        issues.append(log_transform_match_reason(o.name))
    issues.extend(_transform_issues(o))
    return issues


def _transform_issues(o: ObjectiveSpec) -> list[str]:
    """Validation issues for the typed ``transform`` union on one objective.

    Direction rules read ``ObjectiveSpec.effective_mode`` (never the raw
    ``minimize`` boolean) so validation and target construction — which
    resolves direction the same way — can never disagree when a
    ``target_mode`` override is present.
    """
    if o.transform is None:
        return []
    issues: list[str] = []
    if o.log_transform:
        issues.append(
            f"objective '{o.name}' sets both log_transform and transform; use exactly one"
        )
    t = o.transform
    if t.kind == ObjectiveTransformKind.LOG and o.effective_mode == TargetMode.MAXIMIZE:
        issues.append(log_transform_maximize_reason(o.name))
    if t.kind == ObjectiveTransformKind.CLAMP and t.bounds is None:
        issues.append(f"transform kind='clamp' requires bounds on objective '{o.name}'")
    if t.kind == ObjectiveTransformKind.POWER and t.exponent is None:
        issues.append(f"transform kind='power' requires exponent on objective '{o.name}'")
    if t.kind == ObjectiveTransformKind.SIGMOID and (t.center is None or t.steepness is None):
        issues.append(
            f"transform kind='sigmoid' requires center and steepness on objective '{o.name}'"
        )
    if o.effective_mode == TargetMode.MATCH:
        issues.append(
            f"transform cannot be combined with target_mode='match' on objective '{o.name}'"
        )
    return issues


def _desirability_issues(spec: OptimizationSpec) -> list[tuple[str, str]]:
    """Validation issues for the desirability scalarization mode."""
    if spec.scalarization != ScalarizationMode.DESIRABILITY:
        if spec.scalarizer is not None:
            return [("scalarizer", "scalarizer requires scalarization='desirability'")]
        return []
    issues: list[tuple[str, str]] = []
    if spec.n_objectives < 2:
        issues.append(
            ("scalarization", "scalarization='desirability' requires at least 2 objectives")
        )
    for idx, o in enumerate(spec.objectives):
        key = f"objectives[{idx}]"
        issues.extend((key, reason) for reason in _desirability_objective_issues(o))
    return issues


def _desirability_objective_issues(o: ObjectiveSpec) -> list[str]:
    """Per-objective validation issues under desirability scalarization."""
    issues: list[str] = []
    if o.weight is not None and o.weight <= 0:
        issues.append(f"desirability weight must be positive on objective '{o.name}'")
    if o.transform is not None or o.log_transform:
        # The desirability path builds only normalized-ramp / match
        # kernels; a declared transformation has no slot in that
        # construction and accepting it would silently optimize the
        # untransformed target.
        issues.append(
            f"transform/log_transform cannot be combined with "
            f"scalarization='desirability' on objective '{o.name}'; "
            "desirability targets are normalized ramps or match kernels"
        )
    if o.effective_mode == TargetMode.MATCH:
        shape = o.match_shape or MatchShape.ABSOLUTE
        if shape not in (MatchShape.BELL, MatchShape.TRIANGULAR):
            issues.append(
                f"desirability requires normalized targets; match objective "
                f"'{o.name}' must use match_shape='bell' or 'triangular'"
            )
    elif o.normalization_bounds is None:
        issues.append(
            f"desirability requires normalization_bounds on objective '{o.name}' "
            "(the raw-value range mapped onto [0, 1])"
        )
    return issues


def _assert_no_objective_issues(spec: OptimizationSpec) -> None:
    """Raise ``ValueError`` when the objective surface is invalid for BayBE."""
    issues = objective_support_issues(spec)
    if issues:
        msg = "; ".join(reason for _key, reason in issues)
        raise ValueError(msg)


def _build_match_target(o: ObjectiveSpec) -> NumericalTarget:
    """Build a match-a-target-value BayBE target (``NumericalTarget.match_*``).

    Shape semantics follow the BayBE targets userguide
    (https://emdgroup.github.io/baybe/stable/userguide/targets.html):
    ``match_scale`` is the bell's ``sigma`` and the triangle's total base
    ``width``; absolute/quadratic take no scale.
    """
    if o.target_value is None:  # pragma: no cover - guarded by validation
        msg = f"target_mode='match' requires target_value on objective '{o.name}'"
        raise ValueError(msg)
    shape = o.match_shape or MatchShape.ABSOLUTE
    if shape == MatchShape.ABSOLUTE:
        return NumericalTarget.match_absolute(o.name, match_value=o.target_value)
    if shape == MatchShape.QUADRATIC:
        return NumericalTarget.match_quadratic(o.name, match_value=o.target_value)
    if shape == MatchShape.BELL:
        return NumericalTarget.match_bell(o.name, match_value=o.target_value, sigma=o.match_scale)  # ty: ignore[invalid-argument-type]
    if shape == MatchShape.TRIANGULAR:
        return NumericalTarget.match_triangular(
            o.name, match_value=o.target_value, width=o.match_scale
        )
    # Explicit final-branch check: a future MatchShape member must fail
    # loudly here instead of silently routing to the last constructor.
    msg = f"Unhandled match_shape '{shape.value}' on objective '{o.name}'"
    raise ValueError(msg)


def _missing_transform_field_error(o: ObjectiveSpec, field_name: str) -> ValueError:
    """Internal-invariant error: validation should have rejected this spec."""
    return ValueError(f"transform on objective '{o.name}' is missing required field '{field_name}'")


def _apply_transform(target: NumericalTarget, o: ObjectiveSpec) -> NumericalTarget:
    """Chain the typed ``transform`` union onto a plain min/max target."""
    t = o.transform
    if t is None:
        return target
    if t.kind == ObjectiveTransformKind.LOG:
        return target.log()
    if t.kind == ObjectiveTransformKind.CLAMP:
        if t.bounds is None:  # pragma: no cover - guarded by validation
            raise _missing_transform_field_error(o, "bounds")
        return target.clamp(min=t.bounds[0], max=t.bounds[1])
    if t.kind == ObjectiveTransformKind.POWER:
        if t.exponent is None:  # pragma: no cover - guarded by validation
            raise _missing_transform_field_error(o, "exponent")
        return target.power(int(t.exponent))
    if t.kind == ObjectiveTransformKind.SIGMOID:
        return _build_sigmoid_target(o)
    # Explicit final-branch check: a future ObjectiveTransformKind member
    # must fail loudly here instead of silently routing to the sigmoid path.
    msg = f"Unhandled transform kind '{t.kind.value}' on objective '{o.name}'"
    raise ValueError(msg)


def _build_sigmoid_target(o: ObjectiveSpec) -> NumericalTarget:
    """Build a normalized-sigmoid target from ``center`` / ``steepness``.

    The logistic ``1 / (1 + exp(-k (x - c)))`` passes through
    ``(c - 1/k, 1/(1+e))`` and ``(c + 1/k, e/(1+e))``; feeding those two
    anchor points to ``NumericalTarget.normalized_sigmoid`` reconstructs
    exactly the requested curve. ``minimize=True`` flips the direction
    (descending sigmoid), matching the plain-target semantics. Direction
    is resolved through ``effective_mode`` like every other target
    builder, so a ``target_mode`` override wins over the raw ``minimize``
    boolean here too.
    """
    t = o.transform
    if t is None or t.center is None or t.steepness is None:  # pragma: no cover
        raise _missing_transform_field_error(o, "center/steepness")
    lower_y = 1.0 / (1.0 + math.e)
    offset = 1.0 / t.steepness
    return NumericalTarget.normalized_sigmoid(
        o.name,
        anchors=[(t.center - offset, lower_y), (t.center + offset, 1.0 - lower_y)],
        minimize=o.effective_mode == TargetMode.MINIMIZE,
    )


def _build_baybe_target(o: ObjectiveSpec) -> NumericalTarget:
    """Build one BayBE target from the neutral objective spec.

    Plain minimize/maximize objectives keep the historical construction —
    the legacy ``log_transform=True`` boolean produces the identical
    ``NumericalTarget(...).log()`` chain (byte-compatible with stored
    specs). ``target_mode='match'`` dispatches to the
    ``NumericalTarget.match_*`` constructors, and the typed ``transform``
    union chains BayBE's target transformations. Note that BayBE applies
    target transformations as part of the acquisition objective, not as a
    surrogate outcome transform — the GP itself still fits the raw target
    scale, which the backend surfaces as a ``DEGRADED`` capability report.
    """
    if o.effective_mode == TargetMode.MATCH:
        return _build_match_target(o)
    minimize = o.effective_mode == TargetMode.MINIMIZE
    if o.transform is not None and o.transform.kind == ObjectiveTransformKind.SIGMOID:
        return _build_sigmoid_target(o)
    target = NumericalTarget(name=o.name, minimize=minimize)
    if o.transform is not None:
        return _apply_transform(target, o)
    if not o.log_transform:
        return target
    if not minimize:
        raise ValueError(log_transform_maximize_reason(o.name))
    return target.log()


def _build_desirability_target(o: ObjectiveSpec) -> NumericalTarget:
    """Build one normalized target for the desirability scalarization.

    Minimize/maximize objectives become ``normalized_ramp`` targets over
    the declared ``normalization_bounds`` (descending for minimize);
    bell/triangular match targets are normalized kernels already and pass
    through :func:`_build_match_target` unchanged. Mirrors the BayBE
    desirability example
    (https://emdgroup.github.io/baybe/stable/userguide/objectives.html).
    """
    mode = o.effective_mode
    if mode == TargetMode.MATCH:
        return _build_match_target(o)
    if o.normalization_bounds is None:  # pragma: no cover - guarded by validation
        msg = f"desirability requires normalization_bounds on objective '{o.name}'"
        raise ValueError(msg)
    return NumericalTarget.normalized_ramp(
        o.name,
        cutoffs=o.normalization_bounds,
        descending=mode == TargetMode.MINIMIZE,
    )


# Maps the neutral scalarizer kind to BayBE's Scalarizer enum member name.
_SCALARIZER_MAP: dict[ScalarizerKind, str] = {
    ScalarizerKind.MEAN: "MEAN",
    ScalarizerKind.GEOM_MEAN: "GEOM_MEAN",
}


def spec_to_objective(
    spec: OptimizationSpec,
) -> SingleTargetObjective | ParetoObjective | DesirabilityObjective:
    """Convert OptimizationSpec to BayBE Objective.

    Single-objective specs produce SingleTargetObjective. Multi-objective
    specs produce ParetoObjective (default) or DesirabilityObjective when
    ``spec.scalarization == 'desirability'`` — a weighted scalarization of
    normalized targets with the spec-level ``scalarizer`` (BayBE default:
    geometric mean).
    """
    _assert_no_objective_issues(spec)
    if spec.n_objectives > 1 and spec.scalarization == ScalarizationMode.DESIRABILITY:
        targets = [_build_desirability_target(o) for o in spec.objectives]
        weights = [o.weight if o.weight is not None else 1.0 for o in spec.objectives]
        scalarizer = _SCALARIZER_MAP[spec.scalarizer or ScalarizerKind.GEOM_MEAN]
        return DesirabilityObjective(targets, weights=weights, scalarizer=scalarizer)  # ty: ignore[invalid-argument-type]

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


def _referenced_types(
    constraint: ConstraintSpec,
    parameters: list[ParameterSpec],
) -> set[ParameterType]:
    """Parameter types referenced by ``constraint`` (empty when unknown names)."""
    by_name = {p.name: p for p in parameters}
    if any(name not in by_name for name in constraint.parameters):
        return set()
    return {by_name[name].type for name in constraint.parameters}


def _cardinality_issue(constraint: ConstraintSpec) -> str | None:
    """Validate the cardinality bounds against the referenced parameter count."""
    lo = constraint.min_cardinality
    hi = constraint.max_cardinality
    if lo is None and hi is None:
        return "CARDINALITY constraint requires min_cardinality and/or max_cardinality"
    n = len(constraint.parameters)
    if (lo is not None and not 0 <= lo <= n) or (hi is not None and not 0 <= hi <= n):
        return (
            f"CARDINALITY bounds must lie within [0, {n}] for "
            f"{len(constraint.parameters)} referenced parameters"
        )
    if lo is not None and hi is not None and lo > hi:
        return "CARDINALITY requires min_cardinality <= max_cardinality"
    if (lo is None or lo == 0) and (hi is None or hi == n):
        # BayBE's CardinalityConstraint rejects this combination at
        # construction ("No constraint ... is required"): a lower bound of
        # zero with an upper bound equal to the parameter count permits
        # every activation count, so the constraint has no effect.
        return (
            f"CARDINALITY with min_cardinality={lo or 0} and "
            f"max_cardinality={hi if hi is not None else n} over {n} "
            "parameters constrains nothing; tighten a bound or drop the "
            "constraint"
        )
    return None


def _set_based_support(
    constraint: ConstraintSpec,
    parameters: list[ParameterSpec],
) -> tuple[bool, str | None]:
    """Support decision for the label/set-based discrete constraint family.

    BayBE's ``DiscreteNoLabelDuplicates`` / ``DiscreteLinkedParameters`` /
    ``DiscretePermutationInvariance`` constraints operate on the discrete
    subspace, so every referenced parameter must be discrete or
    categorical — a continuous member has no label to compare. Each type
    additionally needs a feasible row over the *effective* value pools
    (``active_values``-restricted where set — BayBE builds the
    categorical product from the active values): linkage requires the
    pools to overlap (:func:`_linked_domain_issue`), while
    no-label-duplicates and permutation invariance require a
    distinct-value assignment to exist
    (:func:`_distinct_assignment_issue` — BayBE's permutation constraint
    also prunes every row with a repeated label).
    """
    types = _referenced_types(constraint, parameters)
    if not types:
        return False, _constraint_unsupported_reason(
            ConstraintTargetClass.UNKNOWN, constraint, parameters
        )
    if ParameterType.CONTINUOUS in types:
        return False, (
            f"{constraint.type.value} constraints operate on discrete/categorical "
            f"parameters only; {sorted(constraint.parameters)} includes a "
            "continuous parameter"
        )
    if len(constraint.parameters) < 2:
        return False, f"{constraint.type.value} requires at least 2 parameters"
    if constraint.type == ConstraintType.LINKED_PARAMETERS:
        issue = _linked_domain_issue(constraint, parameters)
    else:
        issue = _distinct_assignment_issue(constraint, parameters)
    if issue is not None:
        return False, issue
    return True, None


def _effective_member_pool(p: ParameterSpec) -> frozenset[Any] | None:
    """Effective feasibility-value pool of one referenced parameter, or ``None``.

    Numeric-discrete grids compare on float equality; every label-based
    parameter compares on its string labels, restricted to
    ``parameter_options['baybe'].active_values`` when set — BayBE builds
    the categorical/substance/custom product from the *active* values
    (and a task parameter recommends only its active values), so
    feasibility must be judged on the active pool, not the declared one.
    ``None`` means the pool cannot be derived here (malformed member
    declaration) — the parameter validators own that failure, so the
    feasibility checks abstain.
    """
    if p.type == ParameterType.DISCRETE:
        if p.values is not None:  # noqa: PD011 — neutral dataclass field
            return frozenset(float(v) for v in p.values)  # noqa: PD011
        try:
            return frozenset(_integer_grid_from_bounds(p))
        except ValueError:
            return None
    opts = extract_baybe_parameter_options(p.parameter_options)
    if opts.active_values:
        return frozenset(str(v) for v in opts.active_values)
    if not p.categories:
        return None
    return frozenset(str(c) for c in p.categories)


def _member_pools(
    constraint: ConstraintSpec,
    parameters: list[ParameterSpec],
) -> list[frozenset[Any]] | None:
    """Effective pools of every referenced parameter, or ``None`` to abstain."""
    by_name = {p.name: p for p in parameters}
    members = [by_name[name] for name in constraint.parameters if name in by_name]
    if len(members) != len(constraint.parameters):
        return None  # unknown names are rejected by the type check above
    pools = [_effective_member_pool(m) for m in members]
    if any(pool is None for pool in pools):
        return None  # malformed members are rejected by their own validators
    return [pool for pool in pools if pool is not None]


def _has_distinct_assignment(pools: list[frozenset[Any]]) -> bool:
    """True when each pool can take a value no other pool member repeats.

    Exact system-of-distinct-representatives check via backtracking;
    pools are visited smallest-first so infeasible sets prune early. The
    pool count is the constraint's referenced-parameter count, which is
    small by construction.
    """
    ordered = sorted(pools, key=lambda pool: len(pool))
    used: set[Any] = set()

    def assign(index: int) -> bool:
        if index == len(ordered):
            return True
        for value in ordered[index]:
            if value not in used:
                used.add(value)
                if assign(index + 1):
                    return True
                used.discard(value)
        return False

    return assign(0)


def _distinct_assignment_issue(
    constraint: ConstraintSpec,
    parameters: list[ParameterSpec],
) -> str | None:
    """Reject duplicate-forbidding constraints that no candidate row can satisfy.

    ``DiscreteNoLabelDuplicatesConstraint`` drops every row with a
    repeated value, and ``DiscretePermutationInvarianceConstraint``
    additionally prunes repeated-label rows while deduplicating
    permutations (verified against BayBE 0.15.0) — so unless the
    referenced parameters' effective pools admit an all-distinct
    assignment, the discrete subspace filters to zero rows and the first
    suggestion fails. Rejecting here turns that guaranteed failure into
    an intake-time report.
    """
    pools = _member_pools(constraint, parameters)
    if pools is None:
        return None
    if _has_distinct_assignment(pools):
        return None
    return (
        f"{constraint.type.value} requires every referenced parameter to take "
        f"a distinct value in each candidate row, but the effective value "
        f"pools of {sorted(constraint.parameters)} (active_values-restricted "
        "where set) admit no such assignment — every candidate row would be "
        "filtered out. Widen the value lists or active_values, or drop the "
        "constraint"
    )


def _linked_domain_issue(
    constraint: ConstraintSpec,
    parameters: list[ParameterSpec],
) -> str | None:
    """Reject linkage over value domains that can never hold identical values.

    ``DiscreteLinkedParametersConstraint`` keeps only rows whose
    referenced columns hold identical values (verified against BayBE
    0.15.0), so disjoint effective domains — or a mix of numeric grids
    and string labels, which never compare equal in the experimental
    representation — filter the discrete subspace down to zero rows.
    Rejecting here turns a guaranteed empty-space failure at suggestion
    time into an intake-time report.
    """
    by_name = {p.name: p for p in parameters}
    members = [by_name[name] for name in constraint.parameters if name in by_name]
    if len(members) != len(constraint.parameters):
        return None  # unknown names are rejected by the type check above
    families = {"numeric" if m.type == ParameterType.DISCRETE else "label" for m in members}
    if len(families) > 1:
        return (
            f"{constraint.type.value} cannot link numeric grids with "
            f"categorical labels: {sorted(constraint.parameters)} mixes both "
            "families, and a numeric value never equals a string label, so "
            "every candidate row would be filtered out"
        )
    pools = _member_pools(constraint, parameters)
    if pools is None:
        return None
    if not frozenset.intersection(*pools):
        return (
            f"{constraint.type.value} requires the referenced parameters to "
            f"share at least one value in their effective pools "
            f"(active_values-restricted where set); the domains of "
            f"{sorted(constraint.parameters)} are disjoint, so every candidate "
            "row would be filtered out. Align the value lists or drop the "
            "constraint"
        )
    return None


def _cardinality_support(
    constraint: ConstraintSpec,
    target_class: ConstraintTargetClass,
    parameters: list[ParameterSpec],
) -> tuple[bool, str | None]:
    """Support decision for cardinality (sparsity) constraints.

    BayBE offers both :class:`ContinuousCardinalityConstraint` and
    :class:`DiscreteCardinalityConstraint`; categorical labels have no
    numeric zero, so categorical/hybrid targets are rejected. Every
    continuous member's bounds must include zero unconditionally (BayBE
    raises an exception group at search-space build otherwise), and
    whenever the constraint can force a member to zero
    (``max_cardinality`` below the referenced-parameter count) every
    discrete member's grid must contain a zero value (the filtered
    subspace would otherwise be silently empty).
    """
    if target_class not in (ConstraintTargetClass.CONTINUOUS, ConstraintTargetClass.DISCRETE):
        return False, _constraint_unsupported_reason(target_class, constraint, parameters)
    issue = _cardinality_issue(constraint)
    if issue is not None:
        return False, issue
    issue = _cardinality_zero_issue(constraint, parameters)
    if issue is not None:
        return False, issue
    return True, None


def _parameter_grid_contains_zero(p: ParameterSpec) -> bool:
    """True when a discrete parameter's value grid contains zero.

    Bounds-only discrete parameters span the integer grid over
    ``[ceil(lo), floor(hi)]`` (the :func:`_integer_grid_from_bounds`
    contract), which contains zero exactly when the rounded bounds
    straddle it.
    """
    if p.values is not None:  # noqa: PD011 — neutral dataclass field
        # Exact comparison is deliberate: grid values are user-declared
        # literals and BayBE's cardinality semantics count exact zeros.
        return any(float(v) == 0.0 for v in p.values)  # noqa: PD011
    if p.bounds is not None:
        return math.ceil(p.bounds[0]) <= 0 <= math.floor(p.bounds[1])
    return False


def _cardinality_zero_issue(
    constraint: ConstraintSpec,
    parameters: list[ParameterSpec],
) -> str | None:
    """Reject cardinality constraints whose members cannot take the value zero.

    Continuous members must span zero unconditionally:
    ``ContinuousCardinalityConstraint`` validates every member's bounds at
    search-space build regardless of whether the cardinality bounds ever
    force a zero. Discrete members only need a zero grid value when the
    constraint can actually force one (``max_cardinality`` below the
    referenced-parameter count) — BayBE builds the unforced discrete case
    fine.
    """
    hi = constraint.max_cardinality
    can_force_zero = hi is not None and hi < len(constraint.parameters)
    by_name = {p.name: p for p in parameters}
    for name in constraint.parameters:
        p = by_name.get(name)
        if p is None:
            continue  # unknown names are rejected by the target classification
        if p.type == ParameterType.CONTINUOUS:
            if p.bounds is not None and not (p.bounds[0] <= 0.0 <= p.bounds[1]):
                return (
                    "CARDINALITY over continuous parameters requires every "
                    "member's bounds to include zero (BayBE validates the "
                    "bounds at search-space build for any max_cardinality), "
                    f"but '{name}' has bounds {p.bounds}; widen the bounds or "
                    "drop the parameter from the constraint"
                )
        elif can_force_zero and not _parameter_grid_contains_zero(p):
            return (
                f"CARDINALITY with max_cardinality={hi} can force parameter "
                f"'{name}' to zero, but its declared value grid contains no "
                "zero; add 0 to the values or drop the parameter from the "
                "constraint"
            )
    return None


def baybe_constraint_support(
    constraint: ConstraintSpec,
    parameters: list[ParameterSpec],
) -> tuple[bool, str | None]:
    """Return ``(is_supported, reason)`` for a BayBE constraint mapping.

    Used by both :func:`spec_to_constraints` and the BayBE backend's
    ``validate_capabilities`` so capability reporting and construction
    agree. The mapping rules:

    * Continuous-only: sum_*/linear map to
      :class:`ContinuousLinearConstraint` (``is_interpoint`` supported);
      ``CARDINALITY`` maps to :class:`ContinuousCardinalityConstraint`.
    * Discrete-only: sum_* map to :class:`DiscreteSumConstraint`,
      product_* to :class:`DiscreteProductConstraint`, ``CARDINALITY`` to
      :class:`DiscreteCardinalityConstraint`. ``LINEAR`` is **not**
      supported on a discrete-only target because BayBE has no general
      ``DiscreteLinearConstraint`` — pretending the coefficients are
      uniform would silently drop user-supplied weights.
    * Set-based (no_label_duplicates / linked_parameters /
      permutation_invariance): any discrete/categorical parameter set.
    * Hybrid: not supported by BayBE (deliberate exclusion, preserved).
    * Categorical-targeted arithmetic: not supported by BayBE
      (deliberate exclusion, preserved).
    """
    if constraint.type in SET_BASED_CONSTRAINT_TYPES:
        return _set_based_support(constraint, parameters)
    target_class = classify_constraint_target(constraint, parameters)
    if constraint.type == ConstraintType.CARDINALITY:
        return _cardinality_support(constraint, target_class, parameters)
    return _arithmetic_support(constraint, target_class, parameters)


def _arithmetic_support(
    constraint: ConstraintSpec,
    target_class: ConstraintTargetClass,
    parameters: list[ParameterSpec],
) -> tuple[bool, str | None]:
    """Support decision for the arithmetic (sum/product/linear) family."""
    if constraint.is_interpoint and target_class != ConstraintTargetClass.CONTINUOUS:
        return False, (
            "is_interpoint applies to continuous linear/sum constraints only; "
            f"constraint over {sorted(constraint.parameters)} is "
            f"{target_class.value}"
        )
    if target_class == ConstraintTargetClass.CONTINUOUS:
        if constraint.type in PRODUCT_CONSTRAINT_TYPES:
            return False, (
                "product constraints are discrete-only on BayBE "
                "(DiscreteProductConstraint); continuous parameters "
                f"{sorted(constraint.parameters)} cannot be product-constrained"
            )
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
    :class:`ContinuousLinearConstraint`. Numerical-discrete sum
    constraints map to BayBE :class:`DiscreteSumConstraint` so finite
    mixture grids and integer-sum constraints are expressed natively
    instead of being forced through the continuous mapping. Hybrid,
    categorical, and discrete-LINEAR constraints raise ``ValueError`` —
    BayBE has no matching native construct and silently dropping the
    constraint would let an "auto" selection produce a SearchSpace BayBE
    cannot construct.
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
        ok, reason = baybe_constraint_support(c, declared_params)
        if not ok:
            raise ValueError(
                reason or _constraint_unsupported_reason(nc.target_class, c, declared_params)
            )
        baybe_constraints.append(_build_baybe_constraint(c, nc.target_class))

    return baybe_constraints if baybe_constraints else None


def _build_baybe_constraint(
    c: ConstraintSpec,
    target_class: ConstraintTargetClass,
) -> BayBEConstraint:
    """Dispatch one (already support-checked) constraint to its BayBE builder."""
    if c.type in SET_BASED_CONSTRAINT_TYPES:
        return _build_set_based_constraint(c)
    if c.type == ConstraintType.CARDINALITY:
        return _build_cardinality_constraint(c, target_class)
    if target_class == ConstraintTargetClass.DISCRETE:
        if c.type in PRODUCT_CONSTRAINT_TYPES:
            return _build_product_constraint(c)
        return _build_discrete_constraint(c)
    return _build_continuous_constraint(c)


def _build_set_based_constraint(
    c: ConstraintSpec,
) -> (
    DiscreteNoLabelDuplicatesConstraint
    | DiscreteLinkedParametersConstraint
    | DiscretePermutationInvarianceConstraint
):
    """Build a BayBE label/set-relationship constraint.

    See the BayBE constraints userguide
    (https://emdgroup.github.io/baybe/stable/userguide/constraints.html):
    ``DiscreteNoLabelDuplicatesConstraint`` forbids repeated values within
    a row, ``DiscreteLinkedParametersConstraint`` forces identical values,
    and ``DiscretePermutationInvarianceConstraint`` deduplicates
    permutations of the parameter group. The permutation constraint also
    drops rows whose group slots hold *equal* values (verified against
    BayBE 0.15.0: ``(x, x)`` is filtered alongside the reordered
    ``(y, x)`` duplicate of ``(x, y)``) — callers that need "same value
    in two slots" must not combine it with this constraint.
    """
    params = list(c.parameters)
    if c.type == ConstraintType.NO_LABEL_DUPLICATES:
        return DiscreteNoLabelDuplicatesConstraint(parameters=params)
    if c.type == ConstraintType.LINKED_PARAMETERS:
        return DiscreteLinkedParametersConstraint(parameters=params)
    return DiscretePermutationInvarianceConstraint(parameters=params)


def _build_cardinality_constraint(
    c: ConstraintSpec,
    target_class: ConstraintTargetClass,
) -> ContinuousCardinalityConstraint | DiscreteCardinalityConstraint:
    """Build a BayBE sparsity constraint bounding the count of nonzero values."""
    kwargs: dict[str, Any] = {"parameters": list(c.parameters)}
    if c.min_cardinality is not None:
        kwargs["min_cardinality"] = int(c.min_cardinality)
    if c.max_cardinality is not None:
        kwargs["max_cardinality"] = int(c.max_cardinality)
    if target_class == ConstraintTargetClass.CONTINUOUS:
        return ContinuousCardinalityConstraint(**kwargs)
    return DiscreteCardinalityConstraint(**kwargs)


def _build_product_constraint(c: ConstraintSpec) -> DiscreteProductConstraint:
    """Build a BayBE discrete product constraint with a ThresholdCondition.

    Mirrors :func:`_build_discrete_constraint`'s tolerance convention for
    equality-style operators.
    """
    operator = _DISCRETE_OPERATOR_MAP[c.type]
    tolerance: float | None = (
        DISCRETE_EQUALITY_TOLERANCE if operator in _EQUALITY_OPERATORS else None
    )
    threshold = ThresholdCondition(
        threshold=float(c.value),
        operator=operator,
        tolerance=tolerance,
    )
    return DiscreteProductConstraint(parameters=list(c.parameters), condition=threshold)


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


# BayBE ThresholdCondition operators that require an explicit tolerance
# (equality-style); ordering operators must keep tolerance=None or BayBE
# raises during validation.
_EQUALITY_OPERATORS: frozenset[str] = frozenset({"=", "==", "!="})

# Tolerance attached to equality-style ThresholdConditions over
# numerical-discrete grids; matches BayBE's own convention.
DISCRETE_EQUALITY_TOLERANCE = 1e-8


def _build_continuous_constraint(c: ConstraintSpec) -> ContinuousLinearConstraint:
    """Build a BayBE ContinuousLinearConstraint from a neutral ConstraintSpec.

    ``is_interpoint=True`` switches the constraint to BayBE's interpoint
    semantics: the aggregate holds across all points of one recommended
    batch instead of per point (BayBE constraints userguide,
    https://emdgroup.github.io/baybe/stable/userguide/constraints.html).
    """
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
        interpoint=c.is_interpoint,  # ty: ignore[unknown-argument]
    )


def _build_discrete_constraint(
    c: ConstraintSpec,
) -> DiscreteSumConstraint:
    """Build a BayBE discrete sum constraint with a ThresholdCondition.

    Only the three SUM-type neutral constraints reach this builder —
    dispatch routes product types to :func:`_build_product_constraint`.
    """
    operator = _DISCRETE_OPERATOR_MAP[c.type]
    tolerance: float | None = (
        DISCRETE_EQUALITY_TOLERANCE if operator in _EQUALITY_OPERATORS else None
    )
    threshold = ThresholdCondition(
        threshold=float(c.value),
        operator=operator,
        tolerance=tolerance,
    )
    return DiscreteSumConstraint(parameters=list(c.parameters), condition=threshold)


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
        uses_log = obj.log_transform or (
            obj.transform is not None and obj.transform.kind == ObjectiveTransformKind.LOG
        )
        if not uses_log:
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
