"""Shared intermediate representation for OptimizationSpec → backend converters.

The neutral :class:`~bo_engine.types.OptimizationSpec` is the IR every backend
consumes. Both the BoTorch and BayBE converters need the same dispatch /
classification primitives — search-space type, per-constraint target class —
to decide which native construct (``ContinuousLinearConstraint`` vs.
``DiscreteSumConstraint``, ``classify_search_space`` branch, etc.) to emit.

Centralizing these helpers here keeps the two engine converters and the
backends' ``validate_capabilities`` reports aligned: capability decisions
and converter construction reference the *same* classification call, so a
constraint cannot be reported as ``SUPPORTED`` while the converter then
fails to build it (and vice versa).

The classifiers operate on plain :class:`OptimizationSpec` parameters; the
optional :func:`normalize_spec` bundles the per-spec classification results
so callers that visit multiple constraints in a single pass do not have to
re-run the dispatch for each lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from bo_engine.types import (
    ConstraintSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)


class ConstraintTargetClass(StrEnum):
    """Classification of a constraint by the parameter types it touches.

    Values are intentionally lowercase strings so the enum is interchangeable
    with the legacy string return values used by older callers (the BayBE
    converter and its tests). Equality with a bare string still holds
    because :class:`StrEnum` derives from :class:`str`.
    """

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    HYBRID = "hybrid"
    CATEGORICAL = "categorical"
    UNKNOWN = "unknown"


def classify_constraint_target(
    constraint: ConstraintSpec,
    parameters: list[ParameterSpec],
) -> ConstraintTargetClass:
    """Classify a constraint by the parameter types it references.

    Returns :data:`ConstraintTargetClass.UNKNOWN` whenever **any** referenced
    parameter is missing from ``parameters`` — a mix of known and unknown
    names is just as fatal during native constraint construction as an
    entirely unknown reference, so both must surface the same way.
    """
    by_name = {p.name: p for p in parameters}
    missing = [name for name in constraint.parameters if name not in by_name]
    if missing or not constraint.parameters:
        return ConstraintTargetClass.UNKNOWN
    types = {by_name[name].type for name in constraint.parameters}
    if ParameterType.CATEGORICAL in types:
        return ConstraintTargetClass.CATEGORICAL
    if types == {ParameterType.CONTINUOUS}:
        return ConstraintTargetClass.CONTINUOUS
    if types == {ParameterType.DISCRETE}:
        return ConstraintTargetClass.DISCRETE
    return ConstraintTargetClass.HYBRID


@dataclass(frozen=True)
class NormalizedConstraint:
    """A constraint paired with its pre-computed target classification."""

    spec: ConstraintSpec
    target_class: ConstraintTargetClass


@dataclass(frozen=True)
class NormalizedSpec:
    """OptimizationSpec with per-constraint classifications pre-computed.

    The neutral spec itself is the IR; this struct just memoizes the
    classification step so a converter visiting every constraint does not
    have to re-run the dispatch (or invent its own).
    """

    spec: OptimizationSpec
    constraints: tuple[NormalizedConstraint, ...]


def normalize_spec(spec: OptimizationSpec) -> NormalizedSpec:
    """Pre-compute per-constraint classification for ``spec``.

    Idempotent and cheap; safe to call once at the entry to a converter.
    """
    return NormalizedSpec(
        spec=spec,
        constraints=tuple(
            NormalizedConstraint(
                spec=c,
                target_class=classify_constraint_target(c, spec.parameters),
            )
            for c in spec.constraints
        ),
    )
