"""Discrete search-space size budgeting and deterministic subsampling.

BayBE's ``SearchSpace.from_product`` materializes the full Cartesian
product of all discrete/categorical parameters, and ``Campaign.to_json``
embeds the resulting dataframes — so a large-categorical campaign blows up
memory, recommend runtime, *and* the persisted backend state (the original
incident produced a 1.34 GB state that PostgreSQL's 1 GiB protocol limit
killed mid-UPDATE). This module implements BayBE's own documented answer
for combinatorially large spaces: measure first via
``SubspaceDiscrete.estimate_product_space_size`` and, above a configured
budget, build the discrete subspace from an explicit **deterministically
subsampled** candidate list (direct ``SubspaceDiscrete`` construction over
the sampled frame — see :func:`build_subsampled_discrete_subspace` for why
``from_dataframe`` is bypassed).

Determinism is load-bearing: the candidate set must be identical across
processes and the ``_rebuild_from_observations`` fallback, so the sampler
is seeded from a blake2b hash of the canonical spec content — never
Python's ``hash()`` (PYTHONHASHSEED) and never a global RNG.

Reference: BayBE search-space userguide
(https://emdgroup.github.io/baybe/stable/userguide/searchspace.html)
— ``estimate_product_space_size`` / ``from_dataframe``; PostgreSQL
protocol message-length limit
(https://www.postgresql.org/docs/current/protocol-message-formats.html).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from baybe.parameters import NumericalDiscreteParameter
from baybe.parameters.base import DiscreteParameter
from baybe.searchspace import SubspaceDiscrete
from baybe.utils.dataframe import normalize_input_dtypes

from bo_engine.transforms import count_discrete_combinations
from bo_engine.types import ObservationData, OptimizationSpec, ParameterSpec, ParameterType
from bo_engine_baybe.constants import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MAX_SEARCHSPACE_STATE_BYTES,
    EXP_LABEL_BYTES_UPPER_BOUND,
    MIN_VIABLE_SUBSAMPLE_CANDIDATES,
    OBJECT_CELL_POINTER_BYTES,
    SEARCHSPACE_ESTIMATE_INFLATION_FACTOR,
    SUBSAMPLE_TOPUP_MAX_ROUNDS,
    SUBSTANCE_COMP_WIDTH_UPPER_BOUND,
)
from bo_engine_baybe.options import (
    BayBEParameterRole,
    extract_baybe_backend_options,
    extract_baybe_parameter_options,
)

if TYPE_CHECKING:
    from baybe.searchspace.discrete import MemorySize

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiscreteSpaceBudget:
    """Resolved size budget for the enumerated discrete subspace.

    ``exceeds_budget`` is the branch decision:  ``False`` keeps the
    historical ``SearchSpace.from_product`` path byte-for-byte;  ``True``
    routes construction through the deterministic subsampler with
    ``n_candidates`` rows.
    """

    n_combinations: int
    estimated_state_bytes: float
    byte_budget: int
    max_candidates: int
    n_candidates: int
    exceeds_budget: bool


def estimate_discrete_state_size(
    parameters: Sequence[DiscreteParameter],
) -> MemorySize:
    """Pure wrapper around BayBE's product-space size estimation.

    Isolated so tests can mock the (pre-constraint) estimate; per the
    BayBE search-space userguide the estimate ignores constraints — a
    heavily-constrained space may be flagged although its *filtered*
    product would fit, an accepted v1 trade-off because applying the
    pandas-path constraints requires materializing the unfiltered product
    anyway.
    """
    return SubspaceDiscrete.estimate_product_space_size(list(parameters))


def resolve_discrete_space_budget(
    spec: OptimizationSpec,
    discrete_parameters: Sequence[DiscreteParameter],
) -> DiscreteSpaceBudget:
    """Resolve the budget decision for the spec's discrete subspace.

    Bytes are the primary bound (§ "bound bytes, not rows": with OHE the
    computational representation is ``n_rows × Σ category counts`` floats,
    so a row cap alone does not bound bytes); the row cap additionally
    bounds recommend-time exhaustive scoring. The row budget for the
    subsample path is derived from the byte budget via the estimated
    bytes-per-row.
    """
    options = extract_baybe_backend_options(spec.backend_options)
    byte_budget = (
        options.max_searchspace_state_bytes
        if options.max_searchspace_state_bytes is not None
        else DEFAULT_MAX_SEARCHSPACE_STATE_BYTES
    )
    max_candidates = (
        options.max_candidates if options.max_candidates is not None else DEFAULT_MAX_CANDIDATES
    )
    n_combinations = count_discrete_combinations(spec)
    if n_combinations <= 1 or not discrete_parameters:
        return DiscreteSpaceBudget(
            n_combinations=n_combinations,
            estimated_state_bytes=0.0,
            byte_budget=byte_budget,
            max_candidates=max_candidates,
            n_candidates=n_combinations,
            exceeds_budget=False,
        )
    estimate = estimate_discrete_state_size(discrete_parameters)
    estimated_bytes = (
        float(estimate.exp_rep_bytes) + float(estimate.comp_rep_bytes)
    ) * SEARCHSPACE_ESTIMATE_INFLATION_FACTOR
    exceeds = estimated_bytes > byte_budget or n_combinations > max_candidates
    n_rows = max(int(estimate.exp_rep_shape[0]), 1)
    bytes_per_row = max(estimated_bytes / n_rows, 1.0)
    n_candidates = min(max_candidates, math.floor(byte_budget / bytes_per_row))
    n_candidates = max(n_candidates, MIN_VIABLE_SUBSAMPLE_CANDIDATES)
    return DiscreteSpaceBudget(
        n_combinations=n_combinations,
        estimated_state_bytes=estimated_bytes,
        byte_budget=byte_budget,
        max_candidates=max_candidates,
        n_candidates=min(n_candidates, n_combinations),
        exceeds_budget=exceeds,
    )


def _comp_width_upper_bound(p: ParameterSpec) -> int | None:
    """Safe over-estimate of one parameter's computational-representation width.

    Categorical: one column per category under OHE (one under INT).
    Numerical-discrete: one column. Substance / custom roles: the widest
    curated encoding respectively the declared descriptor table. Returns
    ``None`` ("inconclusive") for a substance parameter carrying
    ``kwargs_fingerprint`` — the passthrough can raise the fingerprint
    dimensionality beyond any fixed bound, so the prescreen must defer
    to the precise, parameter-built estimate instead of over-promising.
    """
    opts = extract_baybe_parameter_options(p.parameter_options)
    if p.type == ParameterType.CATEGORICAL:
        if opts.role == BayBEParameterRole.SUBSTANCE:
            if opts.kwargs_fingerprint is not None:
                return None
            return SUBSTANCE_COMP_WIDTH_UPPER_BOUND
        if opts.role == BayBEParameterRole.CUSTOM and opts.custom_descriptors:
            return max(len(row) for row in opts.custom_descriptors.values())
        if opts.encoding is not None and opts.encoding.value == "INT":
            return 1
        return len(p.categories or [])
    return 1


def _exp_label_bytes_bound(p: ParameterSpec) -> int:
    """Safe per-cell over-estimate of one parameter's experimental-rep bytes.

    BayBE's ``estimate_product_space_size`` measures *actual* label bytes
    (pandas ``memory_usage(deep=True)`` sums ``sys.getsizeof`` per cell
    plus one object pointer), so neither a fixed constant nor a
    character-count bound stays an upper bound for every label: PEP 393
    strings store 1, 2, or 4 bytes per character depending on the widest
    code point, making long SMILES labels *and* wide-Unicode chemical
    names (Greek letters, CJK) realistic under-estimation cases.
    Measuring the declared labels with the same ``sys.getsizeof`` keeps
    the bound exact; the historical floor preserves previous decisions
    for short-label and float cells.
    """
    max_label_bytes = max(
        (sys.getsizeof(str(label)) for label in p.categories or ()),
        default=0,
    )
    return max(EXP_LABEL_BYTES_UPPER_BOUND, OBJECT_CELL_POINTER_BYTES + max_label_bytes)


def cheap_budget_prescreen(spec: OptimizationSpec) -> DiscreteSpaceBudget | None:
    """Construction-free below-budget proof, or ``None`` when inconclusive.

    Uses per-parameter *upper bounds* on the computational width, so a
    non-``None`` result is a guarantee that the precise estimate would
    also land below budget — letting ``validate_capabilities`` and the
    warning surface skip building BayBE parameter objects (and in
    particular skip computing substance descriptor tables) for every
    ordinarily-sized spec. Returns ``None`` when the bounds cannot prove
    "below budget"; callers then fall back to the precise, parameter-built
    estimate.
    """
    options = extract_baybe_backend_options(spec.backend_options)
    byte_budget = (
        options.max_searchspace_state_bytes
        if options.max_searchspace_state_bytes is not None
        else DEFAULT_MAX_SEARCHSPACE_STATE_BYTES
    )
    max_candidates = (
        options.max_candidates if options.max_candidates is not None else DEFAULT_MAX_CANDIDATES
    )
    n_combinations = count_discrete_combinations(spec)
    if n_combinations <= 1:
        return DiscreteSpaceBudget(
            n_combinations=n_combinations,
            estimated_state_bytes=0.0,
            byte_budget=byte_budget,
            max_candidates=max_candidates,
            n_candidates=n_combinations,
            exceeds_budget=False,
        )
    if n_combinations > max_candidates:
        return None
    float_bytes = np.dtype(np.float64).itemsize
    width_bounds = [
        _comp_width_upper_bound(p)
        for p in spec.parameters
        if p.type in (ParameterType.CATEGORICAL, ParameterType.DISCRETE)
    ]
    if any(bound is None for bound in width_bounds):
        # A parameter's computational width has no safe fixed bound
        # (substance fingerprint-kwargs passthrough) — the prescreen must
        # abstain rather than declare "below budget" on a guess.
        return None
    width_bound = sum(bound for bound in width_bounds if bound is not None)
    exp_bytes_per_row_bound = sum(_exp_label_bytes_bound(p) for p in spec.parameters)
    bytes_bound = (
        n_combinations
        * (float_bytes * width_bound + exp_bytes_per_row_bound)
        * SEARCHSPACE_ESTIMATE_INFLATION_FACTOR
    )
    if bytes_bound > byte_budget:
        return None
    return DiscreteSpaceBudget(
        n_combinations=n_combinations,
        estimated_state_bytes=bytes_bound,
        byte_budget=byte_budget,
        max_candidates=max_candidates,
        n_candidates=n_combinations,
        exceeds_budget=False,
    )


def _canonical_spec_payload(spec: OptimizationSpec) -> dict[str, Any]:
    """Canonical, sorted description of the sampled search-space content."""
    return {
        "parameters": sorted(
            (
                p.name,
                str(p.type),
                tuple(p.bounds) if p.bounds is not None else None,
                # Neutral dataclass grid field, not a pandas accessor.
                tuple(float(v) for v in p.values) if p.values is not None else None,  # noqa: PD011
                tuple(p.categories) if p.categories is not None else None,
                json.dumps(p.parameter_options, sort_keys=True, default=str)
                if p.parameter_options
                else None,
            )
            for p in spec.parameters
        ),
        "constraints": sorted(
            # Optional fields are normalized to (is_set, value) pairs so
            # two near-identical constraints (e.g. one with coefficients,
            # one without) stay sortable — a bare ``None`` next to a tuple
            # or int would raise TypeError inside sorted().
            (
                str(c.type),
                tuple(sorted(c.parameters)),
                float(c.value),
                (c.coefficients is not None, tuple(c.coefficients or ())),
                (c.min_cardinality is not None, c.min_cardinality or 0),
                (c.max_cardinality is not None, c.max_cardinality or 0),
                c.is_interpoint,
            )
            for c in spec.constraints
        ),
    }


def _spec_content_seed(spec: OptimizationSpec) -> int:
    """Deterministic sampler seed derived from the canonical spec content.

    blake2b of the canonical payload — never Python's ``hash()``
    (PYTHONHASHSEED-dependent) and never the process-global RNG, so the
    candidate set is identical across processes, restarts, and the
    ``_rebuild_from_observations`` fallback (the
    ``_observation_fingerprint`` precedent).
    """
    encoded = json.dumps(_canonical_spec_payload(spec), sort_keys=True, default=str)
    digest = hashlib.blake2b(encoded.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _value_pools(discrete_parameters: Sequence[DiscreteParameter]) -> dict[str, np.ndarray]:
    """Per-parameter candidate value pools typed by the *declared* parameter class.

    The dtype comes from the parameter's own type, never from value
    coercibility: only :class:`NumericalDiscreteParameter` grids are
    float64; every label-based parameter (categorical / substance / task
    / custom) keeps object dtype even when all its labels happen to be
    numeric strings ("0", CAS-like IDs, …). Try-float-first coercion
    previously turned such labels into a float column, which BayBE's
    ``normalize_input_dtypes`` never converts back for categorical
    parameters — crashing the subspace merge without observations, or
    silently producing an almost-all-NaN computational representation
    (and a corrupted GP fit) with observations unioned in.
    """
    pools: dict[str, np.ndarray] = {}
    for p in discrete_parameters:
        if isinstance(p, NumericalDiscreteParameter):
            pools[p.name] = np.asarray(list(p.values), dtype=np.float64)
        else:
            pools[p.name] = np.asarray(list(p.values), dtype=object)
    return pools


def _sample_rows(
    rng: np.random.Generator,
    pools: dict[str, np.ndarray],
    n_rows: int,
) -> pd.DataFrame:
    """Vectorized independent per-column sampling of ``n_rows`` rows.

    Column dtypes follow the pool arrays built by :func:`_value_pools`
    (declared-type driven), so the resulting frame matches BayBE's
    expected experimental dtypes without a lossy conversion pass.
    """
    return pd.DataFrame({name: rng.choice(pool, size=n_rows) for name, pool in pools.items()})


def _filter_constraint_violations(
    df: pd.DataFrame,
    baybe_constraints: list[Any] | None,
) -> pd.DataFrame:
    """Drop rows violating BayBE discrete constraints.

    Reuses BayBE's own ``DiscreteConstraint.get_invalid`` so the sampled
    frame agrees with what ``from_product`` + constraint filtering would
    have produced — ``SubspaceDiscrete.from_dataframe`` takes no
    ``constraints`` argument (verified signature), hence the manual pass.
    """
    if not baybe_constraints:
        return df
    from baybe.constraints.base import DiscreteConstraint

    for constraint in baybe_constraints:
        if not isinstance(constraint, DiscreteConstraint):
            continue
        if not set(constraint.parameters).issubset(df.columns):
            continue
        invalid = constraint.get_invalid(df)
        if len(invalid):
            df = df.drop(index=invalid)
    return df


def _observation_rows(
    spec: OptimizationSpec,
    discrete_names: list[str],
    observations: list[ObservationData] | None,
    pending_points: list[dict[str, Any]] | None,
) -> pd.DataFrame | None:
    """Observed + pending parameter configurations restricted to discrete columns.

    These rows are unioned into the candidate frame so BayBE's
    per-candidate metadata ledger can mark them and the
    ``allow_recommending_already_measured/_recommended=False`` contracts
    keep working on the subsampled space.
    """
    _ = spec
    rows: list[dict[str, Any]] = [
        {name: obs.parameter_values.get(name) for name in discrete_names}
        for obs in observations or []
    ]
    rows.extend(
        {name: pending.get(name) for name in discrete_names}
        for pending in pending_points or []
        if isinstance(pending, dict)
    )
    if not rows:
        return None
    frame = pd.DataFrame(rows, columns=discrete_names).dropna()
    return frame if not frame.empty else None


def _normalize_union_rows(
    frame: pd.DataFrame,
    discrete_parameters: Sequence[DiscreteParameter],
) -> pd.DataFrame:
    """Coerce union-in rows to the declared pool dtypes before matching.

    Observed/pending values arrive as raw JSON-decoded scalars: a label
    submitted as the number ``7`` must compare equal to the declared
    category ``"7"``, and a numeric-discrete measurement within the
    parameter's declared ``tolerance`` of a grid value counts as that
    grid value (mirroring BayBE's own tolerance-based measurement
    matching). Numeric values inside the tolerance are snapped to the
    nearest grid value so the canonical candidate row enters the frame;
    values that still miss the pools afterwards are genuinely
    out-of-pool and are dropped (with a warning) downstream.
    """
    frame = frame.copy()
    for p in discrete_parameters:
        if p.name not in frame.columns:
            continue
        if isinstance(p, NumericalDiscreteParameter):
            column = pd.to_numeric(frame[p.name], errors="coerce")
            tolerance = float(p.tolerance)
            if tolerance > 0.0:
                grid = np.asarray(list(p.values), dtype=np.float64)
                values = column.to_numpy(dtype=np.float64)
                nearest = grid[np.abs(values[:, None] - grid[None, :]).argmin(axis=1)]
                within = np.abs(values - nearest) <= tolerance
                column = pd.Series(np.where(within, nearest, values), index=frame.index)
            frame[p.name] = column
        else:
            frame[p.name] = frame[p.name].astype(str)
    return frame


def _drop_out_of_pool_rows(
    frame: pd.DataFrame,
    pools: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Drop union-in rows whose values are outside the declared value pools.

    A drifted observed/pending value (e.g. a label renamed since the
    campaign was created) would otherwise produce an all-NaN
    computational-representation row — the one validation the bypassed
    ``SubspaceDiscrete.from_dataframe`` used to provide. Server-side
    submit validation makes this defense-in-depth, so out-of-pool rows
    are dropped with a warning rather than failing the whole build.
    """
    mask = pd.Series(data=True, index=frame.index)
    for name, pool in pools.items():
        column = frame[name]
        if pool.dtype == np.float64:
            column = pd.to_numeric(column, errors="coerce")
        mask &= column.isin(pool.tolist())
    n_dropped = int((~mask).sum())
    if n_dropped:
        logger.warning(
            "Dropped %d observed/pending row(s) with values outside the "
            "declared parameter value pools before subspace construction.",
            n_dropped,
        )
    return frame[mask]


def sample_discrete_candidates(
    spec: OptimizationSpec,
    discrete_parameters: Sequence[DiscreteParameter],
    n_candidates: int,
    baybe_constraints: list[Any] | None = None,
    observations: list[ObservationData] | None = None,
    pending_points: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Deterministically sample a bounded candidate frame (no materialization).

    Rows are drawn by independent per-parameter choice (vectorized over
    ``n_candidates``), deduplicated, constraint-filtered via BayBE's own
    ``get_invalid``, and topped up for a bounded number of rounds. The
    constraint filter always runs on the *merged* frame — cross-row
    constraints (BayBE's permutation-invariance ``get_invalid`` marks
    duplicates within the frame it sees) would otherwise admit round-2+
    representatives of classes already kept in round 1. Observed and
    pending configurations are unioned in first (pool-membership checked,
    deduplicated), and the filter runs once more over the union so the
    observed rows win as the canonical representatives of their class.

    Raises:
        ValueError: When constraint filtering cannot reach the minimum
            viable candidate count within the bounded top-up rounds
            (near-infeasible constraint set).
    """
    rng = np.random.default_rng(_spec_content_seed(spec))
    pools = _value_pools(discrete_parameters)
    discrete_names = [p.name for p in discrete_parameters]

    frame: pd.DataFrame | None = None
    for _round in range(SUBSAMPLE_TOPUP_MAX_ROUNDS):
        missing = n_candidates - (0 if frame is None else len(frame))
        if missing <= 0:
            break
        sampled = _sample_rows(rng, pools, missing * 2)
        merged = sampled if frame is None else pd.concat([frame, sampled], ignore_index=True)
        merged = merged.drop_duplicates(ignore_index=True)
        frame = _filter_constraint_violations(merged, baybe_constraints).reset_index(drop=True)
    if frame is None:
        frame = pd.DataFrame(columns=discrete_names)
    frame = frame.head(n_candidates)

    # The floor guards against constraint-driven sampling starvation only:
    # when the requested candidate count is itself below the floor, the
    # space is small enough to enumerate and a short feasible frame is the
    # correct result (parity with the from_product path on tiny spaces).
    if len(frame) < MIN_VIABLE_SUBSAMPLE_CANDIDATES <= n_candidates:
        msg = (
            f"Subsampling the discrete search space produced only {len(frame)} "
            f"feasible candidate(s) after {SUBSAMPLE_TOPUP_MAX_ROUNDS} rounds "
            f"(minimum viable: {MIN_VIABLE_SUBSAMPLE_CANDIDATES}). The "
            "constraint set is (near-)infeasible for sampling; relax the "
            "constraints or reduce the space."
        )
        raise ValueError(msg)

    extra = _observation_rows(spec, discrete_names, observations, pending_points)
    if extra is not None:
        extra = _normalize_union_rows(extra, discrete_parameters)
        extra = _drop_out_of_pool_rows(extra, pools)
    if extra is not None and not extra.empty:
        # Observed/pending rows first: drop_duplicates and the cross-row
        # constraint filter both keep first occurrences, so unioned rows
        # survive as the canonical representatives of their class.
        unioned = pd.concat([extra, frame], ignore_index=True).drop_duplicates(ignore_index=True)
        frame = _filter_constraint_violations(unioned, baybe_constraints).reset_index(drop=True)
    return frame


def build_subsampled_discrete_subspace(
    spec: OptimizationSpec,
    discrete_parameters: Sequence[DiscreteParameter],
    budget: DiscreteSpaceBudget,
    baybe_constraints: list[Any] | None = None,
    observations: list[ObservationData] | None = None,
    pending_points: list[dict[str, Any]] | None = None,
) -> SubspaceDiscrete:
    """Build the bounded discrete subspace by direct ``SubspaceDiscrete`` construction.

    The sampled candidate frame is passed straight to the
    ``SubspaceDiscrete`` constructor together with the explicit parameter
    objects (``SubspaceDiscrete.from_dataframe`` is deliberately bypassed
    — see the inline note at the construction site), so encodings,
    ``SubstanceParameter`` and ``TaskParameter`` objects from
    ``spec_to_parameters`` are preserved.
    """
    candidates = sample_discrete_candidates(
        spec,
        discrete_parameters,
        budget.n_candidates,
        baybe_constraints=baybe_constraints,
        observations=observations,
        pending_points=pending_points,
    )
    candidates = normalize_input_dtypes(candidates, discrete_parameters)
    logger.warning(
        "BayBE discrete search space subsampled: %d of ~%d combinations "
        "(estimated serialized state %.1f MB > budget %.1f MB or row cap %d).",
        len(candidates),
        budget.n_combinations,
        budget.estimated_state_bytes / 1e6,
        budget.byte_budget / 1e6,
        budget.max_candidates,
        extra={
            "outcome": "searchspace_subsampled",
            "n_candidates": len(candidates),
            "n_combinations": budget.n_combinations,
            "estimated_state_bytes": budget.estimated_state_bytes,
            "byte_budget": budget.byte_budget,
            "max_candidates": budget.max_candidates,
        },
    )
    # Direct construction instead of ``SubspaceDiscrete.from_dataframe``:
    # the sampled values come from the parameters' own value pools by
    # construction, and from_dataframe's per-unique-value ``is_in_range``
    # validation is O(rows x pool size) in Python — ~100 s per 1000 rows
    # on two 10k-value grids (measured), defeating the runtime half of
    # the safeguard. The explicit ``parameters=`` intent (preserving
    # encodings / SubstanceParameter / TaskParameter objects) is kept by
    # passing the objects straight into the constructor; dtype
    # normalization above mirrors from_dataframe's own step.
    # ``comp_rep`` is derived by the attrs default hook — ty cannot see
    # attrs field defaults (the Campaign/Settings call-site precedent).
    return SubspaceDiscrete(  # ty: ignore[missing-argument]
        parameters=tuple(discrete_parameters),
        exp_rep=candidates,
    )


def subsample_warning(budget: DiscreteSpaceBudget) -> str:
    """User-facing warning appended to suggestion batches on the subsampled path."""
    return (
        f"BayBE search space subsampled: optimizing over {budget.n_candidates} "
        f"deterministically sampled candidates out of ~{budget.n_combinations} "
        f"combinations (estimated full-space state "
        f"{budget.estimated_state_bytes / 1e6:.0f} MB exceeds the "
        f"{budget.byte_budget / 1e6:.0f} MB budget or the "
        f"{budget.max_candidates}-candidate cap). Consider "
        "parameter_options['baybe'].encoding='INT', fewer categories, "
        "backend='botorch', or raising backend_options['baybe'].max_candidates."
    )
