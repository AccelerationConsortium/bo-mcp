"""Initial-design generation for Bayesian Optimization campaigns.

Split from :mod:`bo_engine.suggestions` to give Sobol-based initial-design
draws, exclusion / deduplication helpers, and the purely-categorical
exhaustion fallback their own module. The BO acquisition pipeline imports
:func:`generate_initial_design` directly; legacy callers still using
``from bo_engine.suggestions import generate_initial_design`` continue
to work because :mod:`bo_engine.suggestions` re-exports the public names
from this module.

Reference: low-discrepancy initial designs are the standard warm-start
strategy for sample-efficient BO. The implementation uses the Owen-style
scrambled Sobol sequence from :class:`torch.quasirandom.SobolEngine`,
which delivers asymptotic O((log N)^d / N) discrepancy on the unit
hypercube (Owen 1995; see also the BoTorch tutorials).
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

from bo_engine.constants import (
    DUPLICATE_DETECTION_TOLERANCE,
    INITIAL_DESIGN_MULTIPLIER,
    NUMERICAL_EPSILON,
    SAFE_DIVISION_EPSILON,
)
from bo_engine.constraints import (
    _get_parameter_indices,
    apply_sum_constraint,
)
from bo_engine.device import get_device, get_dtype
from bo_engine.reproducibility import derive_seed
from bo_engine.transforms import (
    SearchSpaceType,
    classify_search_space,
    count_categorical_combinations,
    decode_categorical,
    get_bounds_tensor,
    get_n_dims,
)
from bo_engine.types import (
    ConstraintType,
    ObservationData,
    OptimizationSpec,
    ParameterType,
)

logger = logging.getLogger(__name__)

# Oversampling factor used when filtering Sobol draws against a non-empty
# exclusion set — draw ~OVERSAMPLE_FACTOR * n_points candidates so the
# post-filter still leaves enough unique designs for the requested batch.
SOBOL_EXCLUSION_OVERSAMPLE_FACTOR = 4


class SearchSpaceExhaustedError(RuntimeError):
    """Raised when the finite search space cannot yield more unique points.

    Triggered when the caller asks for ``n_points`` initial-design draws
    but every candidate is already in the supplied exclusion set *and*
    the space is small/finite enough that no further fresh combinations
    exist.  Only applicable to purely-categorical spaces today; continuous
    spaces have effectively infinite cardinality and fall back to Sobol
    continuation instead.
    """

    def __init__(
        self,
        *,
        n_requested: int,
        n_available: int,
        n_total_combinations: int | None = None,
        subsampled: bool = False,
        n_full_combinations: int | None = None,
        max_candidates: int | None = None,
    ) -> None:
        """Build an exhaustion error referencing the requested, available and total counts.

        ``subsampled=True`` marks that the exhausted candidate set was a
        bounded subsample of a larger space (the BayBE large-categorical
        safeguard): ``n_total_combinations`` then counts the *subsample*,
        ``n_full_combinations`` the estimated full product space, and
        ``max_candidates`` the active row budget — so callers can
        recommend raising the budget instead of terminating a campaign
        whose real space is far from exhausted.
        """
        msg = (
            f"Cannot generate {n_requested} unique initial-design points: "
            f"only {n_available} unseen combinations remain"
        )
        if n_total_combinations is not None:
            msg += f" ({n_available}/{n_total_combinations} total)"
        if subsampled and n_full_combinations is not None:
            msg += (
                f"; the candidate set is a bounded subsample of a "
                f"~{n_full_combinations}-combination space"
            )
        msg += "."
        super().__init__(msg)
        self.n_requested = n_requested
        self.n_available = n_available
        self.n_total_combinations = n_total_combinations
        self.subsampled = subsampled
        self.n_full_combinations = n_full_combinations
        self.max_candidates = max_candidates


def _values_match(
    a: object,
    b: object,
    is_continuous: bool,
    tolerance: float,
) -> bool:
    """Compare a single pair of parameter values by type semantics."""
    if not is_continuous:
        return a == b
    if not isinstance(a, (int, float, str, bytes)) or not isinstance(b, (int, float, str, bytes)):
        return False
    try:
        return abs(float(a) - float(b)) <= tolerance
    except (TypeError, ValueError):
        return False


def _design_matches(
    design: dict[str, Any],
    other: dict[str, Any],
    spec: OptimizationSpec,
    tolerance: float,
) -> bool:
    """Check whether two decoded parameter dicts describe the same point.

    Categorical / discrete params must match exactly; continuous params
    must match within ``tolerance`` (absolute).  Missing keys in either
    side are treated as non-matching so we never silently accept partial
    candidates.
    """
    for param in spec.parameters:
        name = param.name
        if name not in design or name not in other:
            return False
        is_continuous = param.type == ParameterType.CONTINUOUS
        if not _values_match(design[name], other[name], is_continuous, tolerance):
            return False
    return True


def _exclude_known_designs(
    designs: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    spec: OptimizationSpec,
    tolerance: float,
) -> list[dict[str, Any]]:
    """Return ``designs`` with entries that match any ``excluded`` dropped.

    The comparison uses :func:`_design_matches`, so categorical equality
    is exact and continuous equality is within ``tolerance``.  Relative
    ordering is preserved so the caller still sees the first-N behaviour
    of the Sobol sequence.
    """
    if not excluded:
        return list(designs)
    return [
        d
        for d in designs
        if not any(_design_matches(d, other, spec, tolerance) for other in excluded)
    ]


def _deduplicate_designs(
    designs: list[dict[str, Any]],
    spec: OptimizationSpec,
    tolerance: float,
) -> list[dict[str, Any]]:
    """Drop within-batch duplicates while preserving order."""
    kept: list[dict[str, Any]] = []
    for d in designs:
        if not any(_design_matches(d, k, spec, tolerance) for k in kept):
            kept.append(d)
    return kept


def _guard_categorical_space_exhaustion(
    spec: OptimizationSpec,
    observations: list[ObservationData],
    batch_size: int,
) -> None:
    """Raise if every unique combination of a purely-categorical space is observed.

    This short-circuits before the BO and initial-design paths so callers
    get a clean ``SearchSpaceExhaustedError`` instead of an opaque failure
    from ``optimize_acqf_discrete`` (which raises when ``X_avoid`` covers
    the entire choice set) or a warning-return from
    :func:`generate_initial_design`.
    """
    if classify_search_space(spec) != SearchSpaceType.PURELY_CATEGORICAL:
        return
    n_total = count_categorical_combinations(spec)
    if n_total <= 0:
        return
    observed_params = [obs.parameter_values for obs in observations]
    unique_observed = _deduplicate_designs(observed_params, spec, DUPLICATE_DETECTION_TOLERANCE)
    if len(unique_observed) >= n_total:
        raise SearchSpaceExhaustedError(
            n_requested=batch_size,
            n_available=0,
            n_total_combinations=n_total,
        )


def _enumerate_unobserved_categorical_combinations(
    spec: OptimizationSpec,
    excluded: list[dict[str, Any]],
    tolerance: float,
) -> list[dict[str, Any]]:
    """Enumerate categorical combinations not present in ``excluded``.

    Only valid for purely-categorical spaces.  Used as a deterministic
    fallback when Sobol continuation cannot supply enough unseen points.
    """
    cat_axes: list[list[str]] = []
    for param in spec.parameters:
        if param.type != ParameterType.CATEGORICAL or param.categories is None:
            msg = "Enumeration fallback requires a purely-categorical space."
            raise ValueError(msg)
        cat_axes.append(list(param.categories))

    param_names = [p.name for p in spec.parameters]
    unseen: list[dict[str, Any]] = []
    for combo in itertools.product(*cat_axes):
        candidate = dict(zip(param_names, combo, strict=True))
        if not any(_design_matches(candidate, e, spec, tolerance) for e in excluded):
            unseen.append(candidate)
    return unseen


def generate_initial_design(
    spec: OptimizationSpec,
    n_points: int | None = None,
    *,
    n_drawn: int = 0,
    excluded_points: list[dict[str, Any]] | None = None,
    dedup_tolerance: float = SAFE_DIVISION_EPSILON,
) -> list[dict[str, Any]]:
    """Generate initial design using Sobol sequence.

    Applies constraints (sum_equals, sum_less_than, sum_greater_than) to
    project samples onto the constraint surface.

    Consecutive calls on the same campaign should pass ``n_drawn`` equal
    to the number of points already produced for that campaign so the
    low-discrepancy Sobol sequence continues where it left off instead of
    restarting with a fresh scramble.  Determinism requires
    ``spec.random_seed`` to be set; otherwise ``SobolEngine`` uses an
    OS-level scramble that changes on every construction.

    When ``excluded_points`` is supplied, any candidate that matches one
    of them (exactly for categorical/discrete, within ``dedup_tolerance``
    for continuous) is dropped.  For purely-categorical spaces the
    function falls back to enumerating the remaining combinations if
    Sobol continuation cannot supply enough fresh designs, and raises
    :class:`SearchSpaceExhaustedError` only when the space is truly
    exhausted.

    Args:
        spec: Campaign specification.
        n_points: Number of points to generate (default: 2 * n_dims + 1).
        n_drawn: Number of prior Sobol draws to skip (fast-forward).
        excluded_points: Points that must not appear in the output.
        dedup_tolerance: Absolute tolerance for continuous equality.

    Returns:
        List of parameter value dictionaries of length ``n_points``.

    Raises:
        SearchSpaceExhaustedError: Purely-categorical space whose remaining
            unseen combinations are fewer than ``n_points``.
    """
    n_dims = get_n_dims(spec)

    if n_points is None:
        n_points = spec.initial_design_size or (INITIAL_DESIGN_MULTIPLIER * spec.n_parameters + 1)

    excluded: list[dict[str, Any]] = list(excluded_points) if excluded_points else []

    # Fast-forward only makes sense when the Sobol sequence is stable across
    # calls (i.e. ``spec.random_seed`` is set).  With ``seed=None`` every
    # construction of ``SobolEngine`` picks a fresh OS-level scramble, so
    # continuing from position ``n_drawn`` of a different sequence buys us
    # nothing and changes user-visible output for existing unseeded tests.
    effective_n_drawn = n_drawn if spec.random_seed is not None else 0

    # Oversample Sobol draws only when we actually need to filter — for
    # continuous campaigns with no exclusion the extra draws would be wasted
    # and (worse) would shift the position-0 point, regressing deterministic
    # tutorial-style tests that pin specific Sobol outputs.
    space_type = classify_search_space(spec)
    is_finite_space = space_type != SearchSpaceType.CONTINUOUS
    need_filter = bool(excluded) and is_finite_space
    draw_count = (
        max(n_points, n_points * SOBOL_EXCLUSION_OVERSAMPLE_FACTOR) if need_filter else n_points
    )
    designs = _draw_sobol_designs(spec, n_dims, draw_count, n_drawn=effective_n_drawn)

    if excluded:
        designs = _exclude_known_designs(designs, excluded, spec, dedup_tolerance)
    if is_finite_space:
        designs = _deduplicate_designs(designs, spec, dedup_tolerance)

    if len(designs) >= n_points:
        return designs[:n_points]

    # Sobol alone could not supply enough unique designs.  For purely
    # categorical spaces, fall back to deterministic enumeration — this
    # is the only way to certify exhaustion.
    space_type = classify_search_space(spec)
    if space_type == SearchSpaceType.PURELY_CATEGORICAL:
        already_selected = list(excluded) + list(designs)
        extra = _enumerate_unobserved_categorical_combinations(
            spec, already_selected, dedup_tolerance
        )
        designs = designs + extra
        designs = _deduplicate_designs(designs, spec, dedup_tolerance)
        if len(designs) < n_points:
            total = count_categorical_combinations(spec) or None
            raise SearchSpaceExhaustedError(
                n_requested=n_points,
                n_available=len(designs),
                n_total_combinations=total,
            )
        return designs[:n_points]

    # Continuous / mixed spaces are effectively infinite; surface a warning
    # but return what we have so downstream callers can decide.
    logger.warning(
        "Sobol continuation produced %d unique designs for a batch of %d "
        "after filtering against %d excluded points; returning the smaller batch.",
        len(designs),
        n_points,
        len(excluded),
    )
    return designs


def _draw_sobol_designs(
    spec: OptimizationSpec,
    n_dims: int,
    draw_count: int,
    *,
    n_drawn: int,
) -> list[dict[str, Any]]:
    """Sample ``draw_count`` Sobol points and decode them to parameter dicts.

    Honours ``spec.random_seed`` (when non-None) so consecutive calls with
    the same seed produce the same low-discrepancy sequence, and advances
    past ``n_drawn`` prior draws via ``fast_forward``.

    The Sobol seed is routed through
    :func:`bo_engine.reproducibility.derive_seed` with role tag
    ``"sobol:initial_design"`` so it shares the single source of truth for
    per-phase seed derivation. ``spec.random_seed=None`` is preserved as
    "OS-level scramble" (Sobol's documented behavior) — the helper only
    fires when a master seed exists.
    """
    if spec.random_seed is None:
        sobol_seed: int | None = None
    else:
        sobol_seed = derive_seed(spec.random_seed, "sobol:initial_design")
    sobol = SobolEngine(dimension=n_dims, scramble=True, seed=sobol_seed)
    if n_drawn > 0:
        sobol.fast_forward(n_drawn)
    samples = sobol.draw(draw_count).to(device=get_device(), dtype=get_dtype())

    bounds = get_bounds_tensor(spec)
    lower = bounds[0]
    upper = bounds[1]
    scaled_samples = samples * (upper - lower) + lower
    scaled_samples = _apply_constraints_to_samples(scaled_samples, spec, bounds)

    return [decode_categorical(scaled_samples[i], spec) for i in range(draw_count)]


def _apply_constraints_to_samples(
    samples: Tensor,
    spec: OptimizationSpec,
    bounds: Tensor,
) -> Tensor:
    """Apply constraints to projected samples.

    For sum_equals constraints, normalizes parameters to satisfy the constraint.
    For inequality constraints, clips values as needed.

    Args:
        samples: Tensor of shape (n_samples, n_dims)
        spec: Optimization specification with constraints
        bounds: Tensor of shape (2, n_dims) with lower and upper bounds

    Returns:
        Samples with constraints applied
    """
    result = samples.clone()

    for constraint in spec.constraints:
        param_indices = _get_parameter_indices(constraint.parameters, spec)

        if constraint.type == ConstraintType.SUM_EQUALS:
            # Project onto sum = value constraint surface
            result = apply_sum_constraint(result, param_indices, constraint.value)
            # Clip to bounds after normalization
            result = torch.clamp(result, bounds[0], bounds[1])
            # Re-apply constraint after clipping (may need iteration for tight bounds)
            result = apply_sum_constraint(result, param_indices, constraint.value)

        elif constraint.type == ConstraintType.SUM_LESS_THAN:
            # Scale down if sum exceeds limit
            selected = result[..., param_indices]
            current_sum = selected.sum(dim=-1, keepdim=True)
            excess_mask = current_sum > constraint.value
            if excess_mask.any():
                scale = torch.where(
                    excess_mask,
                    constraint.value / current_sum,
                    torch.ones_like(current_sum),
                )
                result[..., param_indices] = selected * scale

        elif constraint.type == ConstraintType.SUM_GREATER_THAN:
            # Scale up if sum is below minimum
            selected = result[..., param_indices]
            current_sum = selected.sum(dim=-1, keepdim=True)
            deficit_mask = current_sum < constraint.value
            if deficit_mask.any():
                # Avoid division by zero: when ``current_sum`` is below
                # NUMERICAL_EPSILON treat it as zero and replace with 1
                # so the per-batch scale factor below does not blow up.
                safe_sum = torch.where(
                    current_sum.abs() < NUMERICAL_EPSILON,
                    torch.ones_like(current_sum),
                    current_sum,
                )
                scale = torch.where(
                    deficit_mask,
                    constraint.value / safe_sum,
                    torch.ones_like(current_sum),
                )
                result[..., param_indices] = selected * scale
            # Clip to bounds
            result = torch.clamp(result, bounds[0], bounds[1])

    return result
