"""BOBackend protocol — the interface all BO backends must implement.

This module defines the contract between the MCP server and any
optimization backend (BoTorch, BayBE, BOFire, Ax, etc.). All methods
accept and return plain Python types — never framework-specific objects
like torch.Tensor or pandas.DataFrame.

Usage:
    from bo_engine.backend import BOBackend, Feature

    def my_operation(backend: BOBackend, ...):
        result = backend.generate_suggestions(spec, observations, ...)

The companion :mod:`bo_engine.backend_base` module provides the
recommended :class:`BaseBackend` abstract class — concrete backends
inherit from it to get sensible defaults for the optional members of
this protocol. Pure ``Protocol`` implementations remain supported for
third-party plugins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from bo_engine.progress import ProgressCallback
from bo_engine.types import ObservationData, OptimizationSpec

if TYPE_CHECKING:
    from bo_engine.backend_base import BackendValidationResult


class DiagnosticSection(StrEnum):
    """Valid section names for :meth:`BOBackend.compute_diagnostics`.

    Using this enum (instead of bare strings) lets the type checker catch
    typos and gives new backend implementers a definitive list.
    """

    OBJECTIVES = "objectives"
    MODEL = "model"
    OUTLIERS = "outliers"
    SUGGESTIONS_TENSOR = "suggestions_tensor"


class Feature(StrEnum):
    """Capabilities a backend can advertise."""

    MULTI_OBJECTIVE = "multi_objective"
    CONSTRAINTS = "constraints"
    OUTCOME_CONSTRAINTS = "outcome_constraints"
    COST_AWARE = "cost_aware"
    TRANSFER_LEARNING = "transfer_learning"
    MULTI_FIDELITY = "multi_fidelity"
    HIGH_DIMENSIONAL = "high_dimensional"
    CATEGORICAL = "categorical"
    MIXED_SEARCH_SPACE = "mixed_search_space"
    INPUT_WARPING = "input_warping"


@dataclass
class SuggestionBatch:
    """Result of suggestion generation.

    Returned by :meth:`BOBackend.generate_suggestions` and
    :meth:`BOBackend.generate_initial_design`.
    """

    suggestions: list[dict[str, Any]]
    """List of (parameter_values, provenance_dict) tuples."""

    method_info: dict[str, Any] = field(default_factory=dict)
    """Method selection metadata (model type, acquisition function, etc.)."""

    backend_state: dict[str, Any] | None = None
    """Opaque backend state to persist between iterations.

    **Serialization contract:** The server stores this value via
    ``json.dumps()`` and restores it with ``json.loads()``.  All values
    must therefore be JSON-serializable (``str``, ``int``, ``float``,
    ``bool``, ``None``, ``list``, or ``dict``).  Putting non-serializable
    objects (tensors, numpy arrays, fitted models) here will crash at
    persist time.
    """

    warnings: list[str] = field(default_factory=list)


@dataclass
class BatchDiversityMetrics:
    """Diversity metrics for a batch of suggestions."""

    min_pairwise_distance: float
    mean_pairwise_distance: float
    diversity_score: float
    is_diverse: bool


@dataclass
class DuplicateInfo:
    """Information about a detected duplicate."""

    index: int
    is_exact: bool
    parameter_distance: float


@runtime_checkable
class BOBackend(Protocol):
    """Interface that all BO backends must implement.

    Methods accept :class:`OptimizationSpec` and
    :class:`ObservationData` (plain Python types from ``bo_engine.types``)
    and return plain Python dicts/dataclasses — never tensors.
    """

    @property
    def name(self) -> str:
        """Human-readable backend name."""
        ...

    @property
    def supported_features(self) -> frozenset[Feature]:
        """Set of features this backend supports."""
        ...

    # ----- Validation -----

    def validate_spec(self, spec: OptimizationSpec) -> list[str]:
        """Check whether this backend can handle *spec*.

        Returns a (possibly empty) list of warnings.  Each warning
        describes a spec feature that the backend does not support and
        will silently ignore (e.g. ``"TuRBO is not supported by BayBE;
        ignored"``).

        Raise ``ValueError`` if the spec is fundamentally incompatible
        and optimization cannot proceed.  For gracefully-degraded
        operation return warnings instead.

        The default implementation returns ``[]`` (no warnings). New
        code should prefer :meth:`validate_capabilities` for the
        structured per-feature, per-option capability report.
        """
        ...

    def validate_capabilities(self, spec: OptimizationSpec) -> BackendValidationResult:
        """Return a spec-aware capability descriptor.

        Backends implement this to report per-feature and per-option
        capability classifications (``SUPPORTED`` / ``DEGRADED`` /
        ``IGNORED`` / ``UNSUPPORTED``). ``resolve_backend_name("auto",
        ...)`` consumes :attr:`BackendValidationResult.is_compatible`
        instead of the coarse ``supported_features`` boolean set.

        Implementations may fall back to the default
        :class:`bo_engine.backend_base.BaseBackend` implementation, which
        derives a report from ``supported_features`` plus the active
        attributes on ``spec``.
        """
        ...

    # ----- Suggestion Generation -----

    def generate_initial_design(
        self,
        spec: OptimizationSpec,
        n_points: int,
    ) -> list[dict[str, Any]]:
        """Generate space-filling initial design points.

        Returns:
            List of parameter-value dicts.
        """
        ...

    def generate_suggestions(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
        batch_size: int,
        iteration: int,
        backend_state: dict[str, Any] | None = None,
        pending_points: list[dict[str, Any]] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> SuggestionBatch:
        """Generate model-guided suggestions.

        Args:
            spec: Problem specification.
            observations: Past observations.
            batch_size: How many suggestions to generate.
            iteration: Current iteration number.
            backend_state: Opaque state from a prior call (e.g. TuRBO).
            pending_points: Parameter-value dicts for in-flight suggestions
                that have not yet produced a result.  Backends that support
                batch / parallel acquisition (e.g. BoTorch) should forward
                these to the acquisition optimizer as ``X_pending`` so new
                candidates do not cluster around the pending batch.
            progress_callback: Optional synchronous hook invoked at coarse
                milestones (GP fit start, acquisition start/done, etc.).
                See :class:`bo_engine.progress.ProgressEvent` for the
                contract. Backends are free to ignore the callback —
                ``None`` is the default and recovers the silent behavior.

        Returns:
            SuggestionBatch with suggestions and updated state.
        """
        ...

    # ----- Metrics & Diagnostics -----

    def compute_hypervolume(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
    ) -> float | None:
        """Compute hypervolume indicator for multi-objective campaigns.

        Returns None for single-objective or insufficient data.
        """
        ...

    def detect_duplicates(
        self,
        new_params: dict[str, Any],
        existing_params: list[dict[str, Any]],
        tolerance: float,
    ) -> list[DuplicateInfo]:
        """Detect near-duplicate parameter configurations."""
        ...

    def update_state_after_results(
        self,
        spec: OptimizationSpec,
        new_observations: list[ObservationData],
        backend_state: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Update backend-specific state after new results.

        For BoTorch this updates TuRBO trust region. Other backends may
        update their own internal state. Returns None if no state to track.
        """
        ...

    def compute_batch_diversity(
        self,
        spec: OptimizationSpec,
        candidates: list[dict[str, Any]],
    ) -> BatchDiversityMetrics | None:
        """Compute diversity metrics for a batch of candidate points."""
        ...

    def select_methods(
        self,
        spec: OptimizationSpec,
        n_observations: int,
    ) -> dict[str, Any]:
        """Select and explain the methods used for this problem."""
        ...

    def compute_diagnostics(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
        sections: frozenset[str] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Compute model-based diagnostics for a campaign.

        This handles all computation that requires ML framework access
        (model fitting, LOO-CV, feature importance, outlier detection,
        multi-objective Pareto/hypervolume, exploration metrics).

        Server-side concerns (campaign state, suggestion provenance,
        caching, formatting) are NOT part of this method.

        Args:
            spec: Problem specification.
            observations: All observations for the campaign.
            sections: Which diagnostic sections to compute.  When *None*,
                compute all.  Valid values are defined in
                :class:`DiagnosticSection`.
            progress_callback: Optional synchronous hook for coarse
                milestone reporting (see
                :class:`bo_engine.progress.ProgressEvent`). Backends may
                ignore it.

        Returns:
            Dictionary with computed diagnostics.  Keys depend on
            requested sections.  Returns empty sections as ``None``.
        """
        ...
