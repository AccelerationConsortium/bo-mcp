"""Result provenance tracking for Bayesian Optimization.

This module provides functions to track the provenance of results,
linking them back to the suggestions and model versions that generated
them. This is critical for reproducibility and debugging.

Section 3.6 - Missing Trust-Building Features

References:
    - W3C PROV-DM: The PROV Data Model (for provenance concepts)
    - Herschel et al. "A Survey on Provenance" ACM Computing Surveys 2017
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class ProvenanceEventType(Enum):
    """Types of provenance events."""

    SUGGESTION_GENERATED = "suggestion_generated"
    RESULT_SUBMITTED = "result_submitted"
    PARAMETERS_MODIFIED = "parameters_modified"
    MODEL_TRAINED = "model_trained"
    CAMPAIGN_CREATED = "campaign_created"


@dataclass
class ModelSnapshot:
    """Snapshot of model configuration at a point in time.

    Attributes:
        model_type: Type of GP model (e.g., "SingleTaskGP", "ModelListGP").
        kernel_type: Type of kernel used.
        use_input_warping: Whether input warping is enabled.
        lengthscales: Fitted lengthscale values (if available).
        noise_variance: Fitted noise variance.
        n_training_points: Number of training points.
        timestamp: When this snapshot was created.
        config_hash: Hash of the full configuration for comparison.
    """

    model_type: str
    kernel_type: str
    use_input_warping: bool
    lengthscales: dict[str, float] | None
    noise_variance: float | None
    n_training_points: int
    timestamp: datetime
    config_hash: str


@dataclass
class SuggestionProvenance:
    """Provenance information for a suggestion.

    Attributes:
        suggestion_id: Unique identifier for the suggestion.
        campaign_id: Campaign that generated this suggestion.
        iteration: Optimization iteration number.
        batch_index: Index within the batch.
        model_snapshot: Model state when suggestion was generated.
        acquisition_method: Acquisition function used.
        acquisition_value: Value of acquisition function at this point.
        parameters: Suggested parameter values.
        random_seed: Random seed used for generation.
        timestamp: When the suggestion was generated.
    """

    suggestion_id: str
    campaign_id: str
    iteration: int
    batch_index: int
    model_snapshot: ModelSnapshot
    acquisition_method: str
    acquisition_value: float | None
    parameters: dict[str, float]
    random_seed: int | None
    timestamp: datetime


@dataclass
class ResultProvenance:
    """Provenance information for a submitted result.

    Attributes:
        result_id: Unique identifier for the result.
        suggestion_provenance: Provenance of the suggestion that led here.
        parameters_as_executed: Actual parameters used (may differ from suggestion).
        parameter_deviation: How much parameters deviated from suggestion.
        objective_values: Observed objective values.
        constraint_values: Observed constraint values (if any).
        submitted_by: Who submitted the result.
        source: Source of the result (manual, automated, etc.).
        timestamp: When the result was submitted.
        notes: Additional notes or context.
    """

    result_id: str
    suggestion_provenance: SuggestionProvenance | None
    parameters_as_executed: dict[str, float]
    parameter_deviation: float
    objective_values: dict[str, float]
    constraint_values: dict[str, float] | None
    submitted_by: str | None
    source: str
    timestamp: datetime
    notes: str | None = None


@dataclass
class ProvenanceEvent:
    """A single event in the provenance chain.

    Attributes:
        event_type: Type of the event.
        event_id: Unique identifier.
        timestamp: When the event occurred.
        entity_id: ID of the affected entity (suggestion, result, etc.).
        data: Event-specific data.
        previous_event_id: ID of the previous event in the chain.
    """

    event_type: ProvenanceEventType
    event_id: str
    timestamp: datetime
    entity_id: str
    data: dict[str, Any]
    previous_event_id: str | None = None


@dataclass
class ProvenanceChain:
    """Complete provenance chain for an entity.

    Attributes:
        entity_id: ID of the entity (result, suggestion).
        entity_type: Type of entity.
        events: Ordered list of provenance events.
        summary: Human-readable summary.
    """

    entity_id: str
    entity_type: str
    events: list[ProvenanceEvent]
    summary: str


class ProvenanceTracker:
    """Tracks provenance information throughout optimization.

    This class maintains a record of all provenance events and can
    reconstruct the full history of any result or suggestion.
    """

    def __init__(self) -> None:
        """Initialize the provenance tracker."""
        self._events: list[ProvenanceEvent] = []
        self._suggestions: dict[str, SuggestionProvenance] = {}
        self._results: dict[str, ResultProvenance] = {}
        self._model_snapshots: list[ModelSnapshot] = []

    def record_model_training(
        self,
        model_type: str,
        kernel_type: str,
        use_input_warping: bool,
        lengthscales: dict[str, float] | None,
        noise_variance: float | None,
        n_training_points: int,
    ) -> ModelSnapshot:
        """Record a model training event.

        Args:
            model_type: Type of model trained.
            kernel_type: Kernel used.
            use_input_warping: Whether input warping was used.
            lengthscales: Fitted lengthscales.
            noise_variance: Fitted noise variance.
            n_training_points: Number of training points.

        Returns:
            ModelSnapshot for later reference.
        """
        config_dict = {
            "model_type": model_type,
            "kernel_type": kernel_type,
            "use_input_warping": use_input_warping,
            "lengthscales": lengthscales,
            "noise_variance": noise_variance,
            "n_training_points": n_training_points,
        }
        config_hash = _compute_config_hash(config_dict)

        snapshot = ModelSnapshot(
            model_type=model_type,
            kernel_type=kernel_type,
            use_input_warping=use_input_warping,
            lengthscales=lengthscales,
            noise_variance=noise_variance,
            n_training_points=n_training_points,
            timestamp=datetime.now(UTC),
            config_hash=config_hash,
        )

        self._model_snapshots.append(snapshot)

        # Record event
        event = ProvenanceEvent(
            event_type=ProvenanceEventType.MODEL_TRAINED,
            event_id=_generate_id("model"),
            timestamp=snapshot.timestamp,
            entity_id=config_hash,
            data=config_dict,
            previous_event_id=self._events[-1].event_id if self._events else None,
        )
        self._events.append(event)

        return snapshot

    def record_suggestion(
        self,
        suggestion_id: str,
        campaign_id: str,
        iteration: int,
        batch_index: int,
        model_snapshot: ModelSnapshot,
        acquisition_method: str,
        acquisition_value: float | None,
        parameters: dict[str, float],
        random_seed: int | None = None,
    ) -> SuggestionProvenance:
        """Record a suggestion generation event.

        Args:
            suggestion_id: Unique ID for this suggestion.
            campaign_id: Campaign that generated this.
            iteration: Optimization iteration.
            batch_index: Index in the batch.
            model_snapshot: Model state at generation time.
            acquisition_method: Acquisition function used.
            acquisition_value: Acquisition value.
            parameters: Suggested parameters.
            random_seed: Random seed used.

        Returns:
            SuggestionProvenance object.
        """
        provenance = SuggestionProvenance(
            suggestion_id=suggestion_id,
            campaign_id=campaign_id,
            iteration=iteration,
            batch_index=batch_index,
            model_snapshot=model_snapshot,
            acquisition_method=acquisition_method,
            acquisition_value=acquisition_value,
            parameters=parameters,
            random_seed=random_seed,
            timestamp=datetime.now(UTC),
        )

        self._suggestions[suggestion_id] = provenance

        # Record event
        event = ProvenanceEvent(
            event_type=ProvenanceEventType.SUGGESTION_GENERATED,
            event_id=_generate_id("sugg"),
            timestamp=provenance.timestamp,
            entity_id=suggestion_id,
            data={
                "campaign_id": campaign_id,
                "iteration": iteration,
                "batch_index": batch_index,
                "acquisition_method": acquisition_method,
                "acquisition_value": acquisition_value,
                "parameters": parameters,
            },
            previous_event_id=self._events[-1].event_id if self._events else None,
        )
        self._events.append(event)

        return provenance

    def record_result(
        self,
        result_id: str,
        suggestion_id: str | None,
        parameters_as_executed: dict[str, float],
        objective_values: dict[str, float],
        constraint_values: dict[str, float] | None = None,
        submitted_by: str | None = None,
        source: str = "manual",
        notes: str | None = None,
    ) -> ResultProvenance:
        """Record a result submission event.

        Args:
            result_id: Unique ID for this result.
            suggestion_id: ID of the suggestion that led to this (if any).
            parameters_as_executed: Actual parameters used.
            objective_values: Observed objective values.
            constraint_values: Observed constraint values.
            submitted_by: Who submitted this.
            source: Source of the result.
            notes: Additional notes.

        Returns:
            ResultProvenance object.
        """
        # Get suggestion provenance if available
        suggestion_provenance = self._suggestions.get(suggestion_id) if suggestion_id else None

        # Compute parameter deviation
        deviation = 0.0
        if suggestion_provenance:
            for key, value in parameters_as_executed.items():
                if key in suggestion_provenance.parameters:
                    deviation += abs(value - suggestion_provenance.parameters[key])

        provenance = ResultProvenance(
            result_id=result_id,
            suggestion_provenance=suggestion_provenance,
            parameters_as_executed=parameters_as_executed,
            parameter_deviation=deviation,
            objective_values=objective_values,
            constraint_values=constraint_values,
            submitted_by=submitted_by,
            source=source,
            timestamp=datetime.now(UTC),
            notes=notes,
        )

        self._results[result_id] = provenance

        # Record event
        event = ProvenanceEvent(
            event_type=ProvenanceEventType.RESULT_SUBMITTED,
            event_id=_generate_id("result"),
            timestamp=provenance.timestamp,
            entity_id=result_id,
            data={
                "suggestion_id": suggestion_id,
                "parameters": parameters_as_executed,
                "parameter_deviation": deviation,
                "objectives": objective_values,
                "source": source,
            },
            previous_event_id=self._events[-1].event_id if self._events else None,
        )
        self._events.append(event)

        return provenance

    def record_parameter_modification(
        self,
        suggestion_id: str,
        original_parameters: dict[str, float],
        modified_parameters: dict[str, float],
        reason: str | None = None,
    ) -> ProvenanceEvent:
        """Record when parameters are modified from original suggestion.

        Args:
            suggestion_id: ID of the modified suggestion.
            original_parameters: Original suggested values.
            modified_parameters: Modified values.
            reason: Reason for modification.

        Returns:
            ProvenanceEvent for this modification.
        """
        event = ProvenanceEvent(
            event_type=ProvenanceEventType.PARAMETERS_MODIFIED,
            event_id=_generate_id("mod"),
            timestamp=datetime.now(UTC),
            entity_id=suggestion_id,
            data={
                "original": original_parameters,
                "modified": modified_parameters,
                "reason": reason,
            },
            previous_event_id=self._events[-1].event_id if self._events else None,
        )
        self._events.append(event)

        return event

    def get_suggestion_provenance(self, suggestion_id: str) -> SuggestionProvenance | None:
        """Get provenance for a suggestion.

        Args:
            suggestion_id: ID of the suggestion.

        Returns:
            SuggestionProvenance or None if not found.
        """
        return self._suggestions.get(suggestion_id)

    def get_result_provenance(self, result_id: str) -> ResultProvenance | None:
        """Get provenance for a result.

        Args:
            result_id: ID of the result.

        Returns:
            ResultProvenance or None if not found.
        """
        return self._results.get(result_id)

    def get_provenance_chain(self, entity_id: str) -> ProvenanceChain | None:
        """Get the full provenance chain for an entity.

        Reconstructs the complete history of events related to this entity.

        Args:
            entity_id: ID of the entity.

        Returns:
            ProvenanceChain or None if entity not found.
        """
        # Find all events related to this entity
        related_events = [e for e in self._events if e.entity_id == entity_id]

        if not related_events:
            return None

        # Determine entity type
        if entity_id in self._results:
            entity_type = "result"
            # Include suggestion events if linked
            result = self._results[entity_id]
            if result.suggestion_provenance:
                sugg_id = result.suggestion_provenance.suggestion_id
                sugg_events = [e for e in self._events if e.entity_id == sugg_id]
                related_events = sugg_events + related_events
        elif entity_id in self._suggestions:
            entity_type = "suggestion"
        else:
            entity_type = "unknown"

        # Sort by timestamp
        related_events.sort(key=lambda e: e.timestamp)

        # Generate summary
        summary = _generate_provenance_summary(entity_id, entity_type, related_events)

        return ProvenanceChain(
            entity_id=entity_id,
            entity_type=entity_type,
            events=related_events,
            summary=summary,
        )

    def get_all_events(self) -> list[ProvenanceEvent]:
        """Get all recorded provenance events.

        Returns:
            List of all ProvenanceEvent objects.
        """
        return self._events.copy()

    def export_to_dict(self) -> dict[str, Any]:
        """Export all provenance data to a dictionary.

        Useful for serialization and storage.

        Returns:
            Dictionary representation of all provenance data.
        """
        return {
            "events": [
                {
                    "event_type": e.event_type.value,
                    "event_id": e.event_id,
                    "timestamp": e.timestamp.isoformat(),
                    "entity_id": e.entity_id,
                    "data": e.data,
                    "previous_event_id": e.previous_event_id,
                }
                for e in self._events
            ],
            "n_suggestions": len(self._suggestions),
            "n_results": len(self._results),
            "n_model_snapshots": len(self._model_snapshots),
        }


def compute_parameter_deviation(
    suggested: dict[str, float],
    executed: dict[str, float],
    bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Compute detailed deviation between suggested and executed parameters.

    Args:
        suggested: Suggested parameter values.
        executed: Actually executed parameter values.
        bounds: Parameter bounds for normalization.

    Returns:
        Dictionary with per-parameter and aggregate deviation metrics.
    """
    deviations: dict[str, float] = {}
    normalized_deviations: dict[str, float] = {}

    for key in suggested:
        if key in executed:
            diff = abs(executed[key] - suggested[key])
            deviations[key] = diff

            if bounds and key in bounds:
                range_val = bounds[key][1] - bounds[key][0]
                if range_val > 0:
                    normalized_deviations[key] = diff / range_val

    total_deviation = sum(deviations.values())
    max_deviation_key = max(deviations, key=lambda k: deviations[k]) if deviations else None
    max_deviation = deviations[max_deviation_key] if max_deviation_key else 0.0

    avg_normalized = (
        sum(normalized_deviations.values()) / len(normalized_deviations)
        if normalized_deviations
        else 0.0
    )

    return {
        "per_parameter": deviations,
        "normalized": normalized_deviations,
        "total": total_deviation,
        "max_parameter": max_deviation_key,
        "max_deviation": max_deviation,
        "avg_normalized": avg_normalized,
        "is_exact": total_deviation == 0.0,
    }


def _format_result_provenance(provenance: ResultProvenance) -> list[str]:
    """Format a ResultProvenance into report lines."""
    lines = [
        "=== Result Provenance ===",
        f"Result ID: {provenance.result_id}",
        f"Submitted: {provenance.timestamp.isoformat()}",
        f"Source: {provenance.source}",
    ]
    if provenance.submitted_by:
        lines.append(f"Submitted by: {provenance.submitted_by}")
    lines.append("")
    lines.append("Objective Values:")
    for name, value in provenance.objective_values.items():
        lines.append(f"  {name}: {value}")
    lines.append("")
    lines.append("Parameters (as executed):")
    for name, value in provenance.parameters_as_executed.items():
        lines.append(f"  {name}: {value}")

    if provenance.parameter_deviation > 0:
        lines.append("")
        lines.append(f"Parameter deviation from suggestion: {provenance.parameter_deviation:.6f}")

    if provenance.suggestion_provenance:
        sp = provenance.suggestion_provenance
        lines.extend(
            [
                "",
                "--- Linked Suggestion ---",
                f"Suggestion ID: {sp.suggestion_id}",
                f"Generated: {sp.timestamp.isoformat()}",
                f"Iteration: {sp.iteration}",
                f"Acquisition: {sp.acquisition_method}",
            ]
        )
    return lines


def _format_suggestion_provenance(provenance: SuggestionProvenance) -> list[str]:
    """Format a SuggestionProvenance into report lines."""
    lines = [
        "=== Suggestion Provenance ===",
        f"Suggestion ID: {provenance.suggestion_id}",
        f"Campaign: {provenance.campaign_id}",
        f"Generated: {provenance.timestamp.isoformat()}",
        f"Iteration: {provenance.iteration}, Batch Index: {provenance.batch_index}",
        f"Acquisition Method: {provenance.acquisition_method}",
    ]
    if provenance.acquisition_value is not None:
        lines.append(f"Acquisition Value: {provenance.acquisition_value}")
    if provenance.random_seed is not None:
        lines.append(f"Random Seed: {provenance.random_seed}")
    lines.append("")
    lines.append("Suggested Parameters:")
    for name, value in provenance.parameters.items():
        lines.append(f"  {name}: {value}")
    ms = provenance.model_snapshot
    lines.extend(
        [
            "",
            "Model Snapshot:",
            f"  Type: {ms.model_type}",
            f"  Kernel: {ms.kernel_type}",
            f"  Training Points: {ms.n_training_points}",
        ]
    )
    return lines


def format_provenance_report(provenance: ResultProvenance | SuggestionProvenance) -> str:
    """Format provenance information as a human-readable report.

    Args:
        provenance: ResultProvenance or SuggestionProvenance.

    Returns:
        Formatted string report.
    """
    if isinstance(provenance, ResultProvenance):
        lines = _format_result_provenance(provenance)
    else:
        lines = _format_suggestion_provenance(provenance)
    return "\n".join(lines)


def _compute_config_hash(config: dict[str, Any]) -> str:
    """Compute a hash of a configuration dictionary."""
    # Sort keys for consistent hashing
    json_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def _generate_id(prefix: str) -> str:
    """Generate a unique ID with prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _generate_provenance_summary(
    entity_id: str,
    entity_type: str,
    events: list[ProvenanceEvent],
) -> str:
    """Generate a human-readable summary of a provenance chain."""
    lines = [
        f"Provenance for {entity_type} '{entity_id}'",
        f"Total events: {len(events)}",
        "",
        "Timeline:",
    ]

    for event in events:
        timestamp = event.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"  [{timestamp}] {event.event_type.value}")

    return "\n".join(lines)
