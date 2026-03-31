"""Shared helpers for operations.

Deduplicates converters used by multiple operations (generate_suggestions,
submit_results) so bug fixes apply in one place.
"""

from typing import Any

from bo_engine.turbo import TurboState
from bo_engine.types import ObservationData

from bo_mcp_server.domain import Result


def results_to_observations(results: list[Result]) -> list[ObservationData]:
    """Convert domain Result objects to ObservationData for bo-engine."""
    return [
        ObservationData(
            parameter_values=r.parameter_values,
            objective_values=r.objective_values,
            cost=r.metadata.get("cost") if r.metadata else None,
        )
        for r in results
    ]


def turbo_state_to_dict(state: TurboState) -> dict[str, Any]:
    """Serialize TurboState to dictionary for JSON storage."""
    return {
        "dim": state.dim,
        "batch_size": state.batch_size,
        "length": state.length,
        "length_min": state.length_min,
        "length_max": state.length_max,
        "failure_counter": state.failure_counter,
        "failure_tolerance": state.failure_tolerance,
        "success_counter": state.success_counter,
        "success_tolerance": state.success_tolerance,
        "best_value": state.best_value,
        "restart_triggered": state.restart_triggered,
    }


def dict_to_turbo_state(data: dict[str, Any]) -> TurboState:
    """Deserialize dictionary to TurboState."""
    return TurboState(
        dim=data["dim"],
        batch_size=data["batch_size"],
        length=data["length"],
        length_min=data["length_min"],
        length_max=data["length_max"],
        failure_counter=data["failure_counter"],
        failure_tolerance=data["failure_tolerance"],
        success_counter=data["success_counter"],
        success_tolerance=data["success_tolerance"],
        best_value=data["best_value"],
        restart_triggered=data["restart_triggered"],
    )
