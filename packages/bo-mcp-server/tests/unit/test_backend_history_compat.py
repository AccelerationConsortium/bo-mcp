"""Backward compatibility for the ``initial_design_history`` protocol addition.

``initial_design_history`` was added to ``BOBackend.generate_suggestions`` after
the first public protocol. A third-party plugin implementing the *older*
signature must keep working: the server forwards the new keyword only to
backends that accept it, so a legacy backend is driven via observation/pending
exclusion instead of raising ``TypeError: unexpected keyword argument`` (which
would escape the typed backend-error path).
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from bo_engine.backend import BOBackend, SuggestionBatch
from bo_engine.types import ObjectiveSpec, OptimizationSpec, ParameterSpec, ParameterType
from bo_mcp_server.operations.generate_suggestions import (
    _backend_accepts_initial_design_history,
    _generate_via_backend,
)

_PROVENANCE = {
    "iteration": 1,
    "batch_index": 0,
    "generation_method": "initial_design",
    "random_seed": 1,
}


def _spec() -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
        batch_size=1,
    )


def _batch(value: float) -> SuggestionBatch:
    return SuggestionBatch(
        suggestions=[{"parameter_values": {"x": value}, "provenance": dict(_PROVENANCE)}],
        method_info={},
        backend_state=None,
        warnings=[],
    )


class _LegacyBackend:
    """Duck-typed backend with the pre-``initial_design_history`` signature."""

    name = "legacy"

    def generate_suggestions(
        self,
        spec,
        observations,
        batch_size,
        iteration,
        backend_state=None,
        pending_points=None,
        progress_callback=None,
    ) -> SuggestionBatch:
        _ = (spec, observations, batch_size, iteration, backend_state, pending_points)
        _ = progress_callback
        return _batch(0.25)


class _NewBackend:
    """Duck-typed backend that accepts the new keyword and records it."""

    name = "new"

    def __init__(self) -> None:
        self.received_history: object = "unset"

    def generate_suggestions(
        self,
        spec,
        observations,
        batch_size,
        iteration,
        backend_state=None,
        pending_points=None,
        progress_callback=None,
        initial_design_history=None,
    ) -> SuggestionBatch:
        _ = (spec, observations, batch_size, iteration, backend_state, pending_points)
        _ = progress_callback
        self.received_history = initial_design_history
        return _batch(0.75)


class _KwargsBackend:
    """Backend that absorbs future keywords via ``**kwargs``."""

    name = "kwargs"

    def generate_suggestions(self, spec, observations, batch_size, iteration, **kwargs) -> Any:
        _ = (spec, observations, batch_size, iteration, kwargs)
        return _batch(0.5)


def test_helper_distinguishes_legacy_new_and_kwargs_backends():
    assert _backend_accepts_initial_design_history(cast("BOBackend", _NewBackend())) is True
    assert _backend_accepts_initial_design_history(cast("BOBackend", _KwargsBackend())) is True
    assert _backend_accepts_initial_design_history(cast("BOBackend", _LegacyBackend())) is False


@pytest.mark.asyncio
async def test_legacy_backend_runs_without_receiving_history():
    """A legacy backend must not receive the keyword and must not raise TypeError."""
    backend = _LegacyBackend()
    data, _state, _warnings, _method = await _generate_via_backend(
        cast("BOBackend", backend),
        _spec(),
        [],
        1,
        1,
        None,
        pending_parameter_values=[],
        initial_design_history=[{"x": 0.25}],  # supplied, but must not be forwarded
        progress_callback=None,
    )
    assert len(data) == 1


@pytest.mark.asyncio
async def test_new_backend_receives_history():
    """A backend advertising the parameter is forwarded the authoritative history."""
    backend = _NewBackend()
    history = [{"x": 0.1}, {"x": 0.2}]
    await _generate_via_backend(
        cast("BOBackend", backend),
        _spec(),
        [],
        1,
        1,
        None,
        pending_parameter_values=[],
        initial_design_history=history,
        progress_callback=None,
    )
    assert backend.received_history == history
