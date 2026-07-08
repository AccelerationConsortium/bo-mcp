"""Reusable backend-contract assertions.

The functions here run a backend through the surfaces every concrete
backend has to honor — JSON-serializable responses, deterministic
initial design, structured capability reports, and state envelope
wrapping. The tests in ``packages/bo-engine/tests/test_backend_contract.py``
exercise this module against the BoTorch backend and a tiny fake
backend; the BayBE package re-imports the same helpers so both
implementations stay in lockstep.

The helpers raise ``AssertionError`` on contract violations so pytest
displays the failing assertion verbatim; they return ``None`` on
success.
"""

from __future__ import annotations

import json
from typing import Any

from bo_engine.backend import BOBackend, SuggestionBatch
from bo_engine.backend_base import (
    BackendStateEnvelope,
    BackendValidationResult,
    is_state_envelope,
)
from bo_engine.types import ObservationData, OptimizationSpec


def assert_json_serializable(value: object, *, context: str = "value") -> None:
    """Raise ``AssertionError`` if ``value`` cannot survive ``json.dumps``."""
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        msg = f"{context} is not JSON-serializable: {exc}"
        raise AssertionError(msg) from exc


def assert_validate_capabilities(
    backend: BOBackend,
    spec: OptimizationSpec,
) -> BackendValidationResult:
    """Verify ``validate_capabilities`` returns a typed result with the expected identity.

    Asserts that the call yields a :class:`BackendValidationResult` and that
    its ``backend`` field matches ``backend.name``.
    """
    result = backend.validate_capabilities(spec)
    assert isinstance(result, BackendValidationResult), (
        f"validate_capabilities must return BackendValidationResult, got {type(result)!r}"
    )
    assert result.backend == backend.name, (
        f"BackendValidationResult.backend={result.backend!r}, expected {backend.name!r}"
    )
    return result


def assert_parameter_options_schema(backend: BOBackend) -> dict[str, Any] | None:
    """``parameter_options_schema`` returns ``None`` or a JSON-safe object schema.

    Every backend (protocol-only plugins included) must answer the hook.
    A non-``None`` fragment must be a JSON-serializable ``dict`` describing
    an object (so the schema-extension layer can splice it under the
    backend name); ``None`` means "no typed parameter options".
    """
    fragment = backend.parameter_options_schema()
    if fragment is None:
        return None
    assert isinstance(fragment, dict), (
        f"parameter_options_schema must return dict|None, got {type(fragment)!r}"
    )
    assert fragment.get("type") == "object", (
        "parameter_options_schema fragment must describe a JSON object "
        f"(type='object'), got type={fragment.get('type')!r}"
    )
    assert_json_serializable(fragment, context="parameter_options_schema fragment")
    return fragment


def assert_backend_options_schema(backend: BOBackend) -> dict[str, Any] | None:
    """``backend_options_schema`` returns ``None`` or a JSON-safe object schema.

    The per-campaign twin of :func:`assert_parameter_options_schema`:
    every backend (protocol-only plugins included) must answer the hook.
    A non-``None`` fragment must be a JSON-serializable ``dict``
    describing an object; ``None`` means "no typed backend options".
    """
    fragment = backend.backend_options_schema()
    if fragment is None:
        return None
    assert isinstance(fragment, dict), (
        f"backend_options_schema must return dict|None, got {type(fragment)!r}"
    )
    assert fragment.get("type") == "object", (
        "backend_options_schema fragment must describe a JSON object "
        f"(type='object'), got type={fragment.get('type')!r}"
    )
    assert_json_serializable(fragment, context="backend_options_schema fragment")
    return fragment


def assert_initial_design_contract(
    backend: BOBackend,
    spec: OptimizationSpec,
    n_points: int = 3,
) -> list[dict[str, Any]]:
    """Initial-design rows must contain every parameter and be JSON-safe."""
    designs = backend.generate_initial_design(spec, n_points=n_points)
    assert len(designs) == n_points, (
        f"initial-design returned {len(designs)} rows, expected {n_points}"
    )
    expected_names = {p.name for p in spec.parameters}
    for design in designs:
        assert set(design.keys()) >= expected_names, (
            f"initial-design row {design} missing parameter names {expected_names}"
        )
        assert_json_serializable(design, context="initial design row")
    return designs


def assert_suggestion_batch_contract(
    backend: BOBackend,
    spec: OptimizationSpec,
    observations: list[ObservationData],
    batch_size: int = 1,
    iteration: int = 1,
) -> SuggestionBatch:
    """``generate_suggestions`` returns a JSON-safe batch with provenance."""
    batch = backend.generate_suggestions(
        spec=spec,
        observations=observations,
        batch_size=batch_size,
        iteration=iteration,
    )
    assert isinstance(batch, SuggestionBatch)
    assert len(batch.suggestions) == batch_size, (
        f"suggestion batch returned {len(batch.suggestions)} rows, expected {batch_size}"
    )
    for entry in batch.suggestions:
        assert "parameter_values" in entry, "suggestion missing 'parameter_values'"
        assert "provenance" in entry, "suggestion missing 'provenance'"
        assert_json_serializable(entry, context="suggestion entry")
    if batch.backend_state is not None:
        assert_json_serializable(batch.backend_state, context="backend_state")
    assert_json_serializable(batch.method_info, context="method_info")
    return batch


def assert_state_envelope_contract(state: dict[str, Any] | None, backend_name: str) -> None:
    """When state is present and follows the envelope shape, validate it.

    Legacy bare payloads are still allowed — backends that have not yet
    moved to :class:`BackendStateEnvelope` are not in violation.
    """
    if state is None:
        return
    if not is_state_envelope(state):
        return
    envelope = BackendStateEnvelope.from_dict(state)
    assert envelope.backend == backend_name, (
        f"state envelope backend={envelope.backend!r}, expected {backend_name!r}"
    )
    assert envelope.schema_version >= 1, "state envelope schema_version must be >= 1"
