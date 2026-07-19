"""BoTorch persists a raw Sobol continuation cursor in ``backend_state``.

The warm-up offset must be the raw Sobol index the accumulator reached, not the
issued-point count. Under a tight constraint each candidate costs many raw draws,
so an issue-count offset rescans the same window every call and eventually stalls
permanently. These tests pin the cursor round-trip and the resulting liveness.
"""

from __future__ import annotations

import pytest

from bo_engine.backend_base import BackendInputError
from bo_engine.botorch_backend import (
    BoTorchBackend,
    _dict_to_turbo_state,
    _next_sobol_cursor,
    _pack_backend_payload,
    _turbo_state_to_dict,
    _unpack_backend_payload,
)
from bo_engine.turbo import TurboState
from bo_engine.types import (
    ConstraintSpec,
    ConstraintType,
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
    SuggestionResult,
)


def _tight_spec() -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
        constraints=[
            ConstraintSpec(type=ConstraintType.SUM_LESS_THAN, parameters=["x"], value=1e-4)
        ],
        batch_size=1,
        initial_design_size=50,  # keep every call in warm-up
        random_seed=17,
    )


def _sr(sobol_index: int | None) -> SuggestionResult:
    return SuggestionResult(
        parameter_values={"x": 0.0},
        iteration=1,
        batch_index=0,
        generation_method="initial_design",
        random_seed=1,
        sobol_index=sobol_index,
    )


def test_pack_unpack_roundtrip_and_legacy_state():
    # New additive payload round-trips both fields while leaving legacy TuRBO
    # fields top-level for older workers during a rolling deployment.
    turbo = {"length": 0.8, "dim": 1}
    packed = _pack_backend_payload(turbo, 512)
    assert packed == {"length": 0.8, "dim": 1, "sobol_cursor": 512}
    assert _unpack_backend_payload(packed) == (turbo, 512)

    # Legacy bare TurboState dict (no combined-format keys) → cursor None.
    assert _unpack_backend_payload(turbo) == (turbo, None)

    # Nothing to persist collapses to None.
    assert _pack_backend_payload(None, None) is None
    assert _unpack_backend_payload(None) == (None, None)

    # The transient nested development format remains readable.
    nested = {"turbo_state": turbo, "sobol_cursor": 513}
    assert _unpack_backend_payload(nested) == (turbo, 513)


def test_flat_cursor_payload_remains_readable_by_legacy_turbo_deserializer():
    """An older worker ignores the additive cursor once TuRBO state exists."""
    turbo_state = TurboState(dim=2, batch_size=1)
    turbo_dict = _turbo_state_to_dict(turbo_state)
    payload = _pack_backend_payload(turbo_dict, 512)

    assert payload is not None
    restored = _dict_to_turbo_state(payload)
    assert restored == turbo_state


@pytest.mark.parametrize("cursor", [-1, True, "12", 1.5])
def test_invalid_cursor_is_rejected_as_corrupted_state(cursor):
    with pytest.raises(ValueError, match="non-negative integer"):
        _unpack_backend_payload({"sobol_cursor": cursor})


def test_backend_wraps_invalid_cursor_as_typed_input_error():
    backend = BoTorchBackend()
    corrupted = backend.wrap_state({"sobol_cursor": "not-an-int"})

    with pytest.raises(BackendInputError, match="could not restore"):
        backend.generate_suggestions(
            spec=_tight_spec(),
            observations=[],
            batch_size=1,
            iteration=1,
            backend_state=corrupted,
            initial_design_history=[],
        )


def test_next_cursor_advances_past_last_index_or_holds():
    # max(sobol_index) + 1, so the next scan resumes just past the last candidate.
    assert _next_sobol_cursor([_sr(3), _sr(70), _sr(41)], previous=0) == 71
    # A model-guided batch stamps no index → cursor unchanged.
    assert _next_sobol_cursor([_sr(None)], previous=512) == 512
    # Defensive monotonicity: malformed/unseeded result indices cannot rewind
    # an already-persisted cursor.
    assert _next_sobol_cursor([_sr(3)], previous=512) == 512


def test_backend_persists_cursor_so_repeated_rejection_does_not_stall():
    """Round-tripping ``backend_state`` keeps warm-up live under a tight constraint.

    Nine generate+retire cycles must all succeed. Without the persisted cursor the
    ninth call rescans the exhausted window and the backend raises.
    """
    backend = BoTorchBackend()
    spec = _tight_spec()

    history: list[dict] = []
    state: dict | None = None
    xs: list[float] = []
    for _ in range(9):
        batch = backend.generate_suggestions(
            spec=spec,
            observations=[],
            batch_size=1,
            iteration=1,
            backend_state=state,
            initial_design_history=list(history),
        )
        assert len(batch.suggestions) == 1
        params = batch.suggestions[0]["parameter_values"]
        assert params["x"] <= 1e-4 + 1e-9  # feasible
        xs.append(params["x"])
        history.append(params)  # retire → stays excluded
        state = batch.backend_state  # persist the cursor (server round-trip)

    # The persisted cursor is present and advanced well past nine issued points.
    _, cursor = _unpack_backend_payload(backend.unwrap_state(state))
    assert cursor is not None
    assert cursor > len(xs)

    # All nine points are distinct within the engine dedup tolerance.
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            assert abs(xs[i] - xs[j]) > 1e-6
