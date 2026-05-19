"""BoTorch backend wraps library exceptions before the boundary (TODO 8.13).

The audit text: ``packages/bo-engine/src/bo_engine/botorch_backend.py``
lets library exceptions propagate. The server-layer code has nothing
reliable to dispatch on.

The fix: the backend wraps non-domain exceptions with
:func:`wrap_backend_exception`. Domain-level signals
(``SearchSpaceExhaustedError``) continue to propagate unwrapped so
the operations layer can handle them with their existing
``ErrorCode.SEARCH_SPACE_EXHAUSTED`` envelope.

Reference patterns:
- Pydantic v2 wraps internal validation errors with ValidationError
  before they leave the public boundary
  (https://docs.pydantic.dev/latest/concepts/errors/).
- AWS SDK boto3 wraps service errors in ``ClientError`` for the same
  reason — single dispatch type at the caller boundary.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from bo_engine.backend_base import BackendInputError, BackendInternalError
from bo_engine.botorch_backend import BoTorchBackend
from bo_engine.initial_design import SearchSpaceExhaustedError
from bo_engine.types import ObjectiveSpec, OptimizationSpec, ParameterSpec, ParameterType


def _two_param_spec() -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="y", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="z", minimize=False)],
    )


def test_value_error_in_generate_is_wrapped() -> None:
    """A ``ValueError`` raised inside ``generate_next_batch`` surfaces as input error.

    Without the wrap, the operations layer cannot distinguish a
    user-input bug from a transient flake — both arrive as
    ``ValueError``. With it, the operations layer maps the typed
    exception to ``VALIDATION_FAILED`` with a stable retry hint.
    """
    backend = BoTorchBackend()
    spec = _two_param_spec()
    with patch(
        "bo_engine.botorch_backend.generate_next_batch",
        side_effect=ValueError("NaN observation"),
    ):
        with pytest.raises(BackendInputError) as exc_info:
            backend.generate_suggestions(
                spec=spec,
                observations=[],
                batch_size=1,
                iteration=0,
            )
    assert exc_info.value.retryable is False
    # The original ValueError is preserved on __cause__ so operators
    # can still triage the underlying library exception.
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_runtime_error_in_generate_maps_to_internal_error() -> None:
    """Unexpected ``RuntimeError`` (e.g. CUDA OOM) becomes :class:`BackendInternalError`.

    Internal errors are the catch-all bucket operators triage on.
    They must keep ``retryable=False`` so clients do not produce
    retry storms against a deterministic library bug.
    """
    backend = BoTorchBackend()
    spec = _two_param_spec()
    with patch(
        "bo_engine.botorch_backend.generate_next_batch",
        side_effect=RuntimeError("CUDA out of memory"),
    ):
        with pytest.raises(BackendInternalError) as exc_info:
            backend.generate_suggestions(
                spec=spec,
                observations=[],
                batch_size=1,
                iteration=0,
            )
    assert exc_info.value.retryable is False
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_search_space_exhausted_propagates_unwrapped() -> None:
    """The domain ``SearchSpaceExhaustedError`` must not be wrapped.

    The operations layer has a dedicated handler for it that maps
    to ``ErrorCode.SEARCH_SPACE_EXHAUSTED``. Wrapping would
    accidentally re-route it through the BackendError mapper and
    lose the dedicated recovery_action.
    """
    backend = BoTorchBackend()
    spec = _two_param_spec()

    def _exhausted(*_args: Any, **_kwargs: Any) -> None:
        raise SearchSpaceExhaustedError(
            n_requested=10,
            n_available=0,
            n_total_combinations=4,
        )

    with patch("bo_engine.botorch_backend.generate_next_batch", side_effect=_exhausted):
        with pytest.raises(SearchSpaceExhaustedError):
            backend.generate_suggestions(
                spec=spec,
                observations=[],
                batch_size=10,
                iteration=0,
            )
