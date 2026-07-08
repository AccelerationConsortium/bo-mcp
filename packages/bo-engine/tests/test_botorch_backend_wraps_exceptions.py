"""BoTorch backend wraps library exceptions before the boundary.

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

from bo_engine.backend_base import (
    BackendInputError,
    BackendInternalError,
    BackendTransientError,
)
from bo_engine.botorch_backend import BoTorchBackend
from bo_engine.initial_design import SearchSpaceExhaustedError
from bo_engine.models import ModelFittingError
from bo_engine.types import (
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
    TurboConfig,
)


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
    with (
        patch(
            "bo_engine.botorch_backend.generate_next_batch",
            side_effect=ValueError("NaN observation"),
        ),
        pytest.raises(BackendInputError) as exc_info,
    ):
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
    with (
        patch(
            "bo_engine.botorch_backend.generate_next_batch",
            side_effect=RuntimeError("CUDA out of memory"),
        ),
        pytest.raises(BackendInternalError) as exc_info,
    ):
        backend.generate_suggestions(
            spec=spec,
            observations=[],
            batch_size=1,
            iteration=0,
        )
    assert exc_info.value.retryable is False
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_model_fitting_error_maps_to_transient_error() -> None:
    """A ``ModelFittingError`` (GP fit flake) becomes a retryable transient error.

    ``ModelFittingError`` subclasses ``RuntimeError`` and would otherwise
    fall through to the terminal ``BackendInternalError``. The
    ``BackendTransientError`` docstring names exactly this case — a
    singular-matrix / numerical-instability hiccup during GP fitting that
    a different seed may clear — as the canonical retryable failure, so
    the wrapper's domain map must surface it with ``retryable=True`` so
    clients back off and retry instead of treating it as a hard bug.
    """
    backend = BoTorchBackend()
    spec = _two_param_spec()
    with (
        patch(
            "bo_engine.botorch_backend.generate_next_batch",
            side_effect=ModelFittingError(
                "singular covariance matrix",
                original_error=RuntimeError("cholesky failed"),
            ),
        ),
        pytest.raises(BackendTransientError) as exc_info,
    ):
        backend.generate_suggestions(
            spec=spec,
            observations=[],
            batch_size=1,
            iteration=0,
        )
    assert exc_info.value.retryable is True
    assert isinstance(exc_info.value.__cause__, ModelFittingError)


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

    with (
        patch("bo_engine.botorch_backend.generate_next_batch", side_effect=_exhausted),
        pytest.raises(SearchSpaceExhaustedError),
    ):
        backend.generate_suggestions(
            spec=spec,
            observations=[],
            batch_size=10,
            iteration=0,
        )


def test_corrupted_turbo_state_in_generate_raises_typed_error() -> None:
    """A corrupted persisted state envelope surfaces as ``BackendInputError``.

    State restoration runs before ``generate_next_batch``, i.e. outside
    the compute try-block — without its own boundary a valid envelope
    with an empty payload would cross the backend boundary as a raw
    ``KeyError('dim')``, violating the base-class contract that only
    ``BackendError`` subclasses leak.
    """
    backend = BoTorchBackend()
    spec = OptimizationSpec(
        parameters=[
            ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="y", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="z", minimize=True)],
        turbo_config=TurboConfig(),
    )

    with pytest.raises(BackendInputError) as exc_info:
        backend.generate_suggestions(
            spec=spec,
            observations=[],
            batch_size=1,
            iteration=1,
            backend_state={"backend": "botorch", "schema_version": 1, "payload": {}},
        )
    assert "state" in str(exc_info.value).lower()
    assert isinstance(exc_info.value.__cause__, KeyError)


def test_foreign_backend_state_in_update_raises_typed_error() -> None:
    """``update_state_after_results`` types foreign-envelope failures too.

    ``unwrap_state`` raises a raw ``ValueError`` for an envelope written
    by a different backend; the update path must convert it to the same
    typed ``BackendInputError`` as the generate path.
    """
    backend = BoTorchBackend()
    spec = _two_param_spec()

    with pytest.raises(BackendInputError):
        backend.update_state_after_results(
            spec=spec,
            new_observations=[],
            backend_state={"backend": "baybe", "schema_version": 1, "payload": {}},
        )
