"""Tests for the backend-agnostic exception hierarchy.

Covers TODO 8.13: backends translate library exceptions into the
typed :class:`bo_engine.backend_base.BackendError` hierarchy so the
server layer can dispatch on a stable type instead of grepping
exception strings.

The strategy mirrors patterns from
https://docs.python.org/3/tutorial/errors.html#user-defined-exceptions
and the Pydantic v2 exception-translation guidance
(https://docs.pydantic.dev/latest/concepts/errors/) — wrap at the
boundary, surface the original via ``__cause__``, keep ``retryable``
as the load-bearing dispatch flag.
"""

from __future__ import annotations

import pytest

from bo_engine.backend_base import (
    BackendError,
    BackendIncompatibilityError,
    BackendInputError,
    BackendInternalError,
    BackendTransientError,
    wrap_backend_exception,
)


class TestBackendExceptionHierarchy:
    """The four backend exception types form a single retryable hierarchy."""

    def test_subclasses_inherit_from_root(self) -> None:
        """Every typed backend exception must be a :class:`BackendError`."""
        assert issubclass(BackendIncompatibilityError, BackendError)
        assert issubclass(BackendInputError, BackendError)
        assert issubclass(BackendTransientError, BackendError)
        assert issubclass(BackendInternalError, BackendError)

    def test_retryable_flag_matches_documented_semantics(self) -> None:
        """Only the transient subclass is marked retryable.

        Misclassifying a deterministic failure as transient produces
        retry storms; the four-way split is what makes the retry hint
        meaningful.
        """
        assert BackendIncompatibilityError().retryable is False
        assert BackendInputError().retryable is False
        assert BackendInternalError().retryable is False
        assert BackendTransientError().retryable is True

    def test_cause_is_chained_for_diagnostics(self) -> None:
        """The wrapped library exception must remain reachable via ``__cause__``."""
        root = ValueError("bad bounds")
        wrapped = BackendInputError("backend rejected spec", cause=root)
        assert wrapped.__cause__ is root


class TestWrapBackendException:
    """``wrap_backend_exception`` produces the correct subclass per input."""

    def test_value_error_maps_to_input_error(self) -> None:
        """:class:`ValueError` always indicates a caller-input bug."""
        wrapped = wrap_backend_exception(
            ValueError("NaN in observations"),
            backend_name="botorch",
        )
        assert isinstance(wrapped, BackendInputError)
        assert wrapped.retryable is False
        assert wrapped.__cause__ is not None

    def test_type_error_maps_to_input_error(self) -> None:
        """:class:`TypeError` also signals a malformed input shape."""
        wrapped = wrap_backend_exception(
            TypeError("expected float, got str"),
            backend_name="botorch",
        )
        assert isinstance(wrapped, BackendInputError)

    def test_unrecognised_exception_maps_to_internal(self) -> None:
        """Unexpected library exceptions become :class:`BackendInternalError`.

        This is the catch-all bucket operators triage on. Subclassing
        :class:`RuntimeError` is the canonical "something unexpected
        happened in the library" surface in the Python ecosystem.
        """
        wrapped = wrap_backend_exception(
            RuntimeError("CUDA OOM"),
            backend_name="botorch",
        )
        assert isinstance(wrapped, BackendInternalError)
        assert wrapped.retryable is False

    def test_backend_error_passes_through_unchanged(self) -> None:
        """A backend that already classified its error keeps the subtype.

        Re-wrapping would lose the deliberate classification (e.g. a
        backend that detected a transient flake and raised
        :class:`BackendTransientError` must not be down-graded to
        :class:`BackendInternalError`).
        """
        transient = BackendTransientError("optimizer flake")
        result = wrap_backend_exception(transient, backend_name="botorch")
        assert result is transient
        assert result.retryable is True

    def test_backend_name_appears_in_message(self) -> None:
        """The diagnostic identifies the backend that failed."""
        wrapped = wrap_backend_exception(
            ValueError("bad bounds"),
            backend_name="custom-backend",
        )
        assert "custom-backend" in str(wrapped)


class TestBackendErrorRaiseChain:
    """Backends must wrap library exceptions before crossing the boundary."""

    def test_wrap_then_raise_pattern(self) -> None:
        """The expected backend integration pattern preserves ``__cause__``.

        Reference: PEP 3134 (Exception Chaining) — ``raise X from Y``
        sets ``X.__cause__ = Y``. Our wrap helper does the same
        explicitly so backends do not have to repeat the boilerplate.
        """
        root = ValueError("NaN")

        def _raise_root() -> None:
            raise root

        def _wrap_and_reraise() -> None:
            try:
                _raise_root()
            except Exception as exc:
                raise wrap_backend_exception(exc, backend_name="botorch") from exc

        with pytest.raises(BackendInputError) as exc_info:
            _wrap_and_reraise()
        assert exc_info.value.__cause__ is root
