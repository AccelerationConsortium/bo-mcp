"""Tests for ``retryable`` / ``retry_after`` on the error envelope.

Covers TODO 8.14: every structured error response must carry a
stable ``retryable`` boolean and a ``retry_after`` hint (or
``None``) so LLM clients and HTTP retry middleware can decide
between fail-fast and exponential backoff without re-parsing the
recovery action string.

The exhaustive enum-coverage test mirrors the strategy described
in the original audit (TODO 8.14, test strategy): every
:class:`ErrorCode` is exercised and asserts a stable retryable
flag. Patterns echo the AWS / Stripe approach to idempotency
errors documented in
https://stripe.com/docs/api/idempotent_requests and
https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/.
"""

from __future__ import annotations

import pytest
from bo_engine.backend_base import (
    BackendIncompatibilityError,
    BackendInputError,
    BackendInternalError,
    BackendTransientError,
)

from bo_mcp_server.errors import (
    ERROR_CODE_RETRY_HINTS,
    ErrorCode,
    error_code_for_backend_exception,
    make_backend_error_response,
    make_error_response,
)


class TestRetryHintCoverage:
    """Every error code must declare a retry hint."""

    def test_every_error_code_has_retry_hint(self) -> None:
        """No silent ``None`` lookups — every code must opt in or out explicitly."""
        for code in ErrorCode:
            assert code in ERROR_CODE_RETRY_HINTS, (
                f"{code.name} missing entry in ERROR_CODE_RETRY_HINTS"
            )

    def test_retryable_codes_carry_a_backoff_hint(self) -> None:
        """A retryable response without a backoff hint encourages retry storms.

        Concurrency / transient codes always pair retryable=True with
        a positive retry_after; deterministic codes pair retryable=
        False with ``None``.
        """
        for code, (retryable, retry_after) in ERROR_CODE_RETRY_HINTS.items():
            if retryable:
                assert retry_after is not None and retry_after > 0, (
                    f"{code.name} is retryable but has no backoff hint"
                )
            else:
                assert retry_after is None, (
                    f"{code.name} is not retryable but carries a retry_after"
                )


class TestEnvelopeShape:
    """``make_error_response`` always emits ``retryable`` and ``retry_after``."""

    @pytest.mark.parametrize("code", list(ErrorCode))
    def test_every_envelope_carries_retry_fields(self, code: ErrorCode) -> None:
        """Stable shape across every error code so consumers can rely on it."""
        response = make_error_response(code)
        error = response["error"]
        assert "retryable" in error
        assert "retry_after" in error
        retryable, retry_after = ERROR_CODE_RETRY_HINTS[code]
        assert error["retryable"] is retryable
        assert error["retry_after"] == retry_after

    def test_retry_after_can_be_overridden_via_details(self) -> None:
        """Per-call backoff overrides survive into the envelope.

        Lets call sites tighten or relax the suggested backoff
        without redefining a new ErrorCode. Mirrors how
        :func:`make_concurrent_modification_response` already
        smuggles ``retry_after_seconds`` through ``details`` for
        clients that read the legacy field.
        """
        response = make_error_response(
            ErrorCode.CONCURRENT_MODIFICATION,
            details={"retry_after_seconds": 5.0},
        )
        assert response["error"]["retry_after"] == pytest.approx(5.0)


class TestIdempotencyConflictCode:
    """``IDEMPOTENCY_CONFLICT`` is distinct from generic ``VALIDATION_FAILED``."""

    def test_conflict_has_its_own_error_code(self) -> None:
        """Clients can programmatically detect key-reuse via the code field."""
        response = make_error_response(ErrorCode.IDEMPOTENCY_CONFLICT)
        assert response["error"]["code"] == "E015"
        assert response["error"]["retryable"] is False


class TestBackendErrorResponseMapping:
    """``BackendError`` subclasses map to the right ErrorCode + retry hint."""

    @pytest.mark.parametrize(
        ("exception", "expected_code", "expected_retryable"),
        [
            (BackendTransientError("flake"), ErrorCode.BACKEND_TRANSIENT_ERROR, True),
            (BackendInputError("bad bounds"), ErrorCode.VALIDATION_FAILED, False),
            (
                BackendIncompatibilityError("hybrid constraint"),
                ErrorCode.BACKEND_INCOMPATIBILITY,
                False,
            ),
            (BackendInternalError("CUDA OOM"), ErrorCode.BACKEND_INTERNAL_ERROR, False),
        ],
    )
    def test_mapping(
        self,
        exception: object,
        expected_code: ErrorCode,
        expected_retryable: bool,
    ) -> None:
        from bo_engine.backend_base import BackendError

        assert isinstance(exception, BackendError)
        assert error_code_for_backend_exception(exception) == expected_code

        response = make_backend_error_response(exception)
        assert response["error"]["code"] == expected_code.value
        assert response["error"]["retryable"] is expected_retryable
        assert response["error"]["details"]["backend_exception"] == type(exception).__name__

    def test_cause_chain_surfaces_in_details(self) -> None:
        """Operators need the original library exception class name for triage."""
        wrapped = BackendInternalError("unexpected", cause=RuntimeError("CUDA OOM"))
        response = make_backend_error_response(wrapped)
        assert "RuntimeError" in response["error"]["details"]["cause"]
