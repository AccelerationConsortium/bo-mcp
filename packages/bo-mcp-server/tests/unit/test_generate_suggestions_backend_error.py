"""Generation operation maps :class:`BackendError` to typed envelopes.

When the backend raises one of the typed
:class:`bo_engine.backend_base.BackendError` subclasses, the
operation layer must route it through
:func:`bo_mcp_server.errors.make_backend_error_response` so the
envelope carries the right ``ErrorCode`` plus the ``retryable`` /
``retry_after`` hint matched to the subclass — without the
operation layer having to know the per-subclass mapping.

This pins the contract end-to-end so a future backend that raises
a new ``BackendError`` subclass automatically gets the right
envelope behaviour by registering it in
:data:`_BACKEND_ERROR_CODE_MAP`.
"""

from __future__ import annotations

from typing import Any

import pytest
from bo_engine.backend_base import (
    BackendIncompatibilityError,
    BackendInputError,
    BackendInternalError,
    BackendTransientError,
)

from bo_mcp_server.errors import (
    ErrorCode,
    error_code_for_backend_exception,
    make_backend_error_response,
)


@pytest.mark.parametrize(
    ("exc", "expected_code", "expected_retryable"),
    [
        (
            BackendTransientError("optimizer flake"),
            ErrorCode.BACKEND_TRANSIENT_ERROR,
            True,
        ),
        (
            BackendInputError("NaN observations"),
            ErrorCode.VALIDATION_FAILED,
            False,
        ),
        (
            BackendIncompatibilityError("hybrid constraint"),
            ErrorCode.BACKEND_INCOMPATIBILITY,
            False,
        ),
        (
            BackendInternalError("CUDA OOM"),
            ErrorCode.BACKEND_INTERNAL_ERROR,
            False,
        ),
    ],
)
def test_backend_error_subclass_routes_to_expected_envelope(
    exc: Any,
    expected_code: ErrorCode,
    expected_retryable: bool,
) -> None:
    """Each subclass produces a stable envelope shape downstream callers can rely on."""
    assert error_code_for_backend_exception(exc) == expected_code

    response = make_backend_error_response(
        exc, extra_details={"campaign_id": "00000000-0000-0000-0000-000000000000"}
    )
    err = response["error"]
    assert err["code"] == expected_code.value
    assert err["retryable"] is expected_retryable
    if expected_retryable:
        assert isinstance(err["retry_after"], (int, float))
        assert err["retry_after"] > 0
    else:
        assert err["retry_after"] is None
    assert err["details"]["backend_exception"] == type(exc).__name__
    assert err["details"]["campaign_id"] == "00000000-0000-0000-0000-000000000000"


def test_unknown_backend_error_subclass_falls_back_to_internal() -> None:
    """A custom subclass without a registered mapping falls back to INTERNAL_ERROR.

    Operators must never see a backend exception that drops to a
    string-level 500 — the envelope must always carry a stable code
    so dashboards keep working.
    """

    class CustomBackendError(BackendInternalError):
        """Hypothetical custom subclass introduced by a future backend plugin."""

    response = make_backend_error_response(CustomBackendError("novel failure mode"))
    assert response["error"]["code"] == ErrorCode.BACKEND_INTERNAL_ERROR.value
