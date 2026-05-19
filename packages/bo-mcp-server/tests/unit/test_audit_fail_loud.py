"""Audit logging fail-loud-or-fail-fast contract.

When the audit-event row cannot be persisted, the ``AUDIT_FAILURES``
counter always increments — dashboards alarm regardless of mode — and
the parent tool's failure mode is gated by ``AUDIT_FAILURES_FATAL``:

* unset (default): the parent tool keeps running, audit row is dropped.
* set: the audit error propagates as :class:`AuditPersistenceError`
  so the tool surface converts it into an error envelope.

Compliance contexts (SOC2, ISO 27001) require audit gaps to be visible
to the caller; SOC dashboards (e.g. AWS GuardDuty, Datadog Audit
Trail) recommend a counter-backed alarm pattern over log-only signals.
"""

from __future__ import annotations

import pytest
from prometheus_client import Counter
from sqlalchemy.exc import SQLAlchemyError

from bo_mcp_server import audit as audit_module
from bo_mcp_server.audit import AuditPersistenceError, log_tool_call
from bo_mcp_server.metrics import AUDIT_FAILURES


def _audit_failure_count(tool: str) -> float:
    """Read the ``AUDIT_FAILURES`` counter for ``tool``."""
    metric: Counter = AUDIT_FAILURES.labels(tool)
    return metric._value.get()  # type: ignore[attr-defined]  # noqa: SLF001


class _BrokenRepo:
    """EventRepository stand-in that raises on ``save``."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.saved = False

    async def save(self, _event):  # noqa: ANN001 - test stand-in
        raise SQLAlchemyError("simulated audit-table outage")


@pytest.mark.asyncio
async def test_audit_failure_increments_counter_and_tool_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    setup_database,
) -> None:
    """Default mode: failure is logged + counted, the parent tool succeeds."""
    monkeypatch.setenv("AUDIT_FAILURES_FATAL", "false")
    monkeypatch.setattr(audit_module, "EventRepository", _BrokenRepo)
    before = _audit_failure_count("bo_test_tool")

    # No exception expected: the default contract is "swallow + count".
    await log_tool_call(
        tool_name="bo_test_tool",
        input_summary={"k": "v"},
        output_summary={"success": True},
    )

    after = _audit_failure_count("bo_test_tool")
    assert after == before + 1, "AUDIT_FAILURES must increment on persistence error"


@pytest.mark.asyncio
async def test_audit_failure_propagates_when_fatal_flag_set(
    monkeypatch: pytest.MonkeyPatch,
    setup_database,
) -> None:
    """Compliance mode: failure raises ``AuditPersistenceError`` and counter still bumps."""
    monkeypatch.setenv("AUDIT_FAILURES_FATAL", "true")
    monkeypatch.setattr(audit_module, "EventRepository", _BrokenRepo)
    before = _audit_failure_count("bo_strict_tool")

    with pytest.raises(AuditPersistenceError) as exc_info:
        await log_tool_call(
            tool_name="bo_strict_tool",
            input_summary={"k": "v"},
            output_summary={"success": True},
        )

    # __cause__ preserves the original exception so triage can pin the class.
    assert isinstance(exc_info.value.__cause__, SQLAlchemyError)
    after = _audit_failure_count("bo_strict_tool")
    assert after == before + 1, (
        "AUDIT_FAILURES must still increment under fatal mode so dashboards "
        "alarm on the same signal regardless of the policy knob"
    )


@pytest.mark.asyncio
async def test_audit_unexpected_exception_propagates(
    monkeypatch: pytest.MonkeyPatch,
    setup_database,
) -> None:
    """Programming bugs (``AttributeError``, ``KeyError``) bypass the catch-and-count.

    The audit helper deliberately catches only the documented "row failed
    to persist" exception types. ``AttributeError`` from a misconfigured
    event would otherwise be silently buried under either mode and is
    impossible to triage from operator dashboards.
    """

    class _ProgrammingBugRepo:
        def __init__(self, *_args, **_kwargs) -> None:
            self.saved = False

        async def save(self, _event):  # noqa: ANN001 - test stand-in
            raise AttributeError("missing required attribute on event")

    monkeypatch.setenv("AUDIT_FAILURES_FATAL", "false")
    monkeypatch.setattr(audit_module, "EventRepository", _ProgrammingBugRepo)

    with pytest.raises(AttributeError):
        await log_tool_call(
            tool_name="bo_unexpected",
            input_summary={},
            output_summary={},
        )
