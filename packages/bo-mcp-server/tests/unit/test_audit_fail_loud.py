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
from bo_mcp_server.audit import (
    MAX_SUMMARY_VALUE_LENGTH,
    AuditPersistenceError,
    extract_audit_campaign_id,
    log_tool_call,
    summarize_tool_arguments,
    summarize_tool_result,
)
from bo_mcp_server.metrics import AUDIT_FAILURES

pytestmark = pytest.mark.usefixtures("setup_database")


def _audit_failure_count(tool: str) -> float:
    """Read the ``AUDIT_FAILURES`` counter for ``tool``."""
    metric: Counter = AUDIT_FAILURES.labels(tool)
    return metric._value.get()  # type: ignore[attr-defined]


class _BrokenRepo:
    """EventRepository stand-in that raises on ``save``."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.saved = False

    async def save(self, _event):
        msg = "simulated audit-table outage"
        raise SQLAlchemyError(msg)


@pytest.mark.asyncio
async def test_audit_failure_increments_counter_and_tool_succeeds(
    monkeypatch: pytest.MonkeyPatch,
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

        async def save(self, _event):
            msg = "missing required attribute on event"
            raise AttributeError(msg)

    monkeypatch.setenv("AUDIT_FAILURES_FATAL", "false")
    monkeypatch.setattr(audit_module, "EventRepository", _ProgrammingBugRepo)

    with pytest.raises(AttributeError):
        await log_tool_call(
            tool_name="bo_unexpected",
            input_summary={},
            output_summary={},
        )


class TestSummaries:
    """Compact-summary builders used by the tool-boundary audit hook.

    The audit row records *what* was called, never the full payload:
    the summaries must stay bounded regardless of the argument or
    response size so an audit write is always cheap and cannot leak
    a full intake spec into the events table.
    """

    def test_arguments_scalars_pass_through(self) -> None:
        summary = summarize_tool_arguments(
            {"campaign_id": "abc", "batch_size": 3, "dry_run": False, "beta": 0.5, "none": None}
        )
        assert summary == {
            "campaign_id": "abc",
            "batch_size": 3,
            "dry_run": False,
            "beta": 0.5,
            "none": None,
        }

    def test_arguments_long_strings_truncated(self) -> None:
        long_value = "x" * (MAX_SUMMARY_VALUE_LENGTH + 50)
        summary = summarize_tool_arguments({"content": long_value})
        assert len(summary["content"]) == MAX_SUMMARY_VALUE_LENGTH + len("...")
        assert summary["content"].endswith("...")

    def test_arguments_collections_become_shapes(self) -> None:
        summary = summarize_tool_arguments(
            {"intake_data": {"name": "n", "parameters": []}, "results": [1, 2, 3]}
        )
        assert summary["intake_data"] == {"type": "dict", "n_keys": 2}
        assert summary["results"] == {"type": "list", "length": 3}

    def test_result_summary_carries_success_error_code_and_counts(self) -> None:
        summary = summarize_tool_result(
            {
                "success": False,
                "error": {"code": "E003", "message": "nope"},
                "errors": ["nope"],
                "suggestions": [],
            }
        )
        assert summary["success"] is False
        assert summary["error_code"] == "E003"
        assert summary["n_errors"] == 1
        assert summary["n_suggestions"] == 0

    def test_result_summary_handles_non_dict(self) -> None:
        assert summarize_tool_result(["content"]) == {"type": "list"}

    def test_extract_campaign_id_requires_well_formed_uuid(self) -> None:
        valid = "0eabc2be-3ad8-4f9e-9e5a-25e5f4f1a9d0"
        assert extract_audit_campaign_id({"campaign_id": valid}) == valid
        assert extract_audit_campaign_id({"campaign_id": "not-a-uuid"}) is None
        assert extract_audit_campaign_id({"campaign_id": 42}) is None
        assert extract_audit_campaign_id({}) is None

    def test_extract_campaign_id_falls_back_to_result(self) -> None:
        """Creation tools mint the id in the response, not the arguments."""
        minted = "3f6bb0a2-9a24-4a5f-8a63-0d1e5f9c7ab1"
        argument = "0eabc2be-3ad8-4f9e-9e5a-25e5f4f1a9d0"
        assert extract_audit_campaign_id({}, {"campaign_id": minted}) == minted
        # The tool's own argument wins over the result field.
        assert (
            extract_audit_campaign_id({"campaign_id": argument}, {"campaign_id": minted})
            == argument
        )
        # A failed create carries campaign_id=None; no attribution.
        assert extract_audit_campaign_id({}, {"campaign_id": None}) is None
        assert extract_audit_campaign_id({}, {"campaign_id": "not-a-uuid"}) is None
        assert extract_audit_campaign_id({}, ["not-a-dict"]) is None
