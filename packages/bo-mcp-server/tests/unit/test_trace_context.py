"""Tests for the workflow-level trace-id propagation context.

Reference: the W3C trace-context specification recommends using a
single id to correlate "calls that belong to the same logical flow",
exactly the use case agentic multi-step workflows hit. See
https://www.w3.org/TR/trace-context/#trace-id.
"""

import pytest

from bo_mcp_server.response_formatter import (
    VerbosityLevel,
    format_diagnostics_response,
)
from bo_mcp_server.trace_context import bind_trace_id, get_trace_id, trace_id_var


def test_unbound_trace_id_is_none() -> None:
    """No trace bound -> ``get_trace_id`` returns ``None`` and metadata omits the key."""
    assert get_trace_id() is None


def test_bind_trace_id_sets_and_clears() -> None:
    """``bind_trace_id`` is scoped — value is unset on exit even on exception."""
    assert trace_id_var.get() is None

    with bind_trace_id("workflow-1"):
        assert get_trace_id() == "workflow-1"

    assert trace_id_var.get() is None


def test_bind_trace_id_passthrough_on_none() -> None:
    """``None`` is a no-op so callers can pass an optional arg straight through."""
    with bind_trace_id(None):
        assert get_trace_id() is None


def test_bind_trace_id_clears_on_exception() -> None:
    """Context manager restores the previous value when the body raises."""
    with pytest.raises(RuntimeError):
        with bind_trace_id("workflow-err"):
            assert get_trace_id() == "workflow-err"
            raise RuntimeError("boom")
    assert get_trace_id() is None


def test_response_metadata_echoes_trace_id() -> None:
    """When a trace is bound the response ``_metadata`` envelope echoes it."""
    payload = {
        "campaign_id": "cid",
        "campaign_status": "running",
        "n_results": 0,
        "iteration": 0,
        "warnings": [],
        "errors": [],
    }
    with bind_trace_id("workflow-42"):
        formatted = format_diagnostics_response(payload, VerbosityLevel.MINIMAL)
    assert formatted["_metadata"]["trace_id"] == "workflow-42"


def test_response_metadata_omits_trace_id_when_unbound() -> None:
    """Default responses keep the metadata envelope compact (no ``trace_id`` key)."""
    payload = {
        "campaign_id": "cid",
        "campaign_status": "running",
        "n_results": 0,
        "iteration": 0,
        "warnings": [],
        "errors": [],
    }
    formatted = format_diagnostics_response(payload, VerbosityLevel.MINIMAL)
    assert "trace_id" not in formatted["_metadata"]
