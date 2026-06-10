"""Tests for request-id propagation into log records.

Reference: Python's official ``logging`` cookbook recommends
``logging.setLogRecordFactory`` (not a logger-level filter) for
binding per-request metadata onto every emitted ``LogRecord`` because
filters attached to a parent logger do not run for records emitted by
its children. See
https://docs.python.org/3/howto/logging-cookbook.html#using-filters-to-impart-contextual-information
and https://docs.python.org/3/library/logging.html#logging.setLogRecordFactory.
"""

import json
import logging

import pytest
from bo_mcp_server.logging_config import configure_logging
from bo_mcp_server.trace_context import bind_trace_id

from api.request_context import (
    RequestIdLogFilter,
    install_request_id_log_factory,
    install_request_id_log_filter,
    request_id_var,
)

pytestmark = pytest.mark.usefixtures("persisted_user")


def test_factory_emits_dash_outside_request() -> None:
    """No request bound -> records carry the sentinel ``"-"``."""
    install_request_id_log_factory()
    record = logging.getLogRecordFactory()(
        "api.test",
        logging.INFO,
        __file__,
        10,
        "no request",
        (),
        None,
    )
    assert getattr(record, "request_id") == "-"  # noqa: B009


def test_factory_picks_up_active_request_id() -> None:
    """A request id set on the context var lands on every record."""
    install_request_id_log_factory()
    token = request_id_var.set("abc-123")
    try:
        record = logging.getLogRecordFactory()(
            "api.test",
            logging.INFO,
            __file__,
            10,
            "with request",
            (),
            None,
        )
    finally:
        request_id_var.reset(token)
    assert getattr(record, "request_id") == "abc-123"  # noqa: B009


def test_child_logger_records_get_request_id_via_factory(caplog) -> None:
    """Records emitted on a child logger pick up ``request_id``.

    This is the case the previous ``Filter``-only installation missed
    — filters on the root logger don't decorate records that
    originated on child loggers, but the factory hook does.
    """
    install_request_id_log_factory()
    child = logging.getLogger("api.test.child.factory")
    token = request_id_var.set("trace-child")
    try:
        with caplog.at_level(logging.INFO, logger=child.name):
            child.info("from child")
    finally:
        request_id_var.reset(token)

    emitted = [r for r in caplog.records if r.name == child.name]
    assert emitted, "child logger should have emitted at least one record"
    assert all(getattr(r, "request_id", None) == "trace-child" for r in emitted), [
        getattr(r, "request_id", None) for r in emitted
    ]


def test_factory_picks_up_bound_trace_id() -> None:
    """A workflow trace bound via ``bind_trace_id`` lands on every record."""
    install_request_id_log_factory()
    with bind_trace_id("wf-123"):
        record = logging.getLogRecordFactory()(
            "api.test",
            logging.INFO,
            __file__,
            10,
            "with trace",
            (),
            None,
        )
    assert getattr(record, "trace_id", None) == "wf-123"


def test_factory_leaves_trace_id_unset_outside_workflow() -> None:
    """No bound workflow -> the attribute stays absent.

    The JSON handler's ``CorrelationIdFilter`` backfills a stable
    ``null`` for absent fields, and ``Logger.makeRecord`` refuses
    ``extra=`` keys that already exist on a record — so the factory must
    not stamp a placeholder.
    """
    install_request_id_log_factory()
    record = logging.getLogRecordFactory()(
        "api.test",
        logging.INFO,
        __file__,
        10,
        "no trace",
        (),
        None,
    )
    assert not hasattr(record, "trace_id")


def test_bound_trace_id_reaches_json_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Factory attribute -> JSON field, end to end through the formatter.

    Pins the full chain: ``bind_trace_id`` sets the context var, the
    record factory stamps ``trace_id`` on the record, and the JSON
    handler renders it as a top-level key instead of the ``null``
    backfilled by ``CorrelationIdFilter`` for absent fields.
    """
    monkeypatch.setenv("LOG_FORMAT", "json")
    configure_logging(level="INFO")
    install_request_id_log_factory()
    with bind_trace_id("wf-json-1"):
        logging.getLogger("api.test.trace_json").info("workflow step")
    line = capsys.readouterr().err.strip().splitlines()[-1]
    record = json.loads(line)
    assert record["trace_id"] == "wf-json-1"


def test_install_factory_is_idempotent() -> None:
    """Repeated installs must not wrap the factory more than once.

    A second wrap would invoke the previous wrapper (which itself
    invokes the original factory), so each install would add another
    request-id assignment per record.  The lock + sentinel keeps the
    factory chain at exactly one layer.
    """
    install_request_id_log_factory()
    first = logging.getLogRecordFactory()
    install_request_id_log_factory()
    install_request_id_log_factory()
    assert logging.getLogRecordFactory() is first


def test_legacy_filter_still_decorates_when_attached() -> None:
    """The deprecated filter helper keeps working for callers that opt in."""
    logger = logging.getLogger("api.test.legacy.filter")
    logger.filters.clear()
    install_request_id_log_filter(logger)
    matching = [f for f in logger.filters if isinstance(f, RequestIdLogFilter)]
    assert len(matching) == 1


@pytest.mark.asyncio
async def test_request_id_threads_into_log_records_end_to_end(
    api_client, auth_headers, caplog
) -> None:
    """An end-to-end request emits log records tagged with its X-Request-ID.

    Hits ``/api/v1/campaigns/validate`` because the validation handler
    routes through
    ``bo_mcp_server.operations.validate_intake.validate_intake_operation``
    which emits an INFO record while the request context var is bound.
    The success path of ``/health`` is intentionally silent so it would
    not exercise the propagation defect this regression test guards.
    """
    install_request_id_log_factory()
    headers = {**auth_headers, "X-Request-ID": "trace-7777"}
    payload = {
        "intake": {
            "name": "trace-probe",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
    }
    with caplog.at_level(logging.INFO):
        response = await api_client.post(
            "/api/v1/campaigns/validate", headers=headers, json=payload
        )
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "trace-7777"
    tagged = [r for r in caplog.records if getattr(r, "request_id", "-") == "trace-7777"]
    assert tagged, "at least one log record from the request must carry the trace id"


@pytest.mark.asyncio
async def test_trace_id_threads_into_log_records_end_to_end(
    api_client, auth_headers, caplog
) -> None:
    """A request carrying ``X-Trace-Id`` emits log records tagged with it.

    The middleware binds the workflow id via ``bind_trace_id`` for the
    duration of the request; the record factory must stamp it as
    ``trace_id`` so JSON logs correlate with the trace echoed in
    response metadata and audit events. Uses the same INFO-emitting
    validation route as the request-id end-to-end test above.
    """
    install_request_id_log_factory()
    headers = {**auth_headers, "X-Trace-Id": "wf-7777"}
    payload = {
        "intake": {
            "name": "trace-probe",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
    }
    with caplog.at_level(logging.INFO):
        response = await api_client.post(
            "/api/v1/campaigns/validate", headers=headers, json=payload
        )
    assert response.status_code == 200
    assert response.headers["X-Trace-Id"] == "wf-7777"
    tagged = [r for r in caplog.records if getattr(r, "trace_id", None) == "wf-7777"]
    assert tagged, "at least one log record from the request must carry the workflow trace id"
