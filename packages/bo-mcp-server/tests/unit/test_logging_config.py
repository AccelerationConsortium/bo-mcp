"""Tests for structured JSON logging and PII redaction.

The JSON formatter wraps stdlib logging via ``python-json-logger`` so
log shippers can index by correlation id (``trace_id`` /
``request_id`` / ``campaign_id`` / ``user_id``) without regex
parsing. Intake docs may carry proprietary chemistry / formulation
IP, so the PII filter scrubs known-sensitive structured fields by
default.

Reference: the JSON-logging pattern is the same one Honeycomb /
Datadog document for stdlib logging — see
https://github.com/nhairs/python-json-logger.
"""

from __future__ import annotations

import json
import logging

import pytest

from bo_mcp_server.logging_config import (
    _REDACTED_KEYS,
    _REDACTED_SENTINEL,
    CorrelationIdFilter,
    PiiRedactionFilter,
    configure_logging,
)


def _make_record(
    *,
    level: int = logging.INFO,
    message: str = "hello",
    **extras: object,
) -> logging.LogRecord:
    record = logging.LogRecord(
        name="bo_mcp_server.tests",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    for key, value in extras.items():
        setattr(record, key, value)
    return record


class TestPiiRedactionFilter:
    def test_filter_redacts_known_sensitive_extras(self) -> None:
        """``extra={"parameter_values": ...}`` is replaced with the sentinel."""
        filt = PiiRedactionFilter(redact=True)
        record = _make_record(
            parameter_values={"temperature": 25.0, "solvent": "ethanol"},
            campaign_id="cmp-123",
        )
        assert filt.filter(record) is True

        # ``LogRecord`` has no static ``parameter_values`` attribute —
        # routing the access through ``__dict__`` is the type-safe
        # shape for these dynamic extras.
        assert record.__dict__["parameter_values"] == _REDACTED_SENTINEL
        # Non-sensitive correlation ids pass through untouched.
        assert record.__dict__["campaign_id"] == "cmp-123"

    def test_filter_passthrough_when_disabled(self) -> None:
        """The opt-in env override leaves sensitive extras intact."""
        filt = PiiRedactionFilter(redact=False)
        sensitive_payload = {"solvent": "ethanol"}
        record = _make_record(parameter_values=sensitive_payload)
        filt.filter(record)
        assert record.__dict__["parameter_values"] is sensitive_payload

    def test_filter_does_not_touch_unknown_extras(self) -> None:
        """Only fields in the explicit allowlist are redacted."""
        filt = PiiRedactionFilter(redact=True)
        record = _make_record(safe_field="public-info")
        filt.filter(record)
        assert record.__dict__["safe_field"] == "public-info"

    def test_filter_redacts_every_named_sensitive_key(self) -> None:
        """Every key in the redaction set is actually scrubbed."""
        filt = PiiRedactionFilter(redact=True)
        extras: dict[str, object] = {key: f"raw-{key}" for key in _REDACTED_KEYS}
        record = _make_record(**extras)  # ty: ignore[invalid-argument-type]
        filt.filter(record)
        for key in _REDACTED_KEYS:
            assert record.__dict__[key] == _REDACTED_SENTINEL


class TestCorrelationIdFilter:
    def test_filter_sets_missing_correlation_ids_to_none(self) -> None:
        """The four correlation fields are always present after the filter runs."""
        filt = CorrelationIdFilter()
        record = _make_record()
        filt.filter(record)
        for field in ("trace_id", "request_id", "campaign_id", "user_id"):
            assert record.__dict__[field] is None

    def test_filter_preserves_supplied_correlation_ids(self) -> None:
        """Explicit values are not overwritten with ``None``."""
        filt = CorrelationIdFilter()
        record = _make_record(trace_id="trace-1", campaign_id="cmp-9")
        filt.filter(record)
        assert record.__dict__["trace_id"] == "trace-1"
        assert record.__dict__["campaign_id"] == "cmp-9"
        # The missing fields are still backfilled.
        assert record.__dict__["request_id"] is None


class TestConfigureLoggingJsonOutput:
    """End-to-end: configure_logging emits a JSON record carrying the audit fields."""

    def test_json_format_emits_structured_record(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv("LOG_FORMAT", "json")
        monkeypatch.delenv("LOG_DISABLE_PII_REDACTION", raising=False)
        monkeypatch.delenv("LOG_INCLUDE_PARAMETER_NAMES", raising=False)

        configure_logging(level="INFO")
        logger = logging.getLogger("bo_mcp_server.tests.json")
        logger.info(
            "submitted intake",
            extra={
                "trace_id": "trace-xyz",
                "campaign_id": "cmp-42",
                "parameter_values": {"solvent": "dimethyl-sulfoxide"},
            },
        )

        captured = capsys.readouterr()
        # Exactly one JSON line is emitted; tests bind a fresh handler.
        line = captured.err.strip().splitlines()[-1]
        record = json.loads(line)

        assert record["trace_id"] == "trace-xyz"
        assert record["campaign_id"] == "cmp-42"
        assert record["message"] == "submitted intake"
        assert record["level"] == "INFO"
        # The proprietary parameter labels never reach the rendered line.
        assert "dimethyl-sulfoxide" not in line
        assert record["parameter_values"] == _REDACTED_SENTINEL

    def test_log_disable_pii_redaction_is_global(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The canonical override turns off redaction for *every* sensitive key.

        Asserting against multiple keys (parameter_values, email, intake)
        is load-bearing: the previous documentation implied a per-field
        gate, when in fact the switch has always been global. The new
        name makes that explicit, and this test pins the contract.
        """
        monkeypatch.setenv("LOG_FORMAT", "json")
        monkeypatch.setenv("LOG_DISABLE_PII_REDACTION", "1")
        monkeypatch.delenv("LOG_INCLUDE_PARAMETER_NAMES", raising=False)

        configure_logging(level="INFO")
        logger = logging.getLogger("bo_mcp_server.tests.opt_in")
        logger.info(
            "intake snapshot",
            extra={
                "parameter_values": {"solvent": "ethanol"},
                "email": "user@example.com",
                "intake": {"name": "secret-campaign"},
            },
        )
        captured = capsys.readouterr()
        line = captured.err.strip().splitlines()[-1]
        record = json.loads(line)
        # All three sensitive keys carry their raw payloads when the
        # global override is set.
        assert record["parameter_values"] == {"solvent": "ethanol"}
        assert record["email"] == "user@example.com"
        assert record["intake"] == {"name": "secret-campaign"}

    def test_legacy_log_include_parameter_names_still_disables_redaction(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Deployments still on ``LOG_INCLUDE_PARAMETER_NAMES`` keep working.

        The old env name is honoured for backwards compatibility — it
        had always behaved globally, just under a misleading name. We
        pin that the legacy override disables redaction for the full
        :data:`_REDACTED_KEYS` set, matching the new canonical name.
        """
        monkeypatch.setenv("LOG_FORMAT", "json")
        monkeypatch.delenv("LOG_DISABLE_PII_REDACTION", raising=False)
        monkeypatch.setenv("LOG_INCLUDE_PARAMETER_NAMES", "1")

        configure_logging(level="INFO")
        logger = logging.getLogger("bo_mcp_server.tests.legacy_opt_in")
        logger.info(
            "intake snapshot",
            extra={
                "parameter_values": {"solvent": "ethanol"},
                "email": "user@example.com",
            },
        )
        captured = capsys.readouterr()
        line = captured.err.strip().splitlines()[-1]
        record = json.loads(line)
        assert record["parameter_values"] == {"solvent": "ethanol"}
        assert record["email"] == "user@example.com"

    def test_text_format_remains_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Without ``LOG_FORMAT=json``, the legacy text format is used."""
        monkeypatch.delenv("LOG_FORMAT", raising=False)
        configure_logging(level="INFO")
        logger = logging.getLogger("bo_mcp_server.tests.text")
        logger.info("plain message")
        captured = capsys.readouterr()
        # Plain text format does NOT contain JSON braces around the level.
        assert "plain message" in captured.err
        with pytest.raises(json.JSONDecodeError):
            json.loads(captured.err.strip().splitlines()[-1])

    def test_env_log_level_is_read_at_call_time(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``BO_MCP_LOG_LEVEL`` set after import must still take effect.

        Entry points load ``.env`` at startup, after this module has been
        imported — an import-time read would freeze the level at whatever
        the interpreter environment happened to contain.
        """
        monkeypatch.delenv("LOG_FORMAT", raising=False)
        monkeypatch.setenv("BO_MCP_LOG_LEVEL", "DEBUG")
        configure_logging()
        assert logging.getLogger("bo_mcp_server").getEffectiveLevel() == logging.DEBUG

        # An explicit ``level`` argument still wins over the environment.
        configure_logging(level="WARNING")
        assert logging.getLogger("bo_mcp_server").getEffectiveLevel() == logging.WARNING

    def test_idempotent_configure_does_not_double_emit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Calling ``configure_logging`` twice should not duplicate records."""
        monkeypatch.setenv("LOG_FORMAT", "json")
        configure_logging(level="INFO")
        configure_logging(level="INFO")  # second call must replace, not append
        logger = logging.getLogger("bo_mcp_server.tests.dedupe")
        logger.info("once")
        captured = capsys.readouterr()
        emitted_lines = [line for line in captured.err.splitlines() if "tests.dedupe" in line]
        assert len(emitted_lines) == 1
