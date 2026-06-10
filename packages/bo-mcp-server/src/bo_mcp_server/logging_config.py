"""Logging configuration for bo-mcp-server.

This module provides centralized logging configuration. Production
deployments emit JSON-formatted records so log shippers can index by
correlation id (``trace_id`` / ``request_id`` / ``campaign_id`` /
``user_id``) without regex parsing. The plain-text format is preserved
for local development.

Intake documents may carry proprietary chemistry / formulation IP, so
the JSON formatter installs a PII filter that scrubs known-sensitive
keys from log records by default. The filter is conservative — it
operates on the structured fields the formatter sees, not on free-text
``%s``-interpolated messages, so call sites that intentionally
serialize a redacted intake into the message format already control
what reaches the formatter.

Reference: ``python-json-logger`` is the de-facto JSON formatter
adapter for stdlib logging — https://github.com/nhairs/python-json-logger.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Final

from pythonjsonlogger.json import JsonFormatter

# Standard fields emitted by the JSON formatter. The ``fmt`` argument
# of :class:`JsonFormatter` is a space-separated list of ``%(field)s``
# placeholders; the library converts those to top-level keys.
_JSON_FIELDS: Final[str] = (
    "%(asctime)s %(levelname)s %(name)s %(module)s %(message)s "
    "%(trace_id)s %(request_id)s %(campaign_id)s %(user_id)s"
)

# Environment variable that selects the log level. Read at
# :func:`configure_logging` call time — not at import time — so entry
# points that load ``.env`` before configuring logging pick up the
# operator's setting.
_LOG_LEVEL_ENV: Final[str] = "BO_MCP_LOG_LEVEL"

# Fallback log level when the environment variable is unset.
DEFAULT_LOG_LEVEL: Final[str] = "INFO"

# Default format for plain-text log messages (dev mode).
DEFAULT_LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Environment switch: ``LOG_FORMAT=json`` enables the JSON formatter.
# Any other value (including unset) leaves the plain-text default in
# place so local development is unchanged.
_LOG_FORMAT_ENV: Final[str] = "LOG_FORMAT"

# Environment switch: when ``LOG_DISABLE_PII_REDACTION=1`` the PII
# filter is a no-op — the operator has opted in to logging *every*
# field in :data:`_REDACTED_KEYS` (parameter values, emails, intake
# documents, descriptions, metadata, etc.). This is a global disable,
# not a per-key gate; choose it only when the deployment has its own
# log-shipper-side scrubbing. Default is closed: every key in the
# redaction set is replaced with ``[REDACTED]``.
#
# The legacy ``LOG_INCLUDE_PARAMETER_NAMES=1`` name is honoured for
# backwards compatibility with deployments that already set it, but
# is deprecated — the old name implied a per-field gate when the
# override has always been global.
_LOG_DISABLE_PII_REDACTION_ENV: Final[str] = "LOG_DISABLE_PII_REDACTION"
_LEGACY_LOG_INCLUDE_PARAMETER_NAMES_ENV: Final[str] = "LOG_INCLUDE_PARAMETER_NAMES"

# Allowlist of structured-field keys that are always safe to log. The
# filter passes these through verbatim; anything *outside* the
# allowlist that hits a key listed in ``_REDACTED_KEYS`` is replaced
# with ``"[REDACTED]"``.
#
# This is deliberately conservative — additive over time, never
# subtractive — so a new caller cannot silently widen the surface
# without an explicit code review.
_ALLOWED_STRUCTURED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "asctime",
        "levelname",
        "name",
        "module",
        "message",
        # Correlation ids: load-bearing for triage; do not contain IP.
        "trace_id",
        "request_id",
        "campaign_id",
        "user_id",
        "tool",
        # Structured outcome / status metadata.
        "outcome",
        "phase",
        "stage",
        "schema_version",
        "schema_version_seen",
        "schema_version_target",
        "migration",
        "n_observations",
        # Counters / numeric fields that don't carry IP.
        "duration_ms",
        "count",
        "retry_count",
        "attempt",
        "elapsed_seconds",
        # Error metadata.
        "error_code",
        "error_class",
        "retryable",
    }
)

# Keys we positively want to redact when they appear in a record's
# ``extra`` payload. Lives separately from the allowlist so a future
# audit can grep this set without re-deriving it from the inverse.
_REDACTED_KEYS: Final[frozenset[str]] = frozenset(
    {
        "parameter_values",
        "parameter_names",
        "parameter_value",
        "objective_values",
        "objective_value",
        "objective_name",
        "intake",
        "intake_payload",
        "intake_doc",
        "spec_payload",
        "campaign_name",
        "description",
        "metadata",
        "email",
        "user_email",
    }
)

_REDACTED_SENTINEL: Final[str] = "[REDACTED]"


class PiiRedactionFilter(logging.Filter):
    """Scrub known-sensitive structured fields from log records.

    Iterates over the record's ``__dict__`` and replaces any value
    whose key is listed in :data:`_REDACTED_KEYS` with the redaction
    sentinel — *unless* the operator has set the canonical opt-out
    ``LOG_DISABLE_PII_REDACTION=1`` (or the deprecated legacy alias
    ``LOG_INCLUDE_PARAMETER_NAMES=1``), in which case the filter is a
    no-op and every field in :data:`_REDACTED_KEYS` reaches the log
    line verbatim.

    The filter intentionally does NOT touch the formatted
    ``record.message`` (already-interpolated free text). Sanitizing
    free text is brittle and easy to bypass; call sites that build
    log messages from sensitive payloads must redact at the call
    site, not here. The filter exists to catch the structured-extra
    path that a caller might use with ``logger.info("...", extra={...})``.
    """

    def __init__(self, redact: bool = True) -> None:
        """Configure whether the filter redacts sensitive structured-extra keys."""
        super().__init__()
        self._redact = redact

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact structured-extra keys flagged in :data:`_REDACTED_KEYS` on the record."""
        if not self._redact:
            return True
        for key in record.__dict__:
            if key in _REDACTED_KEYS:
                record.__dict__[key] = _REDACTED_SENTINEL
        return True


class CorrelationIdFilter(logging.Filter):
    """Ensure correlation-id fields are present on every record.

    The JSON formatter advertises ``trace_id`` / ``request_id`` /
    ``campaign_id`` / ``user_id`` as top-level keys. Records that do
    not set them via ``extra=`` would otherwise produce
    ``KeyError``-driven 'Unknown field' literals in the JSON output.
    Initialising the fields to ``None`` keeps the schema stable and
    log shippers can index the absent value uniformly.
    """

    _FIELDS: Final[tuple[str, ...]] = (
        "trace_id",
        "request_id",
        "campaign_id",
        "user_id",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        """Ensure correlation-id fields exist on the record, defaulting to ``None``."""
        for key in self._FIELDS:
            if not hasattr(record, key):
                setattr(record, key, None)
        return True


def _resolve_log_format() -> str:
    """Return ``json`` or ``text`` based on the active env override."""
    raw = os.environ.get(_LOG_FORMAT_ENV, "text").strip().lower()
    return "json" if raw == "json" else "text"


_TRUTHY: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})


def _env_bool(name: str) -> bool:
    """Return ``True`` when the named env var is set to a truthy literal."""
    raw = os.environ.get(name, "").strip().lower()
    return raw in _TRUTHY


def _should_redact_sensitive_fields() -> bool:
    """Return ``True`` unless the operator has globally disabled redaction.

    Honours both the canonical ``LOG_DISABLE_PII_REDACTION`` switch and
    the legacy ``LOG_INCLUDE_PARAMETER_NAMES`` name (kept for
    backwards compatibility with deployments that already set the
    older variable). Either set to a truthy literal disables redaction
    for *all* :data:`_REDACTED_KEYS`, not just parameter names — the
    documented behaviour both names now share.
    """
    if _env_bool(_LOG_DISABLE_PII_REDACTION_ENV):
        return False
    return not _env_bool(_LEGACY_LOG_INCLUDE_PARAMETER_NAMES_ENV)


def _build_json_handler() -> logging.Handler:
    """Construct a stderr handler with JSON formatting + filters installed."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter(_JSON_FIELDS, rename_fields={"levelname": "level"}))
    handler.addFilter(CorrelationIdFilter())
    handler.addFilter(PiiRedactionFilter(redact=_should_redact_sensitive_fields()))
    return handler


def _build_text_handler(format_string: str) -> logging.Handler:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(format_string))
    return handler


def configure_logging(
    level: str | None = None,
    format_string: str | None = None,
) -> None:
    """Configure logging for the bo-mcp-server package.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to BO_MCP_LOG_LEVEL env var or INFO.
        format_string: Plain-text log message format. Ignored when the
            JSON formatter is selected via ``LOG_FORMAT=json``.
    """
    env_level = os.environ.get(_LOG_LEVEL_ENV, DEFAULT_LOG_LEVEL)
    log_level_name = (level or env_level).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    log_format = format_string or DEFAULT_LOG_FORMAT

    if _resolve_log_format() == "json":
        handler: logging.Handler = _build_json_handler()
    else:
        handler = _build_text_handler(log_format)

    root = logging.getLogger()
    # Replace existing handlers — calling ``configure_logging`` twice
    # should not accumulate emitters and double-print every record.
    # The list() snapshot is required: ``removeHandler`` mutates
    # ``root.handlers`` and iterating it directly skips entries.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(log_level)

    package_logger = logging.getLogger("bo_mcp_server")
    package_logger.setLevel(log_level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance configured for the module.
    """
    return logging.getLogger(name)


__all__: list[Any] = [
    "DEFAULT_LOG_FORMAT",
    "DEFAULT_LOG_LEVEL",
    "CorrelationIdFilter",
    "PiiRedactionFilter",
    "configure_logging",
    "get_logger",
]
