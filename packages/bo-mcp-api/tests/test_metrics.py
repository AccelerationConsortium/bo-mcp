"""Tests for the Prometheus metrics surface.

Reference: the Prometheus exposition format and ``Counter`` / ``Histogram``
metric types are described at
https://prometheus.io/docs/instrumenting/exposition_formats/ and
https://prometheus.io/docs/concepts/metric_types/ — both of which the
``prometheus_client`` library implements directly.
"""

import pytest


@pytest.mark.asyncio
async def test_metrics_endpoint_exposes_prometheus_format(api_client) -> None:
    """``/metrics`` returns the canonical Prometheus exposition format."""
    response = await api_client.get("/metrics")

    assert response.status_code == 200
    body = response.text
    # Prometheus exposition starts with ``# HELP`` / ``# TYPE`` directives.
    assert "# HELP" in body
    assert "bo_mcp_http_requests_total" in body
    assert "bo_mcp_http_request_duration_seconds" in body
    assert "bo_mcp_in_flight_requests" in body


@pytest.mark.asyncio
async def test_request_counter_records_health_probe(api_client) -> None:
    """An HTTP request increments the request counter for the matched route."""
    await api_client.get("/health")
    response = await api_client.get("/metrics")
    body = response.text

    # The counter for /health should be present with a ``2xx`` status class.
    assert "bo_mcp_http_requests_total" in body
    assert "/health" in body
    assert "2xx" in body


@pytest.mark.asyncio
async def test_metrics_endpoint_does_not_pollute_openapi(api_client) -> None:
    """``include_in_schema=False`` keeps the metrics route out of the OpenAPI doc."""
    response = await api_client.get("/openapi.json")
    schema = response.json()
    assert "/metrics" not in schema.get("paths", {})


@pytest.mark.asyncio
async def test_metrics_endpoint_advertises_domain_instruments(api_client) -> None:
    """The exposition format includes the domain-specific instruments.

    A campaigns counter, a generation-latency histogram, a cache event
    counter, and a DB-pool gauge are all required by the TODO scope —
    asserting their presence here keeps the metric surface from
    silently shrinking.
    """
    response = await api_client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    for metric_name in (
        "bo_mcp_campaigns_created_total",
        "bo_mcp_suggestion_generation_duration_seconds",
        "bo_mcp_diagnostics_cache_events_total",
        "bo_mcp_db_pool_connections",
    ):
        assert metric_name in body, f"missing metric: {metric_name}"


@pytest.mark.asyncio
async def test_metrics_endpoint_samples_db_pool_state(
    api_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``/metrics`` actually samples the DB pool, not just registers the gauge.

    Without the explicit ``sample_db_pool()`` call inside the endpoint
    the gauge would advertise the metric name but never emit a labelled
    sample — operators would see ``bo_mcp_db_pool_connections`` with
    zero data points on every scrape. The test patches the snapshot
    helper to return synthetic numbers because SQLite's ``StaticPool``
    (the test-time engine) doesn't expose the counter methods, so the
    real call would yield no samples; the patched snapshot lets us
    verify the scrape path consults the snapshot and emits gauge lines
    for each pool state.
    """
    from api import metrics as api_metrics

    monkeypatch.setattr(
        api_metrics,
        "snapshot_db_pool",
        lambda: {"checked_out": 1.0, "checked_in": 4.0, "overflow": 0.0, "size": 5.0},
    )
    response = await api_client.get("/metrics")
    body = response.text
    # Prometheus exposition emits one line per (metric, label set):
    # ``bo_mcp_db_pool_connections{state="checked_out"} 1.0``
    assert 'bo_mcp_db_pool_connections{state="checked_out"}' in body
    assert 'bo_mcp_db_pool_connections{state="size"} 5.0' in body


def test_sanitize_unmatched_path_collapses_uuids_and_numbers() -> None:
    """High-cardinality URL segments collapse to placeholder labels.

    Reference: the Prometheus practical guide explicitly warns against
    unbounded label cardinality on a counter
    (https://prometheus.io/docs/practices/instrumentation/#do-not-overuse-labels).
    A 404-scanner hitting paths with concrete UUIDs would otherwise
    blow up the label set on ``bo_mcp_http_requests_total``.
    """
    from api.metrics import _sanitize_unmatched_path

    uuid_path = "/api/v1/wrong/01234567-89ab-cdef-0123-456789abcdef"
    assert _sanitize_unmatched_path(uuid_path) == "/api/v1/wrong/{uuid}"

    numeric_path = "/api/v1/wrong/12345/details"
    assert _sanitize_unmatched_path(numeric_path) == "/api/v1/wrong/{int}/details"

    hex_path = "/api/v1/wrong/deadbeefcafebabe"
    assert _sanitize_unmatched_path(hex_path) == "/api/v1/wrong/{hex}"

    safe_path = "/api/v1/known/path"
    assert _sanitize_unmatched_path(safe_path) == "/api/v1/known/path"


def test_bounded_unmatched_label_overflows_after_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Beyond the cap, fresh unmatched paths collapse onto a single sentinel.

    Even with sanitization, a determined scanner can produce many
    distinct ``/word/word/...`` paths. The cap bounds the label set
    so a series cannot grow without limit.
    """
    from api import metrics as api_metrics

    monkeypatch.setattr(api_metrics, "_unmatched_labels", set())
    monkeypatch.setattr(api_metrics, "_UNMATCHED_LABEL_CAP", 2)

    first = api_metrics._bounded_unmatched_label("/a")
    second = api_metrics._bounded_unmatched_label("/b")
    third = api_metrics._bounded_unmatched_label("/c")
    repeat = api_metrics._bounded_unmatched_label("/a")
    overflow_repeat = api_metrics._bounded_unmatched_label("/d")

    assert first == "/a"
    assert second == "/b"
    # ``/c`` arrives after the cap is reached and is collapsed.
    assert third == api_metrics._UNMATCHED_OVERFLOW
    # Already-tracked entries keep their identity post-cap.
    assert repeat == "/a"
    # Anything new after the cap is overflowed.
    assert overflow_repeat == api_metrics._UNMATCHED_OVERFLOW


@pytest.mark.asyncio
async def test_unmatched_path_with_uuid_does_not_pollute_label_set(api_client) -> None:
    """A 404 with a UUID segment records a single template label, not the raw URL.

    Asserts the contract by scraping ``/metrics`` and verifying the
    UUID appears nowhere in the exposition format (the counter only
    sees the sanitized template).
    """
    uuid_value = "01234567-89ab-cdef-0123-456789abcdef"
    await api_client.get(f"/api/v1/does-not-exist/{uuid_value}")
    response = await api_client.get("/metrics")
    body = response.text
    assert uuid_value not in body
    # The sanitized template is recorded instead.
    assert "/api/v1/does-not-exist/{uuid}" in body
