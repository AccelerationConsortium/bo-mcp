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
