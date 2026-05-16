"""Prometheus metrics surface for the REST API.

Exposes ``/metrics`` in the canonical Prometheus exposition format plus
both HTTP-level and domain-level instrumentation. Metric families:

* ``bo_mcp_http_requests_total`` — labelled by method, route, status class.
* ``bo_mcp_http_request_duration_seconds`` — histogram of request latency.
* ``bo_mcp_in_flight_requests`` — concurrent request gauge.
* ``bo_mcp_campaigns_created_total`` — campaigns persisted by status.
* ``bo_mcp_suggestion_generation_duration_seconds`` — BO-engine fit +
  acquisition latency.
* ``bo_mcp_db_pool_*`` — checked-out / overflow connections + pool size.
* ``bo_mcp_diagnostics_cache_events_total`` — cache hits vs. misses.

Reference: the integration follows the prometheus-client tutorial for ASGI
apps — https://prometheus.github.io/client_python/exporting/asgi/ — and the
``Counter``/``Histogram``/``Gauge`` choice matches the recommendations in
the Prometheus naming conventions guide:
https://prometheus.io/docs/practices/naming/.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable

# Re-export the domain instrument hooks via the ``bo_mcp_server.client``
# facade so the API layer stays inside the facade boundary the
# ``test_facade_imports`` regression enforces.
from bo_mcp_server.client import (  # noqa: F401 — re-exported via /metrics
    observe_suggestion_latency,
    record_campaign_created,
    record_diagnostics_cache,
    snapshot_db_pool,
)
from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

logger = logging.getLogger(__name__)


def _build_http_metrics(
    registry: CollectorRegistry,
) -> tuple[Counter, Histogram, Gauge, Gauge]:
    """Build the HTTP-only metric instruments registered under ``registry``.

    Factored out so tests can supply a private :class:`CollectorRegistry`
    and avoid the global ``Counter``-already-exists collision Prometheus
    raises when a process imports the module twice (e.g. across pytest
    runs that re-import ``api.main``).
    """
    requests = Counter(
        "bo_mcp_http_requests_total",
        "Total HTTP requests served by the REST API.",
        labelnames=("method", "route", "status_class"),
        registry=registry,
    )
    duration = Histogram(
        "bo_mcp_http_request_duration_seconds",
        "Latency of HTTP requests in seconds.",
        labelnames=("method", "route"),
        buckets=(0.005, 0.025, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
        registry=registry,
    )
    in_flight = Gauge(
        "bo_mcp_in_flight_requests",
        "Number of requests currently being processed.",
        registry=registry,
    )
    db_pool = Gauge(
        "bo_mcp_db_pool_connections",
        "SQLAlchemy connection-pool utilization (live samples per call).",
        labelnames=("state",),
        registry=registry,
    )
    return requests, duration, in_flight, db_pool


REQUESTS, REQUEST_DURATION, IN_FLIGHT, DB_POOL = _build_http_metrics(REGISTRY)


def sample_db_pool() -> None:
    """Refresh the SQLAlchemy pool-utilization gauge.

    Reads the per-state counts through :func:`snapshot_db_pool` (which
    lives on the server-side facade). The accessor is best-effort: if
    no engine is initialized or the pool implementation doesn't expose
    these counters, the snapshot is empty and the gauge keeps its last
    value.
    """
    snapshot = snapshot_db_pool()
    for state, value in snapshot.items():
        DB_POOL.labels(state).set(value)


def _route_template(request: Request) -> str:
    """Resolve the matched route template, falling back to the raw path.

    Recording the *template* (``/api/v1/campaigns/{campaign_id}``) rather
    than the concrete URL keeps cardinality bounded: a million unique
    campaign ids would otherwise blow up the metric label set.
    """
    route = request.scope.get("route")
    template = getattr(route, "path", None)
    if isinstance(template, str) and template:
        return template
    return request.url.path


def _status_class(status_code: int) -> str:
    return f"{status_code // 100}xx"


async def _instrument(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Record per-request counters, latency, and in-flight gauge."""
    method = request.method
    IN_FLIGHT.inc()
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed = time.perf_counter() - start
        route = _route_template(request)
        REQUEST_DURATION.labels(method, route).observe(elapsed)
        REQUESTS.labels(method, route, "5xx").inc()
        raise
    finally:
        IN_FLIGHT.dec()
    elapsed = time.perf_counter() - start
    route = _route_template(request)
    REQUEST_DURATION.labels(method, route).observe(elapsed)
    REQUESTS.labels(method, route, _status_class(response.status_code)).inc()
    return response


def install_metrics(app: FastAPI) -> None:
    """Attach the metrics middleware and ``/metrics`` endpoint to ``app``."""
    app.middleware("http")(_instrument)

    @app.get("/metrics", include_in_schema=False)
    def metrics_endpoint() -> Response:
        # Refresh point-in-time gauges before each scrape. The DB pool
        # counters aren't event-driven (no callback fires on
        # checkout/checkin), so without this call ``/metrics`` would
        # always emit the last-set value — usually zero.
        sample_db_pool()
        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST,
        )
