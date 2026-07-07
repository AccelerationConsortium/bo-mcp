"""Mapping of the backend's ``model_correlation`` into health scoring.

A measured ``0.0`` is the "model is uninformative" signal — exactly when
the low-correlation warning matters — so it must reach the health
functions unchanged. Only ``None`` ("backend could not measure it") maps
to NaN, which the engine health functions treat as "unknown" (no warning,
no downgrade). The previous ``or 0.5`` falsy-zero rewrite suppressed the
warning precisely for the worst models.
"""

from __future__ import annotations

import math

from bo_engine.diagnostics import determine_health_status
from bo_mcp_server.operations.get_diagnostics import health_model_correlation


class TestHealthModelCorrelationMapping:
    def test_none_maps_to_nan(self) -> None:
        assert math.isnan(health_model_correlation(None))

    def test_measured_zero_passes_through(self) -> None:
        assert health_model_correlation(0.0) == 0.0

    def test_measured_value_passes_through(self) -> None:
        assert health_model_correlation(0.73) == 0.73

    def test_zero_correlation_lowers_health(self) -> None:
        """End-to-end with the engine scorer: 0.0 must downgrade, None must not."""
        status_zero, warnings_zero = determine_health_status(
            n_results=20,
            hypervolume_improvement=0.1,
            model_correlation=health_model_correlation(0.0),
            iterations_without_improvement=0,
        )
        status_unknown, warnings_unknown = determine_health_status(
            n_results=20,
            hypervolume_improvement=0.1,
            model_correlation=health_model_correlation(None),
            iterations_without_improvement=0,
        )

        assert status_zero == "critical"
        assert any("correlation" in w.lower() for w in warnings_zero)
        assert status_unknown == "healthy"
        assert not any("correlation" in w.lower() for w in warnings_unknown)
