"""``DiagnosticsResponse`` REST schema contract.

The diagnostics route serializes this model with
``response_model_exclude_unset=True`` so the REST body stays byte-equal
to the MCP ``bo_get_diagnostics`` projection at every verbosity. These
unit tests pin the model-level behaviour that makes that possible,
independently of the FastAPI stack:

* the ``minimal`` projection omits the standard-only keys
  (``campaign_status`` / ``n_pending_suggestions`` / ``warnings``); the
  envelope must not re-introduce them as declared defaults, and
* backend-/verbosity-specific metric blocks ride through verbatim via
  ``extra="allow"``.
"""

from __future__ import annotations

from api.schemas.diagnostics import DiagnosticsResponse


def test_excludes_unset_declared_defaults() -> None:
    """A minimal-shaped result keeps its keys; unset declared fields stay absent."""
    minimal = {
        "schema_version": 1,
        "success": True,
        "status": "healthy",
        "iteration": 3,
        "n_results": 5,
        "health": "good",
        "key_metric": {"best_value": 0.5},
        "errors": [],
        "_metadata": {"backend": "botorch"},
    }
    dumped = DiagnosticsResponse.model_validate(minimal).model_dump(exclude_unset=True)
    # Unset standard-only fields are NOT injected as defaults.
    assert "campaign_status" not in dumped
    assert "n_pending_suggestions" not in dumped
    assert "warnings" not in dumped
    # Passthrough metric blocks + ``_metadata`` survive verbatim — exact round trip.
    assert dumped == minimal


def test_passthrough_preserves_unknown_metric_blocks() -> None:
    """Backend-specific metric blocks ride through via ``extra='allow'``."""
    payload = {
        "schema_version": 1,
        "success": True,
        "loo_cv_metrics": {"rmse": 0.1},
        "hyperparameters": {"lengthscale": [1.2, 0.8]},
    }
    dumped = DiagnosticsResponse.model_validate(payload).model_dump(exclude_unset=True)
    assert dumped["loo_cv_metrics"] == {"rmse": 0.1}
    assert dumped["hyperparameters"] == {"lengthscale": [1.2, 0.8]}


def test_declared_standard_fields_are_kept_when_present() -> None:
    """A standard-shaped result keeps every declared field it actually carries."""
    standard = {
        "schema_version": 1,
        "success": True,
        "campaign_status": "running",
        "iteration": 3,
        "n_results": 5,
        "n_pending_suggestions": 2,
        "errors": [],
        "warnings": [],
        "health_status": "healthy",
    }
    dumped = DiagnosticsResponse.model_validate(standard).model_dump(exclude_unset=True)
    assert dumped == standard
