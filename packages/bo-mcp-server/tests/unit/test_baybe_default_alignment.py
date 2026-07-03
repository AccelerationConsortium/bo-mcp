"""Backend alignment behaviors introduced with the BayBE-default switch.

Covers the issue #57 fixes (campaign-aware ``_metadata.backend`` stamp,
backend-truthful diagnostics ``model_info``), the capability-aware intake
dry-run, and the backend-aware capabilities listing.
"""

from __future__ import annotations

import contextvars

import pytest

from bo_mcp_server.backend_context import (
    campaign_backend_scope,
    campaign_backend_var,
    get_campaign_backend,
    set_campaign_backend,
)
from bo_mcp_server.domain import CampaignSpec
from bo_mcp_server.operations.diagnostics.enrichment import get_model_info
from bo_mcp_server.operations.list_capabilities import list_capabilities_operation
from bo_mcp_server.operations.validate_intake import validate_intake_with_capabilities
from bo_mcp_server.response_formatter import get_response_metadata


def _make_intake(**overrides) -> dict:
    intake = {
        "name": "Alignment Test",
        "parameters": [
            {"name": "x0", "type": "continuous", "bounds": [0.0, 1.0]},
            {"name": "x1", "type": "continuous", "bounds": [0.0, 1.0]},
        ],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }
    intake.update(overrides)
    return intake


def _make_spec(backend: str = "baybe") -> CampaignSpec:
    return CampaignSpec.model_validate(_make_intake(backend=backend))


# ---------------------------------------------------------------------------
# _metadata.backend stamp (issue #57 root cause 1)
# ---------------------------------------------------------------------------


class _DummyBackend:
    name = "default-backend"


def _patch_default_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    import bo_mcp_server.backend as backend_module

    def fake_get_backend(name: str | None = None) -> _DummyBackend:
        del name
        return _DummyBackend()

    monkeypatch.setattr(backend_module, "get_backend", fake_get_backend)


def test_metadata_prefers_bound_campaign_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bound campaign backend wins over the server default."""
    _patch_default_backend(monkeypatch)

    def run() -> str:
        set_campaign_backend("baybe")
        return get_response_metadata().backend

    # Fresh context: mimics one request's task-local scope.
    assert contextvars.copy_context().run(run) == "baybe"


def test_metadata_falls_back_to_default_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Campaign-agnostic calls keep stamping the default backend."""
    _patch_default_backend(monkeypatch)

    def run() -> str:
        return get_response_metadata().backend

    assert contextvars.copy_context().run(run) == "default-backend"


def test_set_campaign_backend_ignores_empty() -> None:
    def run() -> str | None:
        set_campaign_backend(None)
        set_campaign_backend("")
        return get_campaign_backend()

    assert contextvars.copy_context().run(run) is None


def test_binding_does_not_leak_across_contexts() -> None:
    contextvars.copy_context().run(set_campaign_backend, "baybe")
    assert campaign_backend_var.get() is None


def test_scope_isolates_sequential_operations() -> None:
    """A binding left by operation N never leaks into operation N+1.

    Same-task sequences (in-process client facade, scripts) rely on the
    transport layers entering ``campaign_backend_scope`` per dispatch.
    """

    def run() -> tuple[str | None, str | None, str | None]:
        with campaign_backend_scope():
            set_campaign_backend("baybe")
            inside_first = get_campaign_backend()
        with campaign_backend_scope():
            inside_second = get_campaign_backend()
        return inside_first, inside_second, get_campaign_backend()

    inside_first, inside_second, after = contextvars.copy_context().run(run)
    assert inside_first == "baybe"
    assert inside_second is None
    assert after is None


def test_scope_restores_outer_binding() -> None:
    def run() -> tuple[str | None, str | None]:
        set_campaign_backend("botorch")
        with campaign_backend_scope():
            set_campaign_backend("baybe")
        return get_campaign_backend(), campaign_backend_var.get()

    restored, raw = contextvars.copy_context().run(run)
    assert restored == "botorch"
    assert raw == "botorch"


# ---------------------------------------------------------------------------
# model_info from backend method report (issue #57 root cause 2)
# ---------------------------------------------------------------------------


def test_model_info_uses_backend_method_report() -> None:
    spec = _make_spec(backend="baybe")
    method_info = {
        "model_type": "BayBE GP",
        "acquisition_function": "qLogNoisyExpectedImprovement",
        "optimization_strategy": "BotorchRecommender (GP-based)",
    }
    info = get_model_info(spec, is_single_objective=True, method_info=method_info)
    assert info["backend"] == "baybe"
    assert info["type"] == "BayBE GP"
    assert info["acquisition_function"] == "qLogNoisyExpectedImprovement"
    assert info["batch_strategy"] == "BotorchRecommender (GP-based)"


def test_model_info_legacy_fallback_without_method_report() -> None:
    spec = _make_spec(backend="botorch")
    info = get_model_info(spec, is_single_objective=True, method_info=None)
    assert info["backend"] == "botorch"
    assert "SingleTaskGP" in info["type"]


# ---------------------------------------------------------------------------
# Capability-aware intake dry-run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_intake_runs_capability_check() -> None:
    """A BayBE-pinned spec with a BoTorch-only knob fails the dry-run.

    Previously this passed ``bo_validate_intake`` and only failed at
    ``bo_create_campaign`` — the dry-run/create divergence.
    """
    intake = _make_intake(backend="baybe", use_input_warping=True)
    result = await validate_intake_with_capabilities(intake)
    assert result["valid"] is False
    assert result["errors"]
    assert result["spec"] is None


@pytest.mark.asyncio
async def test_validate_intake_resolves_auto_and_stays_valid() -> None:
    intake = _make_intake()
    result = await validate_intake_with_capabilities(intake)
    assert result["valid"] is True
    assert result["backend"] in ("baybe", "botorch")
    assert result["spec"]["backend"] == result["backend"]


# ---------------------------------------------------------------------------
# Backend-aware capabilities listing
# ---------------------------------------------------------------------------


def test_list_capabilities_reports_named_backend() -> None:
    result = list_capabilities_operation("botorch")
    assert result["backend"] == "botorch"
    assert set(result["available_backends"]) >= {"baybe", "botorch"}
    # The default is environment-configurable (BO_BACKEND, local .env), so
    # assert shape, not value — the shipped default is covered by
    # test_settings.py::test_default_values_when_env_unset.
    assert result["default_backend"] in result["available_backends"]


def test_list_capabilities_reports_baybe_backend() -> None:
    result = list_capabilities_operation("baybe")
    assert result["backend"] == "baybe"
    assert result["supported_features"]


def test_list_capabilities_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="no-such-backend"):
        list_capabilities_operation("no-such-backend")
