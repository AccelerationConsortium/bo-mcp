"""Backend alignment behaviors introduced with the BayBE-default switch.

Covers the issue #57 fixes (campaign-aware ``_metadata.backend`` stamp,
backend-truthful diagnostics ``model_info``), the capability-aware intake
dry-run, and the backend-aware capabilities listing.
"""

from __future__ import annotations

import asyncio
import contextvars
from uuid import uuid4

import pytest

from bo_mcp_server.backend_context import (
    campaign_backend_scope,
    campaign_backend_var,
    get_campaign_backend,
    set_campaign_backend,
    with_campaign_backend_scope,
)
from bo_mcp_server.domain import Campaign, CampaignSpec
from bo_mcp_server.operations.diagnostics.enrichment import get_model_info
from bo_mcp_server.operations.list_campaigns import _build_summary
from bo_mcp_server.operations.list_capabilities import list_capabilities_operation
from bo_mcp_server.operations.validate_intake import validate_intake_with_capabilities
from bo_mcp_server.protocol_context import bind_protocol
from bo_mcp_server.response_formatter import VerbosityLevel, get_response_metadata


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


def test_campaign_spec_default_backend_is_baybe() -> None:
    """The domain default matches the project-wide BayBE default.

    Aligned with migration ``016_default_backend_baybe`` (DB server
    default) and the ``BO_BACKEND`` settings default. Historical rows are
    safe: the column-adding migration backfilled them to ``"botorch"`` in
    the database, so this literal cannot relabel them.
    """
    assert CampaignSpec.model_validate(_make_intake()).backend == "baybe"


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


def test_metadata_backend_source_names_campaign_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``backend_source`` discloses that ``backend`` is the campaign's own.

    ``_metadata.backend`` is polysemous by design (campaign backend on
    campaign-scoped calls, server default otherwise — issue #57). Issue
    #82 showed adjacent responses can disagree without the reader being
    able to tell which meaning each carries; the discriminator makes the
    stamp self-describing.
    """
    _patch_default_backend(monkeypatch)

    def run() -> tuple[str, str]:
        set_campaign_backend("baybe")
        metadata = get_response_metadata()
        return metadata.backend, metadata.backend_source

    assert contextvars.copy_context().run(run) == ("baybe", "campaign")


def test_metadata_backend_source_names_server_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Campaign-agnostic calls disclose default provenance (issue #82)."""
    _patch_default_backend(monkeypatch)

    def run() -> tuple[str, str]:
        metadata = get_response_metadata()
        return metadata.backend, metadata.backend_source

    assert contextvars.copy_context().run(run) == ("default-backend", "server_default")


def test_metadata_protocol_defaults_to_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a bound transport, the in-process default stays "mcp"."""
    _patch_default_backend(monkeypatch)
    assert get_response_metadata().protocol == "mcp"


def test_metadata_protocol_prefers_bound_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transport bound at the dispatch boundary wins over the default.

    Regression (issue #82 follow-up review): once REST envelopes started
    forwarding ``_metadata``, the shared operations' ``protocol="mcp"``
    default surfaced in REST bodies as false transport provenance. The
    REST middleware now binds ``"rest"`` via
    :mod:`bo_mcp_server.protocol_context`.
    """
    _patch_default_backend(monkeypatch)
    with bind_protocol("rest"):
        assert get_response_metadata().protocol == "rest"
    assert get_response_metadata().protocol == "mcp"


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


def test_cross_campaign_scope_neutralizes_leaked_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross-campaign operations stamp the default even after a same-task bind.

    Regression for the REST↔MCP parity break surfaced by ``backend_source``:
    ``batch_get_status`` / ``compare_campaigns`` called in-process right
    after ``create_campaign`` inherited the created campaign's binding and
    mislabeled their multi-campaign envelope as campaign-scoped (issue #82).
    """
    _patch_default_backend(monkeypatch)

    @with_campaign_backend_scope
    async def probe() -> tuple[str, str]:
        metadata = get_response_metadata()
        return metadata.backend, metadata.backend_source

    def run() -> tuple[str, str]:
        set_campaign_backend("baybe")  # binding leaked by a previous operation
        return asyncio.run(probe())

    assert contextvars.copy_context().run(run) == ("default-backend", "server_default")


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
# Per-campaign backend in bo_list_campaigns summaries (issue #82)
# ---------------------------------------------------------------------------


def _make_campaign() -> Campaign:
    return Campaign(spec_id=uuid4(), owner_id=uuid4())


def test_standard_summary_names_campaign_backend() -> None:
    """STANDARD list items say which backend runs each campaign.

    Issue #82: the list envelope's ``_metadata.backend`` stamps the
    server default, so a mixed-backend deployment had no truthful
    per-row backend signal at all.
    """
    spec = _make_spec(backend="baybe")
    summary = _build_summary(_make_campaign(), spec.name, 0, spec, VerbosityLevel.STANDARD)
    assert summary["backend"] == "baybe"


def test_detailed_summary_names_campaign_backend() -> None:
    spec = _make_spec(backend="botorch")
    summary = _build_summary(_make_campaign(), spec.name, 0, spec, VerbosityLevel.DETAILED)
    assert summary["backend"] == "botorch"


def test_minimal_summary_stays_minimal() -> None:
    """MINIMAL keeps its documented three-key shape (~50-token budget)."""
    spec = _make_spec()
    summary = _build_summary(_make_campaign(), spec.name, 0, spec, VerbosityLevel.MINIMAL)
    assert set(summary) == {"campaign_id", "name", "status"}


def test_summary_backend_is_null_without_spec() -> None:
    """A missing spec degrades to ``backend: null``, never a crash."""
    summary = _build_summary(_make_campaign(), "Unknown", 0, None, VerbosityLevel.STANDARD)
    assert summary["backend"] is None


# ---------------------------------------------------------------------------
# model_info from backend method report (issue #57 root cause 2)
# ---------------------------------------------------------------------------


def test_model_info_uses_backend_method_report() -> None:
    spec = _make_spec(backend="baybe")
    method_info = {
        "model_type": "BayBE GP",
        "acquisition_function": "qLogNoisyExpectedImprovement",
        "optimization_strategy": "BotorchRecommender (GP-based)",
        "kernel": "Matern 5/2 (BayBE default GP surrogate)",
    }
    info = get_model_info(spec, method_info=method_info)
    assert info["backend"] == "baybe"
    assert info["type"] == "BayBE GP"
    assert info["acquisition_function"] == "qLogNoisyExpectedImprovement"
    assert info["batch_strategy"] == "BotorchRecommender (GP-based)"
    assert info["kernel"] == "Matern 5/2 (BayBE default GP surrogate)"


def test_model_info_without_method_report_is_generic() -> None:
    """No backend report → nulls, never a guessed model description.

    The former static fallback described BoTorch's model regardless of the
    campaign's backend (issue #57's mislabeling class), so a BayBE campaign
    whose ``select_methods`` failed would read "SingleTaskGP / Matern 5/2".
    The honest contract is ``backend`` plus null model fields.
    """
    spec = _make_spec(backend="baybe")
    info = get_model_info(spec, method_info=None)
    assert info["backend"] == "baybe"
    assert info["type"] is None
    assert info["acquisition_function"] is None
    assert info["batch_strategy"] is None
    assert info["kernel"] is None
    assert "SingleTaskGP" not in str(info)
    assert "Matern" not in str(info)


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
