"""Diagnostics ``model_info`` sourcing and point-of-need backend loading.

Two contracts are pinned here:

1. **Backend-truthful model info.** ``model_info`` fields (including
   ``kernel``) come from the campaign backend's own ``select_methods``
   report; when that report is unavailable the fields are null — never a
   guessed description that could name the wrong backend's model (the
   issue-#57 mislabeling class).
2. **Point-of-need backend loading.** Diagnostics sections that need no
   backend at all (``convergence``, ``constraints`` — pure server-side
   computation) must not cold-load the campaign's backend: on a
   baybe-default server a legacy ``backend="botorch"`` campaign would
   otherwise pay a multi-second torch import for a pure-Python answer.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from bo_engine.backend_base import BackendError
from bo_engine.botorch_backend import BoTorchBackend
from bo_engine.types import (
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)
from bo_engine_baybe.backend import BayBEBackend
from bo_mcp_server.domain import Campaign, CampaignSpec
from bo_mcp_server.operations import get_diagnostics as get_diagnostics_module
from bo_mcp_server.operations.diagnostics.enrichment import get_model_info


def _opt_spec() -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
    )


def _campaign_spec(backend: str) -> CampaignSpec:
    return CampaignSpec.model_validate(
        {
            "name": "Diagnostics fixture",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
            "backend": backend,
        }
    )


def _campaign() -> Campaign:
    return Campaign(spec_id=uuid4(), owner_id=uuid4())


# ---------------------------------------------------------------------------
# model_info completeness from both backends' select_methods reports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("backend_cls", "backend_name"),
    [(BoTorchBackend, "botorch"), (BayBEBackend, "baybe")],
)
def test_select_methods_report_is_model_info_complete(backend_cls, backend_name) -> None:
    """Every field ``get_model_info`` re-keys must be non-null in the report.

    Includes ``kernel``: the backend owns its model description, so the
    diagnostics surface never falls back to a server-side guess.
    """
    method_info = backend_cls().select_methods(_opt_spec(), 3)
    for field in ("model_type", "acquisition_function", "optimization_strategy", "kernel"):
        assert method_info.get(field), f"{backend_name} select_methods missing {field}"

    model_info = get_model_info(_campaign_spec(backend_name), method_info=method_info)
    assert model_info["backend"] == backend_name
    assert model_info["type"] == method_info["model_type"]
    assert model_info["kernel"] == method_info["kernel"]


# ---------------------------------------------------------------------------
# select_methods failure path: no cross-backend text resurrection
# ---------------------------------------------------------------------------


class _FailingSelectMethodsBackend:
    """Stub backend whose method report fails with a typed backend error."""

    name = "baybe"

    def compute_diagnostics(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {}

    def select_methods(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        msg = "forced select_methods failure"
        raise BackendError(msg)


@pytest.mark.asyncio
async def test_select_methods_failure_yields_generic_model_info(monkeypatch) -> None:
    """A BayBE campaign whose report fails must not read as a BoTorch model."""
    stub = _FailingSelectMethodsBackend()

    async def fake_get_backend_async(name: str | None = None) -> _FailingSelectMethodsBackend:
        del name
        return stub

    monkeypatch.setattr(get_diagnostics_module, "get_backend_async", fake_get_backend_async)
    spec = _campaign_spec("baybe")
    diagnostics = await get_diagnostics_module._compute_sections(
        frozenset({"objectives"}),
        spec,
        results=[],
        all_suggestions=[],
        pending_suggestions=[],
        campaign=_campaign(),
    )

    model_info = diagnostics["model_info"]
    assert model_info["backend"] == "baybe"
    assert model_info["type"] is None
    assert model_info["kernel"] is None
    rendered = str(model_info)
    assert "SingleTaskGP" not in rendered
    assert "Matern" not in rendered


# ---------------------------------------------------------------------------
# Point-of-need backend loading (H14 placement)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("sections", [{"convergence"}, {"constraints"}])
async def test_backend_free_sections_never_load_backend(monkeypatch, sections) -> None:
    """``convergence``/``constraints`` requests must not touch the backend loader."""

    async def spy_get_backend_async(_name: str | None = None):
        msg = f"get_backend_async called for backend-free sections {sections}"
        raise AssertionError(msg)

    monkeypatch.setattr(get_diagnostics_module, "get_backend_async", spy_get_backend_async)
    spec = _campaign_spec("botorch")
    diagnostics = await get_diagnostics_module._compute_sections(
        frozenset(sections),
        spec,
        results=[],
        all_suggestions=[],
        pending_suggestions=[],
        campaign=_campaign(),
    )
    assert diagnostics["success"] is True


@pytest.mark.asyncio
async def test_objectives_section_loads_backend_once(monkeypatch) -> None:
    """Sections that need the backend load it exactly once per computation."""
    calls: list[str | None] = []
    stub = _FailingSelectMethodsBackend()

    async def counting_get_backend_async(name: str | None = None):
        calls.append(name)
        return stub

    monkeypatch.setattr(get_diagnostics_module, "get_backend_async", counting_get_backend_async)
    spec = _campaign_spec("baybe")
    await get_diagnostics_module._compute_sections(
        frozenset({"objectives"}),
        spec,
        results=[],
        all_suggestions=[],
        pending_suggestions=[],
        campaign=_campaign(),
    )
    assert calls == ["baybe"]
