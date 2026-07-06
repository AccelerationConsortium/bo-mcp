"""Shared backend-resolution and capability-validation pipeline.

``bo_create_campaign`` and ``bo_validate_intake`` must agree on how a
spec's backend is resolved (``"auto"`` → concrete name), how the
requested/resolved names are stamped into the spec, and how capability
rejections are rendered. Both operations therefore call the helpers
below instead of carrying their own copies — the earlier duplicates had
already drifted apart on the ``errors`` string format.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from bo_engine.backend_base import BackendValidationResult
from bo_mcp_server.backend import get_backend_async, resolve_backend_name
from bo_mcp_server.backend_context import set_campaign_backend
from bo_mcp_server.converters import campaign_spec_to_optimization_spec
from bo_mcp_server.domain import CampaignSpec


@dataclass(frozen=True)
class ResolvedCapabilities:
    """Outcome of the shared resolve → stamp → capability-check pipeline."""

    backend: str
    spec: CampaignSpec
    result: BackendValidationResult


async def resolve_and_validate_capabilities(
    spec_data: dict[str, Any],
) -> ResolvedCapabilities:
    """Resolve the spec's backend, stamp it, and run capability validation.

    Mutates ``spec_data`` in place: ``requested_backend`` records the
    caller's original choice and ``backend`` is replaced with the resolved
    concrete name — exactly what ``create`` persists and ``validate``
    echoes. Resolution runs off-thread because resolving ``"auto"`` may
    cold-load candidate backends (multi-second torch/baybe imports).
    """
    raw_backend = spec_data.get("backend", "auto")
    resolved = await asyncio.to_thread(resolve_backend_name, raw_backend, spec_data)
    spec_data["requested_backend"] = raw_backend
    spec_data["backend"] = resolved
    # Stamp the resolved backend into the response envelope (issue #57).
    set_campaign_backend(resolved)

    spec = CampaignSpec.model_validate(spec_data)
    backend = await get_backend_async(resolved)
    opt_spec = campaign_spec_to_optimization_spec(spec)
    return ResolvedCapabilities(
        backend=resolved,
        spec=spec,
        result=backend.validate_capabilities(opt_spec),
    )


def capability_rejection_errors(
    result: BackendValidationResult,
) -> tuple[list[str], dict[str, list[str]]]:
    """Render unsupported capability reports as ``(errors, field_errors)``.

    One formatter for both operations so the same rejected spec reads
    identically in ``bo_create_campaign`` and ``bo_validate_intake``.
    ``errors`` entries follow the ``path: message`` convention of the
    intake validator. Keyless reports carry no dotted field path, so they
    land only in ``errors`` — never under an empty-string key in
    ``field_errors``, which agents key on to target the offending input.
    """
    errors: list[str] = []
    field_errors: dict[str, list[str]] = {}
    for report in result.unsupported:
        if not report.reason:
            continue
        errors.append(f"{report.key}: {report.reason}" if report.key else str(report.reason))
        if report.key:
            field_errors.setdefault(report.key, []).append(report.reason)
    return errors, field_errors
