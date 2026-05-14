"""Backend-option round-trip tests for TODO 1.66 (MCP-server side).

Covers:

* ``CampaignIntakeInput`` defaults ``random_seed`` to ``None`` (the
  same default REST already used) so omitting the field produces
  identical campaign behavior on both transports.
* ``CampaignIntakeInput`` validation rejects ``backend_options`` whose
  outer keys do not match the chosen backend.
* The canonical ``CampaignSpec.model_validate`` round-trip used by
  :func:`create_campaign_operation` preserves every advanced spec
  field — the previous manual reconstruction dropped them silently.
* ``backend_options`` and ``parameter_options`` survive the conversion
  to ``OptimizationSpec``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bo_mcp_server.converters import campaign_spec_to_optimization_spec
from bo_mcp_server.domain import (
    CampaignIntakeInput,
    CampaignSpec,
    InputParameter,
    Objective,
    ParameterType,
)
from bo_mcp_server.operations.create_campaign import _build_spec_from_dict


def _minimal_intake() -> dict:
    return {
        "name": "Round Trip",
        "parameters": [
            {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
        ],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }


class TestRandomSeedDefault:
    """MCP intake defaults ``random_seed`` to ``None`` (matches REST)."""

    def test_mcp_intake_random_seed_defaults_to_none(self) -> None:
        intake = CampaignIntakeInput.model_validate(_minimal_intake())
        assert intake.random_seed is None

    def test_omitted_seed_propagates_none_to_spec(self) -> None:
        """When seed is omitted, the rebuilt CampaignSpec carries ``None``.

        Previously MCP defaulted to 42 — so the same omitted field
        produced deterministic MCP campaigns but unseeded REST
        campaigns. With the default normalized to ``None`` the
        transports now match.
        """
        intake = CampaignIntakeInput.model_validate(_minimal_intake())
        spec = CampaignSpec.model_validate(intake.model_dump())
        assert spec.random_seed is None


class TestBackendOptionsValidation:
    """``backend_options`` keys must match the selected backend."""

    def test_options_for_matching_backend_accepted(self) -> None:
        payload = {
            **_minimal_intake(),
            "backend": "botorch",
            "backend_options": {"botorch": {"acquisition_optimizer": "lbfgsb"}},
        }
        intake = CampaignIntakeInput.model_validate(payload)
        assert intake.backend_options is not None

    def test_options_for_auto_backend_accepted_with_any_key(self) -> None:
        """``backend=auto`` cannot pre-determine the route, so any key is allowed.

        The selected backend is responsible for ignoring options addressed
        to other backends at validate_capabilities() time.
        """
        payload = {
            **_minimal_intake(),
            "backend": "auto",
            "backend_options": {
                "botorch": {"acquisition_optimizer": "lbfgsb"},
                "baybe": {"encoding": "ohe"},
            },
        }
        intake = CampaignIntakeInput.model_validate(payload)
        assert "botorch" in (intake.backend_options or {})
        assert "baybe" in (intake.backend_options or {})

    def test_options_for_non_matching_backend_rejected(self) -> None:
        payload = {
            **_minimal_intake(),
            "backend": "botorch",
            "backend_options": {"baybe": {"encoding": "ohe"}},
        }
        with pytest.raises(ValidationError, match="backend_options keys"):
            CampaignIntakeInput.model_validate(payload)


class TestCanonicalCampaignSpecRoundTrip:
    """``_build_spec_from_dict`` preserves every field via canonical validation."""

    def test_advanced_fields_round_trip(self) -> None:
        spec = CampaignSpec(
            name="Advanced",
            parameters=[
                InputParameter(
                    name="x",
                    type=ParameterType.CONTINUOUS,
                    bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
                ),
            ],
            objectives=[Objective(name="y", direction="minimize")],
            use_input_warping=True,
            backend="botorch",
            backend_options={"botorch": {"acquisition_optimizer": "lbfgsb"}},
        )
        spec_dict = spec.to_dict()
        rebuilt = _build_spec_from_dict(spec_dict)
        assert rebuilt.use_input_warping is True
        assert rebuilt.backend_options == {"botorch": {"acquisition_optimizer": "lbfgsb"}}

    def test_parameter_options_round_trip(self) -> None:
        spec = CampaignSpec(
            name="Param Options",
            parameters=[
                InputParameter(
                    name="x",
                    type=ParameterType.CONTINUOUS,
                    bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
                    parameter_options={"baybe": {"encoding": "ohe"}},
                ),
            ],
            objectives=[Objective(name="y", direction="minimize")],
        )
        spec_dict = spec.to_dict()
        rebuilt = _build_spec_from_dict(spec_dict)
        assert rebuilt.parameters[0].parameter_options == {"baybe": {"encoding": "ohe"}}

    def test_convert_preserves_backend_options(self) -> None:
        spec = CampaignSpec(
            name="Convert",
            parameters=[
                InputParameter(
                    name="x",
                    type=ParameterType.CONTINUOUS,
                    bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
                    parameter_options={"baybe": {"encoding": "ohe"}},
                ),
            ],
            objectives=[Objective(name="y", direction="minimize")],
            backend_options={"botorch": {"acquisition_optimizer": "lbfgsb"}},
        )
        opt_spec = campaign_spec_to_optimization_spec(spec)
        assert opt_spec.backend_options == {"botorch": {"acquisition_optimizer": "lbfgsb"}}
        assert opt_spec.parameters[0].parameter_options == {"baybe": {"encoding": "ohe"}}
