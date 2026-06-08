"""Typed BayBE-specific option schemas.

The neutral :class:`bo_engine.types.ParameterSpec` and
:class:`bo_engine.types.OptimizationSpec` carry per-backend metadata via
free-form ``dict`` slots (``parameter_options`` / ``backend_options``).
This module defines the BayBE-native shape inside those slots so the
backend can validate user-supplied data once at intake time rather than
crashing with a ``KeyError`` deep inside the converter.

Two Pydantic submodels live here:

* :class:`BayBEParameterOptions` — per-parameter knobs (encoding,
  ``TaskParameter`` active values, substance descriptors,
  candidate-table membership).
* :class:`BayBEBackendOptions` — per-campaign knobs (recommender
  configuration overrides, candidate-table search-space mode).

Both classes use Pydantic's ``"forbid"`` extras policy so misspelled
keys become explicit validation errors instead of silently disappearing
into the opaque dict. The backend's ``validate_capabilities`` consumes
``extract_baybe_*_options`` to surface validation failures as
:class:`~bo_engine.backend_base.CapabilityReport` entries.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BayBEParameterEncoding(StrEnum):
    """Categorical-encoding choices supported by BayBE.

    Mirrors :class:`baybe.parameters.enum.CategoricalEncoding` so the
    neutral spec can carry the user's choice through validation without
    importing BayBE in the domain model.
    """

    OHE = "OHE"
    INT = "INT"


class BayBEParameterRole(StrEnum):
    """Role of a categorical-family BayBE parameter.

    ``categorical`` is the default vanilla one-hot/integer-encoded
    parameter, ``task`` switches to :class:`baybe.parameters.TaskParameter`
    for transfer-learning across related campaigns, and ``substance``
    switches to :class:`baybe.parameters.SubstanceParameter` for
    cheminformatics descriptors.
    """

    CATEGORICAL = "categorical"
    TASK = "task"
    SUBSTANCE = "substance"


class BayBESubstanceEncoding(StrEnum):
    """Substance encoding strategies supported by BayBE.

    Subset of :class:`baybe.parameters.enum.SubstanceEncoding`. Defaults
    follow BayBE's own defaults; the converter passes the string through
    to :class:`baybe.parameters.SubstanceParameter`.
    """

    MORDRED = "MORDRED"
    ECFP = "ECFP"
    RDKIT2DDESCRIPTORS = "RDKIT2DDESCRIPTORS"
    RDKITFINGERPRINT = "RDKITFINGERPRINT"


# Encoding applied to a ``role=substance`` parameter when the caller leaves
# ``substance_encoding`` unset. MORDRED matches BayBE's own
# ``SubstanceParameter`` default (a ~1800-descriptor physicochemical block);
# kept here as a named constant so :func:`spec_to_parameters` never hardcodes
# the string and the documented default lives in exactly one place.
DEFAULT_SUBSTANCE_ENCODING = BayBESubstanceEncoding.MORDRED


class BayBEParameterOptions(BaseModel):
    """Typed BayBE-native parameter metadata.

    Stored as ``parameter_options["baybe"]`` on a neutral
    :class:`~bo_mcp_server.domain.campaign_spec.InputParameter` (or
    :class:`~bo_engine.types.ParameterSpec`). Only fields relevant to the
    parameter's role are consumed by :func:`spec_to_parameters`; foreign
    keys are rejected via ``extra="forbid"``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: BayBEParameterRole = BayBEParameterRole.CATEGORICAL
    encoding: BayBEParameterEncoding | None = None
    active_values: tuple[str, ...] | None = None
    substance_data: dict[str, str] | None = None
    substance_encoding: BayBESubstanceEncoding | None = None


class BayBERecommenderConfig(BaseModel):
    """Recommender configuration overrides.

    The backend currently honors ``switch_after`` to delay the random →
    BO recommender switch (default keeps BayBE's own switch). Future
    growth happens here so the typed surface stays stable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    switch_after: int = Field(default=1, ge=1)


class BayBEBackendOptions(BaseModel):
    """Typed BayBE-native campaign-level options.

    Stored as ``backend_options["baybe"]`` on a neutral
    :class:`~bo_mcp_server.domain.campaign_spec.CampaignSpec`. Validated at
    intake so misshaped payloads cannot reach
    :class:`~bo_engine_baybe.backend.BayBEBackend.generate_suggestions`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    recommender: BayBERecommenderConfig | None = None
    allow_recommending_pending_experiments: bool = False


def extract_baybe_parameter_options(
    raw: dict[str, dict[str, Any]] | None,
) -> BayBEParameterOptions:
    """Return the validated BayBE per-parameter options.

    When ``raw`` is missing or has no ``"baybe"`` key, the default options
    are returned so callers do not have to guard against ``None``. Invalid
    payloads raise ``pydantic.ValidationError`` — the backend's
    ``validate_capabilities`` catches the error and surfaces it as a
    :class:`~bo_engine.backend_base.CapabilityReport` instead of failing
    suggestion generation.
    """
    if not raw or "baybe" not in raw:
        return BayBEParameterOptions()
    return BayBEParameterOptions.model_validate(raw["baybe"])


def extract_baybe_backend_options(
    raw: dict[str, dict[str, Any]] | None,
) -> BayBEBackendOptions:
    """Return the validated BayBE per-campaign options.

    Same defaulting/validation contract as
    :func:`extract_baybe_parameter_options`.
    """
    if not raw or "baybe" not in raw:
        return BayBEBackendOptions()
    return BayBEBackendOptions.model_validate(raw["baybe"])
