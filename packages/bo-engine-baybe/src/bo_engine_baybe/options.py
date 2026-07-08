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

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    for transfer-learning across related campaigns, ``substance``
    switches to :class:`baybe.parameters.SubstanceParameter` for
    cheminformatics descriptors, and ``custom`` switches to
    :class:`baybe.parameters.CustomDiscreteParameter` so the caller can
    supply a precomputed numeric representation per label (e.g. from
    quantum chemistry).
    """

    CATEGORICAL = "categorical"
    TASK = "task"
    SUBSTANCE = "substance"
    CUSTOM = "custom"


class BayBESubstanceEncoding(StrEnum):
    """Substance encoding strategies supported by BayBE.

    Mirrors :class:`baybe.parameters.enum.SubstanceEncoding` (the full
    scikit-fingerprints-backed set as of BayBE 0.15) **minus the
    deprecated ``RDKIT`` alias** (BayBE warns and maps it to
    ``RDKIT2DDESCRIPTORS``). Every member is pinned against the installed
    BayBE by a guard test so a BayBE rename fails CI instead of a user
    campaign. Defaults follow BayBE's own defaults; the converter passes
    the string through to :class:`baybe.parameters.SubstanceParameter`.
    """

    ATOMPAIR = "ATOMPAIR"
    AUTOCORR = "AUTOCORR"
    AVALON = "AVALON"
    BCUT2D = "BCUT2D"
    E3FP = "E3FP"
    ECFP = "ECFP"
    ELECTROSHAPE = "ELECTROSHAPE"
    MORGAN_FP = "MORGAN_FP"
    ERG = "ERG"
    ESTATE = "ESTATE"
    FUNCTIONALGROUPS = "FUNCTIONALGROUPS"
    GETAWAY = "GETAWAY"
    GHOSECRIPPEN = "GHOSECRIPPEN"
    KLEKOTAROTH = "KLEKOTAROTH"
    LAGGNER = "LAGGNER"
    LAYERED = "LAYERED"
    LINGO = "LINGO"
    MACCS = "MACCS"
    MAP = "MAP"
    MHFP = "MHFP"
    MORSE = "MORSE"
    MQNS = "MQNS"
    MORDRED = "MORDRED"
    PATTERN = "PATTERN"
    PHARMACOPHORE = "PHARMACOPHORE"
    PHYSIOCHEMICALPROPERTIES = "PHYSIOCHEMICALPROPERTIES"
    PUBCHEM = "PUBCHEM"
    RDF = "RDF"
    RDKITFINGERPRINT = "RDKITFINGERPRINT"
    RDKIT2DDESCRIPTORS = "RDKIT2DDESCRIPTORS"
    SECFP = "SECFP"
    TOPOLOGICALTORSION = "TOPOLOGICALTORSION"
    USR = "USR"
    USRCAT = "USRCAT"
    VSA = "VSA"
    WHIM = "WHIM"


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
    # Restrict recommendations to a subset of the declared categories while
    # keeping the full set measurable. Valid for every categorical-family
    # role (categorical / task / substance / custom); membership in the
    # declared categories is validated at intake, and the tuple must be
    # non-empty (an empty subset would silently mean "all categories").
    active_values: tuple[str, ...] | None = Field(default=None, min_length=1)
    substance_data: dict[str, str] | None = None
    substance_encoding: BayBESubstanceEncoding | None = None
    # role=substance only: keyword passthroughs for the fingerprint
    # computation and (for conformer-based encodings) the conformer
    # generation — forwarded verbatim to BayBE's SubstanceParameter.
    kwargs_fingerprint: dict[str, Any] | None = None
    kwargs_conformer: dict[str, Any] | None = None
    # Numerical-discrete parameters only: measurement-matching slack used
    # by BayBE when assigning measured values to grid points. Must stay
    # below half the smallest gap between declared values (BayBE's own
    # validator); checked against the declared grid at intake.
    tolerance: float | None = Field(default=None, ge=0.0)
    custom_descriptors: dict[str, dict[str, float]] | None = Field(
        default=None,
        description=(
            "role=custom only. Precomputed numeric representation per category: "
            "{category label: {descriptor name: value}}. Each label becomes one row "
            "of the table BayBE's CustomDiscreteParameter encodes. Rules (enforced at "
            "campaign creation, rejected with a clear error): keys must match the "
            "declared `categories` exactly (no missing/extra labels); at least 2 "
            "categories; every value numeric and finite (no null/NaN/inf); no "
            "descriptor column may be constant across labels (carries no information); "
            "and no two labels may share an identical descriptor vector (ambiguous "
            "representation). Give each label a distinct, informative vector."
        ),
    )
    decorrelate: bool | float = Field(
        default=True,
        description=(
            "role=custom only. Mirrors BayBE CustomDiscreteParameter.decorrelate: "
            "true drops highly correlated descriptor columns, false keeps the table "
            "as-is, or a float in (0, 1) sets the correlation threshold."
        ),
    )


class BayBEInitialRecommender(StrEnum):
    """Initial-design (pre-model) recommender selection.

    ``random`` keeps the historical :class:`baybe.recommenders.RandomRecommender`
    default; ``fps`` selects farthest-point sampling and the clustering
    members select the corresponding clustering recommenders. All
    non-random members require a purely discrete/enumerable search space
    (their BayBE ``compatibility`` is ``SearchSpaceType.DISCRETE``) —
    validated at intake by the capability layer. The sequential/streaming
    meta-recommenders are deliberately not exposed:
    ``StreamingSequentialMetaRecommender`` is non-serializable and would
    break the JSON state envelope.
    """

    RANDOM = "random"
    FPS = "fps"
    KMEANS = "kmeans"
    PAM = "pam"
    GMM = "gmm"


class BayBEHybridSampler(StrEnum):
    """Sampling strategy for the discrete part of hybrid-space optimization.

    Mirrors :class:`baybe.utils.sampling_algorithms.DiscreteSamplingMethod`;
    pinned by a guard test against renames (the ``BayBESubstanceEncoding``
    precedent).
    """

    RANDOM = "Random"
    FPS = "FPS"


class BayBEBayesianRecommenderOptions(BaseModel):
    """Tuning knobs for BayBE's GP-phase :class:`BotorchRecommender`.

    All fields default to ``None`` = keep BayBE's own defaults. Note that
    ``sampling_percentage`` applies to **hybrid** spaces only — purely
    discrete spaces are scored exhaustively by BayBE regardless.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequential_continuous: bool | None = None
    hybrid_sampler: BayBEHybridSampler | None = None
    sampling_percentage: float | None = Field(default=None, gt=0.0, le=1.0)
    n_restarts: int | None = Field(default=None, ge=1)
    n_raw_samples: int | None = Field(default=None, ge=1)


class BayBERecommenderConfig(BaseModel):
    """Recommender configuration overrides.

    ``switch_after`` delays the initial → BO recommender switch and takes
    precedence over the neutral ``OptimizationSpec.initial_design_size``
    knob (explicit BayBE option wins); without either, BayBE switches
    after the first measurement. ``initial_recommender`` selects the
    space-filling phase recommender, and ``bayesian`` tunes the GP-phase
    :class:`BotorchRecommender`. The overall graph stays the two-phase
    meta-recommender.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    switch_after: int = Field(default=1, ge=1)
    initial_recommender: BayBEInitialRecommender = BayBEInitialRecommender.RANDOM
    bayesian: BayBEBayesianRecommenderOptions | None = None


class BayBESurrogateKind(StrEnum):
    """Curated surrogate-model choices.

    Deliberately a subset of BayBE's surrogate zoo: the bandit surrogate
    requires binary targets, and ``CustomONNXSurrogate`` needs a
    user-supplied model artifact with no intake path — both are excluded
    by design. Multi-objective campaigns replicate a non-GP surrogate per
    target via ``CompositeSurrogate.from_replication``.
    """

    GP = "gp"
    RANDOM_FOREST = "random_forest"
    NGBOOST = "ngboost"
    BAYESIAN_LINEAR = "bayesian_linear"
    MEAN_PREDICTION = "mean_prediction"


class BayBEGPPreset(StrEnum):
    """GP hyperparameter presets (``GaussianProcessSurrogate.from_preset``)."""

    BAYBE = "BAYBE"
    BOTORCH = "BOTORCH"
    CHEN = "CHEN"
    EDBO = "EDBO"
    EDBO_SMOOTHED = "EDBO_SMOOTHED"
    HVARFNER = "HVARFNER"


class BayBEKernelKind(StrEnum):
    """Curated GP kernel choices (Matern with tunable nu, RBF)."""

    MATERN = "matern"
    RBF = "rbf"


# Smoothness values accepted by the Matern kernel family (gpytorch contract).
MATERN_ALLOWED_NU: frozenset[float] = frozenset({0.5, 1.5, 2.5})


class BayBEKernelConfig(BaseModel):
    """GP kernel selection (wrapped in a ScaleKernel by the converter)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: BayBEKernelKind
    nu: float | None = None

    @model_validator(mode="after")
    def validate_kernel_fields(self) -> BayBEKernelConfig:
        """``nu`` belongs to the Matern kernel and must be a valid smoothness."""
        if self.nu is not None and self.kind != BayBEKernelKind.MATERN:
            msg = "nu is only valid for the matern kernel"
            raise ValueError(msg)
        if self.nu is not None and self.nu not in MATERN_ALLOWED_NU:
            msg = f"matern nu must be one of {sorted(MATERN_ALLOWED_NU)}"
            raise ValueError(msg)
        return self


class BayBESurrogateConfig(BaseModel):
    """Surrogate-model configuration (``backend_options['baybe'].surrogate``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: BayBESurrogateKind = BayBESurrogateKind.GP
    gp_preset: BayBEGPPreset | None = None
    kernel: BayBEKernelConfig | None = None

    @model_validator(mode="after")
    def validate_gp_only_fields(self) -> BayBESurrogateConfig:
        """Preset/kernel parameterize the GP surrogate only."""
        if self.kind != BayBESurrogateKind.GP and (
            self.gp_preset is not None or self.kernel is not None
        ):
            msg = "gp_preset/kernel are only valid with kind='gp'"
            raise ValueError(msg)
        return self


class BayBEExplainerKind(StrEnum):
    """Explainer backends for SHAP-based feature importance.

    Mirrors ``baybe.insights.shap.EXPLAINERS`` (SHAP explainers plus the
    non-SHAP LIME/MAPLE attributions); pinned by a guard test. The
    non-Kernel SHAP explainers reject categorical experimental
    representations — combine them with ``use_comp_rep=True`` on
    categorical campaigns.
    """

    KERNEL = "KernelExplainer"
    EXACT = "ExactExplainer"
    PERMUTATION = "PermutationExplainer"
    PARTITION = "PartitionExplainer"
    ADDITIVE = "AdditiveExplainer"
    LIME_TABULAR = "LimeTabular"
    MAPLE = "Maple"


class BayBEInsightsOptions(BaseModel):
    """Diagnostics/insights options (``backend_options['baybe'].insights``).

    ``explainer`` selects the SHAP/LIME/MAPLE explainer backend (BayBE
    default: KernelExplainer). ``use_comp_rep`` explains the computational
    (encoded) representation instead of the experimental one.
    ``include_row_level`` adds bounded per-observation attributions to the
    diagnostics payload: rows follow the campaign's measurement order and
    carry no per-row identifiers, and on multi-target campaigns only the
    *first* target's attributions are returned. ``row_level_max_rows``
    overrides the payload bound (default:
    ``bo_engine_baybe.constants.DEFAULT_MAX_ROW_LEVEL_SHAP_ROWS``). Plots
    are deliberately not exposed — binary images do not fit the JSON tool
    contract.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    explainer: BayBEExplainerKind | None = None
    use_comp_rep: bool = False
    include_row_level: bool = False
    row_level_max_rows: int | None = Field(default=None, ge=1)


class BayBEBackendOptions(BaseModel):
    """Typed BayBE-native campaign-level options.

    Stored as ``backend_options["baybe"]`` on a neutral
    :class:`~bo_mcp_server.domain.campaign_spec.CampaignSpec`. Validated at
    intake so misshaped payloads cannot reach
    :class:`~bo_engine_baybe.backend.BayBEBackend.generate_suggestions`.

    The ``allow_recommending_*`` toggles default to ``None`` = keep the
    backend's historical behavior (``False`` for purely discrete spaces,
    BayBE's AUTO otherwise — which resolves the pending flag to ``True``
    on spaces with a continuous part). An explicit
    ``allow_recommending_pending_experiments=False`` is honored on purely
    discrete spaces only: BayBE forbids it whenever the search space has
    a continuous subspace ("for algorithmic reasons" — pending points are
    handled by the acquisition function there, not by candidate
    exclusion), so the capability layer reports that combination
    UNSUPPORTED at intake. ``measurements_must_be_within_tolerance``
    maps to BayBE's ``add_measurements(numerical_measurements_must_be_within_tolerance=...)``
    (default ``True``); relaxing it accepts out-of-tolerance numeric
    values instead of raising — a data-quality trade-off the caller opts
    into explicitly.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    recommender: BayBERecommenderConfig | None = None
    surrogate: BayBESurrogateConfig | None = None
    insights: BayBEInsightsOptions | None = None
    allow_recommending_pending_experiments: bool | None = None
    allow_recommending_already_measured: bool | None = None
    allow_recommending_already_recommended: bool | None = None
    measurements_must_be_within_tolerance: bool | None = None
    # Large-categorical safeguard overrides. ``None`` keeps the named
    # defaults in :mod:`bo_engine_baybe.constants`
    # (DEFAULT_MAX_SEARCHSPACE_STATE_BYTES / DEFAULT_MAX_CANDIDATES); the
    # values bound the serialized size and row count of the enumerated
    # discrete subspace before deterministic subsampling kicks in.
    max_searchspace_state_bytes: int | None = Field(default=None, ge=1)
    max_candidates: int | None = Field(default=None, ge=1)


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
