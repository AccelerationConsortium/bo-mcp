"""Surrogate / kernel / noise-prior construction for the BayBE backend.

Builds the curated ``backend_options['baybe'].surrogate`` selection into a
BayBE surrogate object consumed by ``BotorchRecommender(surrogate_model=...)``.
The curation is deliberately narrow (see
:class:`~bo_engine_baybe.options.BayBESurrogateKind`); widening happens on
request, always behind the pin-guard tests.

Reference: BayBE surrogates userguide
(https://emdgroup.github.io/baybe/stable/userguide/surrogates.html).
"""

from __future__ import annotations

import logging
from typing import cast

from baybe.kernels import (
    LinearKernel,
    MaternKernel,
    PeriodicKernel,
    PolynomialKernel,
    RBFKernel,
    RFFKernel,
    RQKernel,
    ScaleKernel,
)
from baybe.surrogates import (
    BayesianLinearSurrogate,
    GaussianProcessSurrogate,
    MeanPredictionSurrogate,
    NGBoostSurrogate,
    RandomForestSurrogate,
)
from baybe.surrogates.base import SurrogateProtocol

from bo_engine.types import OptimizationSpec
from bo_engine_baybe.options import (
    BayBEBackendOptions,
    BayBEKernelConfig,
    BayBEKernelKind,
    BayBESurrogateConfig,
    BayBESurrogateKind,
)

logger = logging.getLogger(__name__)

# Reason surfaced when a spec sets the neutral ``noise_prior_params`` on
# BayBE. Injecting a parameterized GP noise-prior likelihood requires a
# custom gpytorch object (or a custom likelihood-factory class), and
# BayBE's cattrs campaign serialization cannot restructure either — a
# campaign carrying one would fail every ``Campaign.from_json`` restore
# and silently rebuild from scratch each iteration. The option therefore
# stays BoTorch-only and is reported IGNORED here so ``backend="auto"``
# prefers BoTorch when the caller sets it.
NOISE_PRIOR_IGNORED_REASON = (
    "noise_prior_params targets the BoTorch backend. BayBE's campaign "
    "persistence (Campaign.to_json/from_json) cannot serialize a "
    "parameterized noise-prior likelihood, so honoring it would break "
    "state restoration; the option is ignored on BayBE."
)

# Default Matern smoothness when the kernel config omits ``nu`` — matches
# BayBE's own DefaultKernelFactory (ScaleKernel(MaternKernel(nu=2.5))).
DEFAULT_MATERN_NU = 2.5

# Parameterless surrogate factories for the non-GP curated kinds.
_SIMPLE_SURROGATE_FACTORIES: dict[BayBESurrogateKind, type] = {
    BayBESurrogateKind.RANDOM_FOREST: RandomForestSurrogate,
    BayBESurrogateKind.NGBOOST: NGBoostSurrogate,
    BayBESurrogateKind.BAYESIAN_LINEAR: BayesianLinearSurrogate,
    BayBESurrogateKind.MEAN_PREDICTION: MeanPredictionSurrogate,
}


def _build_kernel(config: BayBEKernelConfig) -> ScaleKernel:
    """Build the configured base kernel wrapped in a ScaleKernel.

    The ScaleKernel wrapper mirrors BayBE's own default kernel factory so
    a curated kernel selection changes the similarity structure, not the
    output-scale handling. ``power``/``num_samples`` are guaranteed
    non-``None`` for their respective kinds by
    :meth:`BayBEKernelConfig.validate_kernel_fields`.
    """
    if config.kind == BayBEKernelKind.MATERN:
        nu = config.nu if config.nu is not None else DEFAULT_MATERN_NU
        kernel = MaternKernel(nu=nu)
    elif config.kind == BayBEKernelKind.RBF:
        kernel = RBFKernel()
    elif config.kind == BayBEKernelKind.LINEAR:
        kernel = LinearKernel()
    elif config.kind == BayBEKernelKind.PERIODIC:
        kernel = (
            PeriodicKernel(period_length_initial_value=config.period_length)
            if config.period_length is not None
            else PeriodicKernel()
        )
    elif config.kind == BayBEKernelKind.POLYNOMIAL:
        kernel = PolynomialKernel(power=cast("int", config.power))
    elif config.kind == BayBEKernelKind.RQ:
        kernel = RQKernel()
    else:
        kernel = RFFKernel(num_samples=cast("int", config.num_samples))
    return ScaleKernel(kernel)


def _build_gp_surrogate(config: BayBESurrogateConfig) -> GaussianProcessSurrogate:
    """Build the (optionally preset/kernel configured) GP surrogate."""
    kernel = _build_kernel(config.kernel) if config.kernel is not None else None
    if config.gp_preset is not None:
        return GaussianProcessSurrogate.from_preset(
            config.gp_preset.value,
            kernel_or_factory=kernel,
        )
    if kernel is not None:
        return GaussianProcessSurrogate(kernel_or_factory=kernel)
    return GaussianProcessSurrogate()


def build_baybe_surrogate(
    spec: OptimizationSpec,
    options: BayBEBackendOptions,
) -> SurrogateProtocol | None:
    """Build the configured surrogate model, or ``None`` for BayBE's default.

    ``None`` is returned when no surrogate config is set, so the historical
    implicit-default-GP path stays byte-identical. Single-output surrogates
    are replicated per target automatically by BayBE's
    ``BotorchRecommender`` (it wraps them in a ``CompositeSurrogate``), so
    no replication handling is needed here. The neutral
    ``noise_prior_params`` is deliberately NOT consumed — see
    :data:`NOISE_PRIOR_IGNORED_REASON`.
    """
    _ = spec
    config = options.surrogate
    if config is None:
        return None
    if config.kind == BayBESurrogateKind.GP:
        return _build_gp_surrogate(config)
    return _SIMPLE_SURROGATE_FACTORIES[config.kind]()
