"""Gaussian Process model creation and fitting.

Supports:
- SingleTaskGP for single-objective optimization
- ModelListGP for multi-objective optimization
- Optional input warping with Kumaraswamy CDF for non-stationary objectives
- Automatic GPU acceleration when available

Output standardization convention
---------------------------------
All GPs in this module attach ``outcome_transform=Standardize(m=1)``. BoTorch
applies the transform during ``SingleTaskGP.__init__`` (the transform is set to
``train()`` mode and invoked on ``train_Y``), so ``model.train_targets`` holds
the standardized targets. ``fit_gpytorch_mll`` therefore calibrates the output
scale, kernel outputscale and noise against unit-scale targets, and
``model.posterior(X)`` automatically untransforms back to the user's original
scale. Use :func:`verify_standardization` to confirm the convention holds on a
fitted model -- e.g. in tests that rely on scale-invariant behavior.

For multi-objective problems ``create_model`` wraps independent ``SingleTaskGP``
instances in a ``ModelListGP`` so each objective is standardized separately.
This is the mechanism that keeps well-behaved acquisition geometry when
objectives span wildly different magnitudes (e.g. yield ~0.5 vs throughput
~500).

v1.0.1: Added create_single_task_model for single-objective
v1.1: Added input warping support
v2.3: Added GPU auto-detection and acceleration
v2.4: Documented output-standardization convention and added
      ``verify_standardization``.
"""

import logging
import math

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.kernels import CategoricalKernel
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import ChainedInputTransform, Normalize, Warp
from botorch.models.transforms.outcome import (
    ChainedOutcomeTransform,
    Log,
    OutcomeTransform,
    Standardize,
)
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.priors import GammaPrior, Prior
from gpytorch.priors.torch_priors import LogNormalPrior
from torch import Tensor

from bo_engine.constants import (
    NOISE_PRIOR_GAMMA_CONCENTRATION,
    NOISE_PRIOR_GAMMA_RATE,
    NOISE_PRIOR_MIN_INFERRED,
    STANDARDIZATION_MEAN_TOLERANCE,
    STANDARDIZATION_STD_FLOOR,
    STANDARDIZATION_VAR_TOLERANCE,
    WARP_PRIOR_LOC,
    WARP_PRIOR_SCALE,
)
from bo_engine.device import ensure_device, get_device, to_device

logger = logging.getLogger(__name__)


class ModelFittingError(RuntimeError):
    """Raised when GP model fitting fails.

    Contains the original exception and a user-friendly message with recovery guidance.
    """

    def __init__(self, message: str, original_error: Exception) -> None:
        super().__init__(message)
        self.original_error = original_error


def _default_noise_prior() -> GammaPrior:
    """Return the default mildly informative ``GammaPrior`` for GP noise.

    Concentrations come from :mod:`bo_engine.constants` and are calibrated for
    targets that have been standardized to unit variance via
    ``Standardize(m=1)``.
    """
    return GammaPrior(NOISE_PRIOR_GAMMA_CONCENTRATION, NOISE_PRIOR_GAMMA_RATE)


def _build_outcome_transform(log_transform: bool) -> OutcomeTransform:
    """Build the outcome transform stack for a single objective.

    By default the GP only standardizes targets (mean 0, unit variance).
    When ``log_transform`` is enabled a :class:`~botorch.models.transforms.outcome.Log`
    transform is applied first via :class:`ChainedOutcomeTransform`, which
    makes the model behave reasonably on multi-decade objectives whose
    raw scale spans several orders of magnitude (e.g. reaction rates or
    contaminant concentrations). BoTorch un-applies both stages on the
    posterior so callers still see results in the user's original scale.

    **Positivity is required.** :class:`Log` operates on the raw target
    values; passing a zero or negative target produces ``-inf`` / ``nan``
    and breaks the downstream ``Standardize`` mean estimate. Callers
    therefore must guarantee strictly positive ``train_Y`` for the
    objective when ``log_transform=True``. The model factories
    (:func:`create_single_task_model`, :func:`create_model`) enforce
    this at construction time so the failure mode is a clear
    ``ValueError`` instead of a numerical NaN cascade.
    """
    if log_transform:
        return ChainedOutcomeTransform(log=Log(), standardize=Standardize(m=1))
    return Standardize(m=1)


# Headroom added on top of ``-min(y)`` when computing the auto-shift; without it
# the shifted target still hits zero exactly and ``Log`` produces ``-inf``.
LOG_AUTO_SHIFT_EPSILON = 1e-6


def compute_log_auto_shift(train_y: Tensor) -> float:
    """Return the additive shift required to make ``train_y`` strictly positive.

    Returns ``0.0`` when the data is already strictly positive (so the
    auto-shift is a no-op and existing campaigns keep their behaviour).
    Otherwise returns ``-min(train_y) + LOG_AUTO_SHIFT_EPSILON`` so the
    smallest observation maps to ``+eps`` after shifting, well above the
    Log transform's ``-inf`` singularity.
    """
    if train_y.numel() == 0:
        return 0.0
    min_value = float(train_y.detach().min().item())
    if min_value > 0:
        return 0.0
    return -min_value + LOG_AUTO_SHIFT_EPSILON


def _assert_positive_for_log_transform(train_y: Tensor, objective_index: int) -> None:
    """Raise ``ValueError`` if any target row is non-positive.

    Enforced at construction time so the failure mode for a
    misconfigured ``log_transform`` campaign is a clear envelope at the
    boundary, not a NaN that propagates through the standardize step
    and surfaces as an opaque BoTorch fit error.
    """
    if not bool(torch.isfinite(train_y).all()):
        raise ValueError(
            f"log_transform=True requires finite targets; objective[{objective_index}] "
            "has NaN or inf values. Drop or impute those rows before fitting."
        )
    if not bool((train_y > 0).all()):
        min_value = float(train_y.min().item())
        raise ValueError(
            f"log_transform=True requires strictly positive targets for "
            f"objective[{objective_index}]; got min={min_value}. Either drop "
            "non-positive observations, pre-shift the target, or disable "
            "log_transform for this objective."
        )


def _build_likelihood(noise_prior: Prior | None) -> GaussianLikelihood:
    """Build a Gaussian likelihood with an explicit, mildly informative prior.

    A floor of ``NOISE_PRIOR_MIN_INFERRED`` is applied via ``GreaterThan`` so
    the inferred noise cannot collapse to zero on multi-scale objectives — a
    known source of singular Cholesky factors during ``fit_gpytorch_mll``.
    """
    prior = noise_prior if noise_prior is not None else _default_noise_prior()
    return GaussianLikelihood(
        noise_prior=prior,
        noise_constraint=GreaterThan(NOISE_PRIOR_MIN_INFERRED),
    )


def _log_fitted_noise(model: SingleTaskGP | ModelListGP, *, fixed_noise: bool) -> None:
    """Emit a debug log of the post-fit noise hyperparameter for drift tracking.

    Targets are standardized, so values close to 0 mean the GP is treating the
    objective as nearly deterministic; values approaching 1 mean noise has
    absorbed most of the signal variance (a silent under-fit symptom).
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return

    if isinstance(model, ModelListGP):
        sub_models: list[SingleTaskGP] = list(model.models)  # ty: ignore[invalid-argument-type]
    else:
        sub_models = [model]

    for idx, gp in enumerate(sub_models):
        likelihood = getattr(gp, "likelihood", None)
        if likelihood is None:
            continue
        noise = getattr(likelihood, "noise", None)
        if noise is None:
            continue
        flat = noise.detach().reshape(-1).to(dtype=torch.float64)
        logger.debug(
            "Fitted noise (model=%d, fixed=%s): min=%.4g mean=%.4g max=%.4g",
            idx,
            fixed_noise,
            float(flat.min().item()),
            float(flat.mean().item()),
            float(flat.max().item()),
        )


def build_mixed_kernel(
    n_total_dims: int,
    categorical_dim_indices: list[int],
) -> Kernel:
    """Build an additive ``RBF(continuous) + CategoricalKernel(one_hot)`` kernel.

    Used when :attr:`OptimizationSpec.use_categorical_kernel` is set on a spec
    that mixes continuous and one-hot categorical columns. The continuous
    block uses the same ARD ``RBFKernel`` shape that ``SingleTaskGP``'s
    default kernel uses (``ScaleKernel(RBFKernel(ard_num_dims=k))``); the
    categorical block uses BoTorch's :class:`CategoricalKernel` restricted
    to the one-hot column indices via ``active_dims`` so it computes
    Hamming-style similarity on the categorical sub-space rather than
    Euclidean distance on the one-hot expansion. The two kernels are
    summed inside a single ``ScaleKernel`` so the model can learn a joint
    outputscale.

    Args:
        n_total_dims: Total number of encoded input dimensions (continuous +
            one-hot categorical columns).
        categorical_dim_indices: Sorted list of column indices that belong
            to one-hot categorical blocks.

    Returns:
        A composite :class:`ScaleKernel` ready to be assigned to
        ``SingleTaskGP.covar_module``.

    Raises:
        ValueError: If the categorical indices fall outside ``[0, n_total_dims)``.
    """
    if any(i < 0 or i >= n_total_dims for i in categorical_dim_indices):
        raise ValueError(
            "categorical_dim_indices must all lie in "
            f"[0, {n_total_dims}); got {categorical_dim_indices}."
        )
    cont_indices = [i for i in range(n_total_dims) if i not in set(categorical_dim_indices)]

    parts: list[Kernel] = []
    if cont_indices:
        rbf = RBFKernel(
            ard_num_dims=len(cont_indices),
            active_dims=tuple(cont_indices),
        )
        parts.append(rbf)
    if categorical_dim_indices:
        cat = CategoricalKernel(
            ard_num_dims=len(categorical_dim_indices),
            active_dims=tuple(categorical_dim_indices),
        )
        parts.append(cat)

    if not parts:
        # Degenerate case (n_total_dims == 0); shouldn't happen for a real spec.
        return ScaleKernel(RBFKernel(ard_num_dims=n_total_dims))

    combined: Kernel = parts[0]
    for kernel in parts[1:]:
        combined = combined + kernel
    return ScaleKernel(combined)


def create_input_transform(
    n_dims: int,
    bounds: Tensor,
    use_input_warping: bool = False,
    *,
    warp_prior_loc: float = WARP_PRIOR_LOC,
    warp_prior_scale: float = WARP_PRIOR_SCALE,
) -> Normalize | ChainedInputTransform:
    """Create input transform with optional warping.

    Transform order is ``normalize → warp``: ``Normalize`` brings raw inputs
    into the unit hypercube, and ``Warp`` (Kumaraswamy CDF) reshapes that
    bounded space to absorb non-stationarity. This ordering matters because
    the Kumaraswamy CDF is defined on ``[0, 1]``; reversing it (warp →
    normalize) requires a custom Kumaraswamy prior calibrated against the
    user's raw input bounds, which is not the case here. The downstream
    ``LogNormalPrior`` on each ``concentration`` therefore assumes the
    *post-normalize* unit-cube domain.

    The ``LogNormalPrior(loc=0, scale=0.75)`` default is the BoTorch stock
    prior. Empirically calibrated alternatives can be passed via
    ``warp_prior_loc`` / ``warp_prior_scale`` for problems where the stock
    prior is too tight (recovers near-identity warps) or too loose (drives
    pathological boundary behaviour). See Eriksson & Snoek 2021 for a
    discussion of warp-prior calibration on non-stationary objectives.

    Args:
        n_dims: Number of input dimensions
        bounds: Parameter bounds of shape (2, n_dims)
        use_input_warping: If True, add Kumaraswamy warping after normalization
        warp_prior_loc: ``loc`` argument for the LogNormal prior on each
            Kumaraswamy concentration parameter. Default matches the BoTorch
            stock prior calibrated for the unit-cube domain.
        warp_prior_scale: ``scale`` argument for the LogNormal prior. Smaller
            values pull both concentrations toward 1 (identity warp); larger
            values allow more pronounced shape transformations.

    Returns:
        Input transform (Normalize or ChainedInputTransform with Warp)
    """
    bounds = to_device(bounds)
    normalize = Normalize(d=n_dims, bounds=bounds)

    if not use_input_warping:
        return normalize

    # Create Kumaraswamy warping with learnable concentration parameters.
    # The prior operates on the post-normalize unit-cube domain (see the
    # docstring above for the chain-order justification).
    device = get_device()
    warp = Warp(
        d=n_dims,
        indices=list(range(n_dims)),
        concentration1_prior=LogNormalPrior(
            loc=torch.tensor(warp_prior_loc, device=device),
            scale=torch.tensor(warp_prior_scale, device=device),
        ),
        concentration0_prior=LogNormalPrior(
            loc=torch.tensor(warp_prior_loc, device=device),
            scale=torch.tensor(warp_prior_scale, device=device),
        ),
    )

    return ChainedInputTransform(
        normalize=normalize,
        warp=warp,
    )


def create_single_task_model(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    use_input_warping: bool = False,
    train_yvar: Tensor | None = None,
    noise_prior: Prior | None = None,
    log_transform: bool = False,
    *,
    categorical_dim_indices: list[int] | None = None,
    auto_shift_for_log: bool = False,
) -> SingleTaskGP:
    """Create a SingleTaskGP for single-objective optimization.

    The returned model owns a ``Standardize(m=1)`` outcome transform. BoTorch
    standardizes ``train_Y`` during ``__init__`` so ``fit_gpytorch_mll`` sees
    unit-scale targets, and untransforms the posterior on inference. See the
    module docstring for the full convention.

    Args:
        train_x: Training inputs of shape (n_samples, n_dims)
        train_y: Training outputs of shape (n_samples, 1) or (n_samples,).
            Raw, unstandardized targets -- do not pre-standardize.
        bounds: Parameter bounds of shape (2, n_dims)
        use_input_warping: If True, use Kumaraswamy input warping
        train_yvar: Optional per-observation noise variance of shape
            (n_samples, 1) or (n_samples,). When supplied, BoTorch builds a
            ``FixedNoiseGaussianLikelihood`` internally and the noise
            hyperparameter is no longer trainable -- ``noise_prior`` is
            therefore ignored in this branch.
        noise_prior: Optional explicit GPyTorch ``Prior`` on the trainable
            noise hyperparameter. Defaults to a mildly informative
            ``GammaPrior`` calibrated for standardized targets (see
            ``bo_engine.constants``). Only used when ``train_yvar`` is None.
        log_transform: If True, apply ``Log`` before ``Standardize(m=1)`` so
            objectives spanning several orders of magnitude (e.g. reaction
            rates) train against a roughly homoskedastic scale. Requires
            strictly positive targets.
        categorical_dim_indices: Optional list of column indices that
            belong to one-hot categorical blocks. When supplied, the GP is
            built with an additive ``RBF(continuous) +
            CategoricalKernel(one_hot)`` kernel via
            :func:`build_mixed_kernel` so categorical similarity is
            Hamming-style rather than Euclidean on the one-hot expansion.
            ``None`` (default) preserves the historical pure-RBF behavior.
        auto_shift_for_log: Only meaningful when ``log_transform=True``.
            ``True`` instructs the factory to compute a one-shot additive
            shift ``-min(train_y) + LOG_AUTO_SHIFT_EPSILON`` when any
            training target is non-positive, apply it before fitting, and
            stash the shift on the returned model as
            ``model._auto_shift_for_log`` so downstream callers can back
            it out of posterior predictions. ``False`` (default) preserves
            the legacy "raise on non-positive" behavior.

    Returns:
        SingleTaskGP model (unfitted)
    """
    if train_yvar is not None:
        train_x, train_y, train_yvar, bounds = ensure_device(train_x, train_y, train_yvar, bounds)
    else:
        train_x, train_y, bounds = ensure_device(train_x, train_y, bounds)

    # Ensure train_y has correct shape (n_samples, 1)
    if train_y.dim() == 1:
        train_y = train_y.unsqueeze(-1)
    if train_yvar is not None and train_yvar.dim() == 1:
        train_yvar = train_yvar.unsqueeze(-1)

    applied_shift = 0.0
    if log_transform and auto_shift_for_log:
        applied_shift = compute_log_auto_shift(train_y)
        if applied_shift > 0:
            logger.warning(
                "log_transform=True with non-positive observations on "
                "objective[0]; applying auto_shift_for_log=%.6g so the Log "
                "transform is well-defined. Posterior predictions are stored "
                "on the shifted scale; subtract %.6g to recover raw values.",
                applied_shift,
                applied_shift,
            )
            train_y = train_y + applied_shift

    input_transform = create_input_transform(
        n_dims=train_x.shape[-1],
        bounds=bounds,
        use_input_warping=use_input_warping,
    )

    if log_transform:
        _assert_positive_for_log_transform(train_y, objective_index=0)

    kwargs: dict = {
        "train_X": train_x,
        "train_Y": train_y,
        "input_transform": input_transform,
        "outcome_transform": _build_outcome_transform(log_transform),
    }
    if categorical_dim_indices:
        kwargs["covar_module"] = build_mixed_kernel(
            n_total_dims=int(train_x.shape[-1]),
            categorical_dim_indices=list(categorical_dim_indices),
        )
    if train_yvar is not None:
        # Heteroskedastic / known-uncertainty path. BoTorch routes Yvar
        # through a FixedNoiseGaussianLikelihood, so the noise hyperparameter
        # is non-trainable and any ``noise_prior`` would have no effect.
        kwargs["train_Yvar"] = train_yvar
    else:
        kwargs["likelihood"] = _build_likelihood(noise_prior)

    model = SingleTaskGP(**kwargs)
    # Stash the applied shift so downstream callers (suggestion provenance,
    # diagnostics) can subtract it before reporting predictions in the
    # user-facing scale.
    if applied_shift > 0:
        model._auto_shift_for_log = applied_shift  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    return model


def create_model(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    use_input_warping: bool = False,
    train_yvar: Tensor | None = None,
    noise_prior: Prior | None = None,
    log_transform: bool | list[bool] = False,
    *,
    categorical_dim_indices: list[int] | None = None,
) -> ModelListGP:
    """Create a ModelListGP for multi-objective optimization.

    Uses independent GP models for each objective. Each per-objective model
    owns its own ``Standardize(m=1)`` outcome transform, so objectives with
    very different magnitudes (e.g. yield ~0.5 vs throughput ~500) are still
    fit against unit-scale targets.

    Args:
        train_x: Training inputs of shape (n_samples, n_dims)
        train_y: Training outputs of shape (n_samples, n_objectives).
            Raw, unstandardized targets -- do not pre-standardize.
        bounds: Parameter bounds of shape (2, n_dims)
        use_input_warping: If True, use Kumaraswamy input warping
        train_yvar: Optional per-objective noise variance of shape
            (n_samples, n_objectives). When supplied, every sub-model is
            built with a ``FixedNoiseGaussianLikelihood`` and ``noise_prior``
            is ignored.
        noise_prior: Optional GPyTorch ``Prior`` shared across sub-models.
            Defaults to a mildly informative ``GammaPrior`` for standardized
            targets. Only used when ``train_yvar`` is None.
        log_transform: If True, applies ``Log`` before ``Standardize`` on
            every sub-model. Pass a list aligned with the objective
            dimension to enable it per-objective; objectives with
            multi-decade magnitudes (e.g. concentrations) typically
            benefit while bounded ones (yield in [0, 1]) do not.

    Returns:
        ModelListGP with one GP per objective
    """
    if train_yvar is not None:
        train_x, train_y, train_yvar, bounds = ensure_device(train_x, train_y, train_yvar, bounds)
    else:
        train_x, train_y, bounds = ensure_device(train_x, train_y, bounds)

    n_objectives = train_y.shape[-1]
    n_dims = train_x.shape[-1]

    if isinstance(log_transform, bool):
        log_flags = [log_transform] * n_objectives
    else:
        if len(log_transform) != n_objectives:
            raise ValueError(
                "log_transform list length must match the number of objectives; "
                f"got {len(log_transform)} flag(s) for {n_objectives} objective(s)."
            )
        log_flags = list(log_transform)

    models = []
    for i in range(n_objectives):
        # Create input transform (each model gets its own)
        input_transform = create_input_transform(
            n_dims=n_dims,
            bounds=bounds,
            use_input_warping=use_input_warping,
        )

        if log_flags[i]:
            _assert_positive_for_log_transform(train_y[:, i : i + 1], objective_index=i)

        kwargs: dict = {
            "train_X": train_x,
            "train_Y": train_y[:, i : i + 1],
            "input_transform": input_transform,
            "outcome_transform": _build_outcome_transform(log_flags[i]),
        }
        if categorical_dim_indices:
            kwargs["covar_module"] = build_mixed_kernel(
                n_total_dims=n_dims,
                categorical_dim_indices=list(categorical_dim_indices),
            )
        if train_yvar is not None:
            kwargs["train_Yvar"] = train_yvar[:, i : i + 1]
        else:
            kwargs["likelihood"] = _build_likelihood(noise_prior)

        models.append(SingleTaskGP(**kwargs))

    return ModelListGP(*models)


def fit_single_task_model(model: SingleTaskGP) -> SingleTaskGP:
    """Fit a SingleTaskGP by maximizing marginal likelihood.

    Args:
        model: SingleTaskGP to fit

    Returns:
        Fitted model

    Raises:
        ModelFittingError: If fitting fails (singular matrix, numerical instability, etc.)
    """
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    try:
        fit_gpytorch_mll(mll)
    except (RuntimeError, torch.linalg.LinAlgError) as e:
        msg = (
            f"GP model fitting failed: {e}. "
            "Consider adding more observations or reducing parameter count."
        )
        logger.error(msg)
        raise ModelFittingError(msg, original_error=e) from e
    _log_fitted_noise(model, fixed_noise=_has_fixed_noise(model))
    return model


def fit_model(model: ModelListGP) -> ModelListGP:
    """Fit the GP model by maximizing marginal likelihood.

    Args:
        model: ModelListGP to fit

    Returns:
        Fitted model

    Raises:
        ModelFittingError: If fitting fails (singular matrix, numerical instability, etc.)
    """
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    try:
        fit_gpytorch_mll(mll)
    except (RuntimeError, torch.linalg.LinAlgError) as e:
        msg = (
            f"Multi-output GP model fitting failed: {e}. "
            "Consider adding more observations or reducing parameter count."
        )
        logger.error(msg)
        raise ModelFittingError(msg, original_error=e) from e
    _log_fitted_noise(model, fixed_noise=_has_fixed_noise(model))
    return model


def _has_fixed_noise(model: SingleTaskGP | ModelListGP) -> bool:
    """Return True iff any sub-model uses a FixedNoiseGaussianLikelihood."""
    if isinstance(model, ModelListGP):
        sub_models: list[SingleTaskGP] = list(model.models)  # ty: ignore[invalid-argument-type]
    else:
        sub_models = [model]
    for gp in sub_models:
        likelihood = getattr(gp, "likelihood", None)
        if likelihood is None:
            continue
        # FixedNoiseGaussianLikelihood exposes `noise_covar` of type FixedGaussianNoise
        noise_covar = getattr(likelihood, "noise_covar", None)
        if noise_covar is not None and not hasattr(noise_covar, "raw_noise"):
            return True
    return False


def create_and_fit_single_task_model(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    use_input_warping: bool = False,
    train_yvar: Tensor | None = None,
    noise_prior: Prior | None = None,
    log_transform: bool = False,
    *,
    categorical_dim_indices: list[int] | None = None,
    auto_shift_for_log: bool = False,
) -> SingleTaskGP:
    """Create and fit a SingleTaskGP.

    Convenience function for single-objective optimization.

    Args:
        train_x: Training inputs of shape (n_samples, n_dims)
        train_y: Training outputs of shape (n_samples, 1) or (n_samples,)
        bounds: Parameter bounds of shape (2, n_dims)
        use_input_warping: If True, use Kumaraswamy input warping
        train_yvar: Optional per-observation noise variance. See
            :func:`create_single_task_model`.
        noise_prior: Optional GPyTorch prior on the trainable noise
            hyperparameter. See :func:`create_single_task_model`.
        log_transform: Forwarded to :func:`create_single_task_model`.
        categorical_dim_indices: Forwarded to :func:`create_single_task_model`.
        auto_shift_for_log: Forwarded to :func:`create_single_task_model`.

    Returns:
        Fitted SingleTaskGP
    """
    model = create_single_task_model(
        train_x,
        train_y,
        bounds,
        use_input_warping=use_input_warping,
        train_yvar=train_yvar,
        noise_prior=noise_prior,
        log_transform=log_transform,
        categorical_dim_indices=categorical_dim_indices,
        auto_shift_for_log=auto_shift_for_log,
    )
    return fit_single_task_model(model)


def create_and_fit_model(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    use_input_warping: bool = False,
    train_yvar: Tensor | None = None,
    noise_prior: Prior | None = None,
    log_transform: bool | list[bool] = False,
    *,
    categorical_dim_indices: list[int] | None = None,
) -> ModelListGP:
    """Create and fit a ModelListGP.

    Convenience function combining create_model and fit_model.

    Args:
        train_x: Training inputs of shape (n_samples, n_dims)
        train_y: Training outputs of shape (n_samples, n_objectives)
        bounds: Parameter bounds of shape (2, n_dims)
        use_input_warping: If True, use Kumaraswamy input warping
        train_yvar: Optional per-observation per-objective noise variance.
            See :func:`create_model`.
        noise_prior: Optional GPyTorch prior shared across sub-models. See
            :func:`create_model`.
        log_transform: Per-objective ``Log → Standardize`` opt-in.
            Forwarded to :func:`create_model`.
        categorical_dim_indices: Forwarded to :func:`create_model`.

    Returns:
        Fitted ModelListGP
    """
    model = create_model(
        train_x,
        train_y,
        bounds,
        use_input_warping=use_input_warping,
        train_yvar=train_yvar,
        noise_prior=noise_prior,
        log_transform=log_transform,
        categorical_dim_indices=categorical_dim_indices,
    )
    return fit_model(model)


def get_model_predictions(
    model: ModelListGP | SingleTaskGP,
    test_x: Tensor,
) -> tuple[Tensor, Tensor]:
    """Get model predictions with uncertainty.

    Args:
        model: Fitted ModelListGP or SingleTaskGP
        test_x: Test inputs of shape (n_test, n_dims)

    Returns:
        Tuple of (mean, variance) each of shape (n_test, n_objectives) or (n_test, 1)
    """
    test_x = to_device(test_x)
    model.eval()

    with torch.no_grad():
        posterior = model.posterior(test_x)
        mean = posterior.mean
        variance = posterior.variance

    # Reshape from (n_test, n_objectives, 1) to (n_test, n_objectives)
    mean = mean.squeeze(-1)
    variance = variance.squeeze(-1)

    return mean, variance


def extract_lengthscales(model: ModelListGP | SingleTaskGP) -> dict[int, Tensor]:
    """Extract lengthscales from model for feature importance analysis.

    For ARD kernels, lengthscales indicate relative importance of parameters.
    Smaller lengthscale = more important (more sensitive to changes).

    Args:
        model: Fitted GP model

    Returns:
        Dictionary mapping objective index to lengthscale tensor
    """
    lengthscales = {}

    if isinstance(model, SingleTaskGP):
        # Single model - access through covar_module attribute
        covar = model.covar_module  # type: ignore[attr-defined]
        ls = covar.base_kernel.lengthscale.detach()  # ty: ignore[call-non-callable, unresolved-attribute]
        lengthscales[0] = ls.squeeze()
    else:
        # ModelListGP
        for i, m in enumerate(model.models):
            covar = m.covar_module  # type: ignore[attr-defined]
            ls = covar.base_kernel.lengthscale.detach()
            lengthscales[i] = ls.squeeze()

    return lengthscales


def get_warping_parameters(model: SingleTaskGP) -> dict[str, Tensor] | None:
    """Extract warping parameters if input warping is used.

    Args:
        model: SingleTaskGP that may have input warping

    Returns:
        Dictionary with 'concentration0' and 'concentration1' tensors, or None
    """
    input_transform = model.input_transform

    # Check if it's a ChainedInputTransform with warping
    if isinstance(input_transform, ChainedInputTransform):
        for transform in input_transform.values():
            if isinstance(transform, Warp):
                c0 = transform.concentration0  # type: ignore[attr-defined]
                c1 = transform.concentration1  # type: ignore[attr-defined]
                return {
                    "concentration0": c0.detach(),  # ty: ignore[call-non-callable]
                    "concentration1": c1.detach(),  # ty: ignore[call-non-callable]
                }

    return None


def verify_standardization(
    model: ModelListGP | SingleTaskGP,
    *,
    mean_atol: float = STANDARDIZATION_MEAN_TOLERANCE,
    var_atol: float = STANDARDIZATION_VAR_TOLERANCE,
) -> list[dict[str, float]]:
    """Verify that BoTorch internalized ``Standardize`` on each sub-model.

    Checks that every GP's stored ``train_targets`` (what MLL is computed
    against) has ~zero mean and ~unit variance -- the invariant that
    downstream acquisition geometry, noise-prior calibration and TuRBO
    tolerances depend on. Logs a warning if the invariant does not hold
    within the given tolerances.

    Args:
        model: Fitted ``SingleTaskGP`` or ``ModelListGP``.
        mean_atol: Absolute tolerance on the standardized-target mean.
        var_atol: Absolute tolerance on ``|var(train_targets) - 1|``. Only
            asserted when the sub-model has at least two observations
            (variance is ill-defined for a single point).

    Returns:
        One diagnostic dict per sub-model with keys ``mean``, ``var``,
        ``n`` and ``standardized`` (bool).

    Raises:
        ValueError: If ``model`` has no ``train_targets`` (not an ExactGP).
    """
    if isinstance(model, ModelListGP):
        sub_models: list[SingleTaskGP] = list(model.models)  # ty: ignore[invalid-argument-type]
    else:
        sub_models = [model]

    reports: list[dict[str, float]] = []
    for idx, gp in enumerate(sub_models):
        targets = getattr(gp, "train_targets", None)
        if targets is None:
            msg = f"Model {idx} has no train_targets -- cannot verify standardization."
            raise ValueError(msg)

        flat = targets.detach().reshape(-1).to(dtype=torch.float64)
        n = int(flat.numel())
        mean = float(flat.mean().item())
        # Sample (unbiased) variance to match BoTorch's `nanstd` normalization
        # -- it divides by n-1, so the post-standardization unbiased variance
        # is exactly 1.0 up to float jitter.
        var = float(flat.var(unbiased=True).item()) if n > 1 else float("nan")

        mean_ok = abs(mean) <= mean_atol
        var_ok = n <= 1 or abs(var - 1.0) <= var_atol
        standardized = mean_ok and var_ok
        if not standardized:
            logger.warning(
                "Model %d targets not unit-scale (mean=%.4g, var=%.4g, n=%d); "
                "outcome_transform may not be attached.",
                idx,
                mean,
                var,
                n,
            )

        reports.append(
            {
                "mean": mean,
                "var": var,
                "n": float(n),
                "standardized": float(standardized),
            }
        )

    return reports


def inspect_standardize_stdvs(
    model: ModelListGP | SingleTaskGP,
) -> list[float]:
    """Return the per-sub-model standardization stddev recorded by ``Standardize``.

    Read-only inspection helper used by :func:`post_fit_verification` to
    surface near-constant training data as a batch warning. Mutating
    ``Standardize.stdvs`` in place would break the forward/inverse
    transform round-trip — the GP was fit with one divisor and the
    posterior is un-transformed with a different one — so we no longer
    floor the stored value. Numerical instabilities from a tiny stddev
    surface to the caller through the warning path instead.

    Args:
        model: Fitted ``SingleTaskGP`` or ``ModelListGP``.

    Returns:
        Per-sub-model minimum stddev across the ``Standardize.stdvs``
        tensor; ``nan`` for sub-models that don't expose a Standardize
        transform.
    """
    if isinstance(model, ModelListGP):
        sub_models: list[SingleTaskGP] = list(model.models)  # ty: ignore[invalid-argument-type]
    else:
        sub_models = [model]

    raw_stds: list[float] = []
    for gp in sub_models:
        transform = getattr(gp, "outcome_transform", None)
        # ChainedOutcomeTransform exposes its children by attribute name; the
        # standardize step is canonically registered as "standardize".
        standardize = transform
        if transform is not None and hasattr(transform, "standardize"):
            standardize = transform.standardize
        stdvs = getattr(standardize, "stdvs", None)
        if stdvs is None:
            raw_stds.append(float("nan"))
            continue
        raw_stds.append(float(stdvs.detach().reshape(-1).min().item()))
    return raw_stds


def floor_standardize_stdvs(
    model: ModelListGP | SingleTaskGP,
) -> list[float]:
    """Deprecated alias of :func:`inspect_standardize_stdvs`.

    The previous implementation clamped ``Standardize.stdvs`` in place
    after fit, which broke the forward/inverse round-trip on
    near-constant data (the GP was fit with the tiny pre-clamp stddev,
    then the posterior was un-transformed with the floored value,
    distorting predictions on the user-facing scale). The function now
    only reports the stored stddev; downstream code should rely on
    :func:`post_fit_verification` warnings to flag near-constant data.
    """
    return inspect_standardize_stdvs(model)


def _label_for_subindex(idx: int, objective_names: list[str] | None) -> str:
    """Return the human-readable label used in standardization warnings."""
    if objective_names and idx < len(objective_names):
        return f"objective '{objective_names[idx]}'"
    return f"sub-model {idx}"


def post_fit_verification(
    model: ModelListGP | SingleTaskGP,
    *,
    objective_names: list[str] | None = None,
    std_floor: float = STANDARDIZATION_STD_FLOOR,
) -> list[str]:
    """Run the post-fit standardization checks and return human-readable warnings.

    Combines :func:`verify_standardization` (the invariant check) and
    :func:`inspect_standardize_stdvs` (the numerical inspection) so the
    suggestion pipeline can surface both signals to the caller as
    :attr:`bo_engine.backend.SuggestionBatch.warnings` entries.

    **No mutation.** A previous version of this helper clamped
    ``Standardize.stdvs`` in place after fit. That broke the forward /
    inverse transform round-trip on near-constant data — the GP was fit
    on standardized targets computed with the raw tiny stddev, but the
    posterior was un-transformed with the floored stddev. We now only
    warn; numerical instability from a near-constant column is the
    caller's responsibility to act on (drop the row, add jitter, or
    rerun with a different objective).

    Args:
        model: Fitted GP (or model list).
        objective_names: Optional names used to attribute warnings to a
            specific objective. When ``None``, warnings reference the
            sub-model index instead.
        std_floor: Threshold below which a sub-model's stddev triggers a
            warning. Defaults to :data:`STANDARDIZATION_STD_FLOOR`.

    Returns:
        List of warning strings (empty when every sub-model passes).
    """
    warnings_list: list[str] = []
    raw_stds = inspect_standardize_stdvs(model)
    for idx, raw_std in enumerate(raw_stds):
        if math.isnan(raw_std) or raw_std >= std_floor:
            continue
        label = _label_for_subindex(idx, objective_names)
        warnings_list.append(
            f"GP for {label} fit on near-constant data: raw stddev "
            f"{raw_std:.3e} is below the recommended floor {std_floor:.0e}. "
            "Posterior predictions may be numerically unstable; verify the "
            "objective is not a replicate-only column or add jitter to the "
            "training data."
        )

    reports = verify_standardization(model)
    for idx, report in enumerate(reports):
        if report["standardized"] >= 1.0:
            continue
        label = _label_for_subindex(idx, objective_names)
        warnings_list.append(
            f"Standardization invariant failed for {label}: "
            f"mean={report['mean']:.3g}, var={report['var']:.3g}. Posterior "
            "predictions may be on an unexpected scale; check that the "
            "outcome transform attached cleanly."
        )
    return warnings_list
