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
      ``verify_standardization`` (TODO §1.42).
"""

import logging

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import ChainedInputTransform, Normalize, Warp
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.priors.torch_priors import LogNormalPrior
from torch import Tensor

from bo_engine.constants import (
    STANDARDIZATION_MEAN_TOLERANCE,
    STANDARDIZATION_VAR_TOLERANCE,
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


def create_input_transform(
    n_dims: int,
    bounds: Tensor,
    use_input_warping: bool = False,
) -> Normalize | ChainedInputTransform:
    """Create input transform with optional warping.

    Args:
        n_dims: Number of input dimensions
        bounds: Parameter bounds of shape (2, n_dims)
        use_input_warping: If True, add Kumaraswamy warping after normalization

    Returns:
        Input transform (Normalize or ChainedInputTransform with Warp)
    """
    bounds = to_device(bounds)
    normalize = Normalize(d=n_dims, bounds=bounds)

    if not use_input_warping:
        return normalize

    # Create Kumaraswamy warping with learnable concentration parameters
    # Prior on concentration: LogNormal encourages mild warping
    device = get_device()
    warp = Warp(
        d=n_dims,
        indices=list(range(n_dims)),
        concentration1_prior=LogNormalPrior(
            loc=torch.tensor(0.0, device=device), scale=torch.tensor(0.75, device=device)
        ),
        concentration0_prior=LogNormalPrior(
            loc=torch.tensor(0.0, device=device), scale=torch.tensor(0.75, device=device)
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

    Returns:
        SingleTaskGP model (unfitted)
    """
    train_x, train_y, bounds = ensure_device(train_x, train_y, bounds)

    # Ensure train_y has correct shape (n_samples, 1)
    if train_y.dim() == 1:
        train_y = train_y.unsqueeze(-1)

    input_transform = create_input_transform(
        n_dims=train_x.shape[-1],
        bounds=bounds,
        use_input_warping=use_input_warping,
    )

    return SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        input_transform=input_transform,
        outcome_transform=Standardize(m=1),
    )


def create_model(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    use_input_warping: bool = False,
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

    Returns:
        ModelListGP with one GP per objective
    """
    train_x, train_y, bounds = ensure_device(train_x, train_y, bounds)

    n_objectives = train_y.shape[-1]
    n_dims = train_x.shape[-1]

    models = []
    for i in range(n_objectives):
        # Create input transform (each model gets its own)
        input_transform = create_input_transform(
            n_dims=n_dims,
            bounds=bounds,
            use_input_warping=use_input_warping,
        )

        # Create single-task GP for each objective
        model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y[:, i : i + 1],
            input_transform=input_transform,
            outcome_transform=Standardize(m=1),
        )
        models.append(model)

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
    return model


def create_and_fit_single_task_model(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    use_input_warping: bool = False,
) -> SingleTaskGP:
    """Create and fit a SingleTaskGP.

    Convenience function for single-objective optimization.

    Args:
        train_x: Training inputs of shape (n_samples, n_dims)
        train_y: Training outputs of shape (n_samples, 1) or (n_samples,)
        bounds: Parameter bounds of shape (2, n_dims)
        use_input_warping: If True, use Kumaraswamy input warping

    Returns:
        Fitted SingleTaskGP
    """
    model = create_single_task_model(train_x, train_y, bounds, use_input_warping)
    return fit_single_task_model(model)


def create_and_fit_model(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    use_input_warping: bool = False,
) -> ModelListGP:
    """Create and fit a ModelListGP.

    Convenience function combining create_model and fit_model.

    Args:
        train_x: Training inputs of shape (n_samples, n_dims)
        train_y: Training outputs of shape (n_samples, n_objectives)
        bounds: Parameter bounds of shape (2, n_dims)
        use_input_warping: If True, use Kumaraswamy input warping

    Returns:
        Fitted ModelListGP
    """
    model = create_model(train_x, train_y, bounds, use_input_warping)
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
            ls = covar.base_kernel.lengthscale.detach()  # ty: ignore[call-non-callable, unresolved-attribute]
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
