"""Gaussian Process model creation and fitting.

Supports:
- SingleTaskGP for single-objective optimization
- ModelListGP for multi-objective optimization
- Optional input warping with Kumaraswamy CDF for non-stationary objectives
- Automatic GPU acceleration when available

v1.0.1: Added create_single_task_model for single-objective
v1.1: Added input warping support
v2.3: Added GPU auto-detection and acceleration
"""

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

from bo_engine.device import ensure_device, get_device, to_device


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

    Args:
        train_x: Training inputs of shape (n_samples, n_dims)
        train_y: Training outputs of shape (n_samples, 1) or (n_samples,)
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

    Uses independent GP models for each objective.

    Args:
        train_x: Training inputs of shape (n_samples, n_dims)
        train_y: Training outputs of shape (n_samples, n_objectives)
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
    """
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def fit_model(model: ModelListGP) -> ModelListGP:
    """Fit the GP model by maximizing marginal likelihood.

    Args:
        model: ModelListGP to fit

    Returns:
        Fitted model
    """
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
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
        ls = covar.base_kernel.lengthscale.detach()
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
                    "concentration0": c0.detach(),
                    "concentration1": c1.detach(),
                }

    return None
