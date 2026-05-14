"""Feature importance analysis for Bayesian Optimization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from botorch.models import ModelListGP, SingleTaskGP

from bo_engine.constants import NUMERICAL_EPSILON

# Optional dependency - imported at module level for clarity
try:
    import shap  # ty: ignore[unresolved-import]

    SHAP_AVAILABLE = True
except ImportError:
    shap = None  # type: ignore[assignment]
    SHAP_AVAILABLE = False

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

GPModel = SingleTaskGP | ModelListGP


def extract_lengthscales(model: GPModel) -> dict[str, torch.Tensor]:
    """Extract length scales from each GP in the model.

    Works with both SingleTaskGP and ModelListGP.
    Returns dict mapping objective name to lengthscale tensor of shape (n_dims,).
    """
    lengthscales = {}

    if isinstance(model, ModelListGP):
        for i, gp in enumerate(model.models):
            covar = gp.covar_module
            # Handle ScaleKernel wrapping base kernel, or direct kernel
            kernel = getattr(covar, "base_kernel", covar)
            lengthscales[f"objective_{i}"] = kernel.lengthscale.detach().squeeze()  # type: ignore[union-attr]
    else:
        # SingleTaskGP
        covar = model.covar_module
        kernel = getattr(covar, "base_kernel", covar)
        lengthscales["objective_0"] = kernel.lengthscale.detach().squeeze()  # type: ignore[union-attr]  # ty: ignore[call-non-callable]

    return lengthscales


def compute_lengthscale_importance(
    lengthscales: dict[str, torch.Tensor],
    param_names: list[str],
) -> dict:
    """Convert length scales to importance scores.

    Importance = 1/lengthscale, normalized to sum to 1.
    Smaller lengthscale = higher sensitivity = more important.
    """
    result: dict = {"by_objective": {}, "aggregate": {}}

    all_importance = []
    for obj_name, ls in lengthscales.items():
        # Clamp lengthscales to NUMERICAL_EPSILON before taking the
        # reciprocal so a near-zero entry (a sign of a poorly conditioned
        # GP fit or a fully active dimension at the boundary) produces a
        # large finite importance instead of inf and downstream NaN. Warn
        # once per objective when the clamp fires so the underlying fit
        # is still visible in logs.
        if torch.any(ls < NUMERICAL_EPSILON):
            logger.warning(
                "Lengthscale below NUMERICAL_EPSILON detected for %s; "
                "clamping before reciprocal to avoid inf importance",
                obj_name,
            )
        safe_ls = torch.clamp(ls, min=NUMERICAL_EPSILON)
        importance = 1.0 / safe_ls
        importance = importance / importance.sum()

        result["by_objective"][obj_name] = {
            name: round(float(imp), 4) for name, imp in zip(param_names, importance, strict=True)
        }
        all_importance.append(importance)

    # Aggregate across objectives (mean)
    if all_importance:
        agg = torch.stack(all_importance).mean(dim=0)
        agg = agg / agg.sum()
        result["aggregate"] = {
            name: round(float(imp), 4) for name, imp in zip(param_names, agg, strict=True)
        }

    return result


def compute_shap_importance(
    model: GPModel,
    train_x: torch.Tensor,
    param_names: list[str],
    n_samples: int = 50,
) -> dict | None:
    """Compute SHAP values for feature importance.

    Uses KernelExplainer. Returns mean |SHAP| per feature, normalized.
    Returns None if shap is not installed.
    """
    if not SHAP_AVAILABLE:
        return None

    def predict_fn(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32)
            posterior = model.posterior(x_tensor)
            # Average across objectives for single importance score
            return posterior.mean.mean(dim=-1).numpy()

    # Use subset as background data
    background = train_x[: min(20, len(train_x))].numpy()
    explainer = shap.KernelExplainer(predict_fn, background)  # ty: ignore[unresolved-attribute]

    # Compute SHAP values
    shap_values = explainer.shap_values(train_x.numpy(), nsamples=n_samples, silent=True)

    # Mean absolute SHAP value per feature, normalized
    importance = np.abs(shap_values).mean(axis=0)
    importance = importance / importance.sum()

    return {name: round(float(imp), 4) for name, imp in zip(param_names, importance, strict=True)}


def compute_feature_importance(
    model: GPModel,
    train_x: torch.Tensor,
    param_names: list[str],
    include_shap: bool = True,
) -> dict:
    """Compute feature importance using length scales and optionally SHAP.

    Args:
        model: Fitted SingleTaskGP or ModelListGP
        train_x: Training data (n_samples, n_dims)
        param_names: Names of input parameters
        include_shap: Whether to compute SHAP (slower but more accurate)

    Returns:
        Dict with 'lengthscale' and optionally 'shap' importance scores.
    """
    # Extract and compute lengthscale-based importance
    lengthscales = extract_lengthscales(model)
    ls_importance = compute_lengthscale_importance(lengthscales, param_names)

    result = {"lengthscale": ls_importance}

    # Optionally compute SHAP importance
    if include_shap and len(train_x) >= 5:
        shap_importance = compute_shap_importance(model, train_x, param_names)
        if shap_importance:
            result["shap"] = shap_importance

    return result
