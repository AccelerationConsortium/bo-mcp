"""Improved Outcome Constraint Modeling for Bayesian Optimization.

This module provides continuous constraint modeling as an alternative to
the binary feasibility approach. Modeling the constraint outcome directly
preserves more information than converting to binary labels.

v2.6: Initial implementation with continuous constraint modeling (Section 2.3)

The Problem with Binary Constraints:
    The current approach converts continuous objective values to binary
    labels (feasible/infeasible) and fits a GP to predict P(feasible).
    This loses valuable information about HOW far a point is from the
    constraint boundary.

The Solution:
    Model the objective outcome directly and compute the probability
    of constraint satisfaction from the predictive distribution.
    This provides:
    1. Better gradient information near the boundary
    2. More accurate probability estimates
    3. Support for expected constraint violation metrics

References:
    - Gardner et al. "Bayesian Optimization with Inequality Constraints"
      ICML 2014
    - Gelbart et al. "Bayesian Optimization with Unknown Constraints"
      UAI 2014
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.metrics import roc_auc_score
from torch import Tensor

from bo_engine.constants import (
    CONSTRAINT_CALIBRATION_N_BINS,
    CONSTRAINT_CALIBRATION_WARN_THRESHOLD,
)
from bo_engine.device import ensure_device, get_device, get_dtype, to_device


class ConstraintModelingMethod(Enum):
    """Method for modeling outcome constraints."""

    BINARY = "binary"  # Current: Convert to binary, fit GP on labels
    CONTINUOUS = "continuous"  # New: Model objective directly
    PROBABILISTIC_CLASSIFICATION = "probabilistic_classification"  # GP classification


@dataclass
class OutcomeConstraintSpec:
    """Specification for an outcome constraint.

    Attributes:
        name: Name of the objective being constrained
        bound: Constraint threshold value
        constraint_type: String indicating constraint type, either "<=" or ">="
    """

    name: str
    bound: float
    constraint_type: str = "<="  # "<=" or ">="

    @property
    def greater_than(self) -> bool:
        """Return True if constraint is objective >= bound."""
        return self.constraint_type == ">="

    @property
    def threshold(self) -> float:
        """Alias for bound for backward compatibility."""
        return self.bound

    @property
    def objective_name(self) -> str:
        """Alias for name for backward compatibility."""
        return self.name


@dataclass
class ConstraintModelConfig:
    """Configuration for constraint model fitting.

    Attributes:
        method: Modeling method (binary, continuous, or probabilistic)
        probability_threshold: P(feasible) threshold for acquisition weighting
        expected_violation_weight: Weight for expected constraint violation
        use_noise_model: Whether to model heteroscedastic noise
    """

    method: ConstraintModelingMethod = ConstraintModelingMethod.CONTINUOUS
    probability_threshold: float = 0.5
    expected_violation_weight: float = 1.0
    use_noise_model: bool = False


@dataclass
class ConstraintModelResult:
    """Result of constraint model fitting.

    Attributes:
        model: Fitted GP model (predicting objective or feasibility)
        constraint_spec: The constraint specification
        method: Method used for modeling
        config: Configuration used
        feasibility_rate: Fraction of training points that are feasible
    """

    model: SingleTaskGP
    constraint_spec: OutcomeConstraintSpec
    method: ConstraintModelingMethod
    config: ConstraintModelConfig
    feasibility_rate: float


def build_constraint_model_continuous(
    train_x: Tensor,
    objective_values: Tensor,
    bounds: Tensor,
    constraint_spec: OutcomeConstraintSpec,
    config: ConstraintModelConfig | None = None,
) -> ConstraintModelResult:
    """Build constraint model using continuous objective modeling.

    Instead of converting to binary labels, this fits a GP directly to
    the objective values. The probability of constraint satisfaction is
    then computed from the predictive distribution.

    Args:
        train_x: Training inputs of shape (n_samples, n_dims)
        objective_values: Objective values of shape (n_samples,) or (n_samples, 1)
        bounds: Parameter bounds of shape (2, n_dims)
        constraint_spec: Constraint specification
        config: Model configuration

    Returns:
        ConstraintModelResult with fitted model

    Example:
        >>> constraint = OutcomeConstraintSpec("cost", threshold=100.0, greater_than=False)
        >>> result = build_constraint_model_continuous(train_x, costs, bounds, constraint)
        >>> # Use result.model to compute P(cost <= 100)
    """
    if config is None:
        config = ConstraintModelConfig(method=ConstraintModelingMethod.CONTINUOUS)

    train_x, objective_values, bounds = ensure_device(train_x, objective_values, bounds)

    # Ensure proper shape
    if objective_values.dim() == 1:
        objective_values = objective_values.unsqueeze(-1)

    n_dims = train_x.shape[-1]

    # Fit GP directly to objective values
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=objective_values,
        input_transform=Normalize(d=n_dims, bounds=bounds),
        outcome_transform=Standardize(m=1),
    )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    # Compute feasibility rate
    if constraint_spec.greater_than:
        feasible = (objective_values.squeeze() >= constraint_spec.threshold).float()
    else:
        feasible = (objective_values.squeeze() <= constraint_spec.threshold).float()
    feasibility_rate = feasible.mean().item()

    return ConstraintModelResult(
        model=model,
        constraint_spec=constraint_spec,
        method=ConstraintModelingMethod.CONTINUOUS,
        config=config,
        feasibility_rate=feasibility_rate,
    )


def build_constraint_model_binary(
    train_x: Tensor,
    objective_values: Tensor,
    bounds: Tensor,
    constraint_spec: OutcomeConstraintSpec,
    config: ConstraintModelConfig | None = None,
) -> ConstraintModelResult:
    """Build constraint model using binary feasibility labels (legacy method).

    This is the original approach that converts continuous values to binary
    and fits a GP to predict P(feasible). Maintained for backward compatibility.

    Args:
        train_x: Training inputs
        objective_values: Objective values
        bounds: Parameter bounds
        constraint_spec: Constraint specification
        config: Model configuration

    Returns:
        ConstraintModelResult with fitted model
    """
    if config is None:
        config = ConstraintModelConfig(method=ConstraintModelingMethod.BINARY)

    train_x, objective_values, bounds = ensure_device(train_x, objective_values, bounds)

    if objective_values.dim() == 1:
        objective_values = objective_values.unsqueeze(-1)

    n_dims = train_x.shape[-1]

    # Convert to binary feasibility
    if constraint_spec.greater_than:
        feasible = (objective_values >= constraint_spec.threshold).double()
    else:
        feasible = (objective_values <= constraint_spec.threshold).double()

    # Fit GP on binary labels
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=feasible,
        input_transform=Normalize(d=n_dims, bounds=bounds),
        outcome_transform=Standardize(m=1),
    )

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    feasibility_rate = feasible.mean().item()

    return ConstraintModelResult(
        model=model,
        constraint_spec=constraint_spec,
        method=ConstraintModelingMethod.BINARY,
        config=config,
        feasibility_rate=feasibility_rate,
    )


def compute_constraint_probability(
    mean_or_model: Tensor | ConstraintModelResult,
    std_or_x: Tensor | None = None,
    bound: float | None = None,
    constraint_type: str | None = None,
) -> Tensor:
    """Compute probability of constraint satisfaction.

    This function supports two signatures, dispatched on the first argument.

    Args:
        mean_or_model: Either the predicted mean tensor (direct signature)
            *or* a fitted :class:`ConstraintModelResult` (model-based
            signature).
        std_or_x: Either the predicted-std tensor (direct) *or* candidate
            points of shape ``(..., n_dims)`` (model-based).
        bound: Constraint threshold (direct signature only; read from the
            model spec otherwise).
        constraint_type: ``"<="`` for less-than, ``">="`` for greater-than
            (direct signature only; read from the model spec otherwise).

    Returns:
        Probability of feasibility
    """
    # Check which signature is being used
    if isinstance(mean_or_model, Tensor) and bound is not None:
        # Simple signature: compute_constraint_probability(mean, std, bound, constraint_type)
        mean = mean_or_model
        std = std_or_x
        if std is None:
            msg = "std is required when using the simple signature"
            raise ValueError(msg)
        if constraint_type is None:
            constraint_type = "<="

        std = std.clamp(min=1e-6)
        z = (bound - mean) / std

        # `>=`: P(objective >= bound) = 1 - Phi((bound - mean) / std)
        # `<=`: P(objective <= bound) = Phi((bound - mean) / std)
        return 1.0 - _standard_normal_cdf(z) if constraint_type == ">=" else _standard_normal_cdf(z)

    # Model-based signature: compute_constraint_probability(model_result, x)
    model_result = mean_or_model
    x = std_or_x

    if not isinstance(model_result, ConstraintModelResult):
        msg = "First argument must be Tensor (for simple signature) or ConstraintModelResult"
        raise TypeError(msg)
    if x is None:
        msg = "x is required when using the model-based signature"
        raise ValueError(msg)

    x = to_device(x)
    model = model_result.model
    model.eval()

    with torch.no_grad():
        posterior = model.posterior(x)
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.sqrt().squeeze(-1).clamp(min=1e-6)

    if model_result.method == ConstraintModelingMethod.CONTINUOUS:
        # Compute P(objective meets constraint) from normal distribution
        threshold = model_result.constraint_spec.threshold

        if model_result.constraint_spec.greater_than:
            # P(objective >= threshold) = 1 - Phi((threshold - mean) / std)
            z = (threshold - mean) / std
            prob = 1.0 - _standard_normal_cdf(z)
        else:
            # P(objective <= threshold) = Phi((threshold - mean) / std)
            z = (threshold - mean) / std
            prob = _standard_normal_cdf(z)
    else:
        # Binary: model predicts P(feasible) directly
        prob = mean.clamp(0.0, 1.0)

    return prob


def compute_expected_constraint_violation(
    model_result: ConstraintModelResult,
    x: Tensor,
) -> Tensor:
    """Compute expected constraint violation at candidate points.

    For continuous modeling, this computes E[max(0, violation)] where
    violation is how much the constraint is violated.

    This provides more information than just P(feasible) because it
    quantifies HOW MUCH the constraint is expected to be violated.

    Args:
        model_result: Fitted constraint model result
        x: Candidate points of shape (..., n_dims)

    Returns:
        Expected violation of shape (...,). Zero if constraint satisfied.
    """
    x = to_device(x)
    model = model_result.model
    model.eval()

    if model_result.method != ConstraintModelingMethod.CONTINUOUS:
        # Binary model doesn't support expected violation
        # Return complement of probability as proxy
        prob = compute_constraint_probability(model_result, x)
        return 1.0 - prob

    with torch.no_grad():
        posterior = model.posterior(x)
        mean = posterior.mean.squeeze(-1)
        std = posterior.variance.sqrt().squeeze(-1).clamp(min=1e-6)

    threshold = model_result.constraint_spec.threshold

    if model_result.constraint_spec.greater_than:
        # Constraint: objective >= threshold
        # Violation: max(0, threshold - objective)
        # E[max(0, threshold - objective)] when objective ~ N(mean, std²)
        z = (threshold - mean) / std
        expected_violation = std * (_standard_normal_pdf(z) + z * _standard_normal_cdf(z))
    else:
        # Constraint: objective <= threshold
        # Violation: max(0, objective - threshold)
        z = (mean - threshold) / std
        expected_violation = std * (_standard_normal_pdf(z) + z * (1 - _standard_normal_cdf(-z)))

    return expected_violation.clamp(min=0.0)


def create_constraint_callable_continuous(
    model_result: ConstraintModelResult,
) -> Callable[[Tensor], Tensor]:
    """Create a constraint callable for use with BoTorch acquisition functions.

    This callable returns positive values when the constraint is satisfied
    (probability above threshold) and negative values otherwise.

    Args:
        model_result: Fitted constraint model result

    Returns:
        Callable compatible with BoTorch's constraints parameter
    """
    threshold = model_result.config.probability_threshold

    def constraint_callable(samples: Tensor) -> Tensor:
        """Constraint function: positive means feasible.

        Args:
            samples: Posterior samples of shape (num_samples, batch_size, 1)

        Returns:
            Constraint values (positive = satisfied)
        """
        # Samples are from the constraint model
        if model_result.method == ConstraintModelingMethod.CONTINUOUS:
            # For continuous, interpret samples as objective predictions
            obj_threshold = model_result.constraint_spec.threshold

            if model_result.constraint_spec.greater_than:
                # Want objective >= threshold
                # Return positive when satisfied
                return samples.squeeze(-1) - obj_threshold
            # Want objective <= threshold
            return obj_threshold - samples.squeeze(-1)
        # For binary, samples are P(feasible) predictions
        return samples.squeeze(-1) - threshold

    return constraint_callable


def build_outcome_constraint_models(
    train_x: Tensor,
    observations: list[dict[str, float]],
    bounds: Tensor,
    constraint_specs: list[OutcomeConstraintSpec],
    config: ConstraintModelConfig | None = None,
) -> list[ConstraintModelResult]:
    """Build constraint models for multiple outcome constraints.

    This is the main entry point for constraint modeling, replacing the
    binary-only approach in suggestions.py.

    Args:
        train_x: Training inputs
        observations: List of observation dicts with objective values
        bounds: Parameter bounds
        constraint_specs: List of constraint specifications
        config: Model configuration (applied to all constraints)

    Returns:
        List of ConstraintModelResult objects
    """
    if config is None:
        config = ConstraintModelConfig()

    results = []
    for spec in constraint_specs:
        # Extract objective values for this constraint
        obj_values = []
        for obs in observations:
            if spec.objective_name not in obs:
                msg = f"Objective '{spec.objective_name}' not found in observations"
                raise ValueError(msg)
            obj_values.append(obs[spec.objective_name])

        obj_tensor = torch.tensor(obj_values, dtype=get_dtype(), device=get_device())

        # Build model based on method
        if config.method == ConstraintModelingMethod.CONTINUOUS:
            result = build_constraint_model_continuous(train_x, obj_tensor, bounds, spec, config)
        else:
            result = build_constraint_model_binary(train_x, obj_tensor, bounds, spec, config)

        results.append(result)

    return results


def assess_constraint_model_quality(
    model_result: ConstraintModelResult,
    train_x: Tensor,
    objective_values: Tensor,
) -> dict[str, Any]:
    """Assess the quality of a fitted constraint model.

    Computes metrics to help understand if the constraint model is
    well-calibrated and provides accurate probability estimates.

    Calibration metrics:

    * ``calibration_error`` — mean absolute deviation between predicted
      feasibility probability and the binary realized label. Cheap and
      intuitive but not a strictly proper scoring rule.
    * ``brier_score`` — mean squared error between predicted probability
      and the binary label (Brier, 1950; "Verification of Forecasts
      Expressed in Terms of Probability"). Strictly proper, so an
      overconfident classifier is penalized even when accuracy is high.
    * ``expected_calibration_error`` — bucket the predictions into
      ``CONSTRAINT_CALIBRATION_N_BINS`` equal-width probability bins, then
      report the weighted average ``|mean(prob) - frac(feasible)|`` per
      bin (Naeini et al., 2015; Guo et al., ICML 2017). Targets the
      reliability-curve gap that drives infeasible-suggestion regret.
    * ``is_calibrated`` — convenience flag mirroring the diagnostics
      threshold ``CONSTRAINT_CALIBRATION_WARN_THRESHOLD``.

    The metrics are computed on the training data the constraint model
    saw; they cannot detect generalization failure but they do detect
    confidently-wrong predictions on the data the model was actually
    fit to.

    Args:
        model_result: Fitted constraint model
        train_x: Training inputs
        objective_values: True objective values

    Returns:
        Dictionary with quality metrics:
        - calibration_error: Mean abs(predicted - actual) feasibility
        - brier_score: Mean squared error of feasibility predictions
        - expected_calibration_error: Reliability-curve gap (ECE, 10 bins)
        - is_calibrated: Whether calibration_error <=
          CONSTRAINT_CALIBRATION_WARN_THRESHOLD
        - auc: Area under ROC curve for binary classification
        - boundary_uncertainty: Mean uncertainty near constraint boundary
        - feasibility_rate: Fraction of feasible training points
        - method: Modeling method used
    """
    train_x, objective_values = ensure_device(train_x, objective_values)

    if objective_values.dim() == 1:
        objective_values = objective_values.unsqueeze(-1)

    spec = model_result.constraint_spec

    # Compute predicted probabilities
    probs = compute_constraint_probability(model_result, train_x)

    # Compute actual feasibility
    if spec.greater_than:
        actual_feasible = (objective_values.squeeze() >= spec.threshold).float()
    else:
        actual_feasible = (objective_values.squeeze() <= spec.threshold).float()

    probs_flat = probs.detach().reshape(-1)
    actual_flat = actual_feasible.detach().reshape(-1)

    # Calibration error (mean absolute deviation — fast sanity check)
    calibration_error = (probs_flat - actual_flat).abs().mean().item()

    # Brier score — strictly proper scoring rule for binary outcomes
    brier_score = ((probs_flat - actual_flat) ** 2).mean().item()

    # Expected calibration error — reliability-curve gap
    expected_calibration_error = _expected_calibration_error(probs_flat, actual_flat)

    is_calibrated = calibration_error <= CONSTRAINT_CALIBRATION_WARN_THRESHOLD

    # Compute AUC if we have both classes
    auc = 0.5  # Default to random
    if actual_feasible.sum() > 0 and actual_feasible.sum() < len(actual_feasible):
        try:
            auc = roc_auc_score(actual_feasible.cpu().numpy(), probs.detach().cpu().numpy())
        except ImportError:
            # Compute simple AUC approximation
            feasible_probs = probs[actual_feasible == 1]
            infeasible_probs = probs[actual_feasible == 0]
            if len(feasible_probs) > 0 and len(infeasible_probs) > 0:
                auc = (
                    (feasible_probs.unsqueeze(1) > infeasible_probs.unsqueeze(0))
                    .float()
                    .mean()
                    .item()
                )

    # Boundary uncertainty: uncertainty at points near threshold
    model_result.model.eval()
    with torch.no_grad():
        posterior = model_result.model.posterior(train_x)
        uncertainties = posterior.variance.squeeze(-1).sqrt()

        # Find points near boundary
        distance_to_threshold = (objective_values.squeeze() - spec.threshold).abs()
        threshold_range = objective_values.std() * 0.5  # Within 0.5 std of threshold
        near_boundary = distance_to_threshold < threshold_range

        if near_boundary.sum() > 0:
            boundary_uncertainty = uncertainties[near_boundary].mean().item()
        else:
            boundary_uncertainty = uncertainties.mean().item()

    return {
        "calibration_error": calibration_error,
        "brier_score": brier_score,
        "expected_calibration_error": expected_calibration_error,
        "is_calibrated": is_calibrated,
        "auc": auc,
        "boundary_uncertainty": boundary_uncertainty,
        "feasibility_rate": model_result.feasibility_rate,
        "method": model_result.method.value,
    }


def compute_outcome_constraint_calibration(
    constraint_specs: list[OutcomeConstraintSpec],
    train_x: Tensor,
    objective_values: dict[str, Tensor],
    bounds: Tensor,
    method: ConstraintModelingMethod = ConstraintModelingMethod.BINARY,
) -> list[dict[str, Any]]:
    """Fit constraint models and assess calibration against realized labels.

    Convenience wrapper for the diagnostics path: takes the same training
    data the suggestion pipeline saw, fits a per-constraint GP using the
    requested ``method`` (defaults to ``BINARY`` to mirror what
    :func:`bo_engine.suggestions._build_outcome_constraint_models` uses at
    suggestion time), and returns the calibration / discrimination metrics
    from :func:`assess_constraint_model_quality` along with a copy of the
    constraint spec so callers can rebuild a per-constraint report.

    The probabilities are evaluated on the training points the GP was fit on
    — this catches confidently-wrong predictions on data the model has
    already seen but cannot detect generalization failure. For held-out
    calibration the caller should pass a cross-validated set.

    Args:
        constraint_specs: Outcome constraints to assess.
        train_x: Training inputs of shape ``(n_samples, n_dims)``.
        objective_values: Map from objective name to the raw observed
            values, shape ``(n_samples,)`` each. Each constraint reads from
            this map by ``OutcomeConstraintSpec.objective_name``.
        bounds: Parameter bounds of shape ``(2, n_dims)``.
        method: Modeling method to use (defaults to ``BINARY`` to match the
            runtime path).

    Returns:
        A list of per-constraint metric dictionaries. Each entry combines
        the keys from :func:`assess_constraint_model_quality` with:
        ``constraint_name`` (the objective name), ``threshold``, and
        ``greater_than``. If a constraint references an objective that is
        not present in ``objective_values`` the entry contains
        ``error``: ``"missing_objective"`` and the calibration metrics are
        omitted.
    """
    config = ConstraintModelConfig(method=method)

    reports: list[dict[str, Any]] = []
    for spec in constraint_specs:
        if spec.objective_name not in objective_values:
            reports.append(
                {
                    "constraint_name": spec.objective_name,
                    "threshold": spec.threshold,
                    "greater_than": spec.greater_than,
                    "error": "missing_objective",
                }
            )
            continue

        obj_tensor = objective_values[spec.objective_name].to(
            dtype=get_dtype(), device=get_device()
        )

        if method == ConstraintModelingMethod.CONTINUOUS:
            result = build_constraint_model_continuous(train_x, obj_tensor, bounds, spec, config)
        else:
            result = build_constraint_model_binary(train_x, obj_tensor, bounds, spec, config)

        metrics = assess_constraint_model_quality(result, train_x, obj_tensor)
        metrics.update(
            {
                "constraint_name": spec.objective_name,
                "threshold": spec.threshold,
                "greater_than": spec.greater_than,
            }
        )
        reports.append(metrics)
    return reports


def _expected_calibration_error(
    probs: Tensor,
    labels: Tensor,
    n_bins: int = CONSTRAINT_CALIBRATION_N_BINS,
) -> float:
    """Compute Expected Calibration Error (ECE) over equal-width bins.

    ECE bins predictions by their forecast probability, computes the gap
    between predicted-mean and empirical-feasibility within each bin, and
    averages weighted by bin occupancy. Empty bins contribute zero.

    References:
        - Naeini, Cooper, Hauskrecht "Obtaining Well Calibrated
          Probabilities Using Bayesian Binning" AAAI 2015.
        - Guo et al., "On Calibration of Modern Neural Networks" ICML 2017.

    Args:
        probs: 1-D tensor of predicted probabilities, values in [0, 1].
        labels: 1-D tensor of binary realized labels (0 or 1).
        n_bins: Number of equal-width probability bins.

    Returns:
        ECE in [0, 1]. Lower is better; 0.0 means perfect reliability.
    """
    n = int(probs.numel())
    if n == 0:
        return 0.0
    probs_clamped = probs.clamp(0.0, 1.0)
    boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=probs.device, dtype=probs.dtype)
    ece = 0.0
    for b in range(n_bins):
        lo = boundaries[b]
        hi = boundaries[b + 1]
        # Last bin is closed on the right so probs == 1.0 lands in a bin.
        if b == n_bins - 1:
            in_bin = (probs_clamped >= lo) & (probs_clamped <= hi)
        else:
            in_bin = (probs_clamped >= lo) & (probs_clamped < hi)
        count = int(in_bin.sum().item())
        if count == 0:
            continue
        bin_conf = float(probs_clamped[in_bin].mean().item())
        bin_acc = float(labels[in_bin].mean().item())
        ece += (count / n) * abs(bin_conf - bin_acc)
    return ece


def _standard_normal_cdf(z: Tensor) -> Tensor:
    """Compute standard normal CDF using PyTorch.

    Args:
        z: Input tensor

    Returns:
        CDF values
    """
    return 0.5 * (1 + torch.erf(z / torch.sqrt(torch.tensor(2.0, device=z.device, dtype=z.dtype))))


def _standard_normal_pdf(z: Tensor) -> Tensor:
    """Compute standard normal PDF using PyTorch.

    Args:
        z: Input tensor

    Returns:
        PDF values
    """
    return torch.exp(-0.5 * z**2) / torch.sqrt(
        torch.tensor(2 * 3.141592653589793, device=z.device, dtype=z.dtype)
    )
