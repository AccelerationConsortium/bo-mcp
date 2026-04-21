"""Thompson Sampling acquisition for Bayesian Optimization.

Thompson Sampling provides an alternative to gradient-based acquisition
function optimization. It naturally provides exploration and can be more
robust for noisy objectives. It also provides uncertainty-aware batch
generation naturally.

Section 3.5 - Missing Trust-Building Features

References:
    - Thompson, W.R. "On the Likelihood that One Unknown Probability Exceeds
      Another in View of the Evidence of Two Samples" (1933)
    - Russo et al. "A Tutorial on Thompson Sampling" (2018)
    - Hernandez-Lobato et al. "Predictive Entropy Search for Efficient Global
      Optimization of Black-box Functions" NeurIPS 2014
    - BoTorch Thompson Sampling: https://botorch.org/tutorials/thompson_sampling/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from botorch.generation.sampling import MaxPosteriorSampling
from botorch.models import ModelListGP, SingleTaskGP
from torch import Tensor

from bo_engine.constants import (
    THOMPSON_BATCH_DIVERSITY_MIN_DISTANCE,
    THOMPSON_NUM_CANDIDATES,
    THOMPSON_NUM_POSTERIOR_SAMPLES,
)
from bo_engine.device import get_device, get_dtype

if TYPE_CHECKING:
    pass


@dataclass
class ThompsonSample:
    """A single sample from Thompson Sampling.

    Attributes:
        parameters: Parameter configuration (1 x d tensor).
        sampled_value: Value from posterior sample at this point.
        posterior_mean: Posterior mean at this point.
        posterior_std: Posterior std at this point.
    """

    parameters: Tensor
    sampled_value: float
    posterior_mean: float
    posterior_std: float


@dataclass
class ThompsonBatch:
    """Batch of suggestions from Thompson Sampling.

    Attributes:
        samples: List of ThompsonSample in the batch.
        parameters_tensor: All parameters stacked (batch_size x d).
        diversity_score: Pairwise diversity score of the batch.
        method_info: Information about the sampling method used.
    """

    samples: list[ThompsonSample]
    parameters_tensor: Tensor
    diversity_score: float
    method_info: str = "Thompson Sampling"


@dataclass
class ThompsonConfig:
    """Configuration for Thompson Sampling.

    Attributes:
        num_candidates: Number of candidates to draw for optimization.
        num_posterior_samples: Number of posterior samples per candidate.
        batch_diversity_min_distance: Minimum distance for diverse batches.
        use_max_posterior_sampling: Use BoTorch's MaxPosteriorSampling.
        seed: Random seed for reproducibility.
    """

    num_candidates: int = THOMPSON_NUM_CANDIDATES
    num_posterior_samples: int = THOMPSON_NUM_POSTERIOR_SAMPLES
    batch_diversity_min_distance: float = THOMPSON_BATCH_DIVERSITY_MIN_DISTANCE
    use_max_posterior_sampling: bool = True
    seed: int | None = None


def generate_thompson_samples(
    model: SingleTaskGP,
    bounds: Tensor,
    n_samples: int = 1,
    config: ThompsonConfig | None = None,
    minimize: bool = True,
) -> ThompsonBatch:
    """Generate suggestions using Thompson Sampling.

    Thompson Sampling works by:
    1. Drawing a sample function from the GP posterior
    2. Finding the optimum of that sample function
    3. Repeating for batch suggestions

    This naturally balances exploration and exploitation.

    Args:
        model: Fitted SingleTaskGP model.
        bounds: Parameter bounds (2 x d tensor).
        n_samples: Number of suggestions to generate.
        config: Thompson Sampling configuration.
        minimize: Whether to minimize (True) or maximize (False).

    Returns:
        ThompsonBatch containing the suggestions.

    Example:
        >>> from bo_engine import generate_thompson_samples
        >>> batch = generate_thompson_samples(model, bounds, n_samples=4)
        >>> for sample in batch.samples:
        ...     print(f"Point: {sample.parameters}, Mean: {sample.posterior_mean:.3f}")

    References:
        - BoTorch Thompson Sampling: https://botorch.org/tutorials/thompson_sampling/
    """
    device = get_device()
    dtype = get_dtype()

    bounds = bounds.to(device=device, dtype=dtype)

    if config is None:
        config = ThompsonConfig()

    if config.seed is not None:
        # Side-effect: mutates the global torch RNG. Required because
        # BoTorch's ``MaxPosteriorSampling`` / manual posterior draws
        # read from the process-wide torch state rather than accepting
        # an explicit :class:`torch.Generator`. Callers that need to
        # isolate this mutation should wrap the call in
        # ``torch.random.fork_rng(devices=[])``.
        torch.manual_seed(config.seed)

    if config.use_max_posterior_sampling:
        # Use BoTorch's efficient MaxPosteriorSampling
        samples = _thompson_via_max_posterior_sampling(
            model=model,
            bounds=bounds,
            n_samples=n_samples,
            num_candidates=config.num_candidates,
            minimize=minimize,
        )
    else:
        # Manual implementation for educational purposes
        samples = _thompson_manual(
            model=model,
            bounds=bounds,
            n_samples=n_samples,
            num_candidates=config.num_candidates,
            minimize=minimize,
        )

    # Build ThompsonSample objects with posterior info
    thompson_samples: list[ThompsonSample] = []

    for i in range(n_samples):
        x = samples[i : i + 1]
        with torch.no_grad():
            posterior = model.posterior(x)
            mean = posterior.mean.item()
            std = posterior.variance.sqrt().item()
            # Get a sampled value for this point
            sampled = posterior.rsample().item()

        thompson_samples.append(
            ThompsonSample(
                parameters=x.squeeze(0),
                sampled_value=sampled,
                posterior_mean=mean,
                posterior_std=std,
            )
        )

    # Compute batch diversity
    diversity_score = _compute_batch_diversity(samples, bounds)

    return ThompsonBatch(
        samples=thompson_samples,
        parameters_tensor=samples,
        diversity_score=diversity_score,
        method_info=f"Thompson Sampling (n_candidates={config.num_candidates})",
    )


def generate_thompson_samples_multi_objective(
    model: ModelListGP,
    bounds: Tensor,
    n_samples: int = 1,
    config: ThompsonConfig | None = None,
    weights: list[float] | None = None,
) -> ThompsonBatch:
    """Generate multi-objective suggestions via Thompson Sampling.

    For multi-objective optimization, we scalarize using random weights
    (like ParEGO) or specified weights, then apply Thompson Sampling.

    Args:
        model: Fitted ModelListGP.
        bounds: Parameter bounds (2 x d tensor).
        n_samples: Number of suggestions to generate.
        config: Thompson Sampling configuration.
        weights: Fixed scalarization weights. If None, random weights per sample.

    Returns:
        ThompsonBatch containing the suggestions.

    References:
        - Knowles "ParEGO: A Hybrid Algorithm with On-Line Landscape
          Approximation" (2006)
    """
    device = get_device()
    dtype = get_dtype()

    bounds = bounds.to(device=device, dtype=dtype)

    if config is None:
        config = ThompsonConfig()

    if config.seed is not None:
        # Side-effect: mutates the global torch RNG for the same reason
        # as the single-objective variant above.
        torch.manual_seed(config.seed)

    n_objectives = len(model.models)

    all_samples: list[Tensor] = []

    for _ in range(n_samples):
        # Random weights if not specified
        if weights is None:
            w = torch.rand(n_objectives, device=device, dtype=dtype)
            w = w / w.sum()  # Normalize to sum to 1
        else:
            w = torch.tensor(weights, device=device, dtype=dtype)

        # Generate candidates
        candidates = _generate_sobol_candidates(
            bounds=bounds,
            n_candidates=config.num_candidates,
        )

        # Draw posterior samples and scalarize
        with torch.no_grad():
            scalarized_samples = torch.zeros(config.num_candidates, device=device, dtype=dtype)

            for obj_idx in range(n_objectives):
                posterior = model.models[obj_idx].posterior(candidates)  # ty: ignore[call-non-callable]
                sample = posterior.rsample()  # 1 x n_candidates x 1
                sample = sample.squeeze()
                scalarized_samples += w[obj_idx] * sample

        # Find best (assuming minimization of scalarized objective)
        best_idx = scalarized_samples.argmin()
        all_samples.append(candidates[best_idx : best_idx + 1])

    samples = torch.cat(all_samples, dim=0)

    # Build ThompsonSample objects
    thompson_samples: list[ThompsonSample] = []

    for i in range(n_samples):
        x = samples[i : i + 1]
        # Use first objective for mean/std (or could aggregate)
        with torch.no_grad():
            posterior = model.models[0].posterior(x)  # ty: ignore[call-non-callable]
            mean = posterior.mean.item()
            std = posterior.variance.sqrt().item()
            sampled = posterior.rsample().item()

        thompson_samples.append(
            ThompsonSample(
                parameters=x.squeeze(0),
                sampled_value=sampled,
                posterior_mean=mean,
                posterior_std=std,
            )
        )

    diversity_score = _compute_batch_diversity(samples, bounds)

    return ThompsonBatch(
        samples=thompson_samples,
        parameters_tensor=samples,
        diversity_score=diversity_score,
        method_info="Thompson Sampling (multi-objective with random scalarization)",
    )


def generate_diverse_thompson_batch(
    model: SingleTaskGP,
    bounds: Tensor,
    n_samples: int,
    min_distance: float | None = None,
    max_attempts: int = 10,
    config: ThompsonConfig | None = None,
    minimize: bool = True,
) -> ThompsonBatch:
    """Generate a diverse batch of Thompson samples.

    Uses repeated sampling with diversity enforcement to ensure
    batch points are well-spread in parameter space.

    Args:
        model: Fitted GP model.
        bounds: Parameter bounds.
        n_samples: Number of samples to generate.
        min_distance: Minimum normalized distance between points.
        max_attempts: Maximum attempts to find diverse point.
        config: Thompson Sampling configuration.
        minimize: Whether minimizing objective.

    Returns:
        ThompsonBatch with diversity-enforced samples.
    """
    device = get_device()
    dtype = get_dtype()

    bounds = bounds.to(device=device, dtype=dtype)

    if config is None:
        config = ThompsonConfig()

    if min_distance is None:
        min_distance = config.batch_diversity_min_distance

    selected_samples: list[ThompsonSample] = []
    selected_tensors: list[Tensor] = []

    for _ in range(n_samples):
        # Try to find a diverse point
        best_sample = None
        best_min_dist = -1.0

        for _ in range(max_attempts):
            # Generate a single Thompson sample
            batch = generate_thompson_samples(
                model=model,
                bounds=bounds,
                n_samples=1,
                config=config,
                minimize=minimize,
            )

            candidate = batch.parameters_tensor

            # Check distance to existing selections
            if not selected_tensors:
                best_sample = batch.samples[0]
                best_min_dist = float("inf")
                break

            # Compute minimum distance to existing points
            stacked = torch.stack(selected_tensors)
            ranges = bounds[1] - bounds[0]
            ranges = torch.clamp(ranges, min=1e-8)
            normalized_candidate = (candidate - bounds[0]) / ranges
            normalized_existing = (stacked - bounds[0]) / ranges

            distances = torch.cdist(normalized_candidate, normalized_existing).squeeze()
            min_dist = distances.min().item()

            if min_dist >= min_distance:
                best_sample = batch.samples[0]
                best_min_dist = min_dist
                break
            elif min_dist > best_min_dist:
                best_sample = batch.samples[0]
                best_min_dist = min_dist

        if best_sample is not None:
            selected_samples.append(best_sample)
            selected_tensors.append(best_sample.parameters.unsqueeze(0))

    if not selected_tensors:
        # Fallback: just generate without diversity
        return generate_thompson_samples(
            model=model,
            bounds=bounds,
            n_samples=n_samples,
            config=config,
            minimize=minimize,
        )

    params_tensor = torch.cat(selected_tensors, dim=0)
    diversity_score = _compute_batch_diversity(params_tensor, bounds)

    return ThompsonBatch(
        samples=selected_samples,
        parameters_tensor=params_tensor,
        diversity_score=diversity_score,
        method_info=f"Diverse Thompson Sampling (min_distance={min_distance:.3f})",
    )


def _thompson_via_max_posterior_sampling(
    model: SingleTaskGP,
    bounds: Tensor,
    n_samples: int,
    num_candidates: int,
    minimize: bool,
) -> Tensor:
    """Thompson Sampling using BoTorch's MaxPosteriorSampling."""
    # Generate candidate points
    candidates = _generate_sobol_candidates(bounds, num_candidates)

    # Use BoTorch's MaxPosteriorSampling
    mps = MaxPosteriorSampling(model=model, replacement=False)

    if minimize:
        # For minimization, we need to negate
        # MaxPosteriorSampling maximizes by default
        # We can sample and find min, or use a workaround
        # Let's sample multiple times and pick minimum
        samples: list[Tensor] = []
        for _ in range(n_samples):
            with torch.no_grad():
                posterior = model.posterior(candidates)
                # Draw one sample per candidate
                sampled_values = posterior.rsample().squeeze()  # num_candidates
                best_idx = sampled_values.argmin()
                samples.append(candidates[best_idx : best_idx + 1])
        return torch.cat(samples, dim=0)
    else:
        # Maximization: use MPS directly
        selected = mps(candidates, num_samples=n_samples)
        return selected


def _thompson_manual(
    model: SingleTaskGP,
    bounds: Tensor,
    n_samples: int,
    num_candidates: int,
    minimize: bool,
) -> Tensor:
    """Manual Thompson Sampling implementation."""
    samples: list[Tensor] = []

    for _ in range(n_samples):
        # Generate candidates
        candidates = _generate_sobol_candidates(bounds, num_candidates)

        with torch.no_grad():
            # Draw posterior samples
            posterior = model.posterior(candidates)
            sampled_values = posterior.rsample().squeeze()

            # Find optimum of this sample
            if minimize:
                best_idx = sampled_values.argmin()
            else:
                best_idx = sampled_values.argmax()

            samples.append(candidates[best_idx : best_idx + 1])

    return torch.cat(samples, dim=0)


def _generate_sobol_candidates(
    bounds: Tensor,
    n_candidates: int,
) -> Tensor:
    """Generate Sobol sequence candidates within bounds."""
    device = bounds.device
    dtype = bounds.dtype
    n_dims = bounds.shape[1]

    # Use Sobol sequence
    sobol_engine = torch.quasirandom.SobolEngine(dimension=n_dims, scramble=True)
    candidates = sobol_engine.draw(n_candidates).to(device=device, dtype=dtype)

    # Scale to bounds
    lower = bounds[0]
    upper = bounds[1]
    candidates = lower + candidates * (upper - lower)

    return candidates


def _compute_batch_diversity(samples: Tensor, bounds: Tensor) -> float:
    """Compute diversity score for a batch of samples."""
    if samples.shape[0] < 2:
        return 1.0

    # Normalize to [0, 1]
    ranges = bounds[1] - bounds[0]
    ranges = torch.clamp(ranges, min=1e-8)
    normalized = (samples - bounds[0]) / ranges

    # Compute pairwise distances
    distances = torch.cdist(normalized, normalized)

    # Get upper triangular (exclude diagonal)
    n = samples.shape[0]
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=samples.device), diagonal=1)
    pairwise_distances = distances[mask]

    if len(pairwise_distances) == 0:
        return 1.0

    # Diversity score: mean distance (higher = more diverse)
    return pairwise_distances.mean().item()


def get_thompson_sampling_summary(batch: ThompsonBatch) -> str:
    """Generate human-readable summary of Thompson Sampling results.

    Args:
        batch: ThompsonBatch to summarize.

    Returns:
        Formatted string summary.
    """
    lines = [
        f"Method: {batch.method_info}",
        f"Batch size: {len(batch.samples)}",
        f"Diversity score: {batch.diversity_score:.3f}",
        "",
        "Samples:",
    ]

    for i, sample in enumerate(batch.samples):
        lines.append(
            f"  {i + 1}. Mean: {sample.posterior_mean:.4f}, Std: {sample.posterior_std:.4f}"
        )

    return "\n".join(lines)
