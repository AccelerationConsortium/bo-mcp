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

import logging
import math
import random
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
from botorch.generation.sampling import MaxPosteriorSampling
from botorch.models import ModelListGP, SingleTaskGP
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from torch import Tensor

from bo_engine.constants import (
    MAX_RANDOM_SEED,
    NUMERICAL_EPSILON,
    PAREGO_AUGMENTED_RHO,
    THOMPSON_BATCH_DIVERSITY_MIN_DISTANCE,
    THOMPSON_NUM_CANDIDATES,
    THOMPSON_NUM_POSTERIOR_SAMPLES,
)
from bo_engine.device import get_device, get_dtype
from bo_engine.reproducibility import GLOBAL_RNG_LOCK, derive_seed

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _resolve_call_seed(config_seed: int | None) -> int:
    """Resolve the torch seed installed inside a ``fork_rng`` block.

    ``fork_rng`` restores the global torch RNG state on exit, so any
    randomness consumed inside the block (Sobol scrambling, posterior
    ``rsample``) is replayed identically by the next call unless each
    call installs its own seed. A configured seed is used verbatim
    (reproducible by contract); otherwise fresh entropy is drawn from
    the stdlib RNG, whose state lives outside the fork and therefore
    advances between calls.
    """
    if config_seed is not None:
        return config_seed
    # Deliberately non-reproducible — no seed was supplied (same contract
    # as bo_engine.suggestions._resolve_acquisition_seed).
    return random.randint(0, MAX_RANDOM_SEED)  # noqa: S311


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

    # fork_rng isolates the global torch RNG for the duration of this
    # call — BoTorch's MaxPosteriorSampling / manual posterior draws
    # read from the process-wide torch state, so without isolation two
    # concurrent calls (e.g. under asyncio.to_thread) would race on the
    # seed and clobber each other's reproducibility. Because fork_rng
    # also RESTORES the state on exit, every call must install its own
    # seed — otherwise consecutive unseeded calls replay the identical
    # draw (see _resolve_call_seed). GLOBAL_RNG_LOCK serializes the
    # section against every other global-RNG snapshot/restore consumer
    # (suggestion pipeline, BayBE seeded scopes) so an overlapping
    # restore cannot roll a concurrent seeded stream back.
    with GLOBAL_RNG_LOCK, torch.random.fork_rng(devices=[]):
        torch.manual_seed(_resolve_call_seed(config.seed))

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
                # Clamp posterior variance to NUMERICAL_EPSILON so finite-
                # precision GP fits that drift into negative variance
                # surface as a near-zero std instead of NaN; warn once
                # per call if the clamp fires so silently broken posteriors
                # are still visible in logs.
                raw_variance = posterior.variance.item()
                if raw_variance < NUMERICAL_EPSILON:
                    logger.warning(
                        "Posterior variance %.3e below NUMERICAL_EPSILON; "
                        "clamping before sqrt to avoid NaN std",
                        raw_variance,
                    )
                clamped_variance = max(raw_variance, NUMERICAL_EPSILON)
                std = clamped_variance**0.5
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


def _raw_observed_outcomes(model: ModelListGP) -> Tensor:
    """Return the model's training outcomes on the raw objective scale.

    Each sub-model stores ``Standardize``-transformed targets; un-transforming
    them recovers the user-scale observations ParEGO needs to normalize each
    objective onto a common ``[0, 1]`` range. Shape ``(n_observations,
    n_objectives)``.
    """
    columns: list[Tensor] = []
    for obj_idx in range(len(model.models)):
        sub = model.models[obj_idx]
        targets = sub.train_targets.reshape(-1, 1)  # ty: ignore[call-non-callable]
        outcome_transform = getattr(sub, "outcome_transform", None)
        if outcome_transform is not None:
            targets, _ = outcome_transform.untransform(targets)
        columns.append(targets.reshape(-1))
    return torch.stack(columns, dim=-1)


def _aggregate_posterior_stats(model: ModelListGP, x: Tensor) -> tuple[float, float]:
    """Aggregate per-objective posterior mean/std at ``x`` across all objectives.

    Returns the mean of the per-objective posterior means and the
    root-mean-square of the per-objective posterior stds, so the reported
    uncertainty reflects every objective rather than only the first one.
    """
    means: list[float] = []
    variances: list[float] = []
    with torch.no_grad():
        for obj_idx in range(len(model.models)):
            posterior = model.models[obj_idx].posterior(x)  # ty: ignore[call-non-callable]
            means.append(float(posterior.mean.mean().item()))
            variances.append(float(posterior.variance.mean().item()))
    mean = sum(means) / len(means)
    std = math.sqrt(sum(variances) / len(variances))
    return mean, std


def generate_thompson_samples_multi_objective(
    model: ModelListGP,
    bounds: Tensor,
    n_samples: int = 1,
    config: ThompsonConfig | None = None,
    weights: list[float] | None = None,
    minimize: bool = True,
) -> ThompsonBatch:
    """Generate multi-objective suggestions via Thompson Sampling with ParEGO.

    Each sample draws random scalarization weights (or uses the supplied fixed
    weights) and combines the per-objective posterior draws with the
    **augmented Tchebycheff** scalarization on objectives normalized to a common
    ``[0, 1]`` range — this is true ParEGO. Unlike a plain weighted sum, the
    Tchebycheff form can reach concave regions of the Pareto front and is not
    dominated by whichever objective happens to have the largest numeric scale.

    Args:
        model: Fitted ModelListGP.
        bounds: Parameter bounds (2 x d tensor).
        n_samples: Number of suggestions to generate.
        config: Thompson Sampling configuration.
        weights: Fixed (non-negative) scalarization weights. If None, fresh
            random weights are drawn per sample.
        minimize: Whether the objectives are minimized (default). The
            direction is encoded as the sign of the scalarization weights, per
            BoTorch's ``get_chebyshev_scalarization`` convention.

    Returns:
        ThompsonBatch containing the suggestions.

    References:
        - Knowles "ParEGO: A Hybrid Algorithm with On-Line Landscape
          Approximation" (2006) — augmented Tchebycheff scalarization.
        - Daulton et al. "Differentiable Expected Hypervolume Improvement"
          NeurIPS 2020 — q-ParEGO via BoTorch's Chebyshev scalarization.
    """
    device = get_device()
    dtype = get_dtype()

    bounds = bounds.to(device=device, dtype=dtype)

    if config is None:
        config = ThompsonConfig()

    # Isolate the global torch RNG for the same reason as the
    # single-objective variant above — see its fork_rng comment (incl.
    # the GLOBAL_RNG_LOCK serialization rationale); the per-call seed
    # likewise keeps consecutive unseeded calls distinct.
    with GLOBAL_RNG_LOCK, torch.random.fork_rng(devices=[]):
        torch.manual_seed(_resolve_call_seed(config.seed))

        n_objectives = len(model.models)
        # Observed outcomes set the per-objective [0, 1] normalization bounds.
        observed_y = _raw_observed_outcomes(model).to(device=device, dtype=dtype)
        # BoTorch's Chebyshev scalarization is written for maximization and
        # expects negative weights on objectives that should be minimized.
        direction = -1.0 if minimize else 1.0

        thompson_samples: list[ThompsonSample] = []
        selected: list[Tensor] = []

        for _ in range(n_samples):
            if weights is None:
                w = torch.rand(n_objectives, device=device, dtype=dtype)
                w = w / w.sum()
            else:
                w = torch.tensor(weights, device=device, dtype=dtype)
            scalarization = get_chebyshev_scalarization(
                weights=direction * w,
                Y=observed_y,
                alpha=PAREGO_AUGMENTED_RHO,
            )

            candidates = _generate_sobol_candidates(
                bounds=bounds,
                n_candidates=config.num_candidates,
            )

            # Draw one joint posterior sample per objective and stack into an
            # (n_candidates, n_objectives) matrix for the scalarization.
            with torch.no_grad():
                objective_samples = [
                    model.models[obj_idx].posterior(candidates).rsample().reshape(-1)  # ty: ignore[call-non-callable]
                    for obj_idx in range(n_objectives)
                ]
                sampled_outcomes = torch.stack(objective_samples, dim=-1)
                # ``get_chebyshev_scalarization`` returns a value to MAXIMIZE
                # (it negates the augmented Tchebycheff cost internally). The
                # optional second arg (X) is passed explicitly for the typed
                # Callable signature.
                scalarized = scalarization(sampled_outcomes, None)

            best_idx = int(scalarized.argmax().item())
            x = candidates[best_idx : best_idx + 1]
            selected.append(x)

            mean, std = _aggregate_posterior_stats(model, x)
            thompson_samples.append(
                ThompsonSample(
                    parameters=x.squeeze(0),
                    sampled_value=float(scalarized[best_idx].item()),
                    posterior_mean=mean,
                    posterior_std=std,
                )
            )

        samples = torch.cat(selected, dim=0)
        diversity_score = _compute_batch_diversity(samples, bounds)

        return ThompsonBatch(
            samples=thompson_samples,
            parameters_tensor=samples,
            diversity_score=diversity_score,
            method_info="Thompson Sampling (multi-objective, augmented Tchebycheff / ParEGO)",
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

    for slot in range(n_samples):
        best_sample = _find_diverse_sample(
            model=model,
            bounds=bounds,
            config=config,
            minimize=minimize,
            slot=slot,
            max_attempts=max_attempts,
            min_distance=min_distance,
            selected_tensors=selected_tensors,
        )
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


def _find_diverse_sample(
    model: SingleTaskGP,
    bounds: Tensor,
    config: ThompsonConfig,
    minimize: bool,
    slot: int,
    max_attempts: int,
    min_distance: float,
    selected_tensors: list[Tensor],
) -> ThompsonSample | None:
    """Draw Thompson samples until one is diverse from prior selections.

    Returns the first sample whose normalized distance to every
    already-selected point is at least ``min_distance``, falling back to
    the most distant attempt when the threshold cannot be met within
    ``max_attempts``.
    """
    best_sample: ThompsonSample | None = None
    best_min_dist = -1.0

    for attempt in range(max_attempts):
        # Each (slot, attempt) needs an independent posterior draw —
        # repeated calls with the SAME seed return the identical point,
        # which would make this retry loop a no-op. A configured seed
        # stays reproducible by deriving a per-attempt seed from it;
        # an unseeded config keeps drawing fresh entropy per call.
        attempt_config = config
        if config.seed is not None:
            attempt_config = replace(
                config,
                seed=derive_seed(config.seed, f"thompson:slot_{slot}:attempt_{attempt}"),
            )

        batch = generate_thompson_samples(
            model=model,
            bounds=bounds,
            n_samples=1,
            config=attempt_config,
            minimize=minimize,
        )

        if not selected_tensors:
            return batch.samples[0]

        # Minimum normalized distance to the already-selected points
        candidate = batch.parameters_tensor
        stacked = torch.stack(selected_tensors)
        ranges = torch.clamp(bounds[1] - bounds[0], min=1e-8)
        normalized_candidate = (candidate - bounds[0]) / ranges
        normalized_existing = (stacked - bounds[0]) / ranges

        distances = torch.cdist(normalized_candidate, normalized_existing).squeeze()
        min_dist = distances.min().item()

        if min_dist >= min_distance:
            return batch.samples[0]
        if min_dist > best_min_dist:
            best_sample = batch.samples[0]
            best_min_dist = min_dist

    return best_sample


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
    # Maximization: use MPS directly
    return mps(candidates, num_samples=n_samples)


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
            best_idx = sampled_values.argmin() if minimize else sampled_values.argmax()

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
    return lower + candidates * (upper - lower)


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
