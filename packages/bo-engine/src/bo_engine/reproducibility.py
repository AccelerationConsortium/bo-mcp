"""Reproducibility guarantees for Bayesian Optimization.

This module provides functions to ensure that BO runs can be reproduced
exactly, given the same inputs and random seeds. This is critical for
debugging and scientific reproducibility.

Section 3.7 - Missing Trust-Building Features

References:
    - PyTorch Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
    - BoTorch Determinism: https://botorch.org/docs/getting_started/
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from bo_engine.constants import MAX_RANDOM_SEED
from bo_engine.device import get_device


@dataclass
class SeedState:
    """Complete random state for reproducibility.

    Attributes:
        master_seed: The primary seed from which others are derived.
        torch_seed: PyTorch random seed.
        numpy_seed: NumPy random seed.
        python_seed: Python random seed.
        cuda_seed: CUDA random seed (if applicable).
    """

    master_seed: int
    torch_seed: int
    numpy_seed: int
    python_seed: int
    cuda_seed: int | None


@dataclass
class ReproducibilityConfig:
    """Configuration for reproducibility settings.

    Attributes:
        master_seed: Primary random seed for the campaign.
        deterministic_algorithms: Use deterministic CUDA algorithms.
        log_seeds: Whether to log seeds used.
        strict_mode: If True, fail on non-reproducible operations.
    """

    master_seed: int
    deterministic_algorithms: bool = True
    log_seeds: bool = True
    strict_mode: bool = False


@dataclass
class IterationSeeds:
    """Seeds for a specific optimization iteration.

    Attributes:
        iteration: The iteration number.
        initial_design_seed: Seed for Sobol/random initial design.
        model_fit_seed: Seed for model fitting.
        acquisition_opt_seed: Seed for acquisition optimization.
        batch_seed: Seed for batch generation.
    """

    iteration: int
    initial_design_seed: int
    model_fit_seed: int
    acquisition_opt_seed: int
    batch_seed: int


@dataclass
class ReproducibilityReport:
    """Report on reproducibility status.

    Attributes:
        is_reproducible: Whether the configuration supports reproducibility.
        config: Reproducibility configuration used.
        seed_log: Log of seeds used per iteration.
        warnings: Any warnings about reproducibility.
        environment_hash: Hash of relevant environment for comparison.
    """

    is_reproducible: bool
    config: ReproducibilityConfig
    seed_log: list[IterationSeeds]
    warnings: list[str] = field(default_factory=list)
    environment_hash: str = ""


class ReproducibilityManager:
    """Manages random seeds and reproducibility for optimization campaigns.

    This class provides deterministic seed derivation so that the same
    campaign run with the same master seed will produce identical results.
    """

    def __init__(self, config: ReproducibilityConfig) -> None:
        """Initialize the reproducibility manager.

        Args:
            config: Reproducibility configuration.
        """
        self._config = config
        self._seed_log: list[IterationSeeds] = []
        self._current_iteration = 0
        self._environment_hash = _compute_environment_hash()

        # Set up global reproducibility settings
        self._initialize_global_settings()

    def _initialize_global_settings(self) -> None:
        """Initialize global reproducibility settings."""
        if self._config.deterministic_algorithms:
            torch.use_deterministic_algorithms(True, warn_only=not self._config.strict_mode)

            # Set CUBLAS workspace config for deterministic behavior
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def set_global_seed(self) -> SeedState:
        """Set all random seeds based on master seed.

        Returns:
            SeedState with all seeds set.
        """
        master = self._config.master_seed

        # Derive seeds from master
        torch_seed = _derive_seed(master, "torch")
        numpy_seed = _derive_seed(master, "numpy")
        python_seed = _derive_seed(master, "python")

        # Set seeds
        torch.manual_seed(torch_seed)
        np.random.seed(numpy_seed)
        random.seed(python_seed)

        # Set CUDA seeds if available
        cuda_seed = None
        if torch.cuda.is_available():
            cuda_seed = _derive_seed(master, "cuda")
            torch.cuda.manual_seed_all(cuda_seed)

        return SeedState(
            master_seed=master,
            torch_seed=torch_seed,
            numpy_seed=numpy_seed,
            python_seed=python_seed,
            cuda_seed=cuda_seed,
        )

    def get_iteration_seeds(self, iteration: int) -> IterationSeeds:
        """Get deterministic seeds for a specific iteration.

        Args:
            iteration: The iteration number.

        Returns:
            IterationSeeds with all seeds for this iteration.
        """
        master = self._config.master_seed

        seeds = IterationSeeds(
            iteration=iteration,
            initial_design_seed=_derive_seed(master, f"initial_{iteration}"),
            model_fit_seed=_derive_seed(master, f"model_{iteration}"),
            acquisition_opt_seed=_derive_seed(master, f"acq_{iteration}"),
            batch_seed=_derive_seed(master, f"batch_{iteration}"),
        )

        if self._config.log_seeds:
            self._seed_log.append(seeds)

        return seeds

    def set_iteration_seed(self, iteration: int, phase: str) -> int:
        """Set seed for a specific phase of an iteration.

        Args:
            iteration: The iteration number.
            phase: Phase name ("initial", "model", "acquisition", "batch").

        Returns:
            The seed that was set.
        """
        seeds = self.get_iteration_seeds(iteration)

        if phase == "initial":
            seed = seeds.initial_design_seed
        elif phase == "model":
            seed = seeds.model_fit_seed
        elif phase == "acquisition":
            seed = seeds.acquisition_opt_seed
        elif phase == "batch":
            seed = seeds.batch_seed
        else:
            seed = _derive_seed(self._config.master_seed, f"{phase}_{iteration}")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        return seed

    def get_reproducibility_report(self) -> ReproducibilityReport:
        """Generate a report on reproducibility status.

        Returns:
            ReproducibilityReport with detailed information.
        """
        warnings: list[str] = []

        # Check for potential issues
        if not self._config.deterministic_algorithms:
            warnings.append("Deterministic algorithms disabled. Results may vary between runs.")

        if torch.cuda.is_available():
            if not torch.backends.cudnn.deterministic:
                warnings.append("cuDNN deterministic mode not set. GPU operations may vary.")

        device = get_device()
        if device.type == "cuda" and not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
            warnings.append("CUBLAS_WORKSPACE_CONFIG not set. Matrix operations may vary.")

        is_reproducible = len(warnings) == 0 or not self._config.strict_mode

        return ReproducibilityReport(
            is_reproducible=is_reproducible,
            config=self._config,
            seed_log=self._seed_log.copy(),
            warnings=warnings,
            environment_hash=self._environment_hash,
        )

    def verify_iteration(
        self,
        iteration: int,
        expected_seeds: IterationSeeds,
    ) -> bool:
        """Verify that iteration seeds match expected values.

        Args:
            iteration: The iteration to verify.
            expected_seeds: Expected seeds for this iteration.

        Returns:
            True if seeds match, False otherwise.
        """
        actual_seeds = self.get_iteration_seeds(iteration)

        return (
            actual_seeds.initial_design_seed == expected_seeds.initial_design_seed
            and actual_seeds.model_fit_seed == expected_seeds.model_fit_seed
            and actual_seeds.acquisition_opt_seed == expected_seeds.acquisition_opt_seed
            and actual_seeds.batch_seed == expected_seeds.batch_seed
        )


def create_reproducible_sobol(
    d: int,
    n: int,
    seed: int,
    bounds: Tensor,
) -> Tensor:
    """Create a reproducible Sobol sequence.

    Args:
        d: Number of dimensions.
        n: Number of samples.
        seed: Random seed for scrambling.
        bounds: Parameter bounds (2 x d tensor).

    Returns:
        Tensor of shape (n, d) with Sobol samples within bounds.
    """
    device = bounds.device
    dtype = bounds.dtype

    # SobolEngine accepts ``seed`` directly, so no global ``torch.manual_seed``
    # call is needed here. Keeping the scramble seed local avoids leaking
    # into the process-wide torch RNG.
    sobol = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)

    # Draw samples in [0, 1]^d
    samples = sobol.draw(n).to(device=device, dtype=dtype)

    # Scale to bounds
    lower = bounds[0]
    upper = bounds[1]
    samples = lower + samples * (upper - lower)

    return samples


def verify_reproducibility(
    suggestions1: list[dict[str, float]],
    suggestions2: list[dict[str, float]],
    tolerance: float = 1e-6,
) -> tuple[bool, str]:
    """Verify that two sets of suggestions are identical.

    Args:
        suggestions1: First set of suggestions.
        suggestions2: Second set of suggestions.
        tolerance: Numerical tolerance for comparison.

    Returns:
        Tuple of (is_identical, message).
    """
    if len(suggestions1) != len(suggestions2):
        return False, f"Different number of suggestions: {len(suggestions1)} vs {len(suggestions2)}"

    for i, (s1, s2) in enumerate(zip(suggestions1, suggestions2, strict=True)):
        if s1.keys() != s2.keys():
            return False, f"Suggestion {i}: Different parameter names"

        for key in s1:
            diff = abs(s1[key] - s2[key])
            if diff > tolerance:
                return False, f"Suggestion {i}, parameter '{key}': {s1[key]} vs {s2[key]}"

    return True, "All suggestions match within tolerance"


def compute_suggestions_hash(
    suggestions: list[dict[str, float]],
    precision: int = 6,
) -> str:
    """Compute a hash of suggestions for comparison.

    Args:
        suggestions: List of suggestion dictionaries.
        precision: Decimal precision for rounding.

    Returns:
        Hash string.
    """
    # Round values for consistent hashing
    rounded = [{k: round(v, precision) for k, v in s.items()} for s in suggestions]

    # Sort keys for consistency
    json_str = json.dumps(rounded, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]


def get_reproducibility_summary(report: ReproducibilityReport) -> str:
    """Generate human-readable summary of reproducibility report.

    Args:
        report: ReproducibilityReport to summarize.

    Returns:
        Formatted string summary.
    """
    lines = [
        "=== Reproducibility Report ===",
        f"Status: {'Reproducible' if report.is_reproducible else 'NOT Reproducible'}",
        f"Master Seed: {report.config.master_seed}",
        f"Deterministic Algorithms: {report.config.deterministic_algorithms}",
        f"Environment Hash: {report.environment_hash}",
        f"Iterations Logged: {len(report.seed_log)}",
    ]

    if report.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in report.warnings:
            lines.append(f"  - {w}")

    if report.seed_log:
        lines.append("")
        lines.append("Recent Seeds (last 3 iterations):")
        for seeds in report.seed_log[-3:]:
            lines.append(
                f"  Iter {seeds.iteration}: "
                f"init={seeds.initial_design_seed}, "
                f"model={seeds.model_fit_seed}, "
                f"acq={seeds.acquisition_opt_seed}"
            )

    return "\n".join(lines)


def derive_seed(master_seed: int, context: str) -> int:
    """Derive a deterministic per-phase seed from the master seed.

    Every stochastic phase of the engine (initial-design Sobol scrambling,
    acquisition multi-start, fantasy sampling, MCMC chain offsets) should
    obtain its seed through this helper so cross-phase reproducibility is
    a function of the campaign master seed plus a stable role tag rather
    than ad-hoc per-call-site derivations.

    The derivation is a SHA-256 hash of ``f"{master_seed}:{context}"``,
    truncated to a 4-byte unsigned integer and folded into ``[0,
    MAX_RANDOM_SEED)``. The hash gives near-uniform coverage of the seed
    space and is stable across Python versions — picking the same context
    string at the same master seed always returns the same integer.

    Args:
        master_seed: The master random seed (from ``OptimizationSpec.random_seed``
            or a higher-level reproducibility manager).
        context: A stable role tag — e.g. ``"sobol:iter_5"``,
            ``"acquisition:iter_5"``. New stochastic phases should pick
            human-readable tags so seed reuse across phases is impossible
            by construction.

    Returns:
        Derived seed value in ``[0, MAX_RANDOM_SEED)``.
    """
    combined = f"{master_seed}:{context}"
    hash_bytes = hashlib.sha256(combined.encode()).digest()
    # Use first 4 bytes to get an integer
    derived = int.from_bytes(hash_bytes[:4], byteorder="big")
    # Ensure within valid range
    return derived % MAX_RANDOM_SEED


# Backward-compatibility alias for callers that imported the underscore
# name before the helper was promoted. New code should call the public
# ``derive_seed`` directly.
_derive_seed = derive_seed


def _compute_environment_hash() -> str:
    """Compute hash of relevant environment for reproducibility tracking.

    Returns:
        Hash string representing the environment.
    """
    env_info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device": str(get_device()),
    }

    if torch.cuda.is_available():
        env_info["gpu_name"] = torch.cuda.get_device_name(0)

    json_str = json.dumps(env_info, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]
