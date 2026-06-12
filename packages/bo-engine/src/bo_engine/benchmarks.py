"""Standard benchmark functions for testing Bayesian Optimization.

These benchmarks are taken from the official BoTorch documentation and tutorials.
They serve as reference implementations for validating BO algorithms.

References:
- https://botorch.org/docs/tutorials/multi_objective_bo/
- https://botorch.org/docs/tutorials/turbo_1/
- https://botorch.org/docs/tutorials/multi_fidelity_bo/
"""

import math
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

# =============================================================================
# Single-Objective Benchmarks
# =============================================================================


def branin(x: Tensor) -> Tensor:
    """Branin function (2D -> 1D).

    A standard benchmark for single-objective optimization.
    Domain: x1 in [-5, 10], x2 in [0, 15]
    Global minimum: f(x*) ≈ 0.397887 at multiple locations:
        (−π, 12.275), (π, 2.275), (9.42478, 2.475)

    Args:
        x: Input tensor of shape (..., 2)

    Returns:
        Function values of shape (...)
    """
    t1 = x[..., 1] - 5.1 / (4 * math.pi**2) * x[..., 0] ** 2 + 5 / math.pi * x[..., 0] - 6
    t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(x[..., 0])
    return t1**2 + t2 + 10


def branin_bounds() -> Tensor:
    """Return bounds for Branin function."""
    return torch.tensor([[-5.0, 0.0], [10.0, 15.0]])


def hartmann6(x: Tensor) -> Tensor:
    """Hartmann 6D function (6D -> 1D).

    A standard high-dimensional benchmark.
    Domain: x in [0, 1]^6
    Global minimum: f(x*) ≈ -3.32237 at (0.20169, 0.150011, 0.476874,
                                          0.275332, 0.311652, 0.6573)

    Args:
        x: Input tensor of shape (..., 6)

    Returns:
        Function values of shape (...)
    """
    # Materialize the constant tensors on the input's device and dtype so the
    # benchmark runs on GPU inputs (a bare ``torch.tensor`` lands on CPU and
    # would raise a device mismatch) and the output dtype follows the input
    # (float32 inputs stay float32 instead of silently upcasting to float64).
    alpha = torch.tensor([1.0, 1.2, 3.0, 3.2], dtype=x.dtype, device=x.device)
    A = torch.tensor(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ],
        dtype=x.dtype,
        device=x.device,
    )
    P = (
        torch.tensor(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ],
            dtype=x.dtype,
            device=x.device,
        )
        * 1e-4
    )

    inner_sum = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
    for i in range(4):
        inner = torch.sum(A[i] * (x - P[i]) ** 2, dim=-1)
        inner_sum = inner_sum + alpha[i] * torch.exp(-inner)
    return -inner_sum


def hartmann6_bounds() -> Tensor:
    """Return bounds for Hartmann 6D function."""
    return torch.tensor([[0.0] * 6, [1.0] * 6])


def embed_in_high_dim(
    func: Callable,
    x: Tensor,
    relevant_dims: list[int],
    original_bounds: Tensor,
) -> Tensor:
    """Embed a low-dimensional function in high-dimensional space.

    Used to test high-dimensional optimization methods like TuRBO and SAASBO.

    Args:
        func: The original function
        x: High-dimensional input of shape (..., D)
        relevant_dims: Indices of dimensions that affect the function
        original_bounds: Bounds for the original function of shape (2, d)

    Returns:
        Function values of shape (...)
    """
    # Extract relevant dimensions
    x_relevant = x[..., relevant_dims]
    # Unnormalize from [0,1] to original bounds
    lower = original_bounds[0]
    upper = original_bounds[1]
    x_unnorm = x_relevant * (upper - lower) + lower
    return func(x_unnorm)


def branin_50d(x: Tensor) -> Tensor:
    """Branin function embedded in 50D (only dims 0,1 matter).

    Domain: x in [0, 1]^50
    Global minimum: same as Branin, but in first 2 dimensions

    Args:
        x: Input tensor of shape (..., 50)

    Returns:
        Function values of shape (...)
    """
    return embed_in_high_dim(branin, x, [0, 1], branin_bounds())


def branin_100d(x: Tensor) -> Tensor:
    """Branin function embedded in 100D (only dims 0,1 matter).

    Used for testing SAASBO.

    Args:
        x: Input tensor of shape (..., 100)

    Returns:
        Function values of shape (...)
    """
    return embed_in_high_dim(branin, x, [0, 1], branin_bounds())


def branin_30d(x: Tensor) -> Tensor:
    """Branin function embedded in 30D (only dims 0,1 matter).

    Reference: SAASBO tutorial uses 30D embedding.
    Domain: x in [0, 1]^30
    Global minimum: same as Branin, but in first 2 dimensions

    Args:
        x: Input tensor of shape (..., 30)

    Returns:
        Function values of shape (...)
    """
    return embed_in_high_dim(branin, x, [0, 1], branin_bounds())


def ackley_20d(x: Tensor) -> Tensor:
    """20D Ackley function on [0, 1]^20 (normalized from [-5, 10]).

    Reference:
        - TuRBO Tutorial: https://botorch.org/docs/tutorials/turbo_1/
        - Ackley function: https://www.sfu.ca/~ssurjano/ackley.html

    The Ackley function is defined as:
        f(x) = -a * exp(-b * sqrt(mean(x^2))) - exp(mean(cos(c*x))) + a + e

    With standard parameters: a=20, b=0.2, c=2*pi
    Global minimum: f(0, ..., 0) = 0

    In this normalized version, x in [0, 1]^20 is mapped to [-5, 10]^20
    as used in the TuRBO tutorial.

    Args:
        x: Input tensor of shape (..., 20) in [0, 1]^20

    Returns:
        Function values of shape (...)
    """
    # Map [0, 1] to [-5, 10] as in TuRBO tutorial
    x_scaled = x * 15 - 5  # Now in [-5, 10]^20

    a = 20.0
    b = 0.2
    c = 2 * math.pi

    term1 = -a * torch.exp(-b * torch.sqrt(torch.mean(x_scaled**2, dim=-1)))
    term2 = -torch.exp(torch.mean(torch.cos(c * x_scaled), dim=-1))

    return term1 + term2 + a + math.e


def ackley_20d_bounds() -> Tensor:
    """Return bounds for 20D Ackley function (normalized to [0, 1])."""
    return torch.tensor([[0.0] * 20, [1.0] * 20])


def branin_currin_noisy(
    x: Tensor,
    noise_std: tuple[float, float] = (15.19, 0.63),
) -> Tensor:
    """BraninCurrin with additive Gaussian noise.

    Reference: Multi-objective BO tutorial uses noise_std = [15.19, 0.63]

    Args:
        x: Input tensor of shape (..., 2)
        noise_std: Standard deviation of noise for each objective

    Returns:
        Noisy objective values of shape (..., 2)
    """
    y = branin_currin(x)

    noise = torch.randn_like(y)
    noise[..., 0] = noise[..., 0] * noise_std[0]
    noise[..., 1] = noise[..., 1] * noise_std[1]

    return y + noise


def augmented_hartmann_cost(x: Tensor) -> tuple[Tensor, Tensor]:
    """Augmented Hartmann with cost model.

    Reference: Multi-fidelity tutorial
    Cost = 5.0 + fidelity

    Args:
        x: Input tensor of shape (..., 7) where last dim is fidelity

    Returns:
        Tuple of (objective_values, cost_values)
    """
    objective = augmented_hartmann(x)
    fidelity = x[..., 6]
    cost = 5.0 + fidelity

    return objective, cost


# =============================================================================
# Multi-Objective Benchmarks
# =============================================================================


def branin_currin(x: Tensor) -> Tensor:
    """Branin-Currin bi-objective function (2D -> 2D).

    A standard benchmark for multi-objective optimization.
    Domain: x in [0, 1]^2

    Args:
        x: Input tensor of shape (..., 2)

    Returns:
        Objective values of shape (..., 2)
    """
    # Branin (scaled to [0,1]^2)
    x1_branin = x[..., 0] * 15 - 5
    x2_branin = x[..., 1] * 15
    t1 = x2_branin - 5.1 / (4 * math.pi**2) * x1_branin**2 + 5 / math.pi * x1_branin - 6
    t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(x1_branin)
    branin_val = (t1**2 + t2 + 10) / 51.95

    # Currin
    x1_currin = x[..., 0]
    x2_currin = x[..., 1]
    factor1 = 1 - torch.exp(-1 / (2 * x2_currin + 1e-10))
    factor2 = 2300 * x1_currin**3 + 1900 * x1_currin**2 + 2092 * x1_currin + 60
    factor3 = 100 * x1_currin**3 + 500 * x1_currin**2 + 4 * x1_currin + 20
    currin_val = factor1 * factor2 / factor3 / 13.77

    return torch.stack([branin_val, currin_val], dim=-1)


def branin_currin_bounds() -> Tensor:
    """Return bounds for Branin-Currin function."""
    return torch.tensor([[0.0, 0.0], [1.0, 1.0]])


def dtlz2(x: Tensor, n_objectives: int = 2) -> Tensor:
    """DTLZ2 multi-objective function.

    A scalable multi-objective benchmark.
    Domain: x in [0, 1]^d where d >= n_objectives
    Pareto front: lies on a unit hypersphere

    Args:
        x: Input tensor of shape (..., d)
        n_objectives: Number of objectives (default 2)

    Returns:
        Objective values of shape (..., n_objectives)
    """
    # NOTE: k = d - n_objectives + 1 is the number of "distance" variables
    # Not used directly but kept for documentation (standard DTLZ notation)
    g = torch.sum((x[..., n_objectives - 1 :] - 0.5) ** 2, dim=-1)

    objectives = []
    for i in range(n_objectives):
        f_i = (1 + g).clone()
        for j in range(n_objectives - i - 1):
            f_i = f_i * torch.cos(x[..., j] * math.pi / 2)
        if i > 0:
            f_i = f_i * torch.sin(x[..., n_objectives - i - 1] * math.pi / 2)
        objectives.append(f_i)

    return torch.stack(objectives, dim=-1)


def c2_dtlz2(x: Tensor, n_objectives: int = 2, r: float = 0.2) -> tuple[Tensor, Tensor]:
    """C2-DTLZ2: DTLZ2 with a constraint.

    A constrained multi-objective benchmark.
    Constraint: c(x) = min_i[(f_i - 1)^2 + sum_{j!=i} f_j^2 - r^2] >= 0

    Args:
        x: Input tensor of shape (..., d)
        n_objectives: Number of objectives
        r: Constraint parameter (smaller = tighter constraint)

    Returns:
        Tuple of (objectives, constraint_values) where:
        - objectives has shape (..., n_objectives)
        - constraint_values has shape (...,) (negative = infeasible)
    """
    f = dtlz2(x, n_objectives)

    # Compute constraint
    c_terms = []
    for i in range(n_objectives):
        term = (f[..., i] - 1) ** 2
        for j in range(n_objectives):
            if j != i:
                term = term + f[..., j] ** 2
        c_terms.append(term - r**2)

    c = torch.stack(c_terms, dim=-1).min(dim=-1).values

    return f, c


# =============================================================================
# Multi-Fidelity Benchmarks
# =============================================================================


def augmented_hartmann(x: Tensor) -> Tensor:
    """Augmented Hartmann function with fidelity parameter.

    Domain: x in [0, 1]^7 where x[6] is the fidelity parameter
    At fidelity = 1.0, equals standard Hartmann6

    Args:
        x: Input tensor of shape (..., 7) where last dim is fidelity

    Returns:
        Function values of shape (...)
    """
    fidelity = x[..., 6]
    x_6d = x[..., :6]

    h6 = hartmann6(x_6d)

    # Add fidelity-dependent bias (lower fidelity = more bias)
    bias = (1 - fidelity) * (0.5 * torch.sin(2 * math.pi * x[..., 0]) + 0.5)

    return h6 + bias


def augmented_hartmann_bounds() -> Tensor:
    """Return bounds for Augmented Hartmann function."""
    return torch.tensor([[0.0] * 7, [1.0] * 7])


# =============================================================================
# Utility Functions
# =============================================================================


def get_benchmark_spec(name: str) -> dict[str, Any]:
    """Get specification for a benchmark function.

    Args:
        name: Benchmark name

    Returns:
        Dictionary with 'function', 'bounds', 'n_dims', 'n_objectives', 'optimal_value'
    """
    benchmarks = {
        "branin": {
            "function": branin,
            "bounds": branin_bounds(),
            "n_dims": 2,
            "n_objectives": 1,
            "optimal_value": 0.397887,
            "description": "Standard 2D single-objective benchmark",
        },
        "hartmann6": {
            "function": hartmann6,
            "bounds": hartmann6_bounds(),
            "n_dims": 6,
            "n_objectives": 1,
            "optimal_value": -3.32237,
            "description": "6D single-objective benchmark",
        },
        "branin_50d": {
            "function": branin_50d,
            "bounds": torch.tensor([[0.0] * 50, [1.0] * 50]),
            "n_dims": 50,
            "n_objectives": 1,
            "optimal_value": 0.397887,
            "description": "Branin embedded in 50D for TuRBO testing",
        },
        "branin_100d": {
            "function": branin_100d,
            "bounds": torch.tensor([[0.0] * 100, [1.0] * 100]),
            "n_dims": 100,
            "n_objectives": 1,
            "optimal_value": 0.397887,
            "description": "Branin embedded in 100D for SAASBO testing",
        },
        "branin_currin": {
            "function": branin_currin,
            "bounds": branin_currin_bounds(),
            "n_dims": 2,
            "n_objectives": 2,
            "optimal_value": None,  # Multi-objective: Pareto front
            "description": "2D bi-objective benchmark",
        },
        "dtlz2": {
            "function": dtlz2,
            "bounds": torch.tensor([[0.0] * 4, [1.0] * 4]),
            "n_dims": 4,
            "n_objectives": 2,
            "optimal_value": None,  # Pareto front on unit circle
            "description": "Scalable multi-objective benchmark",
        },
        "c2_dtlz2": {
            "function": c2_dtlz2,
            "bounds": torch.tensor([[0.0] * 4, [1.0] * 4]),
            "n_dims": 4,
            "n_objectives": 2,
            "optimal_value": None,
            "has_constraints": True,
            "description": "Constrained multi-objective benchmark",
        },
        "augmented_hartmann": {
            "function": augmented_hartmann,
            "bounds": augmented_hartmann_bounds(),
            "n_dims": 7,  # 6 + 1 fidelity
            "n_objectives": 1,
            "fidelity_dim": 6,
            "optimal_value": -3.32237,
            "description": "Multi-fidelity benchmark",
        },
        "branin_30d": {
            "function": branin_30d,
            "bounds": torch.tensor([[0.0] * 30, [1.0] * 30]),
            "n_dims": 30,
            "n_objectives": 1,
            "optimal_value": 0.397887,
            "description": "Branin embedded in 30D for SAASBO testing",
        },
        "ackley_20d": {
            "function": ackley_20d,
            "bounds": ackley_20d_bounds(),
            "n_dims": 20,
            "n_objectives": 1,
            "optimal_value": 0.0,
            "description": "20D Ackley for TuRBO testing",
        },
    }

    if name not in benchmarks:
        msg = f"Unknown benchmark: {name}. Available: {list(benchmarks.keys())}"
        raise ValueError(msg)

    return benchmarks[name]


def list_benchmarks() -> list[str]:
    """List available benchmark names."""
    return [
        "branin",
        "hartmann6",
        "branin_30d",
        "branin_50d",
        "branin_100d",
        "branin_currin",
        "dtlz2",
        "c2_dtlz2",
        "augmented_hartmann",
        "ackley_20d",
    ]
