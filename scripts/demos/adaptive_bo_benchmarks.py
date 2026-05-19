#!/usr/bin/env python3
"""Benchmark functions for Adaptive BO Demo.

This module provides standard optimization benchmark functions for testing
and demonstrating Bayesian Optimization capabilities.

Supported benchmarks:
- Single-objective: Branin, Hartmann6, Levy, Ackley, Rosenbrock
- Multi-objective: BraninCurrin, ZDT1

Usage:
    from adaptive_bo_benchmarks import get_benchmark, BENCHMARKS

    benchmark = get_benchmark("branin")
    result = benchmark(x1=0.5, x2=0.5)  # {"y": value}
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class BenchmarkFunction(ABC):
    """Abstract base class for benchmark functions."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the benchmark function."""

    @property
    @abstractmethod
    def bounds(self) -> list[tuple[float, float]]:
        """Parameter bounds as list of (min, max) tuples."""

    @property
    @abstractmethod
    def optimum(self) -> float:
        """Known global optimum value."""

    @property
    def n_dims(self) -> int:
        """Number of dimensions."""
        return len(self.bounds)

    @property
    def is_multi_objective(self) -> bool:
        """Whether this is a multi-objective benchmark."""
        return False

    @abstractmethod
    def __call__(self, **params: float) -> dict[str, float]:
        """Evaluate the benchmark function.

        Args:
            **params: Parameter values (x1, x2, etc.)

        Returns:
            Dictionary mapping objective names to values.
        """


@dataclass(frozen=True)
class Branin(BenchmarkFunction):
    """2D Branin function.

    The Branin function has three global minima, all with f ≈ 0.397887.
    Common test function for 2D optimization.

    Search domain: x1 ∈ [-5, 10], x2 ∈ [0, 15]
    Global minima at:
        - (π, 2.275)
        - (-π, 12.275)
        - (9.42478, 2.475)
    """

    @property
    def name(self) -> str:
        return "branin"

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [(-5.0, 10.0), (0.0, 15.0)]

    @property
    def optimum(self) -> float:
        return 0.397887

    def __call__(self, **params: float) -> dict[str, float]:
        x1 = params.get("x1", 0.0)
        x2 = params.get("x2", 0.0)

        a = 1.0
        b = 5.1 / (4.0 * math.pi**2)
        c = 5.0 / math.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * math.pi)

        term1 = a * (x2 - b * x1**2 + c * x1 - r) ** 2
        term2 = s * (1.0 - t) * math.cos(x1)
        term3 = s

        return {"y": term1 + term2 + term3}


@dataclass
class Hartmann6(BenchmarkFunction):
    """6D Hartmann function.

    A challenging 6-dimensional function with 6 local minima.
    Global minimum at approximately f ≈ -3.32237.

    Search domain: xi ∈ [0, 1] for all i
    """

    _alpha: list[float] = field(
        default_factory=lambda: [1.0, 1.2, 3.0, 3.2],
        repr=False,
    )
    _A: list[list[float]] = field(
        default_factory=lambda: [
            [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
            [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
            [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
            [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
        ],
        repr=False,
    )
    _P: list[list[float]] = field(
        default_factory=lambda: [
            [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
            [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
            [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
            [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
        ],
        repr=False,
    )

    @property
    def name(self) -> str:
        return "hartmann6"

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [(0.0, 1.0)] * 6

    @property
    def optimum(self) -> float:
        return -3.32237

    def __call__(self, **params: float) -> dict[str, float]:
        x = [params.get(f"x{i + 1}", 0.5) for i in range(6)]

        outer_sum = 0.0
        for i in range(4):
            inner_sum = 0.0
            for j in range(6):
                inner_sum += self._A[i][j] * (x[j] - self._P[i][j]) ** 2
            outer_sum += self._alpha[i] * math.exp(-inner_sum)

        return {"y": -outer_sum}


@dataclass
class Levy(BenchmarkFunction):
    """N-dimensional Levy function.

    Highly multimodal with global minimum at f = 0 at x = [1, 1, ..., 1].

    Search domain: xi ∈ [-10, 10] for all i
    """

    n_dimensions: int = 4

    @property
    def name(self) -> str:
        return f"levy_{self.n_dimensions}d"

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [(-10.0, 10.0)] * self.n_dimensions

    @property
    def optimum(self) -> float:
        return 0.0

    def __call__(self, **params: float) -> dict[str, float]:
        x = [params.get(f"x{i + 1}", 0.0) for i in range(self.n_dimensions)]
        n = len(x)

        # Transform to w
        w = [1.0 + (xi - 1.0) / 4.0 for xi in x]

        term1 = math.sin(math.pi * w[0]) ** 2

        term2 = 0.0
        for i in range(n - 1):
            term2 += (w[i] - 1.0) ** 2 * (1.0 + 10.0 * math.sin(math.pi * w[i] + 1.0) ** 2)

        term3 = (w[n - 1] - 1.0) ** 2 * (1.0 + math.sin(2.0 * math.pi * w[n - 1]) ** 2)

        return {"y": term1 + term2 + term3}


@dataclass
class Ackley(BenchmarkFunction):
    """N-dimensional Ackley function.

    Characterized by a nearly flat outer region and a large hole at the center.
    Global minimum at f = 0 at x = [0, 0, ..., 0].

    Search domain: xi ∈ [-5, 5] for all i
    """

    n_dimensions: int = 4
    a: float = 20.0
    b: float = 0.2
    c: float = 2.0 * math.pi

    @property
    def name(self) -> str:
        return f"ackley_{self.n_dimensions}d"

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [(-5.0, 5.0)] * self.n_dimensions

    @property
    def optimum(self) -> float:
        return 0.0

    def __call__(self, **params: float) -> dict[str, float]:
        x = [params.get(f"x{i + 1}", 0.0) for i in range(self.n_dimensions)]
        n = len(x)

        sum_sq = sum(xi**2 for xi in x)
        sum_cos = sum(math.cos(self.c * xi) for xi in x)

        term1 = -self.a * math.exp(-self.b * math.sqrt(sum_sq / n))
        term2 = -math.exp(sum_cos / n)
        term3 = self.a + math.e

        return {"y": term1 + term2 + term3}


@dataclass
class Rosenbrock(BenchmarkFunction):
    """N-dimensional Rosenbrock function.

    The global minimum lies in a narrow, parabolic valley. The valley is easy
    to find, but convergence to the minimum is difficult.

    Global minimum at f = 0 at x = [1, 1, ..., 1].
    Search domain: xi ∈ [-5, 10] for all i
    """

    n_dimensions: int = 4

    @property
    def name(self) -> str:
        return f"rosenbrock_{self.n_dimensions}d"

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [(-5.0, 10.0)] * self.n_dimensions

    @property
    def optimum(self) -> float:
        return 0.0

    def __call__(self, **params: float) -> dict[str, float]:
        x = [params.get(f"x{i + 1}", 0.0) for i in range(self.n_dimensions)]
        n = len(x)

        total = 0.0
        for i in range(n - 1):
            term1 = 100.0 * (x[i + 1] - x[i] ** 2) ** 2
            term2 = (x[i] - 1.0) ** 2
            total += term1 + term2

        return {"y": total}


@dataclass(frozen=True)
class BraninCurrin(BenchmarkFunction):
    """2D -> 2 objective Branin-Currin benchmark.

    A standard multi-objective test function combining:
    - Branin function (objective 1)
    - Currin function (objective 2)

    Search domain: x1, x2 ∈ [0, 1]
    Both objectives are to be minimized.
    """

    @property
    def name(self) -> str:
        return "branin_currin"

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [(0.0, 1.0), (0.0, 1.0)]

    @property
    def optimum(self) -> float:
        # Pareto front - no single optimum
        return float("nan")

    @property
    def is_multi_objective(self) -> bool:
        return True

    def __call__(self, **params: float) -> dict[str, float]:
        x1 = params.get("x1", 0.5)
        x2 = params.get("x2", 0.5)

        # Scale x1, x2 from [0, 1] to Branin domain
        x1_scaled = 15 * x1 - 5  # [-5, 10]
        x2_scaled = 15 * x2  # [0, 15]

        # Branin (scaled to [0, 1])
        a = 1.0
        b = 5.1 / (4.0 * math.pi**2)
        c = 5.0 / math.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * math.pi)

        branin = (
            a * (x2_scaled - b * x1_scaled**2 + c * x1_scaled - r) ** 2
            + s * (1.0 - t) * math.cos(x1_scaled)
            + s
        )
        # Normalize Branin to roughly [0, 1]
        y1 = (branin - 0.397887) / 300.0

        # Currin exponential
        x1_currin = max(x1, 1e-8)  # Avoid division by zero
        factor1 = 1.0 - math.exp(-1.0 / (2.0 * x2 + 1e-8))
        factor2 = (2300 * x1_currin**3 + 1900 * x1_currin**2 + 2092 * x1_currin + 60) / (
            100 * x1_currin**3 + 500 * x1_currin**2 + 4 * x1_currin + 20
        )
        y2 = factor1 * factor2 / 13.77  # Normalize to roughly [0, 1]

        return {"y1": y1, "y2": y2}


@dataclass
class ZDT1(BenchmarkFunction):
    """N-dimensional ZDT1 multi-objective benchmark.

    A scalable multi-objective test problem with convex Pareto front.

    Search domain: xi ∈ [0, 1] for all i
    Both objectives are to be minimized.
    """

    n_dimensions: int = 6

    @property
    def name(self) -> str:
        return f"zdt1_{self.n_dimensions}d"

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [(0.0, 1.0)] * self.n_dimensions

    @property
    def optimum(self) -> float:
        # Pareto front - no single optimum
        return float("nan")

    @property
    def is_multi_objective(self) -> bool:
        return True

    def __call__(self, **params: float) -> dict[str, float]:
        x = [params.get(f"x{i + 1}", 0.5) for i in range(self.n_dimensions)]
        n = len(x)

        f1 = x[0]

        g = 1.0 + 9.0 * sum(x[1:]) / (n - 1)
        f2 = g * (1.0 - math.sqrt(f1 / g))

        return {"y1": f1, "y2": f2}


# Registry of available benchmarks
BENCHMARKS: dict[str, type[BenchmarkFunction]] = {
    "branin": Branin,
    "hartmann6": Hartmann6,
    "levy": Levy,
    "ackley": Ackley,
    "rosenbrock": Rosenbrock,
    "branin_currin": BraninCurrin,
    "zdt1": ZDT1,
}

# Aliases for convenience
BENCHMARK_ALIASES: dict[str, str] = {
    "hartmann": "hartmann6",
    "hart6": "hartmann6",
    "bc": "branin_currin",
}


def get_benchmark(name: str, n_dims: int | None = None) -> BenchmarkFunction:
    """Get a benchmark function by name.

    Args:
        name: Name of the benchmark (see BENCHMARKS for available options).
        n_dims: Number of dimensions for scalable benchmarks (Levy, Ackley,
            Rosenbrock, ZDT1). Ignored for fixed-dimension benchmarks.

    Returns:
        Instantiated benchmark function.

    Raises:
        ValueError: If benchmark name is unknown.

    Examples:
        >>> benchmark = get_benchmark("branin")
        >>> result = benchmark(x1=0.5, x2=0.5)
        {"y": 23.45}

        >>> benchmark = get_benchmark("levy", n_dims=10)
        >>> result = benchmark(x1=1.0, x2=1.0, ..., x10=1.0)
        {"y": 0.0}
    """
    # Check aliases
    resolved_name = BENCHMARK_ALIASES.get(name, name)

    if resolved_name not in BENCHMARKS:
        available = list(BENCHMARKS.keys()) + list(BENCHMARK_ALIASES.keys())
        msg = f"Unknown benchmark '{name}'. Available benchmarks: {sorted(available)}"
        raise ValueError(msg)

    benchmark_class = BENCHMARKS[resolved_name]

    # Scalable benchmarks accept n_dimensions argument
    scalable_benchmarks = {Levy, Ackley, Rosenbrock, ZDT1}
    if benchmark_class in scalable_benchmarks and n_dims is not None:
        return benchmark_class(n_dimensions=n_dims)  # type: ignore[call-arg]  # ty: ignore[unknown-argument]

    return benchmark_class()


def list_benchmarks() -> dict[str, dict[str, Any]]:
    """List all available benchmarks with their properties.

    Returns:
        Dictionary mapping benchmark names to their properties.
    """
    result = {}
    for name, cls in BENCHMARKS.items():
        # Create instance to get properties
        if cls in {Levy, Ackley, Rosenbrock, ZDT1}:
            instance = cls(n_dimensions=4)  # type: ignore[call-arg]  # ty: ignore[unknown-argument]
            scalable = True
        else:
            instance = cls()
            scalable = False

        result[name] = {
            "n_dims": instance.n_dims,
            "scalable": scalable,
            "multi_objective": instance.is_multi_objective,
            "optimum": instance.optimum,
            "bounds": instance.bounds,
        }

    return result


if __name__ == "__main__":
    # Quick self-test
    print("Testing benchmark functions...")
    print()

    for name in BENCHMARKS:
        bench = get_benchmark(name)
        # Create params from bounds midpoints
        params = {f"x{i + 1}": (b[0] + b[1]) / 2 for i, b in enumerate(bench.bounds)}
        result = bench(**params)

        print(f"{bench.name}:")
        print(f"  Dimensions: {bench.n_dims}")
        print(f"  Optimum: {bench.optimum}")
        print(f"  Test result: {result}")
        print()

    print("All benchmarks passed self-test.")
