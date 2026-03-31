#!/usr/bin/env python3
"""Backend benchmarking harness.

Compares BoTorch and BayBE backends on standard optimization problems
to justify BackendSelector heuristics (TODO item 4.P5 / 10.6).

Usage:
    # Smoke test (5 seeds, 10 iterations — ~5 min)
    uv run python scripts/benchmark_backends.py --mode smoke

    # Full benchmark (30 seeds, 20 iterations — ~2-3 hours CPU)
    uv run python scripts/benchmark_backends.py --mode full

    # Single problem for debugging
    uv run python scripts/benchmark_backends.py --problem B1-Branin --seeds 3

Results are written to data/benchmark_results/ as JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from bo_engine.backend import SuggestionBatch
from bo_engine.botorch_backend import BoTorchBackend
from bo_engine.types import (
    ConstraintSpec,
    ConstraintType,
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkProblem:
    """Definition of a benchmark optimization problem."""

    name: str
    spec: OptimizationSpec
    true_function: Any  # Callable[dict[str, float], dict[str, float]]
    known_optimum: float | None
    n_initial: int
    n_iterations: int
    batch_size: int


@dataclass
class BenchmarkResult:
    """Results from a single problem × backend × seed run."""

    problem: str
    backend: str
    seed: int
    best_values: list[float] = field(default_factory=list)
    wall_times_ms: list[float] = field(default_factory=list)
    total_evaluations: int = 0
    final_best: float | None = None


@dataclass
class BenchmarkSummary:
    """Aggregated summary for a problem × backend combination."""

    problem: str
    backend: str
    n_seeds: int
    median_final_best: float
    iqr_final_best: tuple[float, float]
    median_wall_time_ms: float
    median_regret: float | None


# ---------------------------------------------------------------------------
# Benchmark problem definitions
# ---------------------------------------------------------------------------


def _branin_eval(params: dict[str, float]) -> dict[str, float]:
    """Evaluate Branin function from parameter dict."""
    x1, x2 = params["x1"], params["x2"]
    t1 = x2 - 5.1 / (4 * math.pi**2) * x1**2 + 5 / math.pi * x1 - 6
    t2 = 10 * (1 - 1 / (8 * math.pi)) * math.cos(x1)
    return {"y": t1**2 + t2 + 10}


def _hartmann6_eval(params: dict[str, float]) -> dict[str, float]:
    """Evaluate Hartmann-6 function from parameter dict."""
    alpha = [1.0, 1.2, 3.0, 3.2]
    A = [
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
    ]
    P = [
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ]
    x = [params[f"x{i}"] for i in range(6)]
    outer = 0.0
    for i in range(4):
        inner = sum(A[i][j] * (x[j] - P[i][j] * 1e-4) ** 2 for j in range(6))
        outer += alpha[i] * math.exp(-inner)
    return {"y": -outer}


def _cat_branin_eval(params: dict[str, float | str]) -> dict[str, float]:
    """Branin with categorical modifier."""
    modifiers = {"low": 0.8, "medium": 1.0, "high": 1.2}
    mod = modifiers.get(str(params.get("category", "medium")), 1.0)
    x1, x2 = float(params["x1"]), float(params["x2"])
    t1 = x2 - 5.1 / (4 * math.pi**2) * x1**2 + 5 / math.pi * x1 - 6
    t2 = 10 * (1 - 1 / (8 * math.pi)) * math.cos(x1)
    return {"y": mod * (t1**2 + t2 + 10)}


def _mixed_space_eval(params: dict[str, Any]) -> dict[str, float]:
    """Synthetic function with mixed parameter types."""
    x1 = float(params.get("x1", 0.5))
    x2 = float(params.get("x2", 0.5))
    x3 = float(params.get("x3", 0.5))
    cat_map = {"A": 0.0, "B": 0.3, "C": -0.2}
    cat1 = cat_map.get(str(params.get("cat1", "A")), 0.0)
    cat2 = cat_map.get(str(params.get("cat2", "A")), 0.0)
    return {"y": (x1 - 0.3) ** 2 + (x2 - 0.7) ** 2 + x3**2 + cat1 + cat2}


def _constrained_branin_eval(params: dict[str, float]) -> dict[str, float]:
    """Branin on [0,1]^4 with sum constraint (evaluated without enforcement)."""
    x = [params.get(f"x{i}", 0.25) for i in range(4)]
    val = sum((xi - 0.5) ** 2 for xi in x) + 0.1 * math.sin(10 * x[0])
    return {"y": val}


def _mixture_eval(params: dict[str, float]) -> dict[str, float]:
    """Mixture formulation objective (4 components, sum=1)."""
    x = [params.get(f"x{i}", 0.25) for i in range(4)]
    return {"y": sum((xi - 0.3) ** 2 for xi in x) + 0.5 * x[0] * x[1]}


def _make_problems(n_initial: int, n_iterations: int, batch_size: int) -> list[BenchmarkProblem]:
    """Create all benchmark problems with given iteration settings."""
    return [
        BenchmarkProblem(
            name="B1-Branin",
            spec=OptimizationSpec(
                parameters=[
                    ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(-5.0, 10.0)),
                    ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 15.0)),
                ],
                objectives=[ObjectiveSpec(name="y", minimize=True)],
                batch_size=batch_size,
            ),
            true_function=_branin_eval,
            known_optimum=0.397887,
            n_initial=n_initial,
            n_iterations=n_iterations,
            batch_size=batch_size,
        ),
        BenchmarkProblem(
            name="B2-Hartmann6",
            spec=OptimizationSpec(
                parameters=[
                    ParameterSpec(name=f"x{i}", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))
                    for i in range(6)
                ],
                objectives=[ObjectiveSpec(name="y", minimize=True)],
                batch_size=batch_size,
            ),
            true_function=_hartmann6_eval,
            known_optimum=-3.32237,
            n_initial=n_initial,
            n_iterations=n_iterations,
            batch_size=batch_size,
        ),
        BenchmarkProblem(
            name="B3-CatBranin",
            spec=OptimizationSpec(
                parameters=[
                    ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(-5.0, 10.0)),
                    ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 15.0)),
                    ParameterSpec(
                        name="category",
                        type=ParameterType.CATEGORICAL,
                        categories=["low", "medium", "high"],
                    ),
                ],
                objectives=[ObjectiveSpec(name="y", minimize=True)],
                batch_size=batch_size,
            ),
            true_function=_cat_branin_eval,
            known_optimum=0.8 * 0.397887,  # "low" modifier
            n_initial=n_initial,
            n_iterations=n_iterations,
            batch_size=batch_size,
        ),
        BenchmarkProblem(
            name="B4-MixedSpace",
            spec=OptimizationSpec(
                parameters=[
                    ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                    ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                    ParameterSpec(name="x3", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                    ParameterSpec(
                        name="cat1",
                        type=ParameterType.CATEGORICAL,
                        categories=["A", "B", "C"],
                    ),
                    ParameterSpec(
                        name="cat2",
                        type=ParameterType.CATEGORICAL,
                        categories=["A", "B", "C"],
                    ),
                ],
                objectives=[ObjectiveSpec(name="y", minimize=True)],
                batch_size=batch_size,
            ),
            true_function=_mixed_space_eval,
            known_optimum=None,  # Complex landscape
            n_initial=n_initial,
            n_iterations=n_iterations,
            batch_size=batch_size,
        ),
        BenchmarkProblem(
            name="B6-Constrained",
            spec=OptimizationSpec(
                parameters=[
                    ParameterSpec(name=f"x{i}", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))
                    for i in range(4)
                ],
                objectives=[ObjectiveSpec(name="y", minimize=True)],
                constraints=[
                    ConstraintSpec(
                        type=ConstraintType.SUM_LESS_THAN,
                        parameters=["x0", "x1", "x2", "x3"],
                        value=2.0,
                    ),
                ],
                batch_size=batch_size,
            ),
            true_function=_constrained_branin_eval,
            known_optimum=None,
            n_initial=n_initial,
            n_iterations=n_iterations,
            batch_size=batch_size,
        ),
        BenchmarkProblem(
            name="B11-Mixture",
            spec=OptimizationSpec(
                parameters=[
                    ParameterSpec(name=f"x{i}", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))
                    for i in range(4)
                ],
                objectives=[ObjectiveSpec(name="y", minimize=True)],
                constraints=[
                    ConstraintSpec(
                        type=ConstraintType.SUM_EQUALS,
                        parameters=["x0", "x1", "x2", "x3"],
                        value=1.0,
                    ),
                ],
                batch_size=batch_size,
            ),
            true_function=_mixture_eval,
            known_optimum=None,
            n_initial=n_initial,
            n_iterations=n_iterations,
            batch_size=batch_size,
        ),
    ]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def _run_single(
    problem: BenchmarkProblem,
    backend: Any,
    seed: int,
) -> BenchmarkResult:
    """Run one problem on one backend with one seed."""
    import random

    import numpy as np
    import torch

    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    result = BenchmarkResult(
        problem=problem.name,
        backend=backend.name,
        seed=seed,
    )

    # Phase 1: Initial design
    observations: list[ObservationData] = []
    designs = backend.generate_initial_design(problem.spec, problem.n_initial)
    for d in designs:
        obj_values = problem.true_function(d)
        observations.append(ObservationData(parameter_values=d, objective_values=obj_values))

    best_so_far = min(obs.objective_values["y"] for obs in observations)
    result.best_values.append(best_so_far)

    # Phase 2: BO iterations
    for iteration in range(1, problem.n_iterations + 1):
        t0 = time.perf_counter()
        try:
            batch: SuggestionBatch = backend.generate_suggestions(
                spec=problem.spec,
                observations=observations,
                batch_size=problem.batch_size,
                iteration=iteration,
            )
        except (RuntimeError, ValueError, TypeError) as e:
            print(
                f"  [WARN] {problem.name}/{backend.name}/seed={seed} "
                f"iteration {iteration} failed: {e}"
            )
            break

        wall_ms = (time.perf_counter() - t0) * 1000
        result.wall_times_ms.append(wall_ms)

        # Evaluate suggestions
        for sugg in batch.suggestions:
            params = sugg["parameter_values"]
            obj_values = problem.true_function(params)
            observations.append(
                ObservationData(parameter_values=params, objective_values=obj_values)
            )
            best_so_far = min(best_so_far, obj_values["y"])

        result.best_values.append(best_so_far)

    result.total_evaluations = len(observations)
    result.final_best = best_so_far
    return result


def run_benchmark(
    problem: BenchmarkProblem,
    backend: Any,
    seeds: list[int],
) -> list[BenchmarkResult]:
    """Run a problem on a backend across multiple seeds."""
    results = []
    for seed in seeds:
        print(f"  Seed {seed}...", end=" ", flush=True)
        r = _run_single(problem, backend, seed)
        print(f"best={r.final_best:.4f}" if r.final_best is not None else "failed")
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def summarize(results: list[BenchmarkResult]) -> BenchmarkSummary:
    """Compute summary statistics for a set of results (same problem+backend)."""
    finals = [r.final_best for r in results if r.final_best is not None]
    wall_times = [_median(r.wall_times_ms) for r in results if r.wall_times_ms]

    # Regret computation deferred until full benchmark run with known optima
    regrets: list[float] = []
    if finals:
        med_final = _median(finals)
    else:
        med_final = float("nan")

    return BenchmarkSummary(
        problem=results[0].problem,
        backend=results[0].backend,
        n_seeds=len(results),
        median_final_best=med_final,
        iqr_final_best=(_percentile(finals, 25), _percentile(finals, 75)) if finals else (0, 0),
        median_wall_time_ms=_median(wall_times) if wall_times else 0,
        median_regret=_median(regrets) if regrets else None,
    )


def print_comparison_table(summaries: list[BenchmarkSummary]) -> None:
    """Print a comparison table of results."""
    # Group by problem
    problems: dict[str, dict[str, BenchmarkSummary]] = {}
    for s in summaries:
        problems.setdefault(s.problem, {})[s.backend] = s

    print("\n" + "=" * 90)
    print(f"{'Problem':<20} {'Backend':<10} {'Median Best':>12} {'IQR':>20} {'Time (ms)':>10}")
    print("-" * 90)

    for prob_name in sorted(problems):
        for backend_name in sorted(problems[prob_name]):
            s = problems[prob_name][backend_name]
            iqr_str = f"[{s.iqr_final_best[0]:.4f}, {s.iqr_final_best[1]:.4f}]"
            print(
                f"{s.problem:<20} {s.backend:<10} {s.median_final_best:>12.4f} "
                f"{iqr_str:>20} {s.median_wall_time_ms:>10.0f}"
            )
        print()
    print("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark BO backends")
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
        help="smoke=5 seeds/10 iter, full=30 seeds/20 iter",
    )
    parser.add_argument("--problem", type=str, default=None, help="Run single problem by name")
    parser.add_argument("--seeds", type=int, default=None, help="Override number of seeds")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/benchmark_results",
        help="Output directory for results JSON",
    )
    args = parser.parse_args()

    if args.mode == "smoke":
        n_seeds = args.seeds or 5
        n_initial, n_iterations, batch_size = 4, 10, 2
    else:
        n_seeds = args.seeds or 30
        n_initial, n_iterations, batch_size = 6, 20, 2

    seeds = list(range(n_seeds))
    problems = _make_problems(n_initial, n_iterations, batch_size)

    if args.problem:
        problems = [p for p in problems if p.name == args.problem]
        if not problems:
            print(f"Unknown problem: {args.problem}")
            return

    # Initialize backends
    backends: list[Any] = [BoTorchBackend()]

    try:
        from bo_engine_baybe import BayBEBackend

        backends.append(BayBEBackend())
    except ImportError:
        print("[WARN] BayBE backend not installed, running BoTorch only")

    # Run benchmarks
    all_results: list[BenchmarkResult] = []
    all_summaries: list[BenchmarkSummary] = []

    for problem in problems:
        print(f"\n{'=' * 60}")
        print(f"Problem: {problem.name}")
        print(
            f"  Dims: {len(problem.spec.parameters)}, "
            f"Iterations: {problem.n_iterations}, "
            f"Batch: {problem.batch_size}"
        )

        for backend in backends:
            # Skip problems the backend can't handle
            has_categorical = any(
                p.type == ParameterType.CATEGORICAL for p in problem.spec.parameters
            )
            if has_categorical and backend.name == "botorch":
                # BoTorch backend handles categoricals via one-hot encoding
                pass

            print(f"\n  Backend: {backend.name}")
            results = run_benchmark(problem, backend, seeds)
            all_results.extend(results)

            summary = summarize(results)
            all_summaries.append(summary)

    # Print comparison
    print_comparison_table(all_summaries)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"benchmark_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output_data = {
        "mode": args.mode,
        "n_seeds": n_seeds,
        "results": [asdict(r) for r in all_results],
        "summaries": [asdict(s) for s in all_summaries],
    }
    results_file.write_text(json.dumps(output_data, indent=2, default=str))
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
