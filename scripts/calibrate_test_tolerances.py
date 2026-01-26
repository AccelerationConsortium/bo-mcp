#!/usr/bin/env python3
"""Calibrate test tolerances via Monte Carlo simulation.

This script runs stochastic BO tests multiple times with different seeds
to determine appropriate tolerance thresholds for CI. The output can be
used to update test assertions with empirically-derived bounds.

Usage:
    uv run python scripts/calibrate_test_tolerances.py

    # Run more iterations for higher confidence
    uv run python scripts/calibrate_test_tolerances.py --iterations 200

    # Save results to file
    uv run python scripts/calibrate_test_tolerances.py --output tolerances.json

References:
    - BoTorch Reproducibility: https://botorch.org/docs/reproducibility
    - PyTorch Randomness: https://pytorch.org/docs/stable/notes/randomness.html
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Add packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "bo-engine" / "src"))

from bo_engine import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
    compute_hypervolume,
    compute_pareto_front,
    generate_next_batch,
)
from bo_engine.benchmarks import branin_currin

# =============================================================================
# Configuration Constants
# =============================================================================

DEFAULT_ITERATIONS = 100
DEFAULT_BO_ITERATIONS = 5
DEFAULT_BATCH_SIZE = 3

# Percentiles to compute for tolerance calibration
PERCENTILES = [50, 75, 90, 95, 99]


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CalibrationResult:
    """Result from a single calibration run."""

    seed: int
    pareto_max: float
    pareto_size: int
    final_hypervolume: float
    hypervolume_improvement: float


@dataclass
class ToleranceReport:
    """Aggregated tolerance report."""

    metric_name: str
    mean: float
    std: float
    min_val: float
    max_val: float
    percentiles: dict[int, float]
    recommended_ci_tolerance: float
    recommended_nightly_tolerance: float


# =============================================================================
# Calibration Functions
# =============================================================================


def create_branin_currin_spec(batch_size: int = DEFAULT_BATCH_SIZE) -> OptimizationSpec:
    """Create spec for Branin-Currin bi-objective optimization."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x0", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[
            ObjectiveSpec(name="branin", minimize=True),
            ObjectiveSpec(name="currin", minimize=True),
        ],
        batch_size=batch_size,
    )


def evaluate_branin_currin(spec: OptimizationSpec, suggestions: list[Any]) -> list[ObservationData]:
    """Evaluate Branin-Currin function for given suggestions."""
    observations = []
    for sugg in suggestions:
        x = torch.tensor([[sugg.parameter_values["x0"], sugg.parameter_values["x1"]]])
        y = branin_currin(x)
        observations.append(
            ObservationData(
                parameter_values=sugg.parameter_values,
                objective_values={"branin": y[0, 0].item(), "currin": y[0, 1].item()},
            )
        )
    return observations


def run_single_calibration(
    seed: int,
    bo_iterations: int = DEFAULT_BO_ITERATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> CalibrationResult:
    """Run a single BO optimization and collect metrics."""
    torch.manual_seed(seed)

    spec = create_branin_currin_spec(batch_size=batch_size)
    observations: list[ObservationData] = []
    ref_point = torch.tensor([1.5, 1.5])

    hypervolumes = []

    for iteration in range(bo_iterations):
        suggestions, _ = generate_next_batch(
            spec, observations, batch_size=batch_size, iteration=iteration
        )
        new_obs = evaluate_branin_currin(spec, suggestions)
        observations.extend(new_obs)

        # Compute metrics
        y = torch.tensor(
            [
                [obs.objective_values["branin"], obs.objective_values["currin"]]
                for obs in observations
            ]
        )
        pareto_y, _ = compute_pareto_front(y)
        hv = compute_hypervolume(pareto_y, ref_point)
        hypervolumes.append(hv)

    # Final metrics
    y = torch.tensor(
        [[obs.objective_values["branin"], obs.objective_values["currin"]] for obs in observations]
    )
    pareto_y, _ = compute_pareto_front(y)

    return CalibrationResult(
        seed=seed,
        pareto_max=pareto_y.max().item(),
        pareto_size=pareto_y.shape[0],
        final_hypervolume=hypervolumes[-1],
        hypervolume_improvement=hypervolumes[-1] - hypervolumes[0] if len(hypervolumes) > 1 else 0,
    )


def compute_tolerance_report(
    results: list[CalibrationResult],
    metric_name: str,
    values: list[float],
    is_upper_bound: bool = True,
) -> ToleranceReport:
    """Compute tolerance statistics for a metric."""
    values_array = np.array(values)

    percentile_values = {p: float(np.percentile(values_array, p)) for p in PERCENTILES}

    # Recommended tolerances:
    # - CI (must always pass): 99th percentile + 10% margin
    # - Nightly (statistical): 95th percentile
    if is_upper_bound:
        recommended_ci = percentile_values[99] * 1.1
        recommended_nightly = percentile_values[95]
    else:
        # For lower bounds (e.g., minimum hypervolume)
        if 1 in percentile_values:
            recommended_ci = percentile_values[1] * 0.9
        else:
            recommended_ci = values_array.min() * 0.9
        if 5 in percentile_values:
            recommended_nightly = percentile_values[5]
        else:
            recommended_nightly = values_array.min()

    return ToleranceReport(
        metric_name=metric_name,
        mean=float(values_array.mean()),
        std=float(values_array.std()),
        min_val=float(values_array.min()),
        max_val=float(values_array.max()),
        percentiles=percentile_values,
        recommended_ci_tolerance=recommended_ci,
        recommended_nightly_tolerance=recommended_nightly,
    )


def run_calibration(
    n_iterations: int = DEFAULT_ITERATIONS,
    bo_iterations: int = DEFAULT_BO_ITERATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    verbose: bool = True,
) -> dict[str, ToleranceReport]:
    """Run full calibration and return tolerance reports."""
    results: list[CalibrationResult] = []

    if verbose:
        print(f"Running {n_iterations} calibration iterations...")
        print(f"  BO iterations per run: {bo_iterations}")
        print(f"  Batch size: {batch_size}")
        print()

    for i in range(n_iterations):
        seed = 42 + i  # Deterministic but varied seeds
        result = run_single_calibration(seed, bo_iterations, batch_size)
        results.append(result)

        if verbose and (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{n_iterations} iterations")

    if verbose:
        print()

    # Compute reports for each metric
    reports = {
        "pareto_max": compute_tolerance_report(
            results, "pareto_max", [r.pareto_max for r in results], is_upper_bound=True
        ),
        "pareto_size": compute_tolerance_report(
            results, "pareto_size", [float(r.pareto_size) for r in results], is_upper_bound=False
        ),
        "final_hypervolume": compute_tolerance_report(
            results,
            "final_hypervolume",
            [r.final_hypervolume for r in results],
            is_upper_bound=False,
        ),
    }

    return reports


def print_report(reports: dict[str, ToleranceReport]) -> None:
    """Print calibration report to console."""
    print("=" * 70)
    print("TEST TOLERANCE CALIBRATION REPORT")
    print("=" * 70)
    print()

    for _name, report in reports.items():
        print(f"Metric: {report.metric_name}")
        print("-" * 40)
        print(f"  Mean:   {report.mean:.4f}")
        print(f"  Std:    {report.std:.4f}")
        print(f"  Min:    {report.min_val:.4f}")
        print(f"  Max:    {report.max_val:.4f}")
        print()
        print("  Percentiles:")
        for p, v in sorted(report.percentiles.items()):
            print(f"    {p}th: {v:.4f}")
        print()
        ci_tol = report.recommended_ci_tolerance
        nightly_tol = report.recommended_nightly_tolerance
        print(f"  Recommended CI tolerance (99% + margin):     {ci_tol:.4f}")
        print(f"  Recommended nightly tolerance (95%):         {nightly_tol:.4f}")
        print()

    print("=" * 70)
    print("RECOMMENDED TEST UPDATES")
    print("=" * 70)
    print()
    print("# test_bo_workflow.py - TestMultiObjectiveWorkflow")
    print()
    pareto_report = reports["pareto_max"]
    ci_tol = pareto_report.recommended_ci_tolerance
    nightly_tol = pareto_report.recommended_nightly_tolerance
    print("# Current: assert pareto_y.max() < 5.0")
    print(f"# Recommended CI:      assert pareto_y.max() < {ci_tol:.1f}")
    print(f"# Recommended Nightly: assert pareto_y.max() < {nightly_tol:.1f}")
    print()
    print("# For invariant-based testing (always passes):")
    print("assert pareto_y.shape[0] >= 2  # Always find multiple Pareto points")
    print("assert hypervolumes[-1] >= hypervolumes[0] - 1e-6  # Non-decreasing")
    print()


def save_report(reports: dict[str, ToleranceReport], output_path: Path) -> None:
    """Save calibration report to JSON file."""
    data = {
        name: {
            "metric_name": report.metric_name,
            "mean": report.mean,
            "std": report.std,
            "min": report.min_val,
            "max": report.max_val,
            "percentiles": report.percentiles,
            "recommended_ci_tolerance": report.recommended_ci_tolerance,
            "recommended_nightly_tolerance": report.recommended_nightly_tolerance,
        }
        for name, report in reports.items()
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Report saved to: {output_path}")


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for calibration script."""
    parser = argparse.ArgumentParser(
        description="Calibrate test tolerances via Monte Carlo simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=DEFAULT_ITERATIONS,
        help=f"Number of calibration iterations (default: {DEFAULT_ITERATIONS})",
    )
    parser.add_argument(
        "--bo-iterations",
        type=int,
        default=DEFAULT_BO_ITERATIONS,
        help=f"BO iterations per run (default: {DEFAULT_BO_ITERATIONS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for BO (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output JSON file path (optional)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    reports = run_calibration(
        n_iterations=args.iterations,
        bo_iterations=args.bo_iterations,
        batch_size=args.batch_size,
        verbose=not args.quiet,
    )

    print_report(reports)

    if args.output:
        save_report(reports, args.output)


if __name__ == "__main__":
    main()
