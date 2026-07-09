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


def create_branin_currin_spec(
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int | None = None,
) -> OptimizationSpec:
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
        random_seed=seed,
    )


def evaluate_branin_currin(suggestions: list[Any]) -> list[ObservationData]:
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
    rng = np.random.default_rng(seed)

    spec = create_branin_currin_spec(batch_size=batch_size, seed=seed)
    observations: list[ObservationData] = []
    ref_point = torch.tensor([1.5, 1.5])

    hypervolumes = []

    for iteration in range(bo_iterations):
        suggestions, _ = generate_next_batch(
            spec, observations, batch_size=batch_size, iteration=iteration, rng=rng
        )
        new_obs = evaluate_branin_currin(suggestions)
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
        recommended_nightly = percentile_values[5] if 5 in percentile_values else values_array.min()

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
    return {
        "pareto_max": compute_tolerance_report(
            "pareto_max", [r.pareto_max for r in results], is_upper_bound=True
        ),
        "pareto_size": compute_tolerance_report(
            "pareto_size", [float(r.pareto_size) for r in results], is_upper_bound=False
        ),
        "final_hypervolume": compute_tolerance_report(
            "final_hypervolume",
            [r.final_hypervolume for r in results],
            is_upper_bound=False,
        ),
    }


def print_report(reports: dict[str, ToleranceReport]) -> None:
    """Print calibration report to console."""
    print("=" * 70)
    print("TEST TOLERANCE CALIBRATION REPORT")
    print("=" * 70)
    print()

    for report in reports.values():
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

    with Path(output_path).open("w") as f:
        json.dump(data, f, indent=2)

    print(f"Report saved to: {output_path}")


# =============================================================================
# Drift Check
# =============================================================================


DEFAULT_DRIFT_THRESHOLD = 0.25
"""Fractional drift threshold (25 percent) for the CI calibration gate.

Tolerances are noisy by design — they aggregate across stochastic
runs — so a tight threshold would surface false alarms. 25 percent
matches the band the audit recommended and is consistent with the
headroom existing assertions already carry.
"""

DEFAULT_DRIFT_ABSOLUTE_FLOOR_BY_METRIC: dict[str, float] = {
    # ``pareto_size`` is an integer count: typical values fall in
    # ``[2, 8]`` and the recommended-CI formula is ``min * 0.9``. A
    # 1-unit shift on ``min`` between runs is a ~50 percent relative
    # drift — mathematically loud, no underlying signal change worth
    # gating CI on. The 1.0-unit floor silences that case while still
    # letting a genuine 3+ unit shift through.
    "pareto_size": 1.0,
}
DEFAULT_DRIFT_CONTINUOUS_FLOOR = 0.05
"""Tight absolute floor for continuous metrics not listed above.

A 95 percent relative drift on ``final_hypervolume`` (baseline ~0.97
→ current ~1.9) is exactly the regression the gate exists to catch;
a single global 1.0-unit floor would silence it. The tight 0.05-unit
default suppresses sub-percent absolute noise on continuous metrics
without blunting the relative gate. Per-metric overrides via
:data:`DEFAULT_DRIFT_ABSOLUTE_FLOOR_BY_METRIC` win when present.
"""


def _relative_drift(baseline: float, current: float) -> float:
    """Return ``|current - baseline| / max(|baseline|, eps)``.

    Uses the absolute baseline magnitude as the denominator so the
    drift band is a *relative* tolerance even when the metric centres
    on zero. The ``eps`` floor (``1e-9``) protects against an
    accidental divide-by-zero from a degenerate baseline.
    """
    denom = max(abs(baseline), 1e-9)
    return abs(current - baseline) / denom


def _resolve_absolute_floor(
    metric_name: str,
    overrides: dict[str, float],
    default: float,
) -> float:
    """Return the absolute drift floor for ``metric_name``.

    Per-metric overrides win when present; metrics without an entry
    fall back to the tight continuous default. This is the seam that
    keeps a single integer-valued metric from blanket-suppressing
    drift on every continuous metric.
    """
    return overrides.get(metric_name, default)


def check_against_baseline(
    reports: dict[str, ToleranceReport],
    baseline_path: Path,
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
    absolute_floor_by_metric: dict[str, float] | None = None,
    continuous_floor: float = DEFAULT_DRIFT_CONTINUOUS_FLOOR,
) -> list[str]:
    """Compare ``reports`` against the JSON baseline at ``baseline_path``.

    Returns the list of human-readable drift descriptions for every
    metric whose recommended CI tolerance has moved by more than
    ``threshold`` (relative fraction) **and** by more than its
    metric-specific absolute floor. An empty list means the
    calibration is still within band.

    Per-metric floors come from ``absolute_floor_by_metric`` (defaults
    to :data:`DEFAULT_DRIFT_ABSOLUTE_FLOOR_BY_METRIC`); metrics without
    an explicit entry use ``continuous_floor``. The split prevents a
    single integer-valued metric from setting a global floor that
    silences a real regression on a continuous metric — a 95 percent
    relative drift on ``final_hypervolume`` clears the tight
    continuous floor even when the absolute delta is sub-unit.

    Both gates must trip simultaneously, so small-magnitude integer
    metrics (``pareto_size`` after the ``min * 0.9`` formula) do not
    surface 1-unit shifts as spurious alarms while continuous-metric
    regressions still escalate cleanly.
    """
    if absolute_floor_by_metric is None:
        absolute_floor_by_metric = DEFAULT_DRIFT_ABSOLUTE_FLOOR_BY_METRIC

    if not baseline_path.exists():
        return [
            f"Baseline file not found at {baseline_path}; cannot compare. "
            "Generate one with `--output` before running `--check`."
        ]

    baseline = json.loads(baseline_path.read_text())
    drifts: list[str] = []
    for name, report in reports.items():
        if name not in baseline:
            drifts.append(f"{name}: missing from baseline")
            continue
        baseline_ci = float(baseline[name]["recommended_ci_tolerance"])
        current_ci = float(report.recommended_ci_tolerance)
        rel_delta = _relative_drift(baseline_ci, current_ci)
        abs_delta = abs(current_ci - baseline_ci)
        floor = _resolve_absolute_floor(name, absolute_floor_by_metric, continuous_floor)
        if rel_delta > threshold and abs_delta > floor:
            drifts.append(
                f"{name}: CI tolerance drifted by {rel_delta * 100:.1f}% "
                f"(abs={abs_delta:.4f}, baseline={baseline_ci:.4f}, "
                f"current={current_ci:.4f}); thresholds: "
                f"{threshold * 100:.0f}% AND {floor:.4f} abs"
            )
    return drifts


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
    parser.add_argument(
        "--check",
        type=Path,
        help=(
            "Compare against the calibration baseline at PATH. Exits "
            "non-zero when any recommended CI tolerance has drifted by "
            "more than --drift-threshold (default 25%%). The nightly CI "
            "job uses this to fail loudly on algorithm-variance "
            "regressions instead of letting calibrated tolerances rot."
        ),
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=DEFAULT_DRIFT_THRESHOLD,
        help=(
            f"Maximum relative drift before --check fails. "
            f"Default {DEFAULT_DRIFT_THRESHOLD:.2f} (25%%)."
        ),
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

    if args.check is not None:
        drifts = check_against_baseline(reports, args.check, threshold=args.drift_threshold)
        if drifts:
            print("=" * 70)
            print("CALIBRATION DRIFT DETECTED")
            print("=" * 70)
            for line in drifts:
                print(f"  {line}")
            print()
            print(
                "Re-run with `--output <path>` to refresh the baseline once "
                "the drift has been triaged."
            )
            sys.exit(1)
        else:
            print("Calibration is within drift threshold.")


if __name__ == "__main__":
    main()
