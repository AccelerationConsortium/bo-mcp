"""Tests reproducing results from BoTorch multi-objective optimization tutorials.

This module validates that our implementation achieves results consistent with
the official BoTorch tutorials and academic publications.

References:
    - Multi-objective BO Tutorial: https://botorch.org/docs/tutorials/multi_objective_bo/
    - Paper: Daulton et al. "Parallel Bayesian Optimization of Multiple
             Noisy Objectives with Expected Hypervolume Improvement" NeurIPS 2021
    - BoTorch Multi-objective: https://botorch.org/docs/multi_objective/
"""

import pytest
import torch
from torch.quasirandom import SobolEngine

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
from bo_engine.benchmarks import (
    branin_currin,
    branin_currin_bounds,
    dtlz2,
)

# =============================================================================
# Constants from BoTorch tutorials
# =============================================================================

# From Multi-objective BO Tutorial:
# https://botorch.org/docs/tutorials/multi_objective_bo/
#
# Problem: BraninCurrin (2 objectives, 2 parameters)
# Initial points: 6 (2*(d+1) where d=2)
# Batch size: 4
# Iterations: 20
# Noise std: [15.19, 0.63]
#
# Results after 20 batches:
# - qNEHVI: ~57.77 hypervolume
# - qNParEGO: ~54.32 hypervolume
# - Random (Sobol): ~0.64 hypervolume

TUTORIAL_INITIAL_POINTS = 6
TUTORIAL_BATCH_SIZE = 4
TUTORIAL_ITERATIONS = 20
TUTORIAL_NOISE_STD = (15.19, 0.63)

# Expected hypervolumes (with tolerance for stochasticity)
EXPECTED_QNEHVI_HYPERVOLUME = 57.77
EXPECTED_QNPAREGO_HYPERVOLUME = 54.32
EXPECTED_RANDOM_HYPERVOLUME = 0.64

# Reference point for hypervolume calculation (from tutorial)
TUTORIAL_REFERENCE_POINT = [1.1, 1.1]

# =============================================================================
# Seeded-campaign and hypervolume-panel configuration
# =============================================================================

# Branin-Currin input dimensionality (x1, x2).
BRANIN_CURRIN_N_PARAMS = 2

# Seed panel for the hypervolume-improvement-vs-Sobol comparison: each seed
# drives one fully reproducible paired trial (campaign seeded via
# ``OptimizationSpec.random_seed``, baseline via ``SobolEngine(seed=...)``).
HVI_VS_SOBOL_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
# Wins required of the panel. Trials are deterministic per environment, so the
# gate must catch a regression in one shot (repetition across nights re-runs
# the same fixed trajectories and adds no evidence): 4-of-5 stops a
# degraded-to-random pipeline (per-trial win prob 0.5) with ~81% one-shot
# probability, while the one seed of slack keeps a single hardware-sensitive
# trajectory from turning the nightly permanently red (5-of-5 would).
HVI_VS_SOBOL_MIN_WINS = 4
HVI_VS_SOBOL_ITERATIONS = 12

# Shared by every seeded Branin-Currin campaign in this module.
CAMPAIGN_BATCH_SIZE = 2

# Seed for the single-trajectory BO-vs-Sobol nightly test. Kept distinct from
# the panel seeds: the engine derives per-iteration seeds from ``random_seed``
# alone, so sharing a seed would replay a prefix of a panel campaign.
BO_VS_RANDOM_SEED = 7

# Slack factor for the single-trajectory BO-vs-Sobol comparison; the strict
# per-seed comparison lives in the panel test.
BO_VS_RANDOM_SLACK = 0.5

# The seed panel runs several ~50 s campaigns inside one test node; the
# project-default 120 s timeout (pyproject.toml) would cut it off outside the
# nightly job's ``--timeout=600``. The marker pins the nightly budget everywhere.
HVI_VS_SOBOL_TIMEOUT_SECONDS = 600


# =============================================================================
# Helpers
# =============================================================================


def _seeded_branin_currin_spec(seed: int) -> OptimizationSpec:
    """Two-parameter Branin-Currin minimization spec seeded for reproducibility.

    ``OptimizationSpec.random_seed`` is the engine's reproducibility contract:
    ``generate_next_batch`` seeds a ``fork_rng``-isolated scope from it, so a
    bare ``torch.manual_seed`` in a test never reaches the BO loop (unseeded
    specs draw fresh OS-entropy seeds by design).
    """
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[
            ObjectiveSpec(name="f1", minimize=True),
            ObjectiveSpec(name="f2", minimize=True),
        ],
        batch_size=CAMPAIGN_BATCH_SIZE,
        random_seed=seed,
    )


def _run_seeded_branin_currin_campaign(seed: int, *, n_iterations: int) -> torch.Tensor:
    """Run one seeded hypervolume-improvement campaign; return observed objectives.

    ``AcquisitionMethod.AUTO`` resolves multi-objective specs to the engine's
    default ``hypervolume_improvement`` method, which builds qLogNEHVI (the
    numerically stable formulation of qNEHVI; see ``create_acquisition``).
    """
    spec = _seeded_branin_currin_spec(seed)
    observations: list[ObservationData] = []
    for iteration in range(n_iterations):
        suggestions, _ = generate_next_batch(spec, observations, iteration=iteration)
        for sugg in suggestions:
            x1 = sugg.parameter_values["x1"]
            x2 = sugg.parameter_values["x2"]
            y = branin_currin(torch.tensor([[x1, x2]], dtype=torch.float64))
            observations.append(
                ObservationData(
                    parameter_values={"x1": x1, "x2": x2},
                    objective_values={"f1": y[0, 0].item(), "f2": y[0, 1].item()},
                )
            )
    return torch.tensor(
        [[obs.objective_values["f1"], obs.objective_values["f2"]] for obs in observations],
        dtype=torch.float64,
    )


def _sobol_baseline_objectives(seed: int, n_points: int) -> torch.Tensor:
    """Evaluate a seeded scrambled-Sobol random-search baseline."""
    sobol = SobolEngine(dimension=BRANIN_CURRIN_N_PARAMS, scramble=True, seed=seed)
    random_x = sobol.draw(n_points).to(torch.float64)
    return branin_currin(random_x)


# =============================================================================
# Test Classes
# =============================================================================


@pytest.mark.tutorial
class TestBraninCurrinMultiObjective:
    """Reproduce multi-objective optimization from BoTorch tutorial.

    Reference:
        - Tutorial: https://botorch.org/docs/tutorials/multi_objective_bo/
        - Paper: Daulton et al. "Parallel Bayesian Optimization of Multiple
                 Noisy Objectives with Expected Hypervolume Improvement" NeurIPS 2021

    Tutorial setup:
        - Problem: BraninCurrin (2 objectives, 2 parameters)
        - Initial points: 6 (2*(d+1) where d=2)
        - Batch size: 4
        - Iterations: 20
        - Noise std: [15.19, 0.63]
    """

    @pytest.mark.smoke
    def test_branin_currin_function_shape(self) -> None:
        """BraninCurrin should return 2 objectives for each input."""
        x = torch.rand(10, 2, dtype=torch.float64)
        y = branin_currin(x)

        assert y.shape == (10, 2), f"Expected shape (10, 2), got {y.shape}"

    @pytest.mark.smoke
    def test_branin_currin_bounds(self) -> None:
        """BraninCurrin bounds should be [0, 1]^2."""
        bounds = branin_currin_bounds()

        assert bounds.shape == (2, 2)
        assert (bounds[0] == 0.0).all()
        assert (bounds[1] == 1.0).all()

    @pytest.mark.smoke
    def test_branin_currin_values_in_range(self) -> None:
        """BraninCurrin objectives should be in reasonable range.

        The tutorial normalizes values, so raw values should be positive
        and bounded.
        """
        torch.manual_seed(42)
        x = torch.rand(100, 2, dtype=torch.float64)
        y = branin_currin(x)

        # Both objectives should be positive
        assert (y >= 0).all(), "Objectives should be non-negative"
        # Scaled objectives should be in reasonable range
        assert (y <= 10).all(), "Objectives should be bounded"

    @pytest.mark.smoke
    def test_multi_objective_optimization_basic(self) -> None:
        """Basic multi-objective optimization should produce valid Pareto front.

        This is a smoke test to verify the workflow runs correctly.
        """
        torch.manual_seed(42)

        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[
                ObjectiveSpec(name="f1", minimize=True),
                ObjectiveSpec(name="f2", minimize=True),
            ],
            batch_size=2,
        )

        observations: list[ObservationData] = []
        n_iterations = 8  # Smoke test uses fewer iterations

        for iteration in range(n_iterations):
            suggestions, _ = generate_next_batch(spec, observations, iteration=iteration)

            for sugg in suggestions:
                x1 = sugg.parameter_values["x1"]
                x2 = sugg.parameter_values["x2"]
                x = torch.tensor([[x1, x2]], dtype=torch.float64)
                y = branin_currin(x)
                observations.append(
                    ObservationData(
                        parameter_values={"x1": x1, "x2": x2},
                        objective_values={"f1": y[0, 0].item(), "f2": y[0, 1].item()},
                    )
                )

        # Check we have reasonable number of observations
        assert len(observations) >= n_iterations * 2

        # Check all objectives are valid
        for obs in observations:
            assert "f1" in obs.objective_values
            assert "f2" in obs.objective_values

    # NOTE: an absolute-hypervolume floor test (``hv > 0.1`` on one campaign)
    # was removed here: the Sobol warm-up alone clears any such floor (seed
    # 42's four warm-up points reach hv 0.34, and hypervolume is monotone in
    # added points), so the assertion could not detect a broken acquisition.
    # Inversion/degradation detection lives in the seed-panel test below;
    # magnitude tracking lives in the nightly calibration-drift gate.

    @pytest.mark.slow
    @pytest.mark.nightly
    @pytest.mark.timeout(HVI_VS_SOBOL_TIMEOUT_SECONDS)
    def test_hypervolume_improvement_beats_sobol(self) -> None:
        """Seeded hypervolume-improvement campaigns must beat Sobol on most seeds.

        ``AcquisitionMethod.AUTO`` resolves these specs to the engine default
        ``hypervolume_improvement``, i.e. qLogNEHVI — the numerically stable
        formulation of the tutorial's qNEHVI with the same acquisition
        semantics. Reference: the BoTorch multi-objective tutorial reports
        qNEHVI at ~57.77 hypervolume vs ~0.64 for quasi-random (Sobol)
        sampling with the same budget
        (https://botorch.org/docs/tutorials/multi_objective_bo/). An
        anti-optimizing hypervolume-improvement loop performs *worse* than
        random search, so BO-above-Sobol separates a working pipeline from an
        inverted one regardless of absolute hypervolume values. Both runs are
        measured against the benchmark-pinned ``TUTORIAL_REFERENCE_POINT``,
        deliberately independent of either run's data so no outlier
        observation can reshape the comparison box; a zero-hypervolume tie
        counts against BO.

        Each trial is reproducible end to end: ``OptimizationSpec.random_seed``
        seeds the campaign (a bare ``torch.manual_seed`` never reaches the
        engine's ``fork_rng``-isolated loop, so the earlier single-trial
        version of this test re-rolled a fresh trajectory every run and
        eventually lost one nightly) and ``SobolEngine(seed=...)`` seeds the
        paired baseline. The panel is a paired-comparison design in the
        spirit of the sign test (Dixon & Mood 1946, "The Statistical Sign
        Test", JASA 41(236):557-566), but the threshold is chosen for
        one-shot regression gating rather than significance under a
        no-better-than-random null: an inverted acquisition (win prob ~0)
        fails at any threshold, and a degraded-to-random pipeline (win prob
        0.5) slips past 4-of-5 only with probability 6/32. For a healthy
        pipeline the false-red rate is an order-of-magnitude estimate, not a
        calibrated one: the pre-rewrite record (~1 loss in ~35 nightly
        trials) was measured under a different baseline and reference-point
        design, and extrapolating it with approximately independent seeds
        suggests roughly 1% per benign dependency reroll. A systemic
        regression instead correlates every seed downward — exactly the
        signal the gate exists to catch.
        """
        ref_point = torch.tensor(TUTORIAL_REFERENCE_POINT, dtype=torch.float64)
        outcomes: list[tuple[int, float, float]] = []
        for seed in HVI_VS_SOBOL_SEEDS:
            bo_y = _run_seeded_branin_currin_campaign(seed, n_iterations=HVI_VS_SOBOL_ITERATIONS)
            random_y = _sobol_baseline_objectives(seed, n_points=bo_y.shape[0])
            hv_bo = compute_hypervolume(bo_y, ref_point)
            hv_random = compute_hypervolume(random_y, ref_point)
            outcomes.append((seed, hv_bo, hv_random))

        wins = sum(1 for _, hv_bo, hv_random in outcomes if hv_bo > hv_random)
        trials = "; ".join(
            f"seed {seed}: BO {hv_bo:.3f} vs Sobol {hv_random:.3f}"
            for seed, hv_bo, hv_random in outcomes
        )
        assert wins >= HVI_VS_SOBOL_MIN_WINS, (
            f"Hypervolume-improvement optimization beat the Sobol baseline in "
            f"only {wins} of {len(HVI_VS_SOBOL_SEEDS)} seeded trials "
            f"(required: {HVI_VS_SOBOL_MIN_WINS}); an inverted acquisition "
            f"loses essentially every trial. Trials: {trials}"
        )


@pytest.mark.tutorial
class TestParetoFront:
    """Test Pareto front computation and properties.

    Reference:
        - https://botorch.org/docs/multi_objective/
        - Uses is_non_dominated() utility
    """

    @pytest.mark.smoke
    def test_pareto_front_is_non_dominated(self) -> None:
        """All returned Pareto points should be non-dominated.

        A point is non-dominated if no other point is better in all objectives.
        """
        torch.manual_seed(42)

        # Create objectives with clear Pareto structure
        # Points on f1 + f2 = 1 line are Pareto optimal for minimization
        objectives = torch.tensor(
            [
                [0.1, 0.9],  # Pareto
                [0.5, 0.5],  # Pareto
                [0.9, 0.1],  # Pareto
                [0.6, 0.6],  # Dominated by (0.5, 0.5)
                [0.8, 0.8],  # Dominated
            ],
            dtype=torch.float64,
        )

        pareto_points, pareto_mask = compute_pareto_front(objectives, minimize=True)

        # Should have exactly 3 Pareto points
        assert pareto_mask.sum() == 3, f"Expected 3 Pareto points, got {pareto_mask.sum()}"

        # All Pareto points should be non-dominated
        for i, p1 in enumerate(pareto_points):
            for j, p2 in enumerate(pareto_points):
                if i != j:
                    # p2 should not dominate p1 (i.e., p2 <= p1 in all dims with < in at least one)
                    dominates = (p2 <= p1).all() and (p2 < p1).any()
                    assert not dominates, f"Point {p2} dominates Pareto point {p1}"

    @pytest.mark.smoke
    def test_pareto_front_maximization(self) -> None:
        """Pareto front should work for maximization problems."""
        torch.manual_seed(42)

        # For maximization, larger is better
        objectives = torch.tensor(
            [
                [0.9, 0.1],  # Pareto
                [0.5, 0.5],  # Pareto
                [0.1, 0.9],  # Pareto
                [0.4, 0.4],  # Dominated
            ],
            dtype=torch.float64,
        )

        _pareto_points, pareto_mask = compute_pareto_front(objectives, minimize=False)

        assert pareto_mask.sum() == 3, f"Expected 3 Pareto points, got {pareto_mask.sum()}"

    @pytest.mark.smoke
    def test_pareto_front_all_dominated(self) -> None:
        """When one point dominates all others, only it should be Pareto."""
        objectives = torch.tensor(
            [
                [0.1, 0.1],  # Dominates all (minimization)
                [0.5, 0.5],
                [0.9, 0.9],
            ],
            dtype=torch.float64,
        )

        _pareto_points, pareto_mask = compute_pareto_front(objectives, minimize=True)

        assert pareto_mask.sum() == 1
        assert pareto_mask[0], "First point should be on Pareto front"


@pytest.mark.tutorial
class TestHypervolume:
    """Test hypervolume computation as used in multi-objective BO.

    Reference:
        - https://botorch.org/docs/tutorials/multi_objective_bo/
        - Hypervolume is the standard metric for multi-objective optimization
    """

    @pytest.mark.smoke
    def test_hypervolume_basic(self) -> None:
        """Hypervolume should be positive for valid Pareto fronts."""
        # Simple 2D case: single point dominates reference
        objectives = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
        ref_point = torch.tensor([1.0, 1.0], dtype=torch.float64)

        hv = compute_hypervolume(objectives, ref_point)

        # Hypervolume should be area of rectangle
        expected = (1.0 - 0.5) * (1.0 - 0.5)  # = 0.25
        assert hv == pytest.approx(expected, rel=0.01), f"Expected {expected}, got {hv}"

    @pytest.mark.smoke
    def test_hypervolume_multiple_points(self) -> None:
        """Hypervolume with multiple Pareto points."""
        # Two points forming an L-shaped Pareto front
        objectives = torch.tensor(
            [
                [0.2, 0.8],
                [0.8, 0.2],
            ],
            dtype=torch.float64,
        )
        ref_point = torch.tensor([1.0, 1.0], dtype=torch.float64)

        hv = compute_hypervolume(objectives, ref_point)

        # Should be larger than with single point
        single_hv = (1.0 - 0.5) * (1.0 - 0.5)
        assert hv > single_hv, "Multiple points should increase hypervolume"

    @pytest.mark.smoke
    def test_reference_point_affects_hypervolume(self) -> None:
        """Different reference points should yield different hypervolumes.

        Hypervolume is computed relative to reference point.
        """
        objectives = torch.tensor([[0.5, 0.5]], dtype=torch.float64)

        ref_point_close = torch.tensor([0.6, 0.6], dtype=torch.float64)
        ref_point_far = torch.tensor([1.0, 1.0], dtype=torch.float64)

        hv_close = compute_hypervolume(objectives, ref_point_close)
        hv_far = compute_hypervolume(objectives, ref_point_far)

        assert hv_close < hv_far, (
            f"Closer reference point should give smaller hypervolume: "
            f"close={hv_close:.4f}, far={hv_far:.4f}"
        )

    @pytest.mark.smoke
    def test_hypervolume_dominated_point_ignored(self) -> None:
        """Dominated points should not increase hypervolume.

        Note: [0.5, 0.5] is NOT dominated by [0.2, 0.8] and [0.8, 0.2] in a
        minimization context because neither point dominates it (one is better
        on obj1, the other on obj2). A truly dominated point must be worse on
        ALL objectives compared to some other point.
        """
        # Pareto points
        pareto_only = torch.tensor(
            [
                [0.2, 0.2],  # Good on both objectives
                [0.1, 0.5],  # Best on obj1
                [0.5, 0.1],  # Best on obj2
            ],
            dtype=torch.float64,
        )

        # Add truly dominated point (worse than [0.2, 0.2] on both objectives)
        with_dominated = torch.tensor(
            [
                [0.2, 0.2],
                [0.1, 0.5],
                [0.5, 0.1],
                [0.3, 0.3],  # Truly dominated by [0.2, 0.2]
            ],
            dtype=torch.float64,
        )

        ref_point = torch.tensor([1.0, 1.0], dtype=torch.float64)

        hv_pareto = compute_hypervolume(pareto_only, ref_point)
        hv_with_dom = compute_hypervolume(with_dominated, ref_point)

        assert hv_pareto == pytest.approx(hv_with_dom, rel=0.01), (
            "Dominated points should not change hypervolume"
        )


@pytest.mark.tutorial
class TestDTLZ2Benchmark:
    """Test DTLZ2 benchmark function as used in multi-objective tutorials.

    Reference:
        - Deb et al. "Scalable Test Problems for Evolutionary Multiobjective Optimization" 2005
    """

    @pytest.mark.smoke
    def test_dtlz2_shape(self) -> None:
        """DTLZ2 should return correct output shape."""
        x = torch.rand(10, 4, dtype=torch.float64)
        y = dtlz2(x, n_objectives=2)

        assert y.shape == (10, 2)

    @pytest.mark.smoke
    def test_dtlz2_pareto_front_on_unit_sphere(self) -> None:
        """DTLZ2 Pareto front lies on unit hypersphere.

        For optimal solutions (where x[m:] = 0.5), objectives should
        satisfy sum(f_i^2) = 1.
        """
        # Create optimal point (distance variables at 0.5)
        x = torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float64)
        y = dtlz2(x, n_objectives=2)

        # Sum of squared objectives should equal 1
        sq_sum = (y**2).sum().item()
        assert sq_sum == pytest.approx(1.0, rel=0.05), (
            f"Pareto optimal point should be on unit sphere: sum(f^2) = {sq_sum}"
        )

    @pytest.mark.smoke
    def test_dtlz2_scalable(self) -> None:
        """DTLZ2 can scale to more objectives."""
        x = torch.rand(5, 6, dtype=torch.float64)

        y2 = dtlz2(x, n_objectives=2)
        y3 = dtlz2(x, n_objectives=3)

        assert y2.shape == (5, 2)
        assert y3.shape == (5, 3)


@pytest.mark.tutorial
class TestMultiObjectiveAcquisition:
    """Test multi-objective acquisition function properties."""

    @pytest.mark.smoke
    def test_mo_suggestions_have_valid_metadata(self) -> None:
        """Multi-objective suggestions should include relevant metadata."""
        torch.manual_seed(42)

        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[
                ObjectiveSpec(name="f1", minimize=True),
                ObjectiveSpec(name="f2", minimize=True),
            ],
            batch_size=2,
        )

        # Need some initial data (2*2+1=5 for BO mode with 2 params)
        observations = [
            ObservationData(
                parameter_values={"x1": 0.2, "x2": 0.3},
                objective_values={"f1": 0.5, "f2": 0.8},
            ),
            ObservationData(
                parameter_values={"x1": 0.7, "x2": 0.6},
                objective_values={"f1": 0.8, "f2": 0.4},
            ),
            ObservationData(
                parameter_values={"x1": 0.5, "x2": 0.5},
                objective_values={"f1": 0.6, "f2": 0.6},
            ),
            ObservationData(
                parameter_values={"x1": 0.3, "x2": 0.8},
                objective_values={"f1": 0.55, "f2": 0.7},
            ),
            ObservationData(
                parameter_values={"x1": 0.8, "x2": 0.2},
                objective_values={"f1": 0.75, "f2": 0.45},
            ),
        ]

        suggestions, _ = generate_next_batch(spec, observations, iteration=1)

        assert len(suggestions) == 2

        for sugg in suggestions:
            # Should have BO metadata for iteration > 0
            assert sugg.generation_method == "bo"
            # Should use multi-objective acquisition
            assert sugg.acquisition_function is not None
            assert (
                "hypervolume_improvement" in sugg.acquisition_function
                or "scalarized_multi_objective" in sugg.acquisition_function
            )

    @pytest.mark.smoke
    def test_mo_initial_design_covers_space(self) -> None:
        """Initial design for MO should provide good space coverage."""
        torch.manual_seed(42)

        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[
                ObjectiveSpec(name="f1", minimize=True),
                ObjectiveSpec(name="f2", minimize=True),
            ],
            batch_size=6,
        )

        suggestions, _ = generate_next_batch(spec, [], iteration=0)

        assert len(suggestions) == 6

        # Check coverage
        points = torch.tensor(
            [[s.parameter_values["x1"], s.parameter_values["x2"]] for s in suggestions],
            dtype=torch.float64,
        )

        # Check that points span the space (min and max not too close)
        x1_range = points[:, 0].max() - points[:, 0].min()
        x2_range = points[:, 1].max() - points[:, 1].min()

        assert x1_range > 0.5, f"x1 range ({x1_range:.2f}) too small"
        assert x2_range > 0.5, f"x2 range ({x2_range:.2f}) too small"


@pytest.mark.tutorial
class TestRandomVsBO:
    """Compare BO performance against random baseline.

    Reference: Tutorial shows BO significantly outperforms random sampling.
    """

    @pytest.mark.slow
    @pytest.mark.nightly
    def test_bo_outperforms_random(self) -> None:
        """BO should achieve better hypervolume than random sampling.

        This is the key result from the multi-objective BO tutorial
        (https://botorch.org/docs/tutorials/multi_objective_bo/). The
        comparison keeps a ``BO_VS_RANDOM_SLACK`` factor so only a grossly
        underperforming campaign fails; the strict per-seed comparison is
        ``test_hypervolume_improvement_beats_sobol``. Both sides are fully
        seeded (the baseline via ``SobolEngine(seed=...)``, the campaign via
        ``OptimizationSpec.random_seed`` — a bare ``torch.manual_seed`` never
        reaches the engine's ``fork_rng``-isolated loop), so the trial is
        deterministic per environment.
        """
        n_total_points = 20

        # Random baseline
        random_y = _sobol_baseline_objectives(BO_VS_RANDOM_SEED, n_points=n_total_points)

        ref_point = torch.tensor(TUTORIAL_REFERENCE_POINT, dtype=torch.float64)
        hv_random = compute_hypervolume(random_y, ref_point)

        # BO optimization with the identical evaluation budget
        n_iterations = n_total_points // CAMPAIGN_BATCH_SIZE
        bo_objectives = _run_seeded_branin_currin_campaign(
            BO_VS_RANDOM_SEED, n_iterations=n_iterations
        )
        hv_bo = compute_hypervolume(bo_objectives, ref_point)

        assert hv_bo >= hv_random * BO_VS_RANDOM_SLACK, (
            f"BO hypervolume ({hv_bo:.4f}) should be comparable or better than "
            f"random ({hv_random:.4f})"
        )
