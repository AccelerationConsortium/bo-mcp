"""Tests for per-restart diagnostics in ``optimize_acquisition``.

Captures the contract: the multi-start optimizer must
expose per-restart acquisition values so we can detect "widespread local
minima" — the failure mode where every restart converges to the same
acquisition basin and the BO loop loses its exploration safety net.

Reference: BoTorch tutorials on acquisition optimization
(https://botorch.org/docs/optimization) emphasize that multi-start L-BFGS-B
quality is checked via the dispersion of per-restart acquisition values,
not just the best. We pin both the DEBUG dump and the WARN-on-collapse
behaviour against synthetic restart distributions.
"""

from __future__ import annotations

import logging
import math

import pytest
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.models.deterministic import GenericDeterministicModel

from bo_engine.acquisition import (
    _log_restart_diagnostics,
    _select_assignments,
    optimize_acquisition,
)
from bo_engine.constants import DUPLICATE_DETECTION_TOLERANCE, RESTART_WARN_TOLERANCE
from bo_engine.initial_design import SearchSpaceExhaustedError
from bo_engine.types import ObjectiveSpec, OptimizationSpec, ParameterSpec, ParameterType


class _SumAcquisition(AcquisitionFunction):
    """Simple acquisition used to exercise the unseen-candidate fallback."""

    def __init__(self) -> None:
        model = GenericDeterministicModel(lambda x: x.sum(dim=-1, keepdim=True))
        super().__init__(model=model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # noqa: N803
        return X.sum(dim=(-1, -2))


@pytest.fixture
def diverse_restart_values() -> torch.Tensor:
    """Healthy dispersion: best is meaningfully above the median."""
    return torch.tensor([10.0, 1.0, 0.5, 0.2, 0.1], dtype=torch.float64)


@pytest.fixture
def collapsed_restart_values() -> torch.Tensor:
    """Pathological collapse: every restart converged to ~the same value."""
    return torch.tensor([1.0, 0.999, 0.998, 0.997, 0.996], dtype=torch.float64)


@pytest.fixture
def matching_candidates() -> torch.Tensor:
    """Five (q=1, d=2) candidates to pair with the per-restart values."""
    return torch.linspace(0.0, 1.0, steps=10, dtype=torch.float64).view(5, 1, 2)


class TestLogRestartDiagnostics:
    """``_log_restart_diagnostics`` writes DEBUG top-3 and WARN on collapse."""

    def test_top_three_emitted_at_debug(
        self,
        diverse_restart_values: torch.Tensor,
        matching_candidates: torch.Tensor,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.DEBUG, logger="bo_engine.acquisition")
        _log_restart_diagnostics(diverse_restart_values, matching_candidates)
        restart_logs = [r for r in caplog.records if "Acquisition restart rank" in r.message]
        assert len(restart_logs) == 3
        # Logged in descending acquisition order — top is rank 1 with value 10.0.
        assert "rank 1" in restart_logs[0].message
        assert "10" in restart_logs[0].message

    def test_no_warning_when_restarts_diverge(
        self,
        diverse_restart_values: torch.Tensor,
        matching_candidates: torch.Tensor,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING, logger="bo_engine.acquisition")
        _log_restart_diagnostics(diverse_restart_values, matching_candidates)
        warning_logs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_logs == []

    def test_warning_when_restarts_collapse(
        self,
        collapsed_restart_values: torch.Tensor,
        matching_candidates: torch.Tensor,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        caplog.set_level(logging.WARNING, logger="bo_engine.acquisition")
        _log_restart_diagnostics(collapsed_restart_values, matching_candidates)
        warning_logs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_logs) == 1
        assert "restart dispersion is tight" in warning_logs[0].message

    def test_single_restart_skips_dispersion_check(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """With one restart there is no median to compare against — no WARN."""
        caplog.set_level(logging.WARNING, logger="bo_engine.acquisition")
        values = torch.tensor([3.14], dtype=torch.float64)
        candidates = torch.zeros(1, 1, 2, dtype=torch.float64)
        _log_restart_diagnostics(values, candidates)
        warning_logs = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warning_logs == []

    def test_relative_gap_threshold_matches_constant(self) -> None:
        """Pin the documented constant — guards against drift between the
        diagnostic message and the threshold the test asserts against."""
        assert 0.0 < RESTART_WARN_TOLERANCE < 1.0


class TestOptimizeAcquisitionExposesDiagnostics:
    """End-to-end: q=1 path captures per-restart values via BoTorch.

    With ``batch_size == 1`` we ask ``optimize_acqf`` for the full set of
    per-restart candidates and route them through ``_log_restart_diagnostics``
    before collapsing to the best restart. Pins that the public interface
    still returns the single best candidate but emits diagnostics on the way.
    """

    def test_q1_logs_top_three_restart_values(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from bo_engine.acquisition import create_single_objective_acquisition
        from bo_engine.models import create_and_fit_single_task_model

        torch.manual_seed(0)
        train_x = torch.rand(8, 2, dtype=torch.float64)
        train_y = torch.rand(8, 1, dtype=torch.float64)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        model = create_and_fit_single_task_model(train_x, train_y, bounds)
        acqf = create_single_objective_acquisition(
            model=model,
            train_x=train_x,
            train_y=train_y,
            maximize=True,
        )

        caplog.set_level(logging.DEBUG, logger="bo_engine.acquisition")
        candidates, values = optimize_acquisition(
            acqf=acqf,
            bounds=bounds,
            batch_size=1,
            num_restarts=4,
            raw_samples=32,
        )
        # Public contract unchanged: best candidate is returned, q-shape preserved.
        assert candidates.shape == (1, 2)
        assert values.numel() == 1

        restart_logs = [r for r in caplog.records if "Acquisition restart rank" in r.message]
        # Up to 3 rank entries — exact count depends on num_restarts but the
        # diagnostic must fire for the q=1 path.
        assert restart_logs, "q=1 path should emit per-restart DEBUG diagnostics"

    def test_q_gt_1_keeps_sequential_path_silent(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """``batch_size > 1`` keeps the existing sequential greedy path and
        skips diagnostics — BoTorch does not support ``return_best_only=False``
        with sequential greedy optimization."""
        from bo_engine.acquisition import create_single_objective_acquisition
        from bo_engine.models import create_and_fit_single_task_model

        torch.manual_seed(0)
        train_x = torch.rand(8, 2, dtype=torch.float64)
        train_y = torch.rand(8, 1, dtype=torch.float64)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        model = create_and_fit_single_task_model(train_x, train_y, bounds)
        acqf = create_single_objective_acquisition(
            model=model,
            train_x=train_x,
            train_y=train_y,
            maximize=True,
        )

        caplog.set_level(logging.DEBUG, logger="bo_engine.acquisition")
        candidates, _ = optimize_acquisition(
            acqf=acqf,
            bounds=bounds,
            batch_size=2,
            num_restarts=4,
            raw_samples=32,
        )
        # Same public contract.
        assert candidates.shape == (2, 2)
        restart_logs = [r for r in caplog.records if "Acquisition restart rank" in r.message]
        assert restart_logs == []

    def test_q1_skips_best_restart_when_it_matches_x_avoid(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The highest-valued restart is ineligible after it was evaluated."""
        restart_candidates = torch.tensor(
            [
                [[0.0, 1.0]],
                [[0.25, 0.75]],
                [[0.8, 0.2]],
            ],
            dtype=torch.float64,
        )
        restart_values = torch.tensor([10.0, 9.0, 8.0], dtype=torch.float64)

        def fake_optimize_acqf(**_kwargs):
            return restart_candidates, restart_values

        monkeypatch.setattr("bo_engine.acquisition.optimize_acqf", fake_optimize_acqf)

        candidates, values = optimize_acquisition(
            acqf=_SumAcquisition(),
            bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
            batch_size=1,
            num_restarts=3,
            raw_samples=16,
            x_avoid=restart_candidates[0],
        )

        assert torch.equal(candidates, restart_candidates[1])
        assert torch.equal(values, restart_values[1:2])

    def test_q1_falls_back_to_an_unseen_point_when_all_restarts_match(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Collapsed restarts must not send the evaluated maximizer again."""
        duplicate = torch.tensor([[[1.0, 1.0]]], dtype=torch.float64)

        def fake_optimize_acqf(**kwargs):
            repeats = kwargs["num_restarts"]
            return duplicate.repeat(repeats, 1, 1), torch.ones(repeats, dtype=torch.float64)

        monkeypatch.setattr("bo_engine.acquisition.optimize_acqf", fake_optimize_acqf)
        torch.manual_seed(7)

        candidates, values = optimize_acquisition(
            acqf=_SumAcquisition(),
            bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
            batch_size=1,
            num_restarts=3,
            raw_samples=16,
            x_avoid=duplicate[0],
        )

        assert candidates.shape == (1, 2)
        assert values.shape == (1,)
        assert not torch.equal(candidates, duplicate[0])
        assert torch.all(candidates >= 0.0)
        assert torch.all(candidates <= 1.0)


class _PeakAcquisition(AcquisitionFunction):
    """Concave surface whose unique maximizer sits at ``(peak, ..., peak)``."""

    def __init__(self, peak: float = 0.5) -> None:
        model = GenericDeterministicModel(lambda x: -(x - peak).pow(2).sum(dim=-1, keepdim=True))
        super().__init__(model=model)
        self._peak = peak

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # noqa: N803
        return -(X - self._peak).pow(2).sum(dim=(-1, -2))


def _one_discrete_param_spec(
    values: list[float] | None,
    bounds: tuple[float, float] | None,
) -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x0", type=ParameterType.DISCRETE, values=values, bounds=bounds)
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
    )


def _collapsing_optimize_acqf(point: torch.Tensor):
    """Fake ``optimize_acqf`` where every restart converges to ``point``."""

    def fake_optimize_acqf(**kwargs):
        repeats = kwargs["num_restarts"]
        return point.repeat(repeats, 1, 1), torch.ones(repeats, dtype=torch.float64)

    return fake_optimize_acqf


class TestAvoidedRestartMatching:
    """Restart exclusion uses the engine's duplicate semantics.

    Reference: near-identical GP inputs are the engine's definition of a
    duplicate experiment (``duplicate_row_mask``, tolerance 1e-6 scaled
    by sqrt(d)); acquisition-level exclusion must share it, since multi-start
    L-BFGS-B reconverges to a maximizer only within optimizer tolerance,
    never bit-exactly (BoTorch ``optimize_acqf`` docs, botorch.org).
    """

    def test_q1_filters_restart_within_duplicate_tolerance(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A restart a hair away from an evaluated point is still a duplicate."""
        restart_candidates = torch.tensor(
            [
                [[5e-8, 1.0]],
                [[0.25, 0.75]],
            ],
            dtype=torch.float64,
        )
        restart_values = torch.tensor([10.0, 9.0], dtype=torch.float64)

        def fake_optimize_acqf(**_kwargs):
            return restart_candidates, restart_values

        monkeypatch.setattr("bo_engine.acquisition.optimize_acqf", fake_optimize_acqf)

        candidates, values = optimize_acquisition(
            acqf=_SumAcquisition(),
            bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
            batch_size=1,
            num_restarts=2,
            raw_samples=16,
            x_avoid=torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        )

        assert torch.equal(candidates, restart_candidates[1])
        assert torch.equal(values, restart_values[1:2])

    def test_q1_snaps_value_grid_before_filtering(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A raw candidate that decodes onto an evaluated grid point is excluded.

        Numeric-discrete parameters are relaxed to a continuous box during
        optimization and snapped to their grid only at decode time, so
        avoided-point matching must compare snapped coordinates.
        """
        restart_candidates = torch.tensor(
            [[[0.49]], [[0.8]], [[0.3]]],
            dtype=torch.float64,
        )
        restart_values = torch.tensor([10.0, 9.0, 8.0], dtype=torch.float64)

        def fake_optimize_acqf(**_kwargs):
            return restart_candidates, restart_values

        monkeypatch.setattr("bo_engine.acquisition.optimize_acqf", fake_optimize_acqf)

        candidates, values = optimize_acquisition(
            acqf=_SumAcquisition(),
            bounds=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
            batch_size=1,
            num_restarts=3,
            raw_samples=16,
            spec=_one_discrete_param_spec(values=[0.0, 0.5, 1.0], bounds=None),
            x_avoid=torch.tensor([[0.5]], dtype=torch.float64),
        )

        # 0.49 and 0.3 both snap to the evaluated 0.5; 0.8 snaps to unseen 1.0.
        # The returned candidate is the canonical grid point (the coordinate
        # that will be executed), valued on the acquisition at that point.
        assert torch.equal(candidates, torch.tensor([[1.0]], dtype=torch.float64))
        assert torch.equal(values, torch.tensor([1.0], dtype=torch.float64))

    def test_q1_rounds_bounds_only_integer_grid_before_filtering(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Bounds-only discrete parameters round to integers before matching."""
        restart_candidates = torch.tensor(
            [[[2.4]], [[3.6]]],
            dtype=torch.float64,
        )
        restart_values = torch.tensor([10.0, 9.0], dtype=torch.float64)

        def fake_optimize_acqf(**_kwargs):
            return restart_candidates, restart_values

        monkeypatch.setattr("bo_engine.acquisition.optimize_acqf", fake_optimize_acqf)

        candidates, values = optimize_acquisition(
            acqf=_SumAcquisition(),
            bounds=torch.tensor([[0.0], [5.0]], dtype=torch.float64),
            batch_size=1,
            num_restarts=2,
            raw_samples=16,
            spec=_one_discrete_param_spec(values=None, bounds=(0.0, 5.0)),
            x_avoid=torch.tensor([[2.0]], dtype=torch.float64),
        )

        # 2.4 rounds to the evaluated 2; 3.6 rounds to unseen 4. The canonical
        # integer is returned, valued on the acquisition at that integer.
        assert torch.equal(candidates, torch.tensor([[4.0]], dtype=torch.float64))
        assert torch.equal(values, torch.tensor([4.0], dtype=torch.float64))

    def test_q1_ranks_snapped_candidates_on_acquisition_surface(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Selection compares acquisition values at executable coordinates.

        With grid ``[0, 1]`` and an acquisition peaked at 0.6, the raw restart
        0.49 outranks 0.8 — but it executes as 0.0, which is *worse* than 0.8's
        executable 1.0. Ranking must therefore use the snapped coordinates,
        not the relaxed optimizer output.
        """
        restart_candidates = torch.tensor(
            [[[0.49]], [[0.8]]],
            dtype=torch.float64,
        )
        raw_values = torch.tensor([10.0, 9.0], dtype=torch.float64)

        def fake_optimize_acqf(**_kwargs):
            return restart_candidates, raw_values

        monkeypatch.setattr("bo_engine.acquisition.optimize_acqf", fake_optimize_acqf)

        candidates, values = optimize_acquisition(
            acqf=_PeakAcquisition(peak=0.6),
            bounds=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
            batch_size=1,
            num_restarts=2,
            raw_samples=16,
            spec=_one_discrete_param_spec(values=[0.0, 1.0], bounds=None),
        )

        # acqf(0.0) = -0.36 < acqf(1.0) = -0.16: the grid point 1.0 wins even
        # though its raw restart lost, and the reported value describes it.
        assert torch.equal(candidates, torch.tensor([[1.0]], dtype=torch.float64))
        assert torch.allclose(values, torch.tensor([-0.16], dtype=torch.float64))

    def test_q1_snapped_restart_violating_constraint_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A feasible raw restart whose snap crosses a constraint is ineligible.

        With grid ``[0, 1]`` and ``x <= 0.6``, the raw restart 0.6 is feasible
        for the optimizer but executes as 1.0, which violates the constraint.
        Canonical candidates must be re-validated against the domain, so the
        feasible executable 0.0 wins despite its lower acquisition value.
        """
        restart_candidates = torch.tensor(
            [[[0.6]], [[0.3]]],
            dtype=torch.float64,
        )
        raw_values = torch.tensor([10.0, 9.0], dtype=torch.float64)

        def fake_optimize_acqf(**_kwargs):
            return restart_candidates, raw_values

        monkeypatch.setattr("bo_engine.acquisition.optimize_acqf", fake_optimize_acqf)

        candidates, values = optimize_acquisition(
            acqf=_SumAcquisition(),
            bounds=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
            batch_size=1,
            num_restarts=2,
            raw_samples=16,
            spec=_one_discrete_param_spec(values=[0.0, 1.0], bounds=None),
            inequality_constraints=[
                (
                    torch.tensor([0]),
                    torch.tensor([-1.0], dtype=torch.float64),
                    -0.6,
                )
            ],
        )

        assert torch.equal(candidates, torch.tensor([[0.0]], dtype=torch.float64))
        assert torch.equal(values, torch.tensor([0.0], dtype=torch.float64))

    def test_real_optimizer_never_returns_avoided_maximizer(self) -> None:
        """End-to-end invariant: the suggestion is never a duplicate experiment.

        Uses the real ``optimize_acqf`` on a concave acquisition whose global
        maximizer is the avoided point, so every restart converges (within
        L-BFGS-B tolerance, not bit-exactly) onto it. Whichever path selects
        the candidate — unseen restart or sampling fallback — the result must
        stay at least the duplicate tolerance away from the evaluated point.
        """
        torch.manual_seed(0)
        avoided = torch.tensor([[0.5, 0.5]], dtype=torch.float64)

        candidates, values = optimize_acquisition(
            acqf=_PeakAcquisition(),
            bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
            batch_size=1,
            num_restarts=4,
            raw_samples=32,
            x_avoid=avoided,
        )

        assert candidates.shape == (1, 2)
        assert values.shape == (1,)
        assert torch.all(candidates >= 0.0)
        assert torch.all(candidates <= 1.0)
        distance = torch.linalg.vector_norm(candidates - avoided).item()
        assert distance >= DUPLICATE_DETECTION_TOLERANCE


class TestContinuousUnseenFallback:
    """The all-restarts-avoided fallback is seeded and constraint-aware.

    Reference: BoTorch's ``get_polytope_samples`` (botorch.org, sampling
    utilities) draws feasible points under linear inequality and equality
    constraints via hit-and-run, which is the documented way to sample thin
    or zero-volume feasible regions that box rejection sampling cannot hit.
    """

    def test_fallback_is_reproducible_for_same_seed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two runs with one campaign seed return the identical candidate."""
        duplicate = torch.tensor([[[0.6, 0.6]]], dtype=torch.float64)
        monkeypatch.setattr(
            "bo_engine.acquisition.optimize_acqf", _collapsing_optimize_acqf(duplicate)
        )

        results = [
            optimize_acquisition(
                acqf=_SumAcquisition(),
                bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
                batch_size=1,
                num_restarts=3,
                raw_samples=16,
                x_avoid=duplicate[0],
                random_seed=123,
            )
            for _ in range(2)
        ]

        assert torch.equal(results[0][0], results[1][0])
        assert torch.equal(results[0][1], results[1][1])
        assert not torch.equal(results[0][0], duplicate[0])

    def test_fallback_honors_equality_constraints(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The fallback samples the zero-volume equality manifold directly."""
        duplicate = torch.tensor([[[0.5, 0.5]]], dtype=torch.float64)
        monkeypatch.setattr(
            "bo_engine.acquisition.optimize_acqf", _collapsing_optimize_acqf(duplicate)
        )

        candidates, values = optimize_acquisition(
            acqf=_SumAcquisition(),
            bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
            batch_size=1,
            num_restarts=3,
            raw_samples=16,
            x_avoid=duplicate[0],
            equality_constraints=[
                (
                    torch.tensor([0, 1]),
                    torch.tensor([1.0, 1.0], dtype=torch.float64),
                    1.0,
                )
            ],
            random_seed=7,
        )

        assert candidates.shape == (1, 2)
        assert values.shape == (1,)
        assert math.isclose(float(candidates.sum()), 1.0, abs_tol=1e-6)
        distance = torch.linalg.vector_norm(candidates - duplicate[0]).item()
        assert distance >= DUPLICATE_DETECTION_TOLERANCE

    def test_fallback_samples_thin_inequality_region(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A feasible region far smaller than the box still yields a candidate."""
        duplicate = torch.tensor([[[0.0005, 0.5]]], dtype=torch.float64)
        monkeypatch.setattr(
            "bo_engine.acquisition.optimize_acqf", _collapsing_optimize_acqf(duplicate)
        )

        candidates, _values = optimize_acquisition(
            acqf=_SumAcquisition(),
            bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
            batch_size=1,
            num_restarts=3,
            raw_samples=16,
            x_avoid=duplicate[0],
            inequality_constraints=[
                (
                    torch.tensor([0]),
                    torch.tensor([-1.0], dtype=torch.float64),
                    -0.001,
                )
            ],
            random_seed=7,
        )

        assert candidates.shape == (1, 2)
        assert float(candidates[0, 0]) <= 0.001 + 1e-9
        distance = torch.linalg.vector_norm(candidates - duplicate[0]).item()
        assert distance >= DUPLICATE_DETECTION_TOLERANCE

    def test_fallback_raises_exhausted_when_grid_is_fully_evaluated(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A fully evaluated finite grid raises the domain exhaustion signal.

        ``SearchSpaceExhaustedError`` is the campaign-level contract already
        interpreted by the backend and server layers; it may only be raised
        after full enumeration proves no unseen point remains.
        """
        duplicate = torch.tensor([[[1.0]]], dtype=torch.float64)
        monkeypatch.setattr(
            "bo_engine.acquisition.optimize_acqf", _collapsing_optimize_acqf(duplicate)
        )

        with pytest.raises(SearchSpaceExhaustedError):
            optimize_acquisition(
                acqf=_SumAcquisition(),
                bounds=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
                batch_size=1,
                num_restarts=3,
                raw_samples=16,
                spec=_one_discrete_param_spec(values=[0.0, 1.0], bounds=None),
                x_avoid=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
            )

    def test_fallback_finds_the_single_unseen_point_of_a_large_grid(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A near-exhausted finite grid is enumerated, not sampled.

        With 1001 grid values and exactly one unseen, sampling a fixed cloud
        can miss the survivor's snapping cell and falsely report exhaustion;
        full enumeration must return the remaining value instead.
        """
        grid = [i / 1000 for i in range(1001)]
        unseen = 0.777
        avoided = torch.tensor([[v] for v in grid if v != unseen], dtype=torch.float64)
        duplicate = torch.tensor([[[0.5]]], dtype=torch.float64)
        monkeypatch.setattr(
            "bo_engine.acquisition.optimize_acqf", _collapsing_optimize_acqf(duplicate)
        )

        candidates, values = optimize_acquisition(
            acqf=_SumAcquisition(),
            bounds=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
            batch_size=1,
            num_restarts=3,
            raw_samples=16,
            spec=_one_discrete_param_spec(values=grid, bounds=None),
            x_avoid=avoided,
        )

        assert torch.equal(candidates, torch.tensor([[unseen]], dtype=torch.float64))
        assert values.shape == (1,)

    def test_fallback_enumeration_respects_linear_constraints(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Enumerated grid points are filtered by native linear constraints.

        Feasible region (``x0 + x1 <= 2`` over the 3x3 grid) holds six points;
        five are evaluated, so only the feasible unseen (1, 1) may be
        returned — never an unseen but infeasible corner such as (2, 2).
        """
        duplicate = torch.tensor([[[0.0, 0.0]]], dtype=torch.float64)
        monkeypatch.setattr(
            "bo_engine.acquisition.optimize_acqf", _collapsing_optimize_acqf(duplicate)
        )
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x0", type=ParameterType.DISCRETE, values=[0.0, 1.0, 2.0]),
                ParameterSpec(name="x1", type=ParameterType.DISCRETE, values=[0.0, 1.0, 2.0]),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        avoided = torch.tensor(
            [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [1.0, 0.0], [2.0, 0.0]],
            dtype=torch.float64,
        )

        candidates, _values = optimize_acquisition(
            acqf=_SumAcquisition(),
            bounds=torch.tensor([[0.0, 0.0], [2.0, 2.0]], dtype=torch.float64),
            batch_size=1,
            num_restarts=3,
            raw_samples=16,
            spec=spec,
            x_avoid=avoided,
            inequality_constraints=[
                (
                    torch.tensor([0, 1]),
                    torch.tensor([-1.0, -1.0], dtype=torch.float64),
                    -2.0,
                )
            ],
        )

        assert torch.equal(candidates, torch.tensor([[1.0, 1.0]], dtype=torch.float64))

    def test_fallback_excludes_grid_values_outside_declared_bounds(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Enumeration intersects explicit ``values`` with declared bounds.

        A spec may carry both a value grid and narrower bounds; only in-bounds
        values are executable, so the out-of-bounds 2.0 must never be proposed.
        """
        duplicate = torch.tensor([[[0.0]]], dtype=torch.float64)
        monkeypatch.setattr(
            "bo_engine.acquisition.optimize_acqf", _collapsing_optimize_acqf(duplicate)
        )

        candidates, _values = optimize_acquisition(
            acqf=_SumAcquisition(),
            bounds=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
            batch_size=1,
            num_restarts=3,
            raw_samples=16,
            spec=_one_discrete_param_spec(values=[0.0, 0.5, 2.0], bounds=(0.0, 1.0)),
            x_avoid=torch.tensor([[0.0]], dtype=torch.float64),
        )

        assert torch.equal(candidates, torch.tensor([[0.5]], dtype=torch.float64))

    def test_fallback_exhaustion_counts_only_in_bounds_grid_values(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A grid whose only in-bounds value is evaluated is exhausted.

        With ``values=[0, 2]`` and bounds ``[0, 1]``, the executable space is
        just ``{0}``; once it is avoided, the correct outcome is the domain
        exhaustion signal — not the out-of-bounds 2.0.
        """
        duplicate = torch.tensor([[[0.0]]], dtype=torch.float64)
        monkeypatch.setattr(
            "bo_engine.acquisition.optimize_acqf", _collapsing_optimize_acqf(duplicate)
        )

        with pytest.raises(SearchSpaceExhaustedError):
            optimize_acquisition(
                acqf=_SumAcquisition(),
                bounds=torch.tensor([[0.0], [1.0]], dtype=torch.float64),
                batch_size=1,
                num_restarts=3,
                raw_samples=16,
                spec=_one_discrete_param_spec(values=[0.0, 2.0], bounds=(0.0, 1.0)),
                x_avoid=torch.tensor([[0.0]], dtype=torch.float64),
            )

    def test_fallback_sampled_hybrid_stays_feasible_after_snapping(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Snapped samples in a hybrid space are re-validated for feasibility.

        Polytope samples satisfy ``x0 <= 0.6`` in relaxed coordinates, but the
        discrete column snaps raw values above 0.5 to the infeasible grid
        point 1.0; those rows must be filtered so the returned canonical
        candidate still satisfies the constraint.
        """
        duplicate = torch.tensor([[[0.0, 0.5]]], dtype=torch.float64)
        monkeypatch.setattr(
            "bo_engine.acquisition.optimize_acqf", _collapsing_optimize_acqf(duplicate)
        )
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x0", type=ParameterType.DISCRETE, values=[0.0, 1.0]),
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )

        candidates, _values = optimize_acquisition(
            acqf=_SumAcquisition(),
            bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
            batch_size=1,
            num_restarts=3,
            raw_samples=16,
            spec=spec,
            x_avoid=duplicate[0],
            inequality_constraints=[
                (
                    torch.tensor([0]),
                    torch.tensor([-1.0], dtype=torch.float64),
                    -0.6,
                )
            ],
            random_seed=11,
        )

        assert candidates.shape == (1, 2)
        assert float(candidates[0, 0]) == 0.0
        assert torch.all(candidates >= 0.0)
        assert torch.all(candidates <= 1.0)
        distance = torch.linalg.vector_norm(candidates - duplicate[0]).item()
        assert distance >= DUPLICATE_DETECTION_TOLERANCE

    def test_fallback_error_without_exhaustion_claim_when_samples_all_seen(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the sampled cloud covers nothing unseen, the error says so.

        A continuous space can never be proven exhausted, so the sampled
        branch must not raise the campaign-terminating exhaustion signal —
        only a ``RuntimeError`` stating the sample found no unseen point.
        """
        duplicate = torch.tensor([[[0.3, 0.3]]], dtype=torch.float64)
        monkeypatch.setattr(
            "bo_engine.acquisition.optimize_acqf", _collapsing_optimize_acqf(duplicate)
        )

        def seen_only_sobol(d: int, n: int, _seed: int, bounds: torch.Tensor) -> torch.Tensor:
            return torch.full((n, d), 0.3, dtype=bounds.dtype, device=bounds.device)

        monkeypatch.setattr("bo_engine.acquisition.create_reproducible_sobol", seen_only_sobol)

        with pytest.raises(RuntimeError, match="unseen alternative") as excinfo:
            optimize_acquisition(
                acqf=_SumAcquisition(),
                bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
                batch_size=1,
                num_restarts=3,
                raw_samples=16,
                x_avoid=duplicate[0],
            )
        assert not isinstance(excinfo.value, SearchSpaceExhaustedError)


class TestDomainBoundsSeparation:
    """Exhaustion and validity are judged against campaign domain bounds.

    A TuRBO trust region is a soft search heuristic: when it falls between
    grid coordinates it contains no executable point, but the campaign is
    not exhausted — terminating it would be wrong (Eriksson et al., 2019,
    TuRBO restarts shrunken regions instead of stopping the campaign).
    """

    def test_trust_region_without_grid_points_does_not_exhaust(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A trust region between grid points must not report exhaustion."""
        duplicate = torch.tensor([[[0.5]]], dtype=torch.float64)
        monkeypatch.setattr(
            "bo_engine.acquisition.optimize_acqf", _collapsing_optimize_acqf(duplicate)
        )

        candidates, _values = optimize_acquisition(
            acqf=_SumAcquisition(),
            bounds=torch.tensor([[0.4], [0.6]], dtype=torch.float64),
            batch_size=1,
            num_restarts=3,
            raw_samples=16,
            spec=_one_discrete_param_spec(values=[0.0, 1.0, 2.0], bounds=None),
            x_avoid=torch.tensor([[0.0]], dtype=torch.float64),
            domain_bounds=torch.tensor([[0.0], [2.0]], dtype=torch.float64),
        )

        # 0.5 snaps to the evaluated 0.0; the trust region [0.4, 0.6] holds no
        # grid point, so the fallback searches the domain: best unseen is 2.0.
        assert torch.equal(candidates, torch.tensor([[2.0]], dtype=torch.float64))

    def test_trust_region_grid_points_are_preferred(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Locality is kept when the trust region holds an unseen grid point."""
        duplicate = torch.tensor([[[0.4]]], dtype=torch.float64)
        monkeypatch.setattr(
            "bo_engine.acquisition.optimize_acqf", _collapsing_optimize_acqf(duplicate)
        )

        candidates, _values = optimize_acquisition(
            acqf=_SumAcquisition(),
            bounds=torch.tensor([[0.9], [1.1]], dtype=torch.float64),
            batch_size=1,
            num_restarts=3,
            raw_samples=16,
            spec=_one_discrete_param_spec(values=[0.0, 1.0, 2.0], bounds=None),
            x_avoid=torch.tensor([[0.0]], dtype=torch.float64),
            domain_bounds=torch.tensor([[0.0], [2.0]], dtype=torch.float64),
        )

        # Unseen grid points are {1, 2}; 2.0 ranks higher on the acquisition
        # but lies outside the trust region — the local 1.0 is preferred.
        assert torch.equal(candidates, torch.tensor([[1.0]], dtype=torch.float64))

    def test_local_restart_beats_globally_better_outside_region(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Restart selection prefers an unseen grid point inside the region.

        The globally best canonical restart (0, outside the trust region)
        must lose to the locally executable 1 — the same locality policy the
        enumerated fallback applies.
        """
        restart_candidates = torch.tensor(
            [[[0.45]], [[0.95]]],
            dtype=torch.float64,
        )
        raw_values = torch.tensor([10.0, 9.0], dtype=torch.float64)

        def fake_optimize_acqf(**_kwargs):
            return restart_candidates, raw_values

        monkeypatch.setattr("bo_engine.acquisition.optimize_acqf", fake_optimize_acqf)

        candidates, values = optimize_acquisition(
            acqf=_PeakAcquisition(peak=0.0),
            bounds=torch.tensor([[0.4], [1.1]], dtype=torch.float64),
            batch_size=1,
            num_restarts=2,
            raw_samples=16,
            spec=_one_discrete_param_spec(values=[0.0, 1.0, 2.0], bounds=None),
            domain_bounds=torch.tensor([[0.0], [2.0]], dtype=torch.float64),
        )

        # Snapped restarts are {0, 1}; acqf(0) = 0 beats acqf(1) = -1, but 0
        # lies outside [0.4, 1.1] — the local grid point 1 must win.
        assert torch.equal(candidates, torch.tensor([[1.0]], dtype=torch.float64))
        assert torch.allclose(values, torch.tensor([-1.0], dtype=torch.float64))

    def test_no_local_restart_widens_to_domain_with_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """With no snapped restart inside the region, domain-wide is allowed."""
        restart_candidates = torch.tensor(
            [[[0.41]], [[0.44]]],
            dtype=torch.float64,
        )
        raw_values = torch.tensor([10.0, 9.0], dtype=torch.float64)

        def fake_optimize_acqf(**_kwargs):
            return restart_candidates, raw_values

        monkeypatch.setattr("bo_engine.acquisition.optimize_acqf", fake_optimize_acqf)

        caplog.set_level(logging.WARNING, logger="bo_engine.acquisition")
        candidates, _values = optimize_acquisition(
            acqf=_SumAcquisition(),
            bounds=torch.tensor([[0.4], [0.45]], dtype=torch.float64),
            batch_size=1,
            num_restarts=2,
            raw_samples=16,
            spec=_one_discrete_param_spec(values=[0.0, 1.0, 2.0], bounds=None),
            domain_bounds=torch.tensor([[0.0], [2.0]], dtype=torch.float64),
        )

        # Both restarts snap to 0.0, outside [0.4, 0.45] — still executable.
        assert torch.equal(candidates, torch.tensor([[0.0]], dtype=torch.float64))
        assert any("full campaign domain" in r.message for r in caplog.records)

    def test_sampled_fallback_widens_to_domain_when_no_local_candidate(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Snapped samples all outside the region widen to the domain."""
        duplicate = torch.tensor([[[0.45, 0.5]]], dtype=torch.float64)
        monkeypatch.setattr(
            "bo_engine.acquisition.optimize_acqf", _collapsing_optimize_acqf(duplicate)
        )
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="d", type=ParameterType.DISCRETE, values=[0.0, 1.0]),
                ParameterSpec(name="c", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )

        caplog.set_level(logging.WARNING, logger="bo_engine.acquisition")
        candidates, _values = optimize_acquisition(
            acqf=_SumAcquisition(),
            bounds=torch.tensor([[0.4, 0.0], [0.6, 1.0]], dtype=torch.float64),
            batch_size=1,
            num_restarts=3,
            raw_samples=16,
            spec=spec,
            x_avoid=duplicate[0],
            domain_bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
        )

        # Every sampled d in [0.4, 0.6] snaps to 0 or 1 — both outside the
        # region's d-range — so selection widens to the campaign domain.
        assert float(candidates[0, 0]) in {0.0, 1.0}
        assert torch.all(candidates >= 0.0)
        assert torch.all(candidates <= 1.0)
        assert any("full campaign domain" in r.message for r in caplog.records)

    def test_exhaustion_requires_the_whole_domain_to_be_evaluated(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With every domain grid point avoided, exhaustion is still raised."""
        duplicate = torch.tensor([[[0.5]]], dtype=torch.float64)
        monkeypatch.setattr(
            "bo_engine.acquisition.optimize_acqf", _collapsing_optimize_acqf(duplicate)
        )

        with pytest.raises(SearchSpaceExhaustedError):
            optimize_acquisition(
                acqf=_SumAcquisition(),
                bounds=torch.tensor([[0.4], [0.6]], dtype=torch.float64),
                batch_size=1,
                num_restarts=3,
                raw_samples=16,
                spec=_one_discrete_param_spec(values=[0.0, 1.0, 2.0], bounds=None),
                x_avoid=torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float64),
                domain_bounds=torch.tensor([[0.0], [2.0]], dtype=torch.float64),
            )


class TestConstraintCoupledDiscreteFallback:
    """Constraints coupling discrete and continuous columns need conditioning.

    A relaxed sample satisfying ``d + c = 1`` almost never still does after
    ``d`` snaps to its grid, so snap-and-filter deterministically loses the
    feasible set. Feasible candidates are generated per canonical discrete
    assignment by substituting it into the constraints and sampling the
    reduced continuous subproblem (BoTorch ``get_polytope_samples``).
    """

    def _hybrid_spec(self, discrete_values: list[float]) -> OptimizationSpec:
        return OptimizationSpec(
            parameters=[
                ParameterSpec(name="d", type=ParameterType.DISCRETE, values=discrete_values),
                ParameterSpec(name="c", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )

    def test_equality_coupled_fallback_finds_the_unseen_assignment(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``d + c = 1`` with (0, 1) avoided must yield exactly (1, 0)."""
        duplicate = torch.tensor([[[0.0, 1.0]]], dtype=torch.float64)
        monkeypatch.setattr(
            "bo_engine.acquisition.optimize_acqf", _collapsing_optimize_acqf(duplicate)
        )

        candidates, _values = optimize_acquisition(
            acqf=_SumAcquisition(),
            bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
            batch_size=1,
            num_restarts=3,
            raw_samples=16,
            spec=self._hybrid_spec([0.0, 1.0]),
            x_avoid=duplicate[0],
            equality_constraints=[
                (
                    torch.tensor([0, 1]),
                    torch.tensor([1.0, 1.0], dtype=torch.float64),
                    1.0,
                )
            ],
            random_seed=5,
        )

        assert torch.allclose(
            candidates, torch.tensor([[1.0, 0.0]], dtype=torch.float64), atol=1e-6
        )

    def test_only_the_feasible_assignment_is_generated(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Assignments whose reduced continuous subproblem is empty are skipped.

        Under ``d + c = 2`` with ``c`` in [0, 1], the assignment d=0 needs
        c=2 (infeasible) and d=1 needs the evaluated c=1 — only d=2 with c=0
        remains.
        """
        duplicate = torch.tensor([[[1.0, 1.0]]], dtype=torch.float64)
        monkeypatch.setattr(
            "bo_engine.acquisition.optimize_acqf", _collapsing_optimize_acqf(duplicate)
        )

        candidates, _values = optimize_acquisition(
            acqf=_SumAcquisition(),
            bounds=torch.tensor([[0.0, 0.0], [2.0, 1.0]], dtype=torch.float64),
            batch_size=1,
            num_restarts=3,
            raw_samples=16,
            spec=self._hybrid_spec([0.0, 1.0, 2.0]),
            x_avoid=duplicate[0],
            equality_constraints=[
                (
                    torch.tensor([0, 1]),
                    torch.tensor([1.0, 1.0], dtype=torch.float64),
                    2.0,
                )
            ],
            random_seed=5,
        )

        assert torch.allclose(
            candidates, torch.tensor([[2.0, 0.0]], dtype=torch.float64), atol=1e-6
        )

    def test_huge_assignment_space_is_subsampled_within_budget(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Above the assignment budget, conditioning subsamples — never reverts.

        Snap-and-filter destroys discrete-coupled equalities, so a 10,001-value
        grid must still be handled by conditional generation: a seeded
        assignment subsample bounded by ``ACQF_FALLBACK_MAX_ASSIGNMENTS``,
        with one sampler call per selected assignment at most.
        """
        from bo_engine import acquisition as acquisition_module
        from bo_engine.constants import ACQF_FALLBACK_MAX_ASSIGNMENTS

        grid = [i / 10000 for i in range(10001)]
        duplicate = torch.tensor([[[0.0, 1.0]]], dtype=torch.float64)
        monkeypatch.setattr(
            "bo_engine.acquisition.optimize_acqf", _collapsing_optimize_acqf(duplicate)
        )

        sampler_calls = {"count": 0}
        real_get_polytope_samples = acquisition_module.get_polytope_samples

        def counting_get_polytope_samples(**kwargs):
            sampler_calls["count"] += 1
            return real_get_polytope_samples(**kwargs)

        monkeypatch.setattr(
            "bo_engine.acquisition.get_polytope_samples", counting_get_polytope_samples
        )

        candidates, _values = optimize_acquisition(
            acqf=_SumAcquisition(),
            bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
            batch_size=1,
            num_restarts=3,
            raw_samples=16,
            spec=self._hybrid_spec(grid),
            x_avoid=duplicate[0],
            equality_constraints=[
                (
                    torch.tensor([0, 1]),
                    torch.tensor([1.0, 1.0], dtype=torch.float64),
                    1.0,
                )
            ],
            random_seed=5,
        )

        assert sampler_calls["count"] <= ACQF_FALLBACK_MAX_ASSIGNMENTS
        d_value, c_value = float(candidates[0, 0]), float(candidates[0, 1])
        # Canonical: d sits exactly on the declared grid.
        assert abs(d_value * 10000 - round(d_value * 10000)) < 1e-6
        # Feasible: the coupled equality holds at the executable coordinate.
        assert abs(d_value + c_value - 1.0) < 1e-6
        # Unseen: not the avoided (0, 1).
        distance = torch.linalg.vector_norm(candidates - duplicate[0]).item()
        assert distance >= DUPLICATE_DETECTION_TOLERANCE

    def test_all_feasible_assignments_avoided_raises_without_exhaustion_claim(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A hybrid space cannot prove exhaustion — plain RuntimeError only."""
        duplicate = torch.tensor([[[0.0, 1.0]]], dtype=torch.float64)
        monkeypatch.setattr(
            "bo_engine.acquisition.optimize_acqf", _collapsing_optimize_acqf(duplicate)
        )

        with pytest.raises(RuntimeError, match="unseen alternative") as excinfo:
            optimize_acquisition(
                acqf=_SumAcquisition(),
                bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
                batch_size=1,
                num_restarts=3,
                raw_samples=16,
                spec=self._hybrid_spec([0.0, 1.0]),
                x_avoid=torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64),
                equality_constraints=[
                    (
                        torch.tensor([0, 1]),
                        torch.tensor([1.0, 1.0], dtype=torch.float64),
                        1.0,
                    )
                ],
                random_seed=5,
            )
        assert not isinstance(excinfo.value, SearchSpaceExhaustedError)


class TestSelectAssignments:
    """Assignment subsampling fills its budget with distinct assignments.

    Drawing with replacement and deduplicating underfills the budget
    (coupon-collector effect: ~40 unique from 64 draws over 65 assignments),
    needlessly raising the odds of missing the only feasible assignment.
    Sampling flat indices without replacement yields exactly
    ``min(cardinality, budget)`` distinct assignments at O(budget) cost.
    """

    def test_budget_is_filled_exactly_with_unique_assignments(self) -> None:
        axis = [float(v) for v in range(65)]
        assignments = _select_assignments([(0, axis)], budget=64, seed=0)

        assert len(assignments) == 64
        assert len(set(assignments)) == 64
        assert all(value in axis for (value,) in assignments)

    def test_same_seed_selects_identical_assignments(self) -> None:
        axes = [(0, [float(v) for v in range(100)]), (1, [0.0, 0.5, 1.0])]
        first = _select_assignments(axes, budget=64, seed=42)
        second = _select_assignments(axes, budget=64, seed=42)

        assert first == second

    def test_huge_cardinality_stays_bounded_and_unique(self) -> None:
        """1e8 assignments: exactly the budget, without materializing the space."""
        axes = [
            (0, [float(v) for v in range(10_000)]),
            (1, [float(v) for v in range(10_000)]),
        ]
        assignments = _select_assignments(axes, budget=64, seed=7)

        assert len(assignments) == 64
        assert len(set(assignments)) == 64
        assert all(d in axes[0][1] and c in axes[1][1] for d, c in assignments)


class TestUnenforcedAvoidancePathsLogDebug:
    """``batch_size > 1`` and mixed spaces document dropped ``x_avoid``."""

    def test_q2_continuous_logs_unenforced_x_avoid(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Sequential-greedy batches state at DEBUG that x_avoid is ignored."""
        batch = torch.tensor([[0.2, 0.4], [0.6, 0.8]], dtype=torch.float64)

        def fake_optimize_acqf(**_kwargs):
            return batch, torch.ones(2, dtype=torch.float64)

        monkeypatch.setattr("bo_engine.acquisition.optimize_acqf", fake_optimize_acqf)

        caplog.set_level(logging.DEBUG, logger="bo_engine.acquisition")
        candidates, _values = optimize_acquisition(
            acqf=_SumAcquisition(),
            bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64),
            batch_size=2,
            num_restarts=2,
            raw_samples=16,
            x_avoid=torch.tensor([[0.2, 0.4]], dtype=torch.float64),
        )

        assert torch.equal(candidates, batch)
        assert any("not enforced for continuous acquisition" in r.message for r in caplog.records)

    def test_mixed_space_logs_unenforced_x_avoid(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The mixed-space dispatch states at DEBUG that x_avoid is ignored."""
        mixed_result = (
            torch.zeros(1, 3, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
        )
        monkeypatch.setattr(
            "bo_engine.acquisition._optimize_mixed", lambda *_args, **_kwargs: mixed_result
        )
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x0", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                ParameterSpec(name="c0", type=ParameterType.CATEGORICAL, categories=["a", "b"]),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )

        caplog.set_level(logging.DEBUG, logger="bo_engine.acquisition")
        candidates, _values = optimize_acquisition(
            acqf=_SumAcquisition(),
            bounds=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64),
            batch_size=1,
            spec=spec,
            x_avoid=torch.tensor([[0.5, 1.0, 0.0]], dtype=torch.float64),
        )

        assert torch.equal(candidates, mixed_result[0])
        assert any("mixed-space" in r.message for r in caplog.records)
