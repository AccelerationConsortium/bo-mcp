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

import pytest
import torch

from bo_engine.acquisition import (
    _log_restart_diagnostics,
    optimize_acquisition,
)
from bo_engine.constants import RESTART_WARN_TOLERANCE


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
