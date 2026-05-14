"""Tests for feature importance module."""

import logging

import pytest
import torch

from bo_engine.feature_importance import (
    compute_lengthscale_importance,
    extract_lengthscales,
)
from bo_engine.models import create_and_fit_model


class TestFeatureImportance:
    """Tests for feature importance functions."""

    def test_extract_lengthscales(self, torch_rng):
        """extract_lengthscales returns lengthscales from fitted model."""
        # Simple 2D problem with 1 objective
        X = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        Y = torch.tensor([[1.0], [2.0], [1.5], [2.5]])  # X1 more important than X2
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

        model = create_and_fit_model(X, Y, bounds)
        lengthscales = extract_lengthscales(model)

        assert "objective_0" in lengthscales
        assert lengthscales["objective_0"].shape == (2,)
        # All lengthscales should be positive
        assert (lengthscales["objective_0"] > 0).all()

    def test_compute_lengthscale_importance(self):
        """compute_lengthscale_importance normalizes inverse lengthscales."""
        # Smaller lengthscale = more important
        lengthscales = {
            "objective_0": torch.tensor([0.5, 2.0]),  # x1 more important
        }
        param_names = ["x1", "x2"]

        importance = compute_lengthscale_importance(lengthscales, param_names)

        # Check structure
        assert "by_objective" in importance
        assert "aggregate" in importance
        assert "objective_0" in importance["by_objective"]

        # x1 should be more important (smaller lengthscale)
        obj_imp = importance["by_objective"]["objective_0"]
        assert obj_imp["x1"] > obj_imp["x2"]

        # Should sum to 1
        total = sum(importance["aggregate"].values())
        assert abs(total - 1.0) < 0.01

    def test_importance_multi_objective(self):
        """Importance aggregates correctly across multiple objectives."""
        lengthscales = {
            "objective_0": torch.tensor([0.5, 1.0]),  # x1 more important
            "objective_1": torch.tensor([1.0, 0.5]),  # x2 more important
        }
        param_names = ["x1", "x2"]

        importance = compute_lengthscale_importance(lengthscales, param_names)

        # Each objective should favor different params
        assert importance["by_objective"]["objective_0"]["x1"] > 0.5
        assert importance["by_objective"]["objective_1"]["x2"] > 0.5

        # Aggregate should be roughly equal
        agg = importance["aggregate"]
        assert abs(agg["x1"] - agg["x2"]) < 0.2

    def test_near_zero_lengthscale_is_clamped(self, caplog: pytest.LogCaptureFixture) -> None:
        """A near-zero lengthscale is clamped to ``NUMERICAL_EPSILON`` before reciprocal.

        Without the clamp, ``1.0 / 0`` would yield ``inf``, which then
        propagates through the normalization as ``nan`` and corrupts the
        downstream importance dict. The clamp keeps the importance
        finite, the warning surface keeps a poorly-conditioned GP fit
        visible to operators.

        Reference: numerical-safety guidance in the project CLAUDE.md
        (do not hardcode bare zero comparisons; clamp before reciprocal).
        """
        lengthscales = {
            "objective_0": torch.tensor([0.0, 1.0]),
        }
        with caplog.at_level(logging.WARNING, logger="bo_engine.feature_importance"):
            importance = compute_lengthscale_importance(lengthscales, ["x1", "x2"])

        # Importance values must remain finite and sum to 1.
        agg = importance["aggregate"]
        assert all(abs(v) < float("inf") for v in agg.values())
        assert abs(sum(agg.values()) - 1.0) < 1e-3
        # Clamp fires a warning that explicitly names the objective.
        assert any("objective_0" in rec.message for rec in caplog.records)

    def test_end_to_end_importance(self, torch_rng):
        """Full pipeline: fit model and compute importance."""
        # Create data where x1 clearly matters more than x2
        X = torch.tensor(
            [
                [0.0, 0.5],
                [0.25, 0.2],
                [0.5, 0.8],
                [0.75, 0.3],
                [1.0, 0.6],
            ]
        )
        # Y strongly depends on x1, weakly on x2
        Y = X[:, 0:1] * 10 + X[:, 1:2] * 0.1
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]])

        model = create_and_fit_model(X, Y, bounds)
        lengthscales = extract_lengthscales(model)
        importance = compute_lengthscale_importance(lengthscales, ["x1", "x2"])

        # x1 should be identified as more important
        assert importance["aggregate"]["x1"] > importance["aggregate"]["x2"]
