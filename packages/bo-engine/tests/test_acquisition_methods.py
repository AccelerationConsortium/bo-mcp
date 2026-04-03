"""Tests for acquisition function methods (v1.0.1 and v1.1)."""

import numpy as np
import pytest
import torch

from bo_engine import (
    AcquisitionMethod,
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
    create_acquisition,
    create_and_fit_model,
    create_and_fit_single_task_model,
    create_multi_objective_acquisition,
    create_single_objective_acquisition,
    generate_next_batch,
    get_reference_point,
    optimize_acquisition,
)


@pytest.fixture
def train_data(torch_rng) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create sample training data."""
    train_x = torch.rand(10, 2, dtype=torch.double)
    train_y = torch.rand(10, 2, dtype=torch.double)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    return train_x, train_y, bounds


class TestSingleObjectiveAcquisitionMethods:
    """Test single-objective acquisition methods."""

    def test_qlognei_acquisition(self, torch_rng) -> None:
        """Test qLogNEI (Noisy Expected Improvement) acquisition."""
        train_x = torch.rand(10, 2, dtype=torch.double)
        train_y = torch.rand(10, 1, dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        model = create_and_fit_single_task_model(train_x, train_y, bounds)
        acqf = create_single_objective_acquisition(
            model=model,
            train_x=train_x,
            train_y=train_y,
            use_noisy=True,  # qLogNEI
        )

        # Should be callable and produce valid outputs
        test_x = torch.rand(5, 1, 2, dtype=torch.double)  # q=1, d=2
        values = acqf(test_x)
        assert values.shape == (5,)

    def test_qlogei_acquisition(self, torch_rng) -> None:
        """Test qLogEI (Expected Improvement) acquisition."""
        train_x = torch.rand(10, 2, dtype=torch.double)
        train_y = torch.rand(10, 1, dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        model = create_and_fit_single_task_model(train_x, train_y, bounds)
        acqf = create_single_objective_acquisition(
            model=model,
            train_x=train_x,
            train_y=train_y,
            use_noisy=False,  # qLogEI
            best_f=train_y.min().item(),
        )

        # Should be callable and produce valid outputs
        test_x = torch.rand(5, 1, 2, dtype=torch.double)
        values = acqf(test_x)
        assert values.shape == (5,)


class TestMultiObjectiveAcquisitionMethods:
    """Test multi-objective acquisition methods."""

    def test_qlognehvi_acquisition(
        self, train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        """Test qLogNEHVI (Noisy Expected Hypervolume Improvement) acquisition."""
        train_x, train_y, bounds = train_data

        model = create_and_fit_model(train_x, train_y, bounds)
        minimize_mask = torch.tensor([True, True], dtype=torch.bool)
        ref_point = get_reference_point(train_y, minimize_mask)

        acqf = create_multi_objective_acquisition(
            model=model,
            ref_point=ref_point,
            train_x=train_x,
            train_y=train_y,
            method=AcquisitionMethod.HYPERVOLUME_IMPROVEMENT,
        )

        # Should be callable and produce valid outputs
        test_x = torch.rand(5, 1, 2, dtype=torch.double)
        values = acqf(test_x)
        assert values.shape == (5,)

    def test_qlognparego_acquisition(
        self, train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        """Test qLogNParEGO (Parallel EGO with Chebyshev scalarization) acquisition."""
        train_x, train_y, bounds = train_data

        model = create_and_fit_model(train_x, train_y, bounds)
        minimize_mask = torch.tensor([True, True], dtype=torch.bool)
        ref_point = get_reference_point(train_y, minimize_mask)

        acqf = create_multi_objective_acquisition(
            model=model,
            ref_point=ref_point,
            train_x=train_x,
            train_y=train_y,
            method=AcquisitionMethod.SCALARIZED_MULTI_OBJ,
        )

        # Should be callable and produce valid outputs
        test_x = torch.rand(5, 1, 2, dtype=torch.double)
        values = acqf(test_x)
        assert values.shape == (5,)


class TestUnifiedAcquisitionCreation:
    """Test unified acquisition creation function."""

    def test_auto_selects_single_objective(self, torch_rng) -> None:
        """Test that AUTO selects qLogNEI for single-objective."""
        train_x = torch.rand(10, 2, dtype=torch.double)
        train_y = torch.rand(10, 1, dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        model = create_and_fit_single_task_model(train_x, train_y, bounds)
        acqf = create_acquisition(
            model=model,
            ref_point=None,
            train_x=train_x,
            train_y=train_y,
            n_objectives=1,
            method=AcquisitionMethod.AUTO,
        )

        assert acqf is not None

    def test_auto_selects_multi_objective(
        self, train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        """Test that AUTO selects qLogNEHVI for multi-objective."""
        train_x, train_y, bounds = train_data

        model = create_and_fit_model(train_x, train_y, bounds)
        minimize_mask = torch.tensor([True, True], dtype=torch.bool)
        ref_point = get_reference_point(train_y, minimize_mask)

        acqf = create_acquisition(
            model=model,
            ref_point=ref_point,
            train_x=train_x,
            train_y=train_y,
            n_objectives=2,
            method=AcquisitionMethod.AUTO,
        )

        assert acqf is not None

    def test_explicit_method_selection(
        self, train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        """Test explicit acquisition method selection."""
        train_x, train_y, bounds = train_data

        model = create_and_fit_model(train_x, train_y, bounds)
        minimize_mask = torch.tensor([True, True], dtype=torch.bool)
        ref_point = get_reference_point(train_y, minimize_mask)

        # Explicit qLogNParEGO selection
        acqf = create_acquisition(
            model=model,
            ref_point=ref_point,
            train_x=train_x,
            train_y=train_y,
            n_objectives=2,
            method=AcquisitionMethod.SCALARIZED_MULTI_OBJ,
        )

        assert acqf is not None


class TestAcquisitionOptimization:
    """Test acquisition function optimization."""

    def test_optimize_single_objective_acquisition(self, torch_rng) -> None:
        """Test optimization of single-objective acquisition."""
        train_x = torch.rand(10, 2, dtype=torch.double)
        train_y = torch.rand(10, 1, dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        model = create_and_fit_single_task_model(train_x, train_y, bounds)
        acqf = create_single_objective_acquisition(
            model=model,
            train_x=train_x,
            train_y=train_y,
        )

        candidates, values = optimize_acquisition(
            acqf=acqf,
            bounds=bounds,
            batch_size=3,
            num_restarts=5,
            raw_samples=64,
        )

        assert candidates.shape == (3, 2)
        assert torch.all(candidates >= 0) and torch.all(candidates <= 1)

    def test_optimize_multi_objective_acquisition(
        self, train_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        """Test optimization of multi-objective acquisition."""
        train_x, train_y, bounds = train_data

        model = create_and_fit_model(train_x, train_y, bounds)
        minimize_mask = torch.tensor([True, True], dtype=torch.bool)
        ref_point = get_reference_point(train_y, minimize_mask)

        acqf = create_multi_objective_acquisition(
            model=model,
            ref_point=ref_point,
            train_x=train_x,
            train_y=train_y,
        )

        candidates, values = optimize_acquisition(
            acqf=acqf,
            bounds=bounds,
            batch_size=3,
            num_restarts=5,
            raw_samples=64,
        )

        assert candidates.shape == (3, 2)
        assert torch.all(candidates >= 0) and torch.all(candidates <= 1)


class TestAcquisitionInWorkflow:
    """Test acquisition methods in full workflow."""

    def test_qlogparego_in_workflow(self, rng: np.random.Generator) -> None:
        """Test qLogNParEGO in a complete optimization workflow."""
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
            acquisition_method=AcquisitionMethod.SCALARIZED_MULTI_OBJ,
        )

        observations = [
            ObservationData(
                parameter_values={"x1": 0.2, "x2": 0.3},
                objective_values={"f1": 1.0, "f2": 2.0},
            ),
            ObservationData(
                parameter_values={"x1": 0.8, "x2": 0.7},
                objective_values={"f1": 2.0, "f2": 1.0},
            ),
            ObservationData(
                parameter_values={"x1": 0.5, "x2": 0.5},
                objective_values={"f1": 1.5, "f2": 1.5},
            ),
            ObservationData(
                parameter_values={"x1": 0.3, "x2": 0.8},
                objective_values={"f1": 1.2, "f2": 1.8},
            ),
            ObservationData(
                parameter_values={"x1": 0.6, "x2": 0.2},
                objective_values={"f1": 1.8, "f2": 1.2},
            ),
        ]

        suggestions, _ = generate_next_batch(spec, observations, iteration=1, rng=rng)

        assert len(suggestions) == 2
        assert all(s.generation_method == "bo" for s in suggestions)
        assert all(s.acquisition_function == "scalarized_multi_objective" for s in suggestions)

    def test_qlognehvi_default_in_workflow(self, rng: np.random.Generator) -> None:
        """Test that qLogNEHVI is the default for multi-objective."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ],
            objectives=[
                ObjectiveSpec(name="f1", minimize=True),
                ObjectiveSpec(name="f2", minimize=False),
            ],
            batch_size=2,
            acquisition_method=AcquisitionMethod.AUTO,  # Should select qLogNEHVI
        )

        observations = [
            ObservationData(
                parameter_values={"x": 0.2},
                objective_values={"f1": 1.0, "f2": 2.0},
            ),
            ObservationData(
                parameter_values={"x": 0.8},
                objective_values={"f1": 2.0, "f2": 1.0},
            ),
            ObservationData(
                parameter_values={"x": 0.5},
                objective_values={"f1": 1.5, "f2": 1.5},
            ),
        ]

        suggestions, _ = generate_next_batch(spec, observations, iteration=1, rng=rng)

        assert len(suggestions) == 2
        assert all(s.acquisition_function == "hypervolume_improvement" for s in suggestions)
