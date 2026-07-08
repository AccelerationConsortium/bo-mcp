"""Tests for constraint handling in Bayesian Optimization.

The constraints module provides utilities for handling input constraints
on parameters, including:
- Sum constraints (parameters must sum to a value)
- Linear constraints (weighted sums must satisfy inequalities)

These tests verify:
1. Constraint callable creation for each constraint type
2. Parameter index resolution (including categorical one-hot encoding)
3. Sum constraint projection
4. Integration with BoTorch optimization
"""

import pytest
import torch
from torch import Tensor

from bo_engine.constraints import (
    _get_parameter_indices,
    apply_sum_constraint,
    build_botorch_linear_constraints,
    create_constraint_callable,
    create_constraints_list,
)
from bo_engine.types import (
    ConstraintSpec,
    ConstraintType,
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)


class TestConstraintSpec:
    """Test ConstraintSpec dataclass."""

    def test_sum_equals_constraint(self) -> None:
        """Sum equals constraint can be created."""
        constraint = ConstraintSpec(
            type=ConstraintType.SUM_EQUALS,
            parameters=["x1", "x2", "x3"],
            value=1.0,
        )

        assert constraint.type == ConstraintType.SUM_EQUALS
        assert constraint.parameters == ["x1", "x2", "x3"]
        assert constraint.value == 1.0
        assert constraint.coefficients is None

    def test_sum_less_than_constraint(self) -> None:
        """Sum less than constraint can be created."""
        constraint = ConstraintSpec(
            type=ConstraintType.SUM_LESS_THAN,
            parameters=["a", "b"],
            value=0.5,
        )

        assert constraint.type == ConstraintType.SUM_LESS_THAN
        assert constraint.value == 0.5

    def test_sum_greater_than_constraint(self) -> None:
        """Sum greater than constraint can be created."""
        constraint = ConstraintSpec(
            type=ConstraintType.SUM_GREATER_THAN,
            parameters=["p1"],
            value=0.2,
        )

        assert constraint.type == ConstraintType.SUM_GREATER_THAN

    def test_linear_constraint(self) -> None:
        """Linear constraint can be created with coefficients."""
        constraint = ConstraintSpec(
            type=ConstraintType.LINEAR,
            parameters=["x", "y", "z"],
            value=10.0,
            coefficients=[1.0, 2.0, 3.0],
        )

        assert constraint.type == ConstraintType.LINEAR
        assert constraint.coefficients == [1.0, 2.0, 3.0]


class TestGetParameterIndices:
    """Test parameter index resolution."""

    def test_continuous_parameters(self) -> None:
        """Resolves indices for continuous parameters."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="a", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
                ParameterSpec(name="b", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
                ParameterSpec(name="c", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )

        indices = _get_parameter_indices(["a", "c"], spec)
        assert indices == [0, 2]

    def test_discrete_parameters(self) -> None:
        """Resolves indices for discrete parameters."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x", type=ParameterType.DISCRETE, values=[1, 2, 3]),
                ParameterSpec(name="y", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
            ],
            objectives=[ObjectiveSpec(name="f", minimize=True)],
        )

        indices = _get_parameter_indices(["x"], spec)
        assert indices == [0]

    def test_categorical_parameters_one_hot(self) -> None:
        """Resolves all one-hot indices for categorical parameters."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
                ParameterSpec(
                    name="cat",
                    type=ParameterType.CATEGORICAL,
                    categories=["red", "green", "blue"],
                ),
                ParameterSpec(name="y", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
            ],
            objectives=[ObjectiveSpec(name="f", minimize=True)],
        )

        # Categorical with 3 categories becomes 3 one-hot columns
        indices = _get_parameter_indices(["cat"], spec)
        assert indices == [1, 2, 3]  # Indices for red, green, blue

    def test_mixed_parameters(self) -> None:
        """Resolves indices for mixed parameter types."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="a", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
                ParameterSpec(
                    name="b",
                    type=ParameterType.CATEGORICAL,
                    categories=["x", "y"],
                ),  # 2 columns
                ParameterSpec(name="c", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
            ],
            objectives=[ObjectiveSpec(name="f", minimize=True)],
        )

        # a=0, b=[1,2], c=3
        indices_a = _get_parameter_indices(["a"], spec)
        indices_b = _get_parameter_indices(["b"], spec)
        indices_c = _get_parameter_indices(["c"], spec)

        assert indices_a == [0]
        assert indices_b == [1, 2]
        assert indices_c == [3]


class TestSumEqualsConstraint:
    """Test SUM_EQUALS constraint callable."""

    @pytest.fixture
    def simple_spec(self) -> OptimizationSpec:
        """Simple spec with 3 continuous parameters."""
        return OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
                ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
                ParameterSpec(name="x3", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )

    def test_satisfied_constraint(self, simple_spec: OptimizationSpec) -> None:
        """Constraint satisfied when sum equals target."""
        constraint = ConstraintSpec(
            type=ConstraintType.SUM_EQUALS,
            parameters=["x1", "x2", "x3"],
            value=1.0,
        )

        callable_fn = create_constraint_callable(constraint, simple_spec)

        # Point where x1 + x2 + x3 = 1.0 (satisfied)
        x = torch.tensor([[0.3, 0.3, 0.4]], dtype=torch.double)
        result = callable_fn(x)

        # Should be >= 0 when satisfied (within epsilon)
        assert result.item() >= 0

    def test_violated_constraint(self, simple_spec: OptimizationSpec) -> None:
        """Constraint violated when sum differs from target."""
        constraint = ConstraintSpec(
            type=ConstraintType.SUM_EQUALS,
            parameters=["x1", "x2", "x3"],
            value=1.0,
        )

        callable_fn = create_constraint_callable(constraint, simple_spec)

        # Point where x1 + x2 + x3 = 0.6 (violated)
        x = torch.tensor([[0.2, 0.2, 0.2]], dtype=torch.double)
        result = callable_fn(x)

        # Should be < 0 when violated
        assert result.item() < 0

    def test_batch_evaluation(self, simple_spec: OptimizationSpec) -> None:
        """Constraint can be evaluated on batch of points."""
        constraint = ConstraintSpec(
            type=ConstraintType.SUM_EQUALS,
            parameters=["x1", "x2"],
            value=0.5,
        )

        callable_fn = create_constraint_callable(constraint, simple_spec)

        # Batch of 3 points
        x = torch.tensor(
            [
                [0.25, 0.25, 0.3],  # Satisfied (0.25 + 0.25 = 0.5)
                [0.3, 0.3, 0.4],  # Violated (0.3 + 0.3 = 0.6)
                [0.1, 0.4, 0.5],  # Satisfied (0.1 + 0.4 = 0.5)
            ],
            dtype=torch.double,
        )

        result = callable_fn(x)

        assert result.shape == (3,)
        assert result[0].item() >= 0  # Satisfied
        assert result[1].item() < 0  # Violated
        assert result[2].item() >= 0  # Satisfied


class TestSumLessThanConstraint:
    """Test SUM_LESS_THAN constraint callable."""

    @pytest.fixture
    def simple_spec(self) -> OptimizationSpec:
        """Simple spec with 3 continuous parameters."""
        return OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
                ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
                ParameterSpec(name="x3", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )

    def test_satisfied_constraint(self, simple_spec: OptimizationSpec) -> None:
        """Constraint satisfied when sum < target."""
        constraint = ConstraintSpec(
            type=ConstraintType.SUM_LESS_THAN,
            parameters=["x1", "x2"],
            value=0.8,
        )

        callable_fn = create_constraint_callable(constraint, simple_spec)

        # x1 + x2 = 0.5 < 0.8 (satisfied)
        x = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.double)
        result = callable_fn(x)

        assert result.item() > 0

    def test_violated_constraint(self, simple_spec: OptimizationSpec) -> None:
        """Constraint violated when sum >= target."""
        constraint = ConstraintSpec(
            type=ConstraintType.SUM_LESS_THAN,
            parameters=["x1", "x2"],
            value=0.4,
        )

        callable_fn = create_constraint_callable(constraint, simple_spec)

        # x1 + x2 = 0.5 > 0.4 (violated)
        x = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.double)
        result = callable_fn(x)

        assert result.item() < 0


class TestSumGreaterThanConstraint:
    """Test SUM_GREATER_THAN constraint callable."""

    @pytest.fixture
    def simple_spec(self) -> OptimizationSpec:
        """Simple spec with 3 continuous parameters."""
        return OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
                ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
                ParameterSpec(name="x3", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )

    def test_satisfied_constraint(self, simple_spec: OptimizationSpec) -> None:
        """Constraint satisfied when sum > target."""
        constraint = ConstraintSpec(
            type=ConstraintType.SUM_GREATER_THAN,
            parameters=["x1", "x2"],
            value=0.3,
        )

        callable_fn = create_constraint_callable(constraint, simple_spec)

        # x1 + x2 = 0.5 > 0.3 (satisfied)
        x = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.double)
        result = callable_fn(x)

        assert result.item() > 0

    def test_violated_constraint(self, simple_spec: OptimizationSpec) -> None:
        """Constraint violated when sum <= target."""
        constraint = ConstraintSpec(
            type=ConstraintType.SUM_GREATER_THAN,
            parameters=["x1", "x2"],
            value=0.6,
        )

        callable_fn = create_constraint_callable(constraint, simple_spec)

        # x1 + x2 = 0.5 < 0.6 (violated)
        x = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.double)
        result = callable_fn(x)

        assert result.item() < 0


class TestLinearConstraint:
    """Test LINEAR constraint callable."""

    @pytest.fixture
    def simple_spec(self) -> OptimizationSpec:
        """Simple spec with 3 continuous parameters."""
        return OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
                ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
                ParameterSpec(name="x3", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )

    def test_satisfied_constraint(self, simple_spec: OptimizationSpec) -> None:
        """Constraint satisfied when weighted sum <= value."""
        # 2*x1 + 3*x2 <= 1.0
        constraint = ConstraintSpec(
            type=ConstraintType.LINEAR,
            parameters=["x1", "x2"],
            value=1.0,
            coefficients=[2.0, 3.0],
        )

        callable_fn = create_constraint_callable(constraint, simple_spec)

        # 2*0.1 + 3*0.2 = 0.8 <= 1.0 (satisfied)
        x = torch.tensor([[0.1, 0.2, 0.5]], dtype=torch.double)
        result = callable_fn(x)

        assert result.item() > 0

    def test_violated_constraint(self, simple_spec: OptimizationSpec) -> None:
        """Constraint violated when weighted sum > value."""
        # 2*x1 + 3*x2 <= 0.5
        constraint = ConstraintSpec(
            type=ConstraintType.LINEAR,
            parameters=["x1", "x2"],
            value=0.5,
            coefficients=[2.0, 3.0],
        )

        callable_fn = create_constraint_callable(constraint, simple_spec)

        # 2*0.3 + 3*0.3 = 1.5 > 0.5 (violated)
        x = torch.tensor([[0.3, 0.3, 0.5]], dtype=torch.double)
        result = callable_fn(x)

        assert result.item() < 0

    def test_negative_coefficients(self, simple_spec: OptimizationSpec) -> None:
        """Linear constraint handles negative coefficients."""
        # x1 - x2 <= 0 (x1 <= x2)
        constraint = ConstraintSpec(
            type=ConstraintType.LINEAR,
            parameters=["x1", "x2"],
            value=0.0,
            coefficients=[1.0, -1.0],
        )

        callable_fn = create_constraint_callable(constraint, simple_spec)

        # 0.2 - 0.5 = -0.3 <= 0 (satisfied)
        x_sat = torch.tensor([[0.2, 0.5, 0.3]], dtype=torch.double)
        # 0.6 - 0.3 = 0.3 > 0 (violated)
        x_viol = torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.double)

        assert callable_fn(x_sat).item() > 0
        assert callable_fn(x_viol).item() < 0


class TestApplySumConstraint:
    """Test sum constraint projection."""

    def test_projects_to_target_sum(self) -> None:
        """Candidates are projected to sum to target."""
        candidates = torch.tensor(
            [
                [0.5, 0.3, 0.2],
                [0.8, 0.1, 0.1],
            ],
            dtype=torch.double,
        )

        param_indices = [0, 1, 2]
        target_sum = 1.0

        result = apply_sum_constraint(candidates, param_indices, target_sum)

        # Each row should sum to target
        for i in range(result.shape[0]):
            row_sum = result[i, param_indices].sum().item()
            assert abs(row_sum - target_sum) < 1e-6

    def test_projects_subset_of_params(self) -> None:
        """Only specified parameters are projected."""
        candidates = torch.tensor(
            [
                [0.4, 0.3, 0.5],
            ],
            dtype=torch.double,
        )

        param_indices = [0, 1]  # Only first two
        target_sum = 0.5

        result = apply_sum_constraint(candidates, param_indices, target_sum)

        # First two should sum to 0.5
        assert abs(result[0, 0].item() + result[0, 1].item() - 0.5) < 1e-6
        # Third should be unchanged
        assert result[0, 2].item() == 0.5

    def test_handles_zero_sum(self) -> None:
        """Handles edge case where current sum is near zero."""
        candidates = torch.tensor(
            [
                [0.0, 0.0, 0.5],
            ],
            dtype=torch.double,
        )

        param_indices = [0, 1]
        target_sum = 0.5

        # Should not divide by zero
        result = apply_sum_constraint(candidates, param_indices, target_sum)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        assert abs(result[0, 0].item() + result[0, 1].item() - target_sum) < 1e-6


class TestCreateConstraintsList:
    """Test creating list of constraint callables."""

    def test_multiple_constraints(self) -> None:
        """Creates callables for multiple constraints."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="a", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
                ParameterSpec(name="b", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
                ParameterSpec(name="c", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
            constraints=[
                ConstraintSpec(
                    type=ConstraintType.SUM_EQUALS,
                    parameters=["a", "b", "c"],
                    value=1.0,
                ),
                ConstraintSpec(
                    type=ConstraintType.SUM_LESS_THAN,
                    parameters=["a", "b"],
                    value=0.7,
                ),
            ],
        )

        constraints_list = create_constraints_list(spec)

        assert len(constraints_list) == 2
        assert all(callable(c) for c in constraints_list)

    def test_empty_constraints(self) -> None:
        """Returns empty list when no constraints."""
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
            constraints=[],
        )

        constraints_list = create_constraints_list(spec)
        assert constraints_list == []


class TestUnknownConstraintType:
    """Test error handling for unknown constraint types."""

    def test_all_constraint_types_are_handled(self) -> None:
        """Every ConstraintType member is either encoded or rejected loudly.

        The BoTorch feasibility-callable encoding exists for the arithmetic
        sum/linear family. The extended families (products, cardinality,
        set-based label constraints) are BayBE-only: the BoTorch backend
        vetoes them at intake via ``validate_capabilities``, and this
        builder must raise a clear, type-naming error — never silently
        fall through — if one reaches it anyway. A new enum member without
        either a handler or the documented rejection fails here.
        """
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
                ParameterSpec(name="y", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
            ],
            objectives=[ObjectiveSpec(name="f", minimize=True)],
        )
        botorch_callable_types = {
            ConstraintType.SUM_EQUALS,
            ConstraintType.SUM_LESS_THAN,
            ConstraintType.SUM_GREATER_THAN,
            ConstraintType.LINEAR,
        }

        for constraint_type in ConstraintType:
            constraint = ConstraintSpec(
                type=constraint_type,
                parameters=["x", "y"],
                value=1.0,
                coefficients=[1.0, 1.0] if constraint_type == ConstraintType.LINEAR else None,
            )
            if constraint_type in botorch_callable_types:
                callable_fn = create_constraint_callable(constraint, spec)
                assert callable(callable_fn)
            else:
                with pytest.raises(ValueError, match=constraint_type.value):
                    create_constraint_callable(constraint, spec)


class TestConstraintWithBoTorchIntegration:
    """Test constraints work with BoTorch optimization patterns."""

    def test_constraint_callable_signature(self) -> None:
        """Constraint callable matches BoTorch expected signature.

        BoTorch constraints expect: callable(X) -> Tensor
        where X is (..., d) and returns (...,) with values >= 0 meaning satisfied.
        """
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
                ParameterSpec(name="y", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
            ],
            objectives=[ObjectiveSpec(name="f", minimize=True)],
        )

        constraint = ConstraintSpec(
            type=ConstraintType.SUM_LESS_THAN,
            parameters=["x", "y"],
            value=1.5,
        )

        callable_fn = create_constraint_callable(constraint, spec)

        # Test with various batch shapes
        x_2d = torch.rand(5, 2, dtype=torch.double)
        x_3d = torch.rand(3, 5, 2, dtype=torch.double)

        result_2d = callable_fn(x_2d)
        result_3d = callable_fn(x_3d)

        assert result_2d.shape == (5,)
        assert result_3d.shape == (3, 5)


class TestBotorchLinearConstraintEncoding:
    """Tuple encodings must follow BoTorch's ``>= rhs`` inequality convention.

    BoTorch's ``optimize_acqf`` enforces each inequality tuple as
    ``sum_i X[indices[i]] * coefficients[i] >= rhs`` — see
    https://botorch.readthedocs.io/en/stable/optim.html#botorch.optim.optimize.optimize_acqf
    ("inequality constraints ... in the form sum_i (X[indices[i]] *
    coefficients[i]) >= rhs"). These tests evaluate each encoded tuple at
    known feasible / infeasible points under that convention and assert
    agreement with the user-declared constraint.
    """

    @pytest.fixture
    def simple_spec(self) -> OptimizationSpec:
        """Spec with 3 continuous parameters used by every encoding case."""
        return OptimizationSpec(
            parameters=[
                ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
                ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
                ParameterSpec(name="x3", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )

    @staticmethod
    def _botorch_satisfied(entry: tuple[Tensor, Tensor, float], x: Tensor) -> bool:
        """Evaluate one BoTorch inequality tuple under the ``>= rhs`` convention."""
        indices, coefficients, rhs = entry
        value = float((x[indices] * coefficients).sum().item())
        return value >= rhs - 1e-12

    def _encode_single(
        self, constraint: ConstraintSpec, spec_template: OptimizationSpec
    ) -> tuple[Tensor, Tensor, float]:
        spec = OptimizationSpec(
            parameters=spec_template.parameters,
            objectives=spec_template.objectives,
            constraints=[constraint],
        )
        inequality, equality, projection = build_botorch_linear_constraints(spec)
        assert equality == []
        assert projection == []
        assert len(inequality) == 1
        return inequality[0]

    def test_sum_less_than_encoding(self, simple_spec: OptimizationSpec) -> None:
        """``x1 + x2 <= 0.8`` must be feasible below the cap, infeasible above."""
        constraint = ConstraintSpec(
            type=ConstraintType.SUM_LESS_THAN, parameters=["x1", "x2"], value=0.8
        )
        entry = self._encode_single(constraint, simple_spec)

        feasible = torch.tensor([0.3, 0.4, 0.9], dtype=torch.double)  # sum 0.7 <= 0.8
        infeasible = torch.tensor([0.5, 0.5, 0.0], dtype=torch.double)  # sum 1.0 > 0.8
        assert self._botorch_satisfied(entry, feasible)
        assert not self._botorch_satisfied(entry, infeasible)

    def test_sum_greater_than_encoding(self, simple_spec: OptimizationSpec) -> None:
        """``x1 + x2 >= 0.3`` must be feasible above the floor, infeasible below."""
        constraint = ConstraintSpec(
            type=ConstraintType.SUM_GREATER_THAN, parameters=["x1", "x2"], value=0.3
        )
        entry = self._encode_single(constraint, simple_spec)

        feasible = torch.tensor([0.2, 0.3, 0.0], dtype=torch.double)  # sum 0.5 >= 0.3
        infeasible = torch.tensor([0.1, 0.1, 0.9], dtype=torch.double)  # sum 0.2 < 0.3
        assert self._botorch_satisfied(entry, feasible)
        assert not self._botorch_satisfied(entry, infeasible)

    def test_linear_encoding(self, simple_spec: OptimizationSpec) -> None:
        """``2·x1 + 3·x2 <= 1.0`` (a ``<=`` intent) must keep its direction."""
        constraint = ConstraintSpec(
            type=ConstraintType.LINEAR,
            parameters=["x1", "x2"],
            value=1.0,
            coefficients=[2.0, 3.0],
        )
        entry = self._encode_single(constraint, simple_spec)

        feasible = torch.tensor([0.1, 0.2, 0.5], dtype=torch.double)  # 0.8 <= 1.0
        infeasible = torch.tensor([0.3, 0.3, 0.5], dtype=torch.double)  # 1.5 > 1.0
        assert self._botorch_satisfied(entry, feasible)
        assert not self._botorch_satisfied(entry, infeasible)

    def test_sum_equals_encoding(self, simple_spec: OptimizationSpec) -> None:
        """Equality constraints are direction-free: ``coeffs @ x == value``."""
        constraint = ConstraintSpec(
            type=ConstraintType.SUM_EQUALS, parameters=["x1", "x2", "x3"], value=1.0
        )
        spec = OptimizationSpec(
            parameters=simple_spec.parameters,
            objectives=simple_spec.objectives,
            constraints=[constraint],
        )
        inequality, equality, projection = build_botorch_linear_constraints(spec)
        assert inequality == []
        assert projection == []
        assert len(equality) == 1

        indices, coefficients, rhs = equality[0]
        x = torch.tensor([0.3, 0.3, 0.4], dtype=torch.double)
        value = float((x[indices] * coefficients).sum().item())
        assert value == pytest.approx(rhs)


class TestConstraintSurfaceReports:
    """Direct per-family reports from the BoTorch constraint capability surface.

    Minor-class gap: the ``is_interpoint`` UNSUPPORTED branch (and the
    per-family type vetoes) previously had no direct unit coverage in
    this package — deleting the branch left the suite green.
    """

    @staticmethod
    def _report_spec(constraint: ConstraintSpec) -> OptimizationSpec:
        return OptimizationSpec(
            parameters=[
                ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
                ParameterSpec(name="y", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
            ],
            objectives=[ObjectiveSpec(name="f", minimize=True)],
            constraints=[constraint],
        )

    def test_interpoint_sum_equals_is_unsupported(self) -> None:
        from bo_engine.botorch_backend import _constraint_surface_reports

        reports = _constraint_surface_reports(
            self._report_spec(
                ConstraintSpec(
                    type=ConstraintType.SUM_EQUALS,
                    parameters=["x", "y"],
                    value=1.0,
                    is_interpoint=True,
                )
            )
        )
        assert len(reports) == 1
        assert reports[0].key == "constraint[0](sum_equals)"
        assert "Interpoint" in reports[0].reason

    def test_every_family_classifies_as_documented(self) -> None:
        from bo_engine.botorch_backend import (
            _BOTORCH_SUPPORTED_CONSTRAINT_TYPES,
            _constraint_surface_reports,
        )

        for constraint_type in ConstraintType:
            constraint = ConstraintSpec(
                type=constraint_type,
                parameters=["x", "y"],
                value=1.0,
                coefficients=[1.0, 1.0] if constraint_type == ConstraintType.LINEAR else None,
            )
            reports = _constraint_surface_reports(self._report_spec(constraint))
            if constraint_type in _BOTORCH_SUPPORTED_CONSTRAINT_TYPES:
                assert reports == [], f"{constraint_type} wrongly reported"
            else:
                assert len(reports) == 1, f"{constraint_type} not reported"
                assert constraint_type.value in reports[0].key
