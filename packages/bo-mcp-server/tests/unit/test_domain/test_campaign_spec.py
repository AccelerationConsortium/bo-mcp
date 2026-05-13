"""Tests for CampaignSpec domain model."""

import pytest
from pydantic import ValidationError

from bo_mcp_server.domain import (
    CampaignSpec,
    Constraint,
    ConstraintType,
    InputParameter,
    Objective,
    ParameterType,
)


class TestInputParameter:
    """Tests for InputParameter validation."""

    def test_continuous_param_requires_bounds(self):
        """Continuous parameter must have bounds."""
        with pytest.raises(ValidationError):
            InputParameter(
                name="temp",
                type=ParameterType.CONTINUOUS,
            )

    def test_continuous_param_valid_bounds(self):
        """Continuous parameter bounds must be valid."""
        with pytest.raises(ValidationError):
            InputParameter(
                name="temp",
                type=ParameterType.CONTINUOUS,
                bounds=(100.0, 20.0),  # Lower > upper  # ty: ignore[invalid-argument-type]
            )

    def test_continuous_param_success(self):
        """Valid continuous parameter."""
        param = InputParameter(
            name="temp",
            type=ParameterType.CONTINUOUS,
            bounds=(20.0, 100.0),  # ty: ignore[invalid-argument-type]
        )
        assert param.name == "temp"
        assert param.bounds is not None
        assert param.bounds.lower == 20.0
        assert param.bounds.upper == 100.0

    def test_discrete_param_requires_values_or_bounds(self):
        """Discrete parameter must have values or bounds."""
        with pytest.raises(ValidationError):
            InputParameter(
                name="count",
                type=ParameterType.DISCRETE,
            )

    def test_discrete_param_with_bounds(self):
        """Discrete parameter with bounds."""
        param = InputParameter(
            name="count",
            type=ParameterType.DISCRETE,
            bounds=(1.0, 10.0),  # ty: ignore[invalid-argument-type]
        )
        assert param.name == "count"

    def test_categorical_param_requires_categories(self):
        """Categorical parameter must have categories."""
        with pytest.raises(ValidationError):
            InputParameter(
                name="catalyst",
                type=ParameterType.CATEGORICAL,
            )

    def test_categorical_param_requires_at_least_2(self):
        """Categorical parameter needs at least 2 categories."""
        with pytest.raises(ValidationError):
            InputParameter(
                name="catalyst",
                type=ParameterType.CATEGORICAL,
                categories=["only_one"],
            )

    def test_categorical_param_success(self):
        """Valid categorical parameter."""
        param = InputParameter(
            name="catalyst",
            type=ParameterType.CATEGORICAL,
            categories=["Pt", "Pd", "Rh"],
        )
        assert param.name == "catalyst"
        assert len(param.categories) == 3  # ty: ignore[invalid-argument-type]


class TestObjective:
    """Tests for Objective validation."""

    def test_valid_minimize_objective(self):
        """Valid minimization objective."""
        obj = Objective(
            name="cost",
            direction="minimize",
        )
        assert obj.is_minimize is True

    def test_valid_maximize_objective(self):
        """Valid maximization objective."""
        obj = Objective(
            name="yield",
            direction="maximize",
        )
        assert obj.is_minimize is False

    def test_invalid_direction(self):
        """Invalid direction raises error."""
        with pytest.raises(ValidationError):
            Objective(
                name="cost",
                direction="invalid",
            )


class TestCampaignSpec:
    """Tests for CampaignSpec validation."""

    def test_valid_campaign_spec(self, sample_campaign_spec: CampaignSpec):
        """Valid campaign spec is created successfully."""
        assert sample_campaign_spec.name == "Test Campaign"
        assert sample_campaign_spec.n_parameters == 2
        assert sample_campaign_spec.n_objectives == 2
        assert sample_campaign_spec.is_multi_objective is True

    def test_campaign_spec_requires_parameters(self):
        """Campaign spec requires at least one parameter."""
        with pytest.raises(ValidationError):
            CampaignSpec(
                name="Test",
                parameters=[],
                objectives=[
                    Objective(name="cost", direction="minimize"),
                ],
            )

    def test_campaign_spec_requires_objectives(self):
        """Campaign spec requires at least one objective."""
        with pytest.raises(ValidationError):
            CampaignSpec(
                name="Test",
                parameters=[
                    InputParameter(
                        name="temp",
                        type=ParameterType.CONTINUOUS,
                        bounds=(0.0, 100.0),  # ty: ignore[invalid-argument-type]
                    ),
                ],
                objectives=[],
            )

    def test_campaign_spec_immutable(self, sample_campaign_spec: CampaignSpec):
        """Campaign spec is immutable."""
        with pytest.raises(ValidationError):
            sample_campaign_spec.name = "New Name"

    def test_constraint_references_valid_parameters(self):
        """Constraint must reference valid parameters."""
        with pytest.raises(ValidationError):
            CampaignSpec(
                name="Test",
                parameters=[
                    InputParameter(
                        name="temp",
                        type=ParameterType.CONTINUOUS,
                        bounds=(0.0, 100.0),  # ty: ignore[invalid-argument-type]
                    ),
                ],
                objectives=[
                    Objective(name="cost", direction="minimize"),
                ],
                constraints=[
                    Constraint(
                        type=ConstraintType.SUM_EQUALS,
                        parameters=["unknown_param"],
                        value=1.0,
                    ),
                ],
            )

    def test_get_parameter(self, sample_campaign_spec: CampaignSpec):
        """get_parameter returns correct parameter or None."""
        param = sample_campaign_spec.get_parameter("temperature")
        assert param is not None
        assert param.name == "temperature"

        missing = sample_campaign_spec.get_parameter("nonexistent")
        assert missing is None


class TestConvergenceToleranceValidation:
    """``convergence_tolerance`` is single-objective only.

    The stopping helper consumes a running-best trajectory over the first
    objective, so multi-objective campaigns must reject the field at create
    time rather than silently misinterpreting it.
    """

    def test_single_objective_convergence_tolerance_accepted(self):
        """One objective + ``convergence_tolerance`` is valid."""
        spec = CampaignSpec(
            name="single",
            parameters=[
                InputParameter(
                    name="x",
                    type=ParameterType.CONTINUOUS,
                    bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
                ),
            ],
            objectives=[Objective(name="y", direction="minimize")],
            convergence_tolerance=0.01,
        )
        assert spec.convergence_tolerance == pytest.approx(0.01)

    def test_multi_objective_convergence_tolerance_rejected(self):
        """Two objectives + ``convergence_tolerance`` raises validation error."""
        with pytest.raises(ValidationError, match="convergence_tolerance"):
            CampaignSpec(
                name="multi",
                parameters=[
                    InputParameter(
                        name="x",
                        type=ParameterType.CONTINUOUS,
                        bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
                    ),
                ],
                objectives=[
                    Objective(name="y1", direction="minimize"),
                    Objective(name="y2", direction="minimize"),
                ],
                convergence_tolerance=0.01,
            )

    def test_multi_objective_without_tolerance_accepted(self):
        """Multi-objective campaigns without the field still validate."""
        spec = CampaignSpec(
            name="multi",
            parameters=[
                InputParameter(
                    name="x",
                    type=ParameterType.CONTINUOUS,
                    bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
                ),
            ],
            objectives=[
                Objective(name="y1", direction="minimize"),
                Objective(name="y2", direction="minimize"),
            ],
        )
        assert spec.convergence_tolerance is None
