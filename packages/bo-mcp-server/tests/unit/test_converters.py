"""Tests for converters module.

Reference: This module tests the shared converter function that transforms domain
CampaignSpec objects to bo-engine OptimizationSpec objects.

See: https://docs.scipy.org/doc/scipy/tutorial/optimize.html for background on
optimization specification patterns.
"""

from bo_engine.types import (
    AcquisitionMethod as BOAcquisitionMethod,
)
from bo_engine.types import (
    ConstraintType as BOConstraintType,
)
from bo_engine.types import (
    ParameterType as BOParameterType,
)
from bo_mcp_server.converters import campaign_spec_to_optimization_spec
from bo_mcp_server.domain import (
    AcquisitionMethod,
    CampaignSpec,
    Constraint,
    ConstraintType,
    FidelityParameter,
    InputParameter,
    Objective,
    OutcomeConstraint,
    ParameterType,
    SaasboConfig,
    TransferLearningConfig,
    TurboConfig,
)


class TestCampaignSpecToOptimizationSpec:
    """Tests for campaign_spec_to_optimization_spec converter."""

    def test_basic_continuous_parameters(self) -> None:
        """Test conversion of basic continuous parameters."""
        spec = CampaignSpec(
            name="test",
            parameters=(
                InputParameter(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),  # ty: ignore[invalid-argument-type]
                InputParameter(name="x2", type=ParameterType.CONTINUOUS, bounds=(-5.0, 5.0)),  # ty: ignore[invalid-argument-type]
            ),
            objectives=(Objective(name="y", direction="minimize"),),
        )

        opt_spec = campaign_spec_to_optimization_spec(spec)

        assert len(opt_spec.parameters) == 2
        assert opt_spec.parameters[0].name == "x1"
        assert opt_spec.parameters[0].type == BOParameterType.CONTINUOUS
        assert opt_spec.parameters[0].bounds == (0.0, 1.0)
        assert opt_spec.parameters[1].name == "x2"
        assert opt_spec.parameters[1].bounds == (-5.0, 5.0)

    def test_categorical_parameters(self) -> None:
        """Test conversion of categorical parameters."""
        spec = CampaignSpec(
            name="test",
            parameters=(
                InputParameter(
                    name="method",
                    type=ParameterType.CATEGORICAL,
                    categories=("A", "B", "C"),
                ),
            ),
            objectives=(Objective(name="y", direction="minimize"),),
        )

        opt_spec = campaign_spec_to_optimization_spec(spec)

        assert len(opt_spec.parameters) == 1
        assert opt_spec.parameters[0].type == BOParameterType.CATEGORICAL
        assert opt_spec.parameters[0].categories == ["A", "B", "C"]

    def test_discrete_parameters(self) -> None:
        """Test conversion of discrete parameters with values."""
        spec = CampaignSpec(
            name="test",
            parameters=(
                InputParameter(
                    name="n_layers",
                    type=ParameterType.DISCRETE,
                    values=(1, 2, 3, 4, 5),
                ),
            ),
            objectives=(Objective(name="y", direction="minimize"),),
        )

        opt_spec = campaign_spec_to_optimization_spec(spec)

        assert opt_spec.parameters[0].type == BOParameterType.DISCRETE
        assert opt_spec.parameters[0].values == [1, 2, 3, 4, 5]

    def test_objective_conversion(self) -> None:
        """Test conversion of objectives with minimize/maximize."""
        spec = CampaignSpec(
            name="test",
            parameters=(
                InputParameter(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),  # ty: ignore[invalid-argument-type]
            ),
            objectives=(
                Objective(name="loss", direction="minimize"),
                Objective(name="accuracy", direction="maximize"),
            ),
        )

        opt_spec = campaign_spec_to_optimization_spec(spec)

        assert len(opt_spec.objectives) == 2
        assert opt_spec.objectives[0].name == "loss"
        assert opt_spec.objectives[0].minimize is True
        assert opt_spec.objectives[1].name == "accuracy"
        assert opt_spec.objectives[1].minimize is False

    def test_constraint_conversion(self) -> None:
        """Test conversion of constraints."""
        spec = CampaignSpec(
            name="test",
            parameters=(
                InputParameter(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),  # ty: ignore[invalid-argument-type]
                InputParameter(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),  # ty: ignore[invalid-argument-type]
            ),
            objectives=(Objective(name="y", direction="minimize"),),
            constraints=(
                Constraint(
                    type=ConstraintType.SUM_EQUALS,
                    parameters=("x1", "x2"),
                    value=1.0,
                ),
            ),
        )

        opt_spec = campaign_spec_to_optimization_spec(spec)

        assert len(opt_spec.constraints) == 1
        assert opt_spec.constraints[0].type == BOConstraintType.SUM_EQUALS
        assert opt_spec.constraints[0].parameters == ["x1", "x2"]
        assert opt_spec.constraints[0].value == 1.0

    def test_outcome_constraint_conversion(self) -> None:
        """Test conversion of outcome constraints."""
        spec = CampaignSpec(
            name="test",
            parameters=(
                InputParameter(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),  # ty: ignore[invalid-argument-type]
            ),
            objectives=(Objective(name="y", direction="minimize"),),
            outcome_constraints=(
                OutcomeConstraint(objective_name="y", threshold=0.5, greater_than=True),
            ),
        )

        opt_spec = campaign_spec_to_optimization_spec(spec)

        assert len(opt_spec.outcome_constraints) == 1
        assert opt_spec.outcome_constraints[0].objective_name == "y"
        assert opt_spec.outcome_constraints[0].threshold == 0.5
        assert opt_spec.outcome_constraints[0].greater_than is True

    def test_acquisition_method_conversion(self) -> None:
        """Test conversion of acquisition method."""
        spec = CampaignSpec(
            name="test",
            parameters=(
                InputParameter(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),  # ty: ignore[invalid-argument-type]
            ),
            objectives=(Objective(name="y", direction="minimize"),),
            acquisition_method=AcquisitionMethod.NOISY_EI,
        )

        opt_spec = campaign_spec_to_optimization_spec(spec)

        assert opt_spec.acquisition_method == BOAcquisitionMethod.NOISY_EI

    def test_fidelity_parameter_conversion(self) -> None:
        """Test conversion of fidelity parameter for multi-fidelity optimization."""
        spec = CampaignSpec(
            name="test",
            parameters=(
                InputParameter(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),  # ty: ignore[invalid-argument-type]
            ),
            objectives=(Objective(name="y", direction="minimize"),),
            fidelity_parameter=FidelityParameter(
                name="fidelity",
                bounds=(0.1, 1.0),  # ty: ignore[invalid-argument-type]
                target=1.0,
                cost_weight=2.0,
                fixed_cost=10.0,
            ),
        )

        opt_spec = campaign_spec_to_optimization_spec(spec)

        assert opt_spec.fidelity_parameter is not None
        assert opt_spec.fidelity_parameter.name == "fidelity"
        assert opt_spec.fidelity_parameter.bounds == (0.1, 1.0)
        assert opt_spec.fidelity_parameter.target == 1.0
        assert opt_spec.fidelity_parameter.cost_weight == 2.0
        assert opt_spec.fidelity_parameter.fixed_cost == 10.0

    def test_transfer_learning_conversion(self) -> None:
        """Test conversion of transfer learning config."""
        spec = CampaignSpec(
            name="test",
            parameters=(
                InputParameter(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),  # ty: ignore[invalid-argument-type]
            ),
            objectives=(Objective(name="y", direction="minimize"),),
            transfer_learning=TransferLearningConfig(
                prior_campaign_ids=("campaign-1", "campaign-2"),
                num_ranking_samples=256,
            ),
        )

        opt_spec = campaign_spec_to_optimization_spec(spec)

        assert opt_spec.transfer_learning is not None
        assert opt_spec.transfer_learning.prior_campaign_ids == ["campaign-1", "campaign-2"]
        assert opt_spec.transfer_learning.num_ranking_samples == 256

    def test_optional_flags_conversion(self) -> None:
        """Test conversion of optional optimization flags."""
        spec = CampaignSpec(
            name="test",
            parameters=(
                InputParameter(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),  # ty: ignore[invalid-argument-type]
            ),
            objectives=(Objective(name="y", direction="minimize"),),
            use_input_warping=True,
            turbo_config=TurboConfig(),
            use_cost_aware=True,
            saasbo_config=SaasboConfig(),
            batch_size=5,
            initial_design_size=20,
        )

        opt_spec = campaign_spec_to_optimization_spec(spec)

        assert opt_spec.use_input_warping is True
        assert opt_spec.use_turbo is True
        assert opt_spec.use_cost_aware is True
        assert opt_spec.use_saasbo is True
        assert opt_spec.batch_size == 5
        assert opt_spec.initial_design_size == 20

    def test_no_fidelity_parameter_returns_none(self) -> None:
        """Test that missing fidelity parameter results in None."""
        spec = CampaignSpec(
            name="test",
            parameters=(
                InputParameter(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),  # ty: ignore[invalid-argument-type]
            ),
            objectives=(Objective(name="y", direction="minimize"),),
        )

        opt_spec = campaign_spec_to_optimization_spec(spec)

        assert opt_spec.fidelity_parameter is None

    def test_no_transfer_learning_returns_none(self) -> None:
        """Test that missing transfer learning config results in None."""
        spec = CampaignSpec(
            name="test",
            parameters=(
                InputParameter(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),  # ty: ignore[invalid-argument-type]
            ),
            objectives=(Objective(name="y", direction="minimize"),),
        )

        opt_spec = campaign_spec_to_optimization_spec(spec)

        assert opt_spec.transfer_learning is None

    def test_linear_constraint_with_coefficients(self) -> None:
        """Test conversion of linear constraint with coefficients."""
        spec = CampaignSpec(
            name="test",
            parameters=(
                InputParameter(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),  # ty: ignore[invalid-argument-type]
                InputParameter(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),  # ty: ignore[invalid-argument-type]
            ),
            objectives=(Objective(name="y", direction="minimize"),),
            constraints=(
                Constraint(
                    type=ConstraintType.LINEAR,
                    parameters=("x1", "x2"),
                    value=2.0,
                    coefficients=(1.0, 2.0),
                ),
            ),
        )

        opt_spec = campaign_spec_to_optimization_spec(spec)

        assert opt_spec.constraints[0].type == BOConstraintType.LINEAR
        assert opt_spec.constraints[0].coefficients == [1.0, 2.0]
