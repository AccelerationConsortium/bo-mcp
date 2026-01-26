"""Converters between domain and bo-engine types."""

from bo_engine.types import (
    AcquisitionMethod as BOAcquisitionMethod,
)
from bo_engine.types import (
    ConstraintSpec,
    FidelityParameterSpec,
    ObjectiveSpec,
    OptimizationSpec,
    OutcomeConstraintSpec,
    ParameterSpec,
    TransferLearningSpec,
)
from bo_engine.types import (
    ConstraintType as BOConstraintType,
)
from bo_engine.types import (
    ParameterType as BOParameterType,
)

from bo_mcp_server.domain import CampaignSpec


def campaign_spec_to_optimization_spec(spec: CampaignSpec) -> OptimizationSpec:
    """Convert domain CampaignSpec to bo-engine OptimizationSpec.

    This is the single source of truth for spec conversion between the MCP server
    domain model and the bo-engine optimization types.

    Args:
        spec: Domain CampaignSpec from bo_mcp_server.domain

    Returns:
        OptimizationSpec for use with bo-engine functions
    """
    # Convert parameters
    parameters = []
    for p in spec.parameters:
        param_type = BOParameterType(p.type.value)
        parameters.append(
            ParameterSpec(
                name=p.name,
                type=param_type,
                bounds=p.bounds,
                values=p.values,
                categories=p.categories,
            )
        )

    # Convert objectives
    objectives = [ObjectiveSpec(name=o.name, minimize=o.is_minimize) for o in spec.objectives]

    # Convert constraints
    constraints = []
    for c in spec.constraints:
        constraint_type = BOConstraintType(c.type.value)
        constraints.append(
            ConstraintSpec(
                type=constraint_type,
                parameters=c.parameters,
                value=c.value,
                coefficients=c.coefficients,
            )
        )

    # Convert outcome constraints
    outcome_constraints = []
    for oc in spec.outcome_constraints:
        outcome_constraints.append(
            OutcomeConstraintSpec(
                objective_name=oc.objective_name,
                threshold=oc.threshold,
                greater_than=oc.greater_than,
            )
        )

    # Map domain AcquisitionMethod to bo-engine AcquisitionMethod
    acquisition_method = BOAcquisitionMethod(spec.acquisition_method.value)

    # Convert fidelity parameter (v2.0 multi-fidelity optimization)
    fidelity_parameter = None
    if spec.fidelity_parameter is not None:
        fidelity_parameter = FidelityParameterSpec(
            name=spec.fidelity_parameter.name,
            bounds=spec.fidelity_parameter.bounds,
            target=spec.fidelity_parameter.target,
            cost_weight=spec.fidelity_parameter.cost_weight,
            fixed_cost=spec.fidelity_parameter.fixed_cost,
        )

    # Convert transfer learning config (v2.0)
    transfer_learning = None
    if spec.transfer_learning is not None:
        transfer_learning = TransferLearningSpec(
            prior_campaign_ids=spec.transfer_learning.prior_campaign_ids,
            num_ranking_samples=spec.transfer_learning.num_ranking_samples,
        )

    return OptimizationSpec(
        parameters=parameters,
        objectives=objectives,
        constraints=constraints,
        batch_size=spec.batch_size,
        initial_design_size=spec.initial_design_size,
        acquisition_method=acquisition_method,
        use_input_warping=spec.use_input_warping,
        use_turbo=spec.use_turbo,
        outcome_constraints=outcome_constraints,
        use_cost_aware=spec.use_cost_aware,
        fidelity_parameter=fidelity_parameter,
        transfer_learning=transfer_learning,
        use_saasbo=spec.use_saasbo,
    )
