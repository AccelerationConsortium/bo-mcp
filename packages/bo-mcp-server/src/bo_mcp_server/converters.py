"""Converters between domain and bo-engine types."""

from bo_engine.saasbo import SAASBOConfig
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
from bo_engine.types import (
    TurboConfig as BOTurboConfig,
)

from bo_mcp_server.domain import CampaignSpec


def campaign_spec_to_optimization_spec(spec: CampaignSpec) -> OptimizationSpec:
    """Convert domain CampaignSpec to bo-engine OptimizationSpec.

    This is the single source of truth for spec conversion between the MCP
    server domain model and the bo-engine optimization types.

    Args:
        spec: Domain CampaignSpec from bo_mcp_server.domain

    Returns:
        OptimizationSpec for use with bo-engine functions
    """
    parameters = [
        ParameterSpec(
            name=p.name,
            type=BOParameterType(p.type.value),
            bounds=(p.bounds.lower, p.bounds.upper) if p.bounds is not None else None,
            values=p.values,
            categories=p.categories,
        )
        for p in spec.parameters
    ]

    objectives = [ObjectiveSpec(name=o.name, minimize=o.is_minimize) for o in spec.objectives]

    constraints = [
        ConstraintSpec(
            type=BOConstraintType(c.type.value),
            parameters=c.parameters,
            value=c.value,
            coefficients=c.coefficients,
        )
        for c in spec.constraints
    ]

    outcome_constraints = [
        OutcomeConstraintSpec(
            objective_name=oc.objective_name,
            threshold=oc.threshold,
            greater_than=oc.greater_than,
        )
        for oc in spec.outcome_constraints
    ]

    acquisition_method = BOAcquisitionMethod(spec.acquisition_method.value)

    # Convert TuRBO config
    turbo_config = None
    if spec.turbo_config is not None:
        turbo_config = BOTurboConfig(
            initial_length=spec.turbo_config.initial_length,
            length_min=spec.turbo_config.length_min,
            length_max=spec.turbo_config.length_max,
            success_tolerance=spec.turbo_config.success_tolerance,
        )

    # Convert SAASBO config
    saasbo_config = None
    if spec.saasbo_config is not None:
        saasbo_config = SAASBOConfig(
            warmup_steps=spec.saasbo_config.warmup_steps,
            num_samples=spec.saasbo_config.num_samples,
            thinning=spec.saasbo_config.thinning,
        )

    # Convert fidelity parameter
    fidelity_parameter = None
    if spec.fidelity_parameter is not None:
        fidelity_parameter = FidelityParameterSpec(
            name=spec.fidelity_parameter.name,
            bounds=(
                spec.fidelity_parameter.bounds.lower,
                spec.fidelity_parameter.bounds.upper,
            ),
            target=spec.fidelity_parameter.target,
            cost_weight=spec.fidelity_parameter.cost_weight,
            fixed_cost=spec.fidelity_parameter.fixed_cost,
        )

    # Convert transfer learning config
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
        turbo_config=turbo_config,
        outcome_constraints=outcome_constraints,
        use_cost_aware=spec.use_cost_aware,
        fidelity_parameter=fidelity_parameter,
        transfer_learning=transfer_learning,
        saasbo_config=saasbo_config,
    )
