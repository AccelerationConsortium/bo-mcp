"""Converters between domain and bo-engine types.

The neutral enums (``ParameterType``, ``ConstraintType``,
``AcquisitionMethod``) live in :mod:`bo_engine.types` and are re-exported
from ``bo_mcp_server.domain``; both names refer to the *same* enum object,
so this converter no longer maps between two enum hierarchies.
"""

from bo_engine.saasbo import SAASBOConfig
from bo_engine.types import (
    AcquisitionOptimizationConfig as BOAcquisitionOptimizationConfig,
)
from bo_engine.types import (
    ConstraintSpec,
    FidelityParameterSpec,
    ObjectiveSpec,
    ObjectiveTransformSpec,
    OptimizationSpec,
    OutcomeConstraintSpec,
    ParameterSpec,
    TargetMode,
    TransferLearningSpec,
)
from bo_engine.types import (
    TurboConfig as BOTurboConfig,
)
from bo_mcp_server.domain import CampaignSpec
from bo_mcp_server.domain.campaign_spec import Objective


def _objective_to_engine(o: Objective) -> ObjectiveSpec:
    """Convert one domain Objective to the engine ObjectiveSpec.

    ``target`` is forwarded as the engine ``target_value`` only in match
    mode: legacy rows may carry an informational ``target`` alongside a
    plain direction, and forwarding it there would trip the engine-side
    "target_value requires target_mode='match'" validation.
    """
    is_match = o.target_mode == TargetMode.MATCH
    transform = (
        ObjectiveTransformSpec(
            kind=o.transform.kind,
            bounds=o.transform.bounds,
            exponent=o.transform.exponent,
            center=o.transform.center,
            steepness=o.transform.steepness,
        )
        if o.transform is not None
        else None
    )
    return ObjectiveSpec(
        name=o.name,
        minimize=o.is_minimize,
        log_transform=o.log_transform,
        target_mode=o.target_mode,
        target_value=o.target if is_match else None,
        match_shape=o.match_shape,
        match_scale=o.match_scale,
        weight=o.weight,
        normalization_bounds=o.normalization_bounds,
        transform=transform,
    )


def campaign_spec_to_optimization_spec(spec: CampaignSpec) -> OptimizationSpec:
    """Convert domain CampaignSpec to bo-engine OptimizationSpec.

    This is the single source of truth for spec conversion between the MCP
    server domain model and the bo-engine optimization types.

    Args:
        spec: Domain CampaignSpec from bo_mcp_server.domain

    Returns:
        OptimizationSpec for use with bo-engine functions
    """
    # Domain value objects use tuples for deep immutability; the bo-engine
    # dataclasses still type sequence fields as ``list`` and skip
    # field-level coercion, so we materialize lists at the boundary.
    parameters = [
        ParameterSpec(
            name=p.name,
            type=p.type,
            bounds=(p.bounds.lower, p.bounds.upper) if p.bounds is not None else None,
            values=list(p.values) if p.values is not None else None,
            categories=list(p.categories) if p.categories is not None else None,
            parameter_options=(
                {k: dict(v) for k, v in p.parameter_options.items()}
                if p.parameter_options
                else None
            ),
        )
        for p in spec.parameters
    ]

    objectives = [_objective_to_engine(o) for o in spec.objectives]

    constraints = [
        ConstraintSpec(
            type=c.type,
            parameters=list(c.parameters),
            # Engine dataclass keeps a neutral 0.0 default for the families
            # that take no arithmetic threshold (cardinality / set-based).
            value=c.value if c.value is not None else 0.0,
            coefficients=list(c.coefficients) if c.coefficients is not None else None,
            min_cardinality=c.min_cardinality,
            max_cardinality=c.max_cardinality,
            is_interpoint=c.is_interpoint,
        )
        for c in spec.constraints
    ]

    outcome_constraints = [
        OutcomeConstraintSpec(
            objective_name=oc.objective_name,
            threshold=oc.threshold,
            greater_than=oc.greater_than,
            feasibility_threshold=oc.feasibility_threshold,
        )
        for oc in spec.outcome_constraints
    ]

    # Convert TuRBO config
    turbo_config = None
    if spec.turbo_config is not None:
        turbo_config = BOTurboConfig(
            initial_length=spec.turbo_config.initial_length,
            length_min=spec.turbo_config.length_min,
            length_max=spec.turbo_config.length_max,
            success_tolerance=spec.turbo_config.success_tolerance,
            failure_tolerance=spec.turbo_config.failure_tolerance,
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

    # Convert transfer learning config. The domain model's deprecated
    # ``temperature`` field is intentionally not forwarded — the engine's
    # ranking-loss RGPE weights have no temperature parameter.
    transfer_learning = None
    if spec.transfer_learning is not None:
        transfer_learning = TransferLearningSpec(
            prior_campaign_ids=list(spec.transfer_learning.prior_campaign_ids),
            num_ranking_samples=spec.transfer_learning.num_ranking_samples,
        )

    # Acquisition-optimizer budget overrides (optional). Carrying these
    # through into the bo-engine OptimizationSpec lets `optimize_acquisition`
    # respect per-campaign tuning instead of falling back to its
    # dimension-adaptive defaults.
    if spec.acquisition_optimization is not None:
        acquisition_optimization = BOAcquisitionOptimizationConfig(
            num_restarts=spec.acquisition_optimization.num_restarts,
            raw_samples=spec.acquisition_optimization.raw_samples,
        )
    else:
        acquisition_optimization = BOAcquisitionOptimizationConfig()

    backend_options: dict[str, dict[str, object]] | None = None
    if spec.backend_options:
        backend_options = {k: dict(v) for k, v in spec.backend_options.items()}

    return OptimizationSpec(
        parameters=parameters,
        objectives=objectives,
        constraints=constraints,
        batch_size=spec.batch_size,
        initial_design_size=spec.initial_design_size,
        random_seed=spec.random_seed,
        acquisition_method=spec.acquisition_method,
        acquisition_beta=spec.acquisition_beta,
        scalarization=spec.scalarization,
        scalarizer=spec.scalarizer,
        use_input_warping=spec.use_input_warping,
        turbo_config=turbo_config,
        outcome_constraints=outcome_constraints,
        use_cost_aware=spec.use_cost_aware,
        fidelity_parameter=fidelity_parameter,
        transfer_learning=transfer_learning,
        saasbo_config=saasbo_config,
        acquisition_optimization=acquisition_optimization,
        max_iterations=spec.max_iterations,
        max_observations=spec.max_observations,
        convergence_tolerance=spec.convergence_tolerance,
        backend_options=backend_options,
        acknowledge_degradations=tuple(spec.acknowledge_degradations),
    )
