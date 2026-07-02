"""Bayesian Optimization engine using BoTorch.

This package provides a standalone BO engine that can be used independently
or integrated with higher-level packages like bo-mcp-server.

Supports (routed through ``generate_next_batch``):
- Single-objective optimization (noisy EI, EI) [v1.0.1]
- Multi-objective optimization (hypervolume improvement, scalarized)
- Input warping for non-stationary objectives [v1.1]
- LOO cross-validation for model quality assessment [v1.1]
- TuRBO for high-dimensional optimization [v1.2]
- Cost-aware optimization (EIpu) [v1.3]
- GPU auto-detection and acceleration [v2.3]

Standalone modules (NOT routed through ``generate_next_batch`` — a spec
requesting them is rejected with a typed error; drive the module entry
points directly):
- Multi-fidelity optimization (qMFKG, ``bo_engine.multifidelity``) [v2.0]
- Transfer learning (RGPE, ``bo_engine.transfer_learning``) [v2.0]
- SAASBO for very high-dimensional optimization (``bo_engine.saasbo``) [v2.0]

Usage:
    pip install bo-engine

    from bo_engine import generate_next_batch, OptimizationSpec
    from bo_engine.types import ParameterSpec, ObjectiveSpec, ParameterType

For benchmarks:
    from bo_engine.benchmarks import branin, branin_currin, list_benchmarks
"""

from bo_engine.acquisition import (
    create_acquisition,
    create_multi_objective_acquisition,
    create_single_objective_acquisition,
    get_best_observed_value,
    optimize_acquisition,
)

# Backend protocol (Step 8)
from bo_engine.backend import (
    BatchDiversityMetrics,
    BOBackend,
    DiagnosticSection,
    DuplicateInfo,
    Feature,
    SuggestionBatch,
)
from bo_engine.backend_base import (
    CURRENT_STATE_ENVELOPE_VERSION,
    BackendError,
    BackendIncompatibilityError,
    BackendInputError,
    BackendInternalError,
    BackendStateEnvelope,
    BackendTransientError,
    BackendValidationResult,
    BaseBackend,
    CapabilityReport,
    CapabilityStatus,
    NormalizedObjective,
    NormalizedParameter,
    NormalizedProblem,
    build_normalized_problem,
    is_state_envelope,
    required_features,
    unwrap_state,
    wrap_backend_exception,
    wrap_state,
)

# New modules for critical missing functionality (v2.5)
from bo_engine.batch_diversity import (
    DiversityMetrics,
    apply_local_penalization,
    compute_batch_diversity,
    enforce_diversity,
    filter_diverse_candidates,
)
from bo_engine.botorch_backend import BoTorchBackend

# Model Calibration (v2.7 - Section 3.3)
from bo_engine.calibration import (
    CalibrationCurveData,
    CalibrationReport,
    CoverageResult,
    assess_multi_objective_calibration,
    compute_calibration_curve,
    compute_calibration_score,
    compute_loo_calibration,
    get_calibration_summary,
)
from bo_engine.constants import (
    CONFIDENCE_HIGH_UNCERTAINTY_THRESHOLD,
    CONFIDENCE_MEDIUM_UNCERTAINTY_THRESHOLD,
    CONSTRAINT_PROBABILITY_THRESHOLD,
    CONSTRAINT_VIOLATION_WEIGHT,
    CV_APPROXIMATE_THRESHOLD,
    CV_CACHE_TTL,
    CV_DEFAULT_K_FOLDS,
    DISCRETE_ENUMERATION_MAX_POINTS,
    HIGH_DIMENSION_WARNING_THRESHOLD,
    INITIAL_DESIGN_MULTIPLIER,
    MAX_RANDOM_SEED,
    MIN_DATA_ABSOLUTE,
    MIN_DATA_PARAM_MULTIPLIER,
    MIN_OBSERVATIONS_FOR_LOO_CV,
    MIN_OBSERVATIONS_FOR_MODEL,
    MIXED_CATEGORICAL_COMBO_THRESHOLD,
    MODEL_SELECTION_CRITERION,
    MODEL_SELECTION_MIN_IMPROVEMENT,
    REFERENCE_POINT_ADAPTATION_RATE,
    REFERENCE_POINT_MIN_MARGIN,
    RGPE_HELPFUL_WEIGHT_THRESHOLD,
    RGPE_NUM_SAMPLES,
    SAASBO_MIN_DIMENSIONS,
    TURBO_MIN_DIMENSIONS,
)
from bo_engine.constraints import (
    apply_sum_constraint,
    create_constraint_callable,
)
from bo_engine.convergence import (
    ConvergenceReport,
    StoppingDecision,
    StoppingReason,
    detect_convergence,
    detect_hypervolume_convergence,
    detect_single_objective_convergence,
    estimate_remaining_iterations,
    evaluate_stopping_decision,
)

# Cross-validation optimization (v2.6 - Section 2.4)
from bo_engine.cross_validation import (
    CVConfig,
    CVMetrics,
    clear_cv_cache,
    compute_cv_for_model_list,
    compute_loo_cv_optimized,
    estimate_cv_time,
)
from bo_engine.device import (
    clear_cache,
    get_device,
    get_device_info,
    get_dtype,
)
from bo_engine.diagnostics import (
    ConstraintSatisfactionMetrics,
    ExplorationExploitationMetrics,
    HyperparameterInfo,
    LOOCVMetrics,
    SingleObjectiveDiagnostics,
    UncertaintyTrend,
    analyze_hypervolume_history,
    assess_model_health,
    compute_best_value,
    compute_campaign_health,
    compute_constraint_satisfaction,
    compute_exploration_exploitation_metrics,
    compute_hypervolume,
    compute_improvement_history,
    compute_loo_cv_for_model,
    compute_loo_cv_metrics,
    compute_observed_hypervolume,
    compute_pareto_front,
    compute_single_objective_improvement_rate,
    compute_single_objective_progress_status,
    compute_uncertainty_trend,
    determine_single_objective_health_status,
    extract_hyperparameters,
)
from bo_engine.interop import (
    BAYBE_BACKEND_NAME,
    BAYBE_CUSTOM_ROLE,
    BAYBE_PARAMETER_ROLE_KEY,
    BAYBE_SUBSTANCE_ROLE,
)
from bo_engine.method_selector import (
    MethodSelection,
    select_methods,
)

# Model selection and comparison (v2.6 - Section 2.5)
from bo_engine.model_selection import (
    DEFAULT_CANDIDATES,
    KernelType,
    ModelCandidate,
    ModelComparisonResult,
    ModelConfiguration,
    ModelSelectionConfig,
    ModelSelectionResult,
    compare_models,
    get_model_selection_summary,
    select_best_model,
)
from bo_engine.model_validation import (
    ModelHealthReport,
    compute_model_convergence_score,
    validate_model_health,
)
from bo_engine.models import (
    ModelFittingError,
    create_and_fit_model,
    create_and_fit_single_task_model,
    create_model,
    create_single_task_model,
    extract_lengthscales,
    fit_model,
    fit_single_task_model,
    floor_standardize_stdvs,
    get_warping_parameters,
    post_fit_verification,
    verify_standardization,
)
from bo_engine.multifidelity import (
    FidelitySpec,
    MultiFidelityConfig,
    create_and_fit_multifidelity_model,
    create_cost_model,
    create_mfkg_acquisition,
    create_multifidelity_model,
    fit_multifidelity_model,
    generate_multifidelity_suggestions,
    optimize_mfkg,
)

# Outcome constraint modeling (v2.6 - Section 2.3)
from bo_engine.outcome_constraints import (
    ConstraintModelConfig,
    ConstraintModelingMethod,
    ConstraintModelResult,
    assess_constraint_model_quality,
    build_constraint_model_binary,
    build_constraint_model_continuous,
    build_outcome_constraint_models,
    compute_constraint_probability,
    compute_expected_constraint_violation,
    compute_outcome_constraint_calibration,
)
from bo_engine.pending_points import (
    PendingPoint,
    PendingPointTracker,
    compute_pending_distance,
    encode_pending_points,
    filter_pending_points,
    penalize_near_pending,
)

# Posterior Predictive Checks (v2.7 - Section 3.4)
from bo_engine.posterior_checks import (
    NormalityTestResult,
    PosteriorCheckReport,
    QQPlotData,
    ResidualAnalysis,
    analyze_residuals,
    check_multi_objective_posteriors,
    check_residual_normality,
    compute_qq_plot_data,
    compute_standardized_residuals,
    get_posterior_check_summary,
    run_posterior_checks,
)

# Prediction Intervals (v2.7 - Section 3.2)
from bo_engine.prediction_intervals import (
    BatchPredictions,
    MultiObjectivePrediction,
    PredictionInterval,
    SuggestionPrediction,
    compute_multi_objective_predictions,
    compute_prediction_intervals,
    compute_suggestion_predictions,
    format_prediction_interval_string,
)
from bo_engine.progress import (
    ProgressCallback,
    ProgressEvent,
)
from bo_engine.progress import (
    emit as emit_progress,
)

# Result Provenance (v2.7 - Section 3.6)
from bo_engine.provenance import (
    ModelSnapshot,
    ProvenanceChain,
    ProvenanceEvent,
    ProvenanceEventType,
    ProvenanceTracker,
    ResultProvenance,
    SuggestionProvenance,
    compute_parameter_deviation,
    format_provenance_report,
)

# Dynamic reference point (v2.6 - Section 2.1)
from bo_engine.reference_point import (
    ReferencePointConfig,
    ReferencePointState,
    ReferencePointStrategy,
    compute_reference_point_quality,
    get_reference_point,
    get_reference_point_dynamic,
    recommend_reference_point,
)

# Reproducibility (v2.7 - Section 3.7)
from bo_engine.reproducibility import (
    IterationSeeds,
    ReproducibilityConfig,
    ReproducibilityManager,
    ReproducibilityReport,
    SeedState,
    compute_suggestions_hash,
    create_reproducible_sobol,
    derive_seed,
    get_reproducibility_summary,
    verify_reproducibility,
)
from bo_engine.result_validation import (
    DuplicateResult,
    OutlierResult,
    compute_loo_standardized_errors,
    detect_duplicates,
    detect_duplicates_batch,
    detect_outliers,
)
from bo_engine.saasbo import (
    SAASBOConfig,
    SAASBOImportance,
    compute_saasbo_importance,
    compute_saasbo_importance_report,
    create_and_fit_saasbo_model,
    create_saasbo_model,
    estimate_saasbo_runtime,
    fit_saasbo_model,
    generate_saasbo_suggestions,
    get_saasbo_lengthscales,
    should_use_saasbo,
)

# Sensitivity Analysis (v2.7 - Section 3.1)
from bo_engine.sensitivity_analysis import (
    LocalSensitivityResult,
    ParameterSensitivity,
    SensitivityReport,
    compute_pareto_sensitivity,
    compute_sensitivity,
    compute_sensitivity_heatmap_data,
    rank_parameters_by_sensitivity,
)
from bo_engine.spec_ir import (
    ConstraintTargetClass,
    NormalizedConstraint,
    NormalizedSpec,
    classify_constraint_target,
    normalize_spec,
)
from bo_engine.suggestions import (
    OutcomeConstraintConfigurationError,
    generate_initial_design,
    generate_next_batch,
    update_turbo_after_evaluation,
)

# Thompson Sampling (v2.7 - Section 3.5)
from bo_engine.thompson_sampling import (
    ThompsonBatch,
    ThompsonConfig,
    ThompsonSample,
    generate_diverse_thompson_batch,
    generate_thompson_samples,
    generate_thompson_samples_multi_objective,
    get_thompson_sampling_summary,
)
from bo_engine.transfer_learning import (
    RGPE,
    PriorTaskData,
    RGPEAcquisition,
    RGPEConfig,
    RGPELogEI,
    create_base_model,
    create_rgpe_model,
    generate_rgpe_suggestions,
    generate_rgpe_suggestions_legacy,
    get_rgpe_weights_explanation,
)
from bo_engine.transforms import (
    SearchSpaceType,
    build_fixed_features_list,
    classify_search_space,
    count_categorical_combinations,
    encode_categorical,
    enumerate_discrete_choices,
    normalize_inputs,
    standardize_outputs,
    unnormalize_inputs,
)
from bo_engine.turbo import (
    TurboState,
    create_turbo_state,
    get_turbo_bounds,
    should_use_turbo,
    update_turbo_state,
)
from bo_engine.types import (
    LEGACY_ACQUISITION_VALUES,
    AcquisitionMethod,
    AcquisitionOptimizationConfig,
    ConstraintSpec,
    ConstraintType,
    FidelityParameterSpec,
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    OutcomeConstraintSpec,
    ParameterSpec,
    ParameterType,
    SuggestionResult,
    TransferLearningSpec,
    TurboConfig,
)

# What-If Analysis (v2.7 - Section 3.8)
# What-If Analysis (v2.7 - Section 3.8)
from bo_engine.whatif import (
    HypotheticalResult,
    ModelImpact,
    ParetoImpact,
    SuggestionImpact,
    WhatIfReport,
    compare_hypotheticals,
    find_most_informative_point,
    get_whatif_summary,
    simulate_multiple_results,
    simulate_result,
)

__all__ = [
    # Cross-backend interop markers (capability routing)
    "BAYBE_BACKEND_NAME",
    "BAYBE_CUSTOM_ROLE",
    "BAYBE_PARAMETER_ROLE_KEY",
    "BAYBE_SUBSTANCE_ROLE",
    # Constants (configurable thresholds)
    "CONFIDENCE_HIGH_UNCERTAINTY_THRESHOLD",
    "CONFIDENCE_MEDIUM_UNCERTAINTY_THRESHOLD",
    "CONSTRAINT_PROBABILITY_THRESHOLD",
    "CONSTRAINT_VIOLATION_WEIGHT",
    "CURRENT_STATE_ENVELOPE_VERSION",
    "CV_APPROXIMATE_THRESHOLD",
    "CV_CACHE_TTL",
    "CV_DEFAULT_K_FOLDS",
    # Model Selection (v2.6 - Section 2.5)
    "DEFAULT_CANDIDATES",
    "DISCRETE_ENUMERATION_MAX_POINTS",
    "HIGH_DIMENSION_WARNING_THRESHOLD",
    "INITIAL_DESIGN_MULTIPLIER",
    # Types
    "LEGACY_ACQUISITION_VALUES",
    "MAX_RANDOM_SEED",
    "MIN_DATA_ABSOLUTE",
    "MIN_DATA_PARAM_MULTIPLIER",
    "MIN_OBSERVATIONS_FOR_LOO_CV",
    "MIN_OBSERVATIONS_FOR_MODEL",
    "MIXED_CATEGORICAL_COMBO_THRESHOLD",
    "MODEL_SELECTION_CRITERION",
    "MODEL_SELECTION_MIN_IMPROVEMENT",
    "REFERENCE_POINT_ADAPTATION_RATE",
    "REFERENCE_POINT_MIN_MARGIN",
    # Transfer Learning / RGPE (v2.0, v2.6)
    "RGPE",
    "RGPE_HELPFUL_WEIGHT_THRESHOLD",
    "RGPE_NUM_SAMPLES",
    "SAASBO_MIN_DIMENSIONS",
    "TURBO_MIN_DIMENSIONS",
    "AcquisitionMethod",
    "AcquisitionOptimizationConfig",
    # Backend Protocol (Step 8)
    "BOBackend",
    # Backend extensibility
    "BackendError",
    "BackendIncompatibilityError",
    "BackendInputError",
    "BackendInternalError",
    "BackendStateEnvelope",
    "BackendTransientError",
    "BackendValidationResult",
    "BaseBackend",
    "BatchDiversityMetrics",
    # Prediction Intervals (v2.7 - Section 3.2)
    "BatchPredictions",
    "BoTorchBackend",
    # Cross-Validation Optimization (v2.6 - Section 2.4)
    "CVConfig",
    "CVMetrics",
    # Model Calibration (v2.7 - Section 3.3)
    "CalibrationCurveData",
    "CalibrationReport",
    "CapabilityReport",
    "CapabilityStatus",
    # Outcome Constraint Modeling (v2.6 - Section 2.3)
    "ConstraintModelConfig",
    "ConstraintModelResult",
    "ConstraintModelingMethod",
    # Agent Usability Diagnostics (v2.4)
    "ConstraintSatisfactionMetrics",
    "ConstraintSpec",
    # Shared spec IR
    "ConstraintTargetClass",
    "ConstraintType",
    # Convergence Detection (v2.5 - Section 1.4)
    "ConvergenceReport",
    "CoverageResult",
    "DiagnosticSection",
    # Batch Diversity (v2.5 - Section 1.5)
    "DiversityMetrics",
    "DuplicateInfo",
    # Result Validation (v2.5 - Sections 1.2, 1.3)
    "DuplicateResult",
    "ExplorationExploitationMetrics",
    "Feature",
    "FidelityParameterSpec",
    # Multi-fidelity BO (v2.0)
    "FidelitySpec",
    "HyperparameterInfo",
    # What-If Analysis (v2.7 - Section 3.8)
    "HypotheticalResult",
    # Reproducibility (v2.7 - Section 3.7)
    "IterationSeeds",
    "KernelType",
    "LOOCVMetrics",
    # Sensitivity Analysis (v2.7 - Section 3.1)
    "LocalSensitivityResult",
    # Method selection (v1.2)
    "MethodSelection",
    "ModelCandidate",
    "ModelComparisonResult",
    "ModelConfiguration",
    "ModelFittingError",
    # Model Validation (v2.5 - Section 1.1)
    "ModelHealthReport",
    "ModelImpact",
    "ModelSelectionConfig",
    "ModelSelectionResult",
    # Result Provenance (v2.7 - Section 3.6)
    "ModelSnapshot",
    "MultiFidelityConfig",
    "MultiObjectivePrediction",
    # Posterior Predictive Checks (v2.7 - Section 3.4)
    "NormalityTestResult",
    "NormalizedConstraint",
    "NormalizedObjective",
    "NormalizedParameter",
    "NormalizedProblem",
    "NormalizedSpec",
    "ObjectiveSpec",
    "ObservationData",
    "OptimizationSpec",
    "OutcomeConstraintConfigurationError",
    "OutcomeConstraintSpec",
    "OutlierResult",
    "ParameterSensitivity",
    "ParameterSpec",
    "ParameterType",
    "ParetoImpact",
    # Pending Points (v2.5 - Section 1.6)
    "PendingPoint",
    "PendingPointTracker",
    "PosteriorCheckReport",
    "PredictionInterval",
    "PriorTaskData",
    # Progress reporting hook
    "ProgressCallback",
    "ProgressEvent",
    "ProvenanceChain",
    "ProvenanceEvent",
    "ProvenanceEventType",
    "ProvenanceTracker",
    "QQPlotData",
    "RGPEAcquisition",
    "RGPEConfig",
    "RGPELogEI",
    # Dynamic Reference Point (v2.6 - Section 2.1)
    "ReferencePointConfig",
    "ReferencePointState",
    "ReferencePointStrategy",
    "ReproducibilityConfig",
    "ReproducibilityManager",
    "ReproducibilityReport",
    "ResidualAnalysis",
    "ResultProvenance",
    # SAASBO (v2.0)
    "SAASBOConfig",
    "SAASBOImportance",
    "SearchSpaceType",
    "SeedState",
    "SensitivityReport",
    "SingleObjectiveDiagnostics",
    "StoppingDecision",
    "StoppingReason",
    "SuggestionBatch",
    "SuggestionImpact",
    "SuggestionPrediction",
    "SuggestionProvenance",
    "SuggestionResult",
    # Thompson Sampling (v2.7 - Section 3.5)
    "ThompsonBatch",
    "ThompsonConfig",
    "ThompsonSample",
    "TransferLearningSpec",
    "TurboConfig",
    # TuRBO (v1.2)
    "TurboState",
    "UncertaintyTrend",
    "WhatIfReport",
    "analyze_hypervolume_history",
    "analyze_residuals",
    "apply_local_penalization",
    # Core functions
    "apply_sum_constraint",
    "assess_constraint_model_quality",
    "assess_model_health",
    "assess_multi_objective_calibration",
    "build_constraint_model_binary",
    "build_constraint_model_continuous",
    "build_fixed_features_list",
    "build_normalized_problem",
    "build_outcome_constraint_models",
    "check_multi_objective_posteriors",
    "check_residual_normality",
    "classify_constraint_target",
    "classify_search_space",
    # Device management (v2.3)
    "clear_cache",
    "clear_cv_cache",
    "compare_hypotheticals",
    "compare_models",
    "compute_batch_diversity",
    "compute_best_value",
    "compute_calibration_curve",
    "compute_calibration_score",
    "compute_campaign_health",
    "compute_constraint_probability",
    "compute_constraint_satisfaction",
    "compute_cv_for_model_list",
    "compute_expected_constraint_violation",
    "compute_exploration_exploitation_metrics",
    "compute_hypervolume",
    "compute_improvement_history",
    "compute_loo_calibration",
    "compute_loo_cv_for_model",
    "compute_loo_cv_metrics",
    "compute_loo_cv_optimized",
    "compute_loo_standardized_errors",
    "compute_model_convergence_score",
    "compute_multi_objective_predictions",
    "compute_observed_hypervolume",
    "compute_outcome_constraint_calibration",
    "compute_parameter_deviation",
    "compute_pareto_front",
    "compute_pareto_sensitivity",
    "compute_pending_distance",
    "compute_prediction_intervals",
    "compute_qq_plot_data",
    "compute_reference_point_quality",
    "compute_saasbo_importance",
    "compute_saasbo_importance_report",
    "compute_sensitivity",
    "compute_sensitivity_heatmap_data",
    "compute_single_objective_improvement_rate",
    "compute_single_objective_progress_status",
    "compute_standardized_residuals",
    "compute_suggestion_predictions",
    "compute_suggestions_hash",
    "compute_uncertainty_trend",
    "count_categorical_combinations",
    "create_acquisition",
    "create_and_fit_model",
    "create_and_fit_multifidelity_model",
    "create_and_fit_saasbo_model",
    "create_and_fit_single_task_model",
    "create_base_model",
    "create_constraint_callable",
    "create_cost_model",
    "create_mfkg_acquisition",
    "create_model",
    "create_multi_objective_acquisition",
    "create_multifidelity_model",
    "create_reproducible_sobol",
    "create_rgpe_model",
    "create_saasbo_model",
    "create_single_objective_acquisition",
    "create_single_task_model",
    "create_turbo_state",
    "derive_seed",
    "detect_convergence",
    "detect_duplicates",
    "detect_duplicates_batch",
    "detect_hypervolume_convergence",
    "detect_outliers",
    "detect_single_objective_convergence",
    "determine_single_objective_health_status",
    "emit_progress",
    "encode_categorical",
    "encode_pending_points",
    "enforce_diversity",
    "enumerate_discrete_choices",
    "estimate_cv_time",
    "estimate_remaining_iterations",
    "estimate_saasbo_runtime",
    "evaluate_stopping_decision",
    "extract_hyperparameters",
    "extract_lengthscales",
    "filter_diverse_candidates",
    "filter_pending_points",
    "find_most_informative_point",
    "fit_model",
    "fit_multifidelity_model",
    "fit_saasbo_model",
    "fit_single_task_model",
    "floor_standardize_stdvs",
    "format_prediction_interval_string",
    "format_provenance_report",
    "generate_diverse_thompson_batch",
    "generate_initial_design",
    "generate_multifidelity_suggestions",
    "generate_next_batch",
    "generate_rgpe_suggestions",
    "generate_rgpe_suggestions_legacy",
    "generate_saasbo_suggestions",
    "generate_thompson_samples",
    "generate_thompson_samples_multi_objective",
    "get_best_observed_value",
    "get_calibration_summary",
    "get_device",
    "get_device_info",
    "get_dtype",
    "get_model_selection_summary",
    "get_posterior_check_summary",
    "get_reference_point",
    "get_reference_point_dynamic",
    "get_reproducibility_summary",
    "get_rgpe_weights_explanation",
    "get_saasbo_lengthscales",
    "get_thompson_sampling_summary",
    "get_turbo_bounds",
    "get_warping_parameters",
    "get_whatif_summary",
    "is_state_envelope",
    "normalize_inputs",
    "normalize_spec",
    "optimize_acquisition",
    "optimize_mfkg",
    "penalize_near_pending",
    "post_fit_verification",
    "rank_parameters_by_sensitivity",
    "recommend_reference_point",
    "required_features",
    "run_posterior_checks",
    "select_best_model",
    "select_methods",
    "should_use_saasbo",
    "should_use_turbo",
    "simulate_multiple_results",
    "simulate_result",
    "standardize_outputs",
    "unnormalize_inputs",
    "unwrap_state",
    "update_turbo_after_evaluation",
    "update_turbo_state",
    "validate_model_health",
    "verify_reproducibility",
    "verify_standardization",
    "wrap_backend_exception",
    "wrap_state",
]
