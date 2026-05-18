"""Bayesian Optimization engine using BoTorch.

This package provides a standalone BO engine that can be used independently
or integrated with higher-level packages like bo-mcp-server.

Supports:
- Single-objective optimization (noisy EI, EI) [v1.0.1]
- Multi-objective optimization (hypervolume improvement, scalarized)
- Input warping for non-stationary objectives [v1.1]
- LOO cross-validation for model quality assessment [v1.1]
- TuRBO for high-dimensional optimization [v1.2]
- Cost-aware optimization (EIpu) [v1.3]
- Multi-fidelity optimization (qMFKG) [v2.0]
- Transfer learning (RGPE) [v2.0]
- SAASBO for very high-dimensional optimization [v2.0]
- GPU auto-detection and acceleration [v2.3]

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
    compute_pareto_front,
    compute_single_objective_improvement_rate,
    compute_single_objective_progress_status,
    compute_uncertainty_trend,
    determine_single_objective_health_status,
    extract_hyperparameters,
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
    get_warping_parameters,
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
    create_constraint_callable_continuous,
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
    compute_qq_plot_data,
    compute_standardized_residuals,
    get_posterior_check_summary,
    run_posterior_checks,
    test_residual_normality,
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
    # Constants (configurable thresholds)
    "CONFIDENCE_HIGH_UNCERTAINTY_THRESHOLD",
    "CONFIDENCE_MEDIUM_UNCERTAINTY_THRESHOLD",
    "CONSTRAINT_PROBABILITY_THRESHOLD",
    "CONSTRAINT_VIOLATION_WEIGHT",
    "CV_APPROXIMATE_THRESHOLD",
    "CV_CACHE_TTL",
    "CV_DEFAULT_K_FOLDS",
    "DISCRETE_ENUMERATION_MAX_POINTS",
    "HIGH_DIMENSION_WARNING_THRESHOLD",
    "INITIAL_DESIGN_MULTIPLIER",
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
    "RGPE_HELPFUL_WEIGHT_THRESHOLD",
    "RGPE_NUM_SAMPLES",
    "SAASBO_MIN_DIMENSIONS",
    "TURBO_MIN_DIMENSIONS",
    # Device management (v2.3)
    "clear_cache",
    "get_device",
    "get_device_info",
    "get_dtype",
    # Core functions
    "apply_sum_constraint",
    "assess_model_health",
    "compute_best_value",
    "compute_hypervolume",
    "compute_improvement_history",
    "compute_loo_cv_for_model",
    "compute_loo_cv_metrics",
    "compute_pareto_front",
    "compute_single_objective_improvement_rate",
    "ModelFittingError",
    "OutcomeConstraintConfigurationError",
    "create_acquisition",
    "create_and_fit_model",
    "create_and_fit_single_task_model",
    "create_constraint_callable",
    "create_model",
    "create_multi_objective_acquisition",
    "create_single_objective_acquisition",
    "create_single_task_model",
    "determine_single_objective_health_status",
    "SearchSpaceType",
    "build_fixed_features_list",
    "classify_search_space",
    "count_categorical_combinations",
    "encode_categorical",
    "enumerate_discrete_choices",
    "extract_lengthscales",
    "fit_model",
    "fit_single_task_model",
    "generate_initial_design",
    "generate_next_batch",
    "update_turbo_after_evaluation",
    "get_best_observed_value",
    "get_reference_point",
    "get_warping_parameters",
    "normalize_inputs",
    "optimize_acquisition",
    "standardize_outputs",
    "unnormalize_inputs",
    # Method selection (v1.2)
    "MethodSelection",
    "select_methods",
    # TuRBO (v1.2)
    "TurboState",
    "create_turbo_state",
    "get_turbo_bounds",
    "should_use_turbo",
    "update_turbo_state",
    # Multi-fidelity BO (v2.0)
    "FidelitySpec",
    "MultiFidelityConfig",
    "create_and_fit_multifidelity_model",
    "create_cost_model",
    "create_mfkg_acquisition",
    "create_multifidelity_model",
    "fit_multifidelity_model",
    "generate_multifidelity_suggestions",
    "optimize_mfkg",
    # Transfer Learning / RGPE (v2.0, v2.6)
    "RGPE",
    "PriorTaskData",
    "RGPEAcquisition",
    "RGPEConfig",
    "RGPELogEI",
    "create_base_model",
    "create_rgpe_model",
    "generate_rgpe_suggestions",
    "generate_rgpe_suggestions_legacy",
    "get_rgpe_weights_explanation",
    # SAASBO (v2.0)
    "SAASBOConfig",
    "SAASBOImportance",
    "compute_saasbo_importance",
    "compute_saasbo_importance_report",
    "create_and_fit_saasbo_model",
    "create_saasbo_model",
    "estimate_saasbo_runtime",
    "fit_saasbo_model",
    "generate_saasbo_suggestions",
    "get_saasbo_lengthscales",
    "should_use_saasbo",
    # Agent Usability Diagnostics (v2.4)
    "ConstraintSatisfactionMetrics",
    "ExplorationExploitationMetrics",
    "HyperparameterInfo",
    "UncertaintyTrend",
    "analyze_hypervolume_history",
    "compute_campaign_health",
    "compute_constraint_satisfaction",
    "compute_exploration_exploitation_metrics",
    "compute_single_objective_progress_status",
    "compute_uncertainty_trend",
    "extract_hyperparameters",
    # Types
    "LEGACY_ACQUISITION_VALUES",
    "TurboConfig",
    "AcquisitionMethod",
    "AcquisitionOptimizationConfig",
    "ConstraintSpec",
    "ConstraintType",
    "FidelityParameterSpec",
    "LOOCVMetrics",
    "ObjectiveSpec",
    "ObservationData",
    "OptimizationSpec",
    "OutcomeConstraintSpec",
    "ParameterSpec",
    "ParameterType",
    "SingleObjectiveDiagnostics",
    "SuggestionResult",
    "TransferLearningSpec",
    # Shared spec IR
    "ConstraintTargetClass",
    "NormalizedConstraint",
    "NormalizedSpec",
    "classify_constraint_target",
    "normalize_spec",
    # Model Validation (v2.5 - Section 1.1)
    "ModelHealthReport",
    "compute_model_convergence_score",
    "validate_model_health",
    # Result Validation (v2.5 - Sections 1.2, 1.3)
    "DuplicateResult",
    "OutlierResult",
    "compute_loo_standardized_errors",
    "detect_duplicates",
    "detect_duplicates_batch",
    "detect_outliers",
    # Convergence Detection (v2.5 - Section 1.4)
    "ConvergenceReport",
    "StoppingDecision",
    "StoppingReason",
    "detect_convergence",
    "detect_hypervolume_convergence",
    "detect_single_objective_convergence",
    "estimate_remaining_iterations",
    "evaluate_stopping_decision",
    # Batch Diversity (v2.5 - Section 1.5)
    "DiversityMetrics",
    "apply_local_penalization",
    "compute_batch_diversity",
    "enforce_diversity",
    "filter_diverse_candidates",
    # Pending Points (v2.5 - Section 1.6)
    "PendingPoint",
    "PendingPointTracker",
    "compute_pending_distance",
    "encode_pending_points",
    "filter_pending_points",
    "penalize_near_pending",
    # Progress reporting hook (TODO 1.48)
    "ProgressCallback",
    "ProgressEvent",
    "emit_progress",
    # Dynamic Reference Point (v2.6 - Section 2.1)
    "ReferencePointConfig",
    "ReferencePointState",
    "ReferencePointStrategy",
    "compute_reference_point_quality",
    "get_reference_point_dynamic",
    "recommend_reference_point",
    # Outcome Constraint Modeling (v2.6 - Section 2.3)
    "ConstraintModelConfig",
    "ConstraintModelingMethod",
    "ConstraintModelResult",
    "assess_constraint_model_quality",
    "build_constraint_model_binary",
    "build_constraint_model_continuous",
    "build_outcome_constraint_models",
    "compute_constraint_probability",
    "compute_expected_constraint_violation",
    "compute_outcome_constraint_calibration",
    "create_constraint_callable_continuous",
    # Cross-Validation Optimization (v2.6 - Section 2.4)
    "CVConfig",
    "CVMetrics",
    "clear_cv_cache",
    "compute_cv_for_model_list",
    "compute_loo_cv_optimized",
    "estimate_cv_time",
    # Model Selection (v2.6 - Section 2.5)
    "DEFAULT_CANDIDATES",
    "KernelType",
    "ModelCandidate",
    "ModelComparisonResult",
    "ModelConfiguration",
    "ModelSelectionConfig",
    "ModelSelectionResult",
    "compare_models",
    "get_model_selection_summary",
    "select_best_model",
    # Sensitivity Analysis (v2.7 - Section 3.1)
    "LocalSensitivityResult",
    "ParameterSensitivity",
    "SensitivityReport",
    "compute_pareto_sensitivity",
    "compute_sensitivity",
    "compute_sensitivity_heatmap_data",
    "rank_parameters_by_sensitivity",
    # Prediction Intervals (v2.7 - Section 3.2)
    "BatchPredictions",
    "MultiObjectivePrediction",
    "PredictionInterval",
    "SuggestionPrediction",
    "compute_multi_objective_predictions",
    "compute_prediction_intervals",
    "compute_suggestion_predictions",
    "format_prediction_interval_string",
    # Model Calibration (v2.7 - Section 3.3)
    "CalibrationCurveData",
    "CalibrationReport",
    "CoverageResult",
    "assess_multi_objective_calibration",
    "compute_calibration_curve",
    "compute_calibration_score",
    "compute_loo_calibration",
    "get_calibration_summary",
    # Posterior Predictive Checks (v2.7 - Section 3.4)
    "NormalityTestResult",
    "PosteriorCheckReport",
    "QQPlotData",
    "ResidualAnalysis",
    "analyze_residuals",
    "check_multi_objective_posteriors",
    "compute_qq_plot_data",
    "compute_standardized_residuals",
    "get_posterior_check_summary",
    "run_posterior_checks",
    "test_residual_normality",
    # Thompson Sampling (v2.7 - Section 3.5)
    "ThompsonBatch",
    "ThompsonConfig",
    "ThompsonSample",
    "generate_diverse_thompson_batch",
    "generate_thompson_samples",
    "generate_thompson_samples_multi_objective",
    "get_thompson_sampling_summary",
    # Result Provenance (v2.7 - Section 3.6)
    "ModelSnapshot",
    "ProvenanceChain",
    "ProvenanceEvent",
    "ProvenanceEventType",
    "ProvenanceTracker",
    "ResultProvenance",
    "SuggestionProvenance",
    "compute_parameter_deviation",
    "format_provenance_report",
    # Reproducibility (v2.7 - Section 3.7)
    "IterationSeeds",
    "ReproducibilityConfig",
    "ReproducibilityManager",
    "ReproducibilityReport",
    "SeedState",
    "compute_suggestions_hash",
    "create_reproducible_sobol",
    "get_reproducibility_summary",
    "verify_reproducibility",
    "verify_standardization",
    # What-If Analysis (v2.7 - Section 3.8)
    "HypotheticalResult",
    "ModelImpact",
    "ParetoImpact",
    "SuggestionImpact",
    "WhatIfReport",
    "compare_hypotheticals",
    "find_most_informative_point",
    "get_whatif_summary",
    "simulate_multiple_results",
    "simulate_result",
    # Backend Protocol (Step 8)
    "BOBackend",
    "BatchDiversityMetrics",
    "BoTorchBackend",
    "DiagnosticSection",
    "DuplicateInfo",
    "Feature",
    "SuggestionBatch",
    # Backend extensibility (TODO 1.69)
    "BackendError",
    "BackendIncompatibilityError",
    "BackendInputError",
    "BackendInternalError",
    "BackendStateEnvelope",
    "BackendTransientError",
    "BackendValidationResult",
    "BaseBackend",
    "CapabilityReport",
    "CapabilityStatus",
    "CURRENT_STATE_ENVELOPE_VERSION",
    "NormalizedObjective",
    "NormalizedParameter",
    "NormalizedProblem",
    "build_normalized_problem",
    "is_state_envelope",
    "required_features",
    "unwrap_state",
    "wrap_backend_exception",
    "wrap_state",
]
