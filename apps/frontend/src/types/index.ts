// Campaign types matching backend domain models

export type ParameterType = 'continuous' | 'discrete' | 'categorical';
export type OptimizationDirection = 'minimize' | 'maximize';
export type CampaignStatus = 'created' | 'active' | 'paused' | 'completed' | 'failed';

export interface InputParameter {
  name: string;
  type: ParameterType;
  description?: string;
  bounds?: [number, number];
  values?: number[];
  categories?: string[];
}

export interface Objective {
  name: string;
  direction: OptimizationDirection;
  unit?: string;
  target?: number;
}

export interface Constraint {
  type: 'sum_equals' | 'sum_leq' | 'sum_geq' | 'linear';
  parameters: string[];
  value: number;
  coefficients?: number[];
}

export interface CampaignSpec {
  id: string;
  name: string;
  description?: string;
  parameters: InputParameter[];
  objectives: Objective[];
  constraints?: Constraint[];
  batch_size: number;
  created_at: string;
}

export interface Campaign {
  id: string;
  spec_id: string;
  owner_id?: string;
  name: string;
  description?: string;
  status: CampaignStatus;
  version?: number;
  iteration: number;
  n_parameters?: number;
  n_objectives?: number;
  created_at: string;
  updated_at: string;
  completed_at?: string;
  spec?: CampaignSpec;
}

export interface SuggestionProvenance {
  iteration: number;
  batch_index: number;
  acquisition_value?: number;
  model_uncertainty?: number;
  generation_method: string;
  // Enhanced provenance fields
  acquisition_function?: string;
  model_type?: string;
  random_seed?: number;
  model_version?: number;
  confidence_level?: 'high' | 'medium' | 'low';
  explanation?: string;
}

export interface Suggestion {
  id: string;
  campaign_id: string;
  parameter_values: Record<string, number | string>;
  provenance: SuggestionProvenance;
  created_at: string;
  executed: boolean;
}

export interface Result {
  id: string;
  campaign_id: string;
  suggestion_id?: string;
  parameter_values: Record<string, number | string>;
  objective_values: Record<string, number>;
  constraint_violations?: Record<string, boolean>;
  submitted_by: string;
  source: string;
  created_at: string;
}

export interface FeatureImportance {
  lengthscale: {
    by_objective: Record<string, Record<string, number>>;
    aggregate: Record<string, number>;
  };
  shap?: Record<string, number>;
}

export interface DiagnosticReport {
  success: boolean;
  campaign_status: CampaignStatus;
  iteration: number;
  n_results: number;
  n_pareto_points?: number;
  n_pending_suggestions?: number;
  hypervolume?: number;
  pareto_front?: Record<string, number>[];
  objective_ranges?: Record<string, { min: number; max: number; direction: string }>;
  health_status?: 'healthy' | 'warning' | 'critical';
  progress_status?: 'improving' | 'stagnant' | 'regressing';
  warnings?: string[];
  errors?: string[];
  model_info?: {
    type: string;
    acquisition_function: string;
    batch_strategy: string;
    kernel: string;
  };
  feature_importance?: FeatureImportance;
}

// API request/response types

export interface IntakeRequest {
  intake: {
    name: string;
    description?: string;
    parameters: InputParameter[];
    objectives: Objective[];
    constraints?: Constraint[];
    batch_size: number;
  };
}

export interface CreateCampaignResponse {
  success: boolean;
  campaign_id?: string;
  spec_id?: string;
  errors?: string[];
  warnings?: string[];
}

export interface MethodSelection {
  model_type: string;
  acquisition_function: string;
  optimization_strategy: string;
  input_transforms: string[];
  explanation: string;
  confidence: 'high' | 'medium' | 'low';
  alternatives: Array<{ model?: string; acquisition?: string; reason: string }>;
  warnings: string[];
}

export interface GenerateSuggestionsResponse {
  success: boolean;
  suggestions?: Suggestion[];
  iteration?: number;
  errors?: string[];
  method_selection?: MethodSelection;
}

export interface SubmitResultsRequest {
  results: {
    parameter_values: Record<string, number | string>;
    objective_values: Record<string, number>;
    suggestion_id?: string;
  }[];
  source: string;
}

export interface SubmitResultsResponse {
  success: boolean;
  result_ids?: string[];
  errors?: string[];
}

export interface ValidationResponse {
  valid: boolean;
  errors?: string[];
  warnings?: string[];
  derived_defaults?: Record<string, unknown>;
}
