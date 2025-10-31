// API Response Types

export interface ConversationMessage {
  session_id: string;
  state: string;
  message: string;
  requires_input: boolean;
  parameters: Record<string, any> | null;
  // Chat continuity: analysis results when available
  recommendation?: AnalysisResult;
  correlation_id?: string;
}

export interface AnalysisStartResponse {
  correlation_id: string;
  status: string;
}

export interface AnalysisStatus {
  status: string;
  progress: number;
  message: string;
}

export interface AnalysisResult {
  status: string;
  correlation_id: string;
  action: string;
  confidence: number;
  timeline: string;
  rationale: string[];
  warnings: string[];
  staged_plan?: any;
  expected_outcome?: any;
  risk_summary?: any;
  cost_estimate?: any;
  utility_scores?: Record<string, number>;
  component_confidences?: Record<string, number>;
  evidence?: {
    market_data?: any;
    intelligence?: any;
    prediction?: any;
  };
  metadata?: any;
  created_at?: string;
  updated_at?: string;
}

export interface Model {
  model_id: string;
  model_type: string;
  currency_pair: string;
  trained_at: string;
  version: string;
  horizons: number[];
  calibration_ok: boolean;
  min_samples: number;
}

export interface TrainingJobResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface TrainingStatus {
  job_id: string;
  status: string;
  progress: number;
  message: string;
  model_id?: string;
  error?: string;
}
