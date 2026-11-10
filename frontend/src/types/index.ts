export type Verdict = 'REAL' | 'FAKE' | 'UNCERTAIN';

export type FusionType = 'feature_fusion' | 'decision_fusion';

export type AttributionMethod = 'SHAP_PartitionExplainer' | 'Approximate_Gradient';

export interface HealthResponse {
  status: string;
  version: string;
  device: string;
  models_loaded: boolean;
  temperature: number;
  rejection_threshold: number;
  database_connected: boolean;
  timestamp: string;
}

export interface DemoPreset {
  id: string;
  text: string;
  image_file: string;
  image_url?: string;
  ground_truth: 'REAL' | 'FAKE';
  category: string;
  source_url: string;
  license: string;
  attribution: string;
  description: string;
}

export interface TokenAttribution {
  token: string;
  score: number; // positive: pushes to FAKE, negative: pushes to REAL
}

export interface VerificationResponse {
  id: string;
  verdict: Verdict;
  raw_confidence: number;
  calibrated_confidence: number;
  prob_real: number;
  prob_fake: number;
  rejection_threshold: number;
  fusion_method: FusionType;
  delta_p_text: number;
  delta_p_image: number;
  gradcam_base64?: string;
  explanation_status: 'PENDING' | 'COMPLETED' | 'UNAVAILABLE';
  job_id?: string;
  text_attributions?: TokenAttribution[];
  attribution_method?: AttributionMethod;
  latency_ms: number;
  timestamp: string;
}

export interface ExplanationResponse {
  job_id: string;
  status: 'PENDING' | 'COMPLETED' | 'FAILED';
  tokens?: TokenAttribution[];
  attribution_method?: AttributionMethod;
  duration_ms?: number;
  error?: string;
}

export interface MetricsData {
  dataset_summary: {
    fakeddit_total: number;
    train_count: number;
    val_count: number;
    test_count: number;
    real_percentage: number;
    fake_percentage: number;
  };
  models_comparison: Array<{
    name: string;
    modality: string;
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    macro_f1?: number;
    fake_f1?: number;
    real_f1?: number;
    specificity?: number;
    false_positive_rate?: number;
    roc_auc: number;
    uncalibrated_ece: number;
    calibrated_ece: number;
    brier_score: number;
    latency_ms: number;
  }>;
  selective_classification: {
    tau_star: number;
    validation_coverage: number;
    validation_selective_accuracy: number;
    test_coverage: number;
    test_selective_accuracy: number;
  };
  source: 'EXPERIMENT_EVALUATION' | 'INITIAL_SPECIFICATION';
}
