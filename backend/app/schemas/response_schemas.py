from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.1.0"
    device: str = "cpu"
    models_loaded: bool = False
    temperature: float = 1.0
    rejection_threshold: float = 0.70
    database_connected: bool = True
    timestamp: str

class TokenAttribution(BaseModel):
    token: str
    score: float

class VerificationResponse(BaseModel):
    id: str
    verdict: Literal["REAL", "FAKE", "UNCERTAIN"]
    raw_confidence: float
    calibrated_confidence: float
    prob_real: float
    prob_fake: float
    rejection_threshold: float
    fusion_method: Literal["feature_fusion", "decision_fusion"]
    delta_p_text: float = 0.0
    delta_p_image: float = 0.0
    gradcam_base64: Optional[str] = None
    explanation_status: Literal["PENDING", "COMPLETED", "UNAVAILABLE"] = "PENDING"
    job_id: Optional[str] = None
    text_attributions: Optional[List[TokenAttribution]] = None
    attribution_method: Optional[Literal["SHAP_PartitionExplainer", "Approximate_Gradient"]] = None
    latency_ms: float
    timestamp: str

class ExplanationJobResponse(BaseModel):
    job_id: str
    status: Literal["PENDING", "COMPLETED", "FAILED"]
    tokens: Optional[List[TokenAttribution]] = None
    attribution_method: Optional[Literal["SHAP_PartitionExplainer", "Approximate_Gradient"]] = None
    duration_ms: Optional[float] = None
    error: Optional[str] = None

class DemoPresetResponse(BaseModel):
    id: str
    text: str
    image_file: str
    image_url: Optional[str] = None
    ground_truth: Literal["REAL", "FAKE"]
    category: str
    source_url: str
    license: str
    attribution: str
    description: str

class ModelBenchmarkRow(BaseModel):
    name: str
    modality: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    uncalibrated_ece: float
    calibrated_ece: float
    brier_score: float
    latency_ms: float

class MetricsResponse(BaseModel):
    dataset_summary: dict
    models_comparison: List[ModelBenchmarkRow]
    selective_classification: dict
    source: Literal["EXPERIMENT_EVALUATION", "INITIAL_SPECIFICATION"]

class HistoryItemResponse(BaseModel):
    id: str
    timestamp: str
    text_content: str
    image_filename: Optional[str] = None
    verdict: str
    calibrated_confidence: float
    rejection_threshold: float
    fusion_method: str
    latency_ms: float
