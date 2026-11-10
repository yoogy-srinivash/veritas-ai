from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime, timezone
import uuid
import time
import hashlib

from backend.app.core.database import get_db
from backend.app.core.models_db import VerificationRecord
from backend.app.schemas.response_schemas import VerificationResponse
from backend.app.ml.job_manager import job_manager
from backend.app.config import settings

router = APIRouter(tags=["Verification"])

@router.post("/verify", response_model=VerificationResponse)
async def verify_content(
    background_tasks: BackgroundTasks,
    text: str = Form(..., min_length=3),
    image: UploadFile = File(None),
    fusion_type: str = Form("feature_fusion"),
    db: Session = Depends(get_db)
):
    start_time = time.time()
    record_id = str(uuid.uuid4())
    job_id = f"job-{hashlib.sha256(text.encode()).hexdigest()[:16]}"

    # Check model weight availability
    model_path = settings.MODELS_DIR / f"{fusion_type}_best.pt"
    if not model_path.exists() and not (settings.MODELS_DIR / "feature_fusion_best.pt").exists():
        # Day 1 honest check: models are trained in Colab on Days 2-3
        raise HTTPException(
            status_code=503,
            detail=(
                "Model weights not yet loaded. Veritas AI models are trained on Google Colab (Days 2 & 3). "
                f"Expected checkpoint at {model_path.name} in backend/models/."
            )
        )

    # In Day 4, real inference pipeline is wired here
    # Placeholder structure adhering strictly to response schema
    latency_ms = round((time.time() - start_time) * 1000, 2)
    timestamp_str = datetime.now(timezone.utc).isoformat()

    return VerificationResponse(
        id=record_id,
        verdict="UNCERTAIN",
        raw_confidence=0.50,
        calibrated_confidence=0.50,
        prob_real=0.50,
        prob_fake=0.50,
        rejection_threshold=settings.DEFAULT_REJECTION_THRESHOLD,
        fusion_method=fusion_type if fusion_type in ["feature_fusion", "decision_fusion"] else "feature_fusion",
        delta_p_text=0.0,
        delta_p_image=0.0,
        gradcam_base64=None,
        explanation_status="PENDING",
        job_id=job_id,
        text_attributions=None,
        attribution_method="SHAP_PartitionExplainer",
        latency_ms=latency_ms,
        timestamp=timestamp_str
    )
