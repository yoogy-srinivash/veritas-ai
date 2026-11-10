from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime, timezone
import json
from backend.app.schemas.response_schemas import HealthResponse
from backend.app.core.database import get_db
from backend.app.config import settings

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
def get_health(db: Session = Depends(get_db)):
    # Verify database connection
    db_connected = False
    try:
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        db_connected = False

    # Check for calibrated parameters
    calib_file = settings.MODELS_DIR / "calibration_params.json"
    temperature = settings.DEFAULT_TEMPERATURE
    rejection_threshold = settings.DEFAULT_REJECTION_THRESHOLD

    if calib_file.exists():
        try:
            with open(calib_file, "r", encoding="utf-8") as f:
                calib_data = json.load(f)
                temperature = calib_data.get("temperature", temperature)
                rejection_threshold = calib_data.get("tau_star", rejection_threshold)
        except Exception:
            pass

    # Check for model weights
    models_exist = (
        (settings.MODELS_DIR / "feature_fusion_best.pt").exists() or
        (settings.MODELS_DIR / "decision_fusion_meta.pt").exists()
    )

    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        device=settings.DEVICE,
        models_loaded=models_exist,
        temperature=temperature,
        rejection_threshold=rejection_threshold,
        database_connected=db_connected,
        timestamp=datetime.now(timezone.utc).isoformat()
    )
