from fastapi import APIRouter
import json
from backend.app.schemas.response_schemas import MetricsResponse
from backend.app.config import settings

router = APIRouter(tags=["Metrics"])

@router.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    results_path = settings.MODELS_DIR / "experiment_results.json"
    
    if results_path.exists():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["source"] = "EXPERIMENT_EVALUATION"
                return data
        except Exception:
            pass

    # Unpopulated specification structure prior to Colab evaluation (zero fabricated metrics)
    return MetricsResponse(
        dataset_summary={
            "fakeddit_target": 5000,
            "train_count": 0,
            "val_count": 0,
            "test_count": 0,
            "real_percentage": 50.0,
            "fake_percentage": 50.0,
            "status": "Pending Colab Training & Test Split Evaluation"
        },
        models_comparison=[],
        selective_classification={
            "tau_star": settings.DEFAULT_REJECTION_THRESHOLD,
            "validation_coverage": 0.0,
            "validation_selective_accuracy": 0.0,
            "test_coverage": 0.0,
            "test_selective_accuracy": 0.0,
            "status": "Pending empirical sweep"
        },
        source="INITIAL_SPECIFICATION"
    )
