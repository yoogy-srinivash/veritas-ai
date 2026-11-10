from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import json
from backend.app.schemas.response_schemas import ExplanationJobResponse, TokenAttribution
from backend.app.ml.job_manager import job_manager
from backend.app.core.database import get_db
from backend.app.core.models_db import ExplanationCacheRecord

router = APIRouter(tags=["Explanations"])

@router.get("/explanations/{job_id}", response_model=ExplanationJobResponse)
def get_explanation_status(job_id: str, db: Session = Depends(get_db)):
    # 1. Check in-memory active job manager
    job = job_manager.get_job(job_id)
    if job:
        token_objs = None
        if job.get("tokens"):
            token_objs = [TokenAttribution(**t) for t in job["tokens"]]

        return ExplanationJobResponse(
            job_id=job["job_id"],
            status=job["status"],
            tokens=token_objs,
            attribution_method=job.get("attribution_method"),
            duration_ms=job.get("duration_ms"),
            error=job.get("error")
        )

    # 2. Check if job_id matches a hash in explanation_cache
    cached = db.query(ExplanationCacheRecord).filter(ExplanationCacheRecord.hash_key == job_id).first()
    if cached:
        try:
            tokens_data = json.loads(cached.tokens_json)
            return ExplanationJobResponse(
                job_id=job_id,
                status="COMPLETED",
                tokens=[TokenAttribution(**t) for t in tokens_data],
                attribution_method=cached.attribution_method,
                duration_ms=cached.duration_ms
            )
        except Exception:
            pass

    raise HTTPException(status_code=404, detail=f"Explanation job '{job_id}' not found")
