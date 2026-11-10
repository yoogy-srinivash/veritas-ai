from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timezone
import uuid

from backend.app.core.database import get_db
from backend.app.core.models_db import VerificationRecord, UserFeedback
from backend.app.schemas.request_schemas import FeedbackRequest
from backend.app.schemas.response_schemas import HistoryItemResponse

router = APIRouter(tags=["History & Feedback"])

@router.get("/history", response_model=List[HistoryItemResponse])
def get_history(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    offset = (page - 1) * limit
    records = (
        db.query(VerificationRecord)
        .order_by(VerificationRecord.timestamp.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return [
        HistoryItemResponse(
            id=r.id,
            timestamp=r.timestamp.isoformat(),
            text_content=r.text_content,
            image_filename=r.image_filename,
            verdict=r.verdict,
            calibrated_confidence=r.calibrated_confidence,
            rejection_threshold=r.rejection_threshold,
            fusion_method=r.fusion_method,
            latency_ms=r.latency_ms
        )
        for r in records
    ]

@router.post("/history/{record_id}/feedback")
def submit_feedback(
    record_id: str,
    feedback: FeedbackRequest,
    db: Session = Depends(get_db)
):
    record = db.query(VerificationRecord).filter(VerificationRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Verification record '{record_id}' not found")

    fb_entry = UserFeedback(
        id=str(uuid.uuid4()),
        verification_id=record_id,
        timestamp=datetime.now(timezone.utc),
        user_verdict=feedback.user_verdict,
        user_suggested_label=feedback.user_suggested_label,
        comments=feedback.comments
    )
    db.add(fb_entry)
    db.commit()

    return {"status": "success", "message": "Feedback submitted successfully"}
