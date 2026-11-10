from sqlalchemy import Column, String, Float, DateTime, Text, ForeignKey
from datetime import datetime, timezone
import uuid
from backend.app.core.database import Base

def generate_uuid() -> str:
    return str(uuid.uuid4())

class VerificationRecord(Base):
    __tablename__ = "verification_records"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    text_content = Column(Text, nullable=False)
    image_filename = Column(String(255), nullable=True)
    verdict = Column(String(20), nullable=False)  # 'REAL', 'FAKE', 'UNCERTAIN'
    raw_confidence = Column(Float, nullable=False)
    calibrated_confidence = Column(Float, nullable=False)
    prob_real = Column(Float, nullable=False)
    prob_fake = Column(Float, nullable=False)
    rejection_threshold = Column(Float, nullable=False)
    fusion_method = Column(String(50), nullable=False)
    text_attribution_json = Column(Text, nullable=True)
    attribution_method = Column(String(50), nullable=True)
    image_gradcam_base64 = Column(Text, nullable=True)
    delta_p_text = Column(Float, default=0.0, nullable=False)
    delta_p_image = Column(Float, default=0.0, nullable=False)
    latency_ms = Column(Float, nullable=False)

class UserFeedback(Base):
    __tablename__ = "user_feedback"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    verification_id = Column(String(36), ForeignKey("verification_records.id"), nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    user_verdict = Column(String(20), nullable=False)  # 'AGREE', 'DISAGREE'
    user_suggested_label = Column(String(20), nullable=True)  # 'REAL', 'FAKE'
    comments = Column(Text, nullable=True)

class ExplanationCacheRecord(Base):
    __tablename__ = "explanation_cache"

    hash_key = Column(String(64), primary_key=True)  # SHA-256 of text
    tokens_json = Column(Text, nullable=False)
    attribution_method = Column(String(50), default="SHAP_PartitionExplainer", nullable=False)
    duration_ms = Column(Float, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
