from pydantic import BaseModel, Field
from typing import Optional, Literal

class FeedbackRequest(BaseModel):
    user_verdict: Literal["AGREE", "DISAGREE"]
    user_suggested_label: Optional[Literal["REAL", "FAKE"]] = None
    comments: Optional[str] = None

class VerificationFormInput(BaseModel):
    text: str = Field(..., min_length=3, description="Textual claim or submission title")
    fusion_type: Literal["feature_fusion", "decision_fusion"] = "feature_fusion"
