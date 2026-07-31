from pydantic import BaseModel, Field
from typing import List

class VisionSegment(BaseModel):
    question_id: str = Field(..., description="The ID of the question (e.g., Q1, Q2)")
    extracted_text: str = Field(..., description="The full handwritten answer text extracted for this question")

class VisionOutput(BaseModel):
    segments: List[VisionSegment] = Field(..., description="List of extracted answers segmented by question")

class ScorerOutput(BaseModel):
    score_fraction: float = Field(..., description="The score fraction awarded between 0.0 and 1.0 based on answer quality")
    reasoning: str = Field(..., description="2-3 sentence explanation of the score")
    missing_concepts: List[str] = Field(default_factory=list, description="List of key concepts missing from the answer")
    strengths: List[str] = Field(default_factory=list, description="List of strengths in the student's answer")
