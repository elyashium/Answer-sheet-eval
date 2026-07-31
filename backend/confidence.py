import numpy as np
from typing import Dict, Any


class ConfidenceCalibrator:
    def __init__(self):
        self.thresholds = {
            "high": 0.75,
            "medium": 0.45,
        }

    def _length_score(self, extracted_text: str, ideal_answer: str) -> float:
        """
        Score based on answer length relative to ideal.
        Uses a bell curve around the ideal length ratio of 1.0.
        """
        if not ideal_answer or not extracted_text:
            return 0.0

        student_len = len(extracted_text.split())
        ideal_len = len(ideal_answer.split())

        if ideal_len == 0:
            return 0.0

        ratio = student_len / ideal_len

        # Bell curve: peaks at ratio=1.0, drops off for too short or too long
        # f(x) = exp(-2 * (x - 1)^2)
        length_score = float(np.exp(-2.0 * (ratio - 1.0) ** 2))

        return round(length_score, 4)

    def _agreement_score(
        self, semantic_similarity: float, llm_score: float
    ) -> float:
        """
        Measure agreement between semantic and LLM scoring layers.
        High agreement = high confidence.
        """
        difference = abs(semantic_similarity - llm_score)
        # Convert difference to agreement (0 diff = 1.0 agreement)
        agreement = 1.0 - min(difference * 2.0, 1.0)
        return round(agreement, 4)

    def _clarity_score(self, extracted_text: str) -> float:
        """
        Estimate OCR clarity based on presence of unclear markers
        and text coherence heuristics.
        """
        if not extracted_text:
            return 0.0

        text = extracted_text.lower()
        penalty = 0.0

        # Penalize unclear markers
        unclear_count = text.count("[unclear]") + text.count("[illegible]")
        penalty += unclear_count * 0.15

        # Penalize very short answers (likely incomplete extraction)
        word_count = len(text.split())
        if word_count < 5:
            penalty += 0.3
        elif word_count < 10:
            penalty += 0.1

        # Penalize excessive special characters (OCR noise)
        special_ratio = sum(1 for c in text if not c.isalnum() and c != " ") / max(
            len(text), 1
        )
        if special_ratio > 0.15:
            penalty += 0.2

        clarity = max(0.0, 1.0 - penalty)
        return round(clarity, 4)

    def compute_composite_score(
        self,
        extracted_text: str,
        ideal_answer: str,
        semantic_similarity: float,
        llm_score: float,
    ) -> float:
        """
        Composite confidence score formula:
        confidence = 0.3 * length_score + 0.3 * agreement_score + 0.4 * clarity_score
        """
        length = self._length_score(extracted_text, ideal_answer)
        agreement = self._agreement_score(semantic_similarity, llm_score)
        clarity = self._clarity_score(extracted_text)

        composite = (0.3 * length) + (0.3 * agreement) + (0.4 * clarity)
        return round(float(np.clip(composite, 0.0, 1.0)), 4)

    def calibrate(
        self,
        extracted_text: str,
        ideal_answer: str,
        semantic_similarity: float,
        llm_score: float,
    ) -> Dict[str, Any]:
        """
        Produce confidence assessment with label and human review flag.
        """
        composite = self.compute_composite_score(
            extracted_text, ideal_answer, semantic_similarity, llm_score
        )

        # Determine label
        if composite >= self.thresholds["high"]:
            label = "high"
            color = "green"
        elif composite >= self.thresholds["medium"]:
            label = "medium"
            color = "yellow"
        else:
            label = "low"
            color = "red"

        # Flag for human review
        needs_human_review = (
            label == "low"
            or (label == "medium" and composite < 0.55)
            or "[unclear]" in extracted_text.lower()
            or abs(semantic_similarity - llm_score) > 0.4
        )

        return {
            "confidence_score": composite,
            "confidence_label": label,
            "confidence_color": color,
            "needs_human_review": needs_human_review,
            "breakdown": {
                "length_score": self._length_score(extracted_text, ideal_answer),
                "agreement_score": self._agreement_score(semantic_similarity, llm_score),
                "clarity_score": self._clarity_score(extracted_text),
            },
        }