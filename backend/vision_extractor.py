import base64
import json
import re
from pathlib import Path
from typing import List, Dict, Any

from groq import AsyncGroq

from backend.config import get_settings


class VisionExtractor:
    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncGroq(api_key=self.settings.GROQ_API_KEY)

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.standard_b64encode(image_file.read()).decode("utf-8")

    def _get_mime_type(self, image_path: str) -> str:
        """Determine MIME type from file extension."""
        suffix = Path(image_path).suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        return mime_map.get(suffix, "image/png")

    def _clean_json_response(self, raw_text: str) -> str:
        """Remove markdown code fences and extract JSON from LLM response."""
        # Try direct JSON parse first
        cleaned = raw_text.strip()

        # Remove markdown code blocks if present
        patterns = [
            r"```json\s*(.*?)\s*```",
            r"```\s*(.*?)\s*```",
            r"`(.*?)`",
        ]

        for pattern in patterns:
            match = re.search(pattern, cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()
                break

        # Remove any leading/trailing non-JSON characters
        json_start = cleaned.find("[")
        json_end = cleaned.rfind("]")

        if json_start == -1:
            json_start = cleaned.find("{")
            json_end = cleaned.rfind("}")

        if json_start != -1 and json_end != -1:
            cleaned = cleaned[json_start : json_end + 1]

        return cleaned

    async def extract_and_segment(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Extract handwritten text from an image and segment by question.
        Returns a list of dicts with question_id and extracted_answer.
        """
        base64_image = self._encode_image_to_base64(image_path)
        mime_type = self._get_mime_type(image_path)

        extraction_prompt = """You are an expert OCR system specialized in reading handwritten answer sheets.

Analyze this handwritten answer sheet image and extract the text for each question answered.

Return ONLY a valid JSON array with the following structure:
[
  {
    "question_id": "Q1",
    "extracted_text": "the full handwritten answer text for question 1"
  },
  {
    "question_id": "Q2",
    "extracted_text": "the full handwritten answer text for question 2"
  }
]

Rules:
- Identify question numbers from the sheet (Q1, Q2, etc. or 1, 2, etc.)
- Normalize question IDs to format "Q1", "Q2", etc.
- Extract the complete answer text for each question
- If text is unclear, provide your best interpretation with [unclear] markers
- If no questions are identifiable, treat the entire text as Q1
- Return ONLY the JSON array, no additional text or explanation"""

        try:
            response = await self.client.chat.completions.create(
                model=self.settings.VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": extraction_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0.1,
                max_tokens=2048,
            )

            raw_response = response.choices[0].message.content
            cleaned_json = self._clean_json_response(raw_response)

            try:
                extracted_data = json.loads(cleaned_json)
            except json.JSONDecodeError:
                # Fallback: treat entire response as single answer
                extracted_data = [
                    {
                        "question_id": "Q1",
                        "extracted_text": raw_response.strip(),
                    }
                ]

            # Validate structure
            validated_data = []
            for item in extracted_data:
                if isinstance(item, dict) and "question_id" in item and "extracted_text" in item:
                    validated_data.append(
                        {
                            "question_id": item["question_id"].upper().replace(" ", ""),
                            "extracted_text": item["extracted_text"].strip(),
                        }
                    )

            if not validated_data:
                validated_data = [
                    {
                        "question_id": "Q1",
                        "extracted_text": raw_response.strip(),
                    }
                ]

            return validated_data

        except Exception as e:
            raise RuntimeError(f"Vision extraction failed: {str(e)}") from e
backend/scorer.py
python


import json
import re
from typing import Dict, Any, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import AsyncGroq

from backend.config import get_settings


class SemanticScorer:
    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncGroq(api_key=self.settings.GROQ_API_KEY)
        self.embedding_model = SentenceTransformer(self.settings.EMBEDDING_MODEL)

    def _compute_cosine_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts using sentence embeddings."""
        embeddings = self.embedding_model.encode([text_a, text_b])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(np.clip(similarity, 0.0, 1.0))

    def _compute_key_concepts_ratio(
        self, student_answer: str, key_concepts: List[str]
    ) -> float:
        """Calculate ratio of key concepts mentioned in the student answer."""
        if not key_concepts:
            return 0.0

        student_lower = student_answer.lower()
        matched = 0

        for concept in key_concepts:
            concept_words = concept.lower().split()
            # Check if majority of words in the concept appear in the answer
            word_matches = sum(1 for w in concept_words if w in student_lower)
            if word_matches >= len(concept_words) * 0.6:
                matched += 1

        return matched / len(key_concepts)

    async def _llm_judge_score(
        self,
        student_answer: str,
        ideal_answer: str,
        question: str,
        max_marks: int,
    ) -> Dict[str, Any]:
        """Use LLM as a judge to evaluate answer quality."""
        judge_prompt = f"""You are an expert examiner evaluating a student's answer.

Question: {question}
Maximum Marks: {max_marks}
Ideal Answer: {ideal_answer}
Student's Answer: {student_answer}

Evaluate the student's answer against the ideal answer. Consider:
1. Factual accuracy
2. Completeness of key concepts covered
3. Clarity of explanation
4. Relevance to the question

Return ONLY a valid JSON object:
{{
  "score_fraction": <float between 0.0 and 1.0>,
  "marks_awarded": <float between 0 and {max_marks}>,
  "reasoning": "<2-3 sentence explanation of the score>",
  "missing_concepts": ["concept1", "concept2"],
  "strengths": ["strength1", "strength2"]
}}"""

        try:
            response = await self.client.chat.completions.create(
                model=self.settings.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise academic examiner. Return only valid JSON.",
                    },
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
            )

            raw_response = response.choices[0].message.content.strip()

            # Clean markdown fences
            cleaned = raw_response
            patterns = [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]
            for pattern in patterns:
                match = re.search(pattern, cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1).strip()
                    break

            result = json.loads(cleaned)
            result["score_fraction"] = float(
                np.clip(result.get("score_fraction", 0.0), 0.0, 1.0)
            )
            result["marks_awarded"] = float(
                np.clip(result.get("marks_awarded", 0.0), 0.0, max_marks)
            )

            return result

        except (json.JSONDecodeError, KeyError) as e:
            # Fallback to semantic-only scoring
            return {
                "score_fraction": 0.5,
                "marks_awarded": max_marks * 0.5,
                "reasoning": f"LLM judge parsing failed ({str(e)}). Falling back to semantic score.",
                "missing_concepts": [],
                "strengths": [],
            }
        except Exception as e:
            return {
                "score_fraction": 0.5,
                "marks_awarded": max_marks * 0.5,
                "reasoning": f"LLM judge error: {str(e)}. Using fallback score.",
                "missing_concepts": [],
                "strengths": [],
            }

    async def score_answer(
        self,
        student_answer: str,
        question_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Score a student answer using weighted blend of semantic similarity and LLM judge.
        """
        ideal_answer = question_data["ideal_answer"]
        key_concepts = question_data.get("key_concepts", [])
        max_marks = question_data["max_marks"]
        question_text = question_data["question"]

        # Layer 1: Cosine similarity
        semantic_similarity = self._compute_cosine_similarity(student_answer, ideal_answer)

        # Layer 2: Key concepts ratio
        concepts_ratio = self._compute_key_concepts_ratio(student_answer, key_concepts)

        # Layer 3: LLM Judge (invoked when similarity is in ambiguous range)
        llm_result = None
        if 0.3 <= semantic_similarity <= 0.85:
            llm_result = await self._llm_judge_score(
                student_answer, ideal_answer, question_text, max_marks
            )
        elif semantic_similarity > 0.85:
            llm_result = {
                "score_fraction": min(1.0, semantic_similarity + 0.05),
                "marks_awarded": max_marks * min(1.0, semantic_similarity + 0.05),
                "reasoning": "High semantic similarity indicates strong answer alignment with ideal response.",
                "missing_concepts": [],
                "strengths": ["Strong alignment with expected answer"],
            }
        else:
            llm_result = await self._llm_judge_score(
                student_answer, ideal_answer, question_text, max_marks
            )

        # Weighted score blending
        semantic_component = semantic_similarity * self.settings.WEIGHT_SEMANTIC
        llm_component = llm_result["score_fraction"] * self.settings.WEIGHT_LLM
        blended_score = semantic_component + llm_component

        # Final marks calculation
        final_marks = round(blended_score * max_marks, 1)
        final_marks = float(np.clip(final_marks, 0.0, max_marks))

        return {
            "question_id": question_data["id"],
            "question": question_text,
            "max_marks": max_marks,
            "marks_awarded": final_marks,
            "blended_score": round(blended_score, 4),
            "semantic_similarity": round(semantic_similarity, 4),
            "concepts_coverage": round(concepts_ratio, 4),
            "llm_score": round(llm_result["score_fraction"], 4),
            "reasoning": llm_result.get("reasoning", ""),
            "missing_concepts": llm_result.get("missing_concepts", []),
            "strengths": llm_result.get("strengths", []),
        }