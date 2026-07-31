import json
from typing import Dict, Any, List
import functools

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import AsyncGroq
import instructor
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import get_settings
from backend.schemas import ScorerOutput


class SemanticScorer:
    def __init__(self):
        self.settings = get_settings()
        # Wrap AsyncGroq with instructor for guaranteed schema output
        self.client = instructor.from_groq(AsyncGroq(api_key=self.settings.GROQ_API_KEY), mode=instructor.Mode.JSON)
        self.embedding_model = SentenceTransformer(self.settings.EMBEDDING_MODEL)

    @functools.lru_cache(maxsize=1000)
    def _get_embedding(self, text: str):
        """Cache embeddings to save compute on identical strings."""
        return self.embedding_model.encode([text])[0]

    def _compute_cosine_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts using sentence embeddings."""
        emb_a = self._get_embedding(text_a)
        emb_b = self._get_embedding(text_b)
        similarity = cosine_similarity([emb_a], [emb_b])[0][0]
        return float(np.clip(similarity, 0.0, 1.0))

    def _compute_key_concepts_ratio(
        self, student_answer: str, key_concepts: List[str]
    ) -> float:
        """Calculate ratio of key concepts mentioned in the student answer based on semantic meaning."""
        if not key_concepts:
            return 0.0

        student_emb = self._get_embedding(student_answer)
        matched = 0

        for concept in key_concepts:
            concept_emb = self._get_embedding(concept)
            sim = cosine_similarity([student_emb], [concept_emb])[0][0]
            if sim > 0.55:
                matched += 1

        return matched / len(key_concepts)

    @retry(
        stop=stop_after_attempt(4), 
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def _llm_judge_score(
        self,
        student_answer: str,
        ideal_answer: str,
        question: str,
        max_marks: int,
    ) -> ScorerOutput:
        """Use LLM as a judge to evaluate answer quality with guaranteed Pydantic schema."""
        judge_prompt = f"""You are an expert examiner evaluating a student's answer.

Question: {question}
Maximum Marks: {max_marks}
Ideal Answer: {ideal_answer}
Student's Answer: {student_answer}

Evaluate the student's answer against the ideal answer. Consider:
1. Factual accuracy
2. Completeness of key concepts covered
3. Clarity of explanation
4. Relevance to the question"""

        try:
            return await self.client.chat.completions.create(
                model=self.settings.LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise academic examiner.",
                    },
                    {"role": "user", "content": judge_prompt},
                ],
                response_model=ScorerOutput,
                temperature=0.2,
                max_tokens=1024,
            )
        except Exception as e:
            # Re-raise to trigger Tenacity retry
            raise

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

        # Layer 1: Cosine similarity (Instant if cached)
        semantic_similarity = self._compute_cosine_similarity(student_answer, ideal_answer)

        # Layer 2: Key concepts ratio (Instant if cached)
        concepts_ratio = self._compute_key_concepts_ratio(student_answer, key_concepts)

        # Layer 3: LLM Judge (invoked when similarity is in ambiguous range)
        llm_result = None
        if 0.3 <= semantic_similarity <= 0.85:
            try:
                llm_response = await self._llm_judge_score(
                    student_answer, ideal_answer, question_text, max_marks
                )
                llm_result = {
                    "score_fraction": float(np.clip(llm_response.score_fraction, 0.0, 1.0)),
                    "reasoning": llm_response.reasoning,
                    "missing_concepts": llm_response.missing_concepts,
                    "strengths": llm_response.strengths,
                }
            except Exception as e:
                # Fallback after all retries exhausted
                llm_result = {
                    "score_fraction": 0.5,
                    "reasoning": f"LLM judge failed after retries ({str(e)}). Falling back.",
                    "missing_concepts": [],
                    "strengths": [],
                }
        elif semantic_similarity > 0.85:
            llm_result = {
                "score_fraction": min(1.0, semantic_similarity + 0.05),
                "reasoning": "High semantic similarity indicates strong answer alignment with ideal response.",
                "missing_concepts": [],
                "strengths": ["Strong alignment with expected answer"],
            }
        else:
            try:
                llm_response = await self._llm_judge_score(
                    student_answer, ideal_answer, question_text, max_marks
                )
                llm_result = {
                    "score_fraction": float(np.clip(llm_response.score_fraction, 0.0, 1.0)),
                    "reasoning": llm_response.reasoning,
                    "missing_concepts": llm_response.missing_concepts,
                    "strengths": llm_response.strengths,
                }
            except Exception as e:
                llm_result = {
                    "score_fraction": 0.5,
                    "reasoning": f"LLM judge failed after retries ({str(e)}). Falling back.",
                    "missing_concepts": [],
                    "strengths": [],
                }

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
