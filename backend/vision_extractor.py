import base64
import io
import json
import re
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image
from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import get_settings


class VisionExtractor:
    def __init__(self):
        self.settings = get_settings()
        # Use raw AsyncGroq client — instructor causes json_validate_failed
        # with vision models when the model returns partial/empty JSON
        self.client = AsyncGroq(api_key=self.settings.GROQ_API_KEY)

    def _compress_and_encode_image(self, image_path: str) -> str:
        """Resizes large images and converts to optimal base64 JPEG."""
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            # Cap at 1024x1024 — enough for handwriting, faster to process
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=82)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _parse_json_response(self, raw_text: str) -> List[Dict[str, Any]]:
        """
        Robustly parse JSON from model response even if it's wrapped
        in markdown code fences or has trailing garbage.
        """
        # Strip markdown fences
        cleaned = re.sub(r"```(?:json)?", "", raw_text, flags=re.IGNORECASE).strip()
        cleaned = cleaned.strip("`").strip()

        # Try direct JSON parse first
        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                # Could be {"segments": [...]} or {"answers": [...]}
                for key in ("segments", "answers", "questions", "results"):
                    if key in data and isinstance(data[key], list):
                        return data[key]
                # Might be a single item wrapped in a dict
                if "question_id" in data:
                    return [data]
        except json.JSONDecodeError:
            pass

        # Try extracting JSON array with regex
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Try extracting JSON object
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, dict) and "question_id" in data:
                    return [data]
            except json.JSONDecodeError:
                pass

        return []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=True,
    )
    async def extract_and_segment(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Extract handwritten text from an image and segment by question.
        Uses raw Groq API (no instructor) and robust JSON parsing.
        """
        base64_image = self._compress_and_encode_image(image_path)

        extraction_prompt = """You are an expert OCR system for handwritten answer sheets.

Look at this image and extract every handwritten answer.

Respond with ONLY a JSON array — no explanation, no markdown, no extra text.

Format:
[
  {"question_id": "Q1", "extracted_text": "full text of student answer here"},
  {"question_id": "Q2", "extracted_text": "full text of student answer here"}
]

Rules:
- If you see question numbers like 1, 2, Q1, Q2 — normalize to Q1, Q2 format
- If there are no clear question numbers, label all content as Q1
- Preserve the student's exact words
- If handwriting is unclear, write your best guess with [unclear] markers"""

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
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=2048,
        )

        raw_text = response.choices[0].message.content or ""
        parsed = self._parse_json_response(raw_text)

        # Normalise keys
        validated = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            qid = (
                item.get("question_id")
                or item.get("questionId")
                or item.get("id")
                or "Q1"
            )
            text = (
                item.get("extracted_text")
                or item.get("text")
                or item.get("answer")
                or ""
            )
            validated.append(
                {
                    "question_id": str(qid).upper().replace(" ", ""),
                    "extracted_text": str(text).strip(),
                }
            )

        if not validated:
            # Last resort: treat entire response as Q1
            fallback_text = raw_text.strip() or "[Unable to extract text]"
            validated = [{"question_id": "Q1", "extracted_text": fallback_text}]

        return validated
