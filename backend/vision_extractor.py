import base64
import io
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image
from groq import AsyncGroq
import instructor
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import get_settings
from backend.schemas import VisionOutput


class VisionExtractor:
    def __init__(self):
        self.settings = get_settings()
        self.client = instructor.from_groq(AsyncGroq(api_key=self.settings.GROQ_API_KEY), mode=instructor.Mode.JSON)

    def _compress_and_encode_image(self, image_path: str) -> str:
        """Resizes large images and converts to optimal base64 JPEG to save bandwidth and tokens."""
        with open(image_path, "rb") as f:
            image_bytes = f.read()
            
        with Image.open(io.BytesIO(image_bytes)) as img:
            # Convert to RGB (drops Alpha channel which VLMs don't need)
            if img.mode != "RGB":
                img = img.convert("RGB")
                
            # Hard limit resolution to 1920x1080 while maintaining aspect ratio
            # Use LANCZOS for high quality downsampling
            img.thumbnail((1920, 1080), Image.Resampling.LANCZOS)
            
            # Compress to JPEG
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _get_mime_type(self) -> str:
        """We always output JPEG now."""
        return "image/jpeg"

    @retry(
        stop=stop_after_attempt(4), 
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def extract_and_segment(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Extract handwritten text from an image and segment by question.
        Returns a list of dicts with question_id and extracted_answer.
        """
        base64_image = self._compress_and_encode_image(image_path)
        mime_type = self._get_mime_type()

        extraction_prompt = """You are an expert OCR system specialized in reading handwritten answer sheets.

Analyze this handwritten answer sheet image and extract the text for each question answered.

Rules:
- Identify question numbers from the sheet (Q1, Q2, etc. or 1, 2, etc.)
- Normalize question IDs to format "Q1", "Q2", etc.
- Extract the complete answer text for each question
- If text is unclear, provide your best interpretation with [unclear] markers
- If no questions are identifiable, treat the entire text as Q1"""

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
                response_model=VisionOutput,
                temperature=0.1,
                max_tokens=2048,
            )

            # Map VisionOutput to expected dict format
            validated_data = []
            for segment in response.segments:
                validated_data.append(
                    {
                        "question_id": segment.question_id.upper().replace(" ", ""),
                        "extracted_text": segment.extracted_text.strip(),
                    }
                )

            if not validated_data:
                # Fallback if somehow empty
                validated_data = [
                    {
                        "question_id": "Q1",
                        "extracted_text": "[unclear] No answers detected",
                    }
                ]

            return validated_data

        except Exception as e:
            # Re-raise for Tenacity retry
            raise
