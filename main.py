import os
import json
import uuid
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from backend.config import get_settings
from backend.vision_extractor import VisionExtractor
from backend.scorer import SemanticScorer
from backend.confidence import ConfidenceCalibrator

# Initialize FastAPI app
app = FastAPI(
    title="Evaluator.ai",
    description="Handwritten Answer Sheet Evaluator powered by AI",
    version="1.0.0",
)

# Load settings
settings = get_settings()

# Ensure temp directory exists
Path(settings.TEMP_DIR).mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize pipeline components
vision_extractor = VisionExtractor()
scorer = SemanticScorer()
confidence_calibrator = ConfidenceCalibrator()


def load_answer_key() -> dict:
    """Load the answer key from JSON file."""
    answer_key_path = Path("data/answer_key.json")
    if not answer_key_path.exists():
        raise FileNotFoundError("Answer key not found at data/answer_key.json")
    with open(answer_key_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/")
async def serve_index():
    """Serve the main frontend page."""
    return FileResponse("static/index.html")


import asyncio

@app.post("/api/evaluate")
async def evaluate_answer_sheets(files: List[UploadFile] = File(...)):
    """
    Accept uploaded answer sheet images, run full evaluation pipeline.
    All images are processed CONCURRENTLY via asyncio.gather for speed.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    allowed_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
    max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024

    # Load answer key once (shared across all concurrent tasks)
    try:
        answer_key = load_answer_key()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    questions_map = {q["id"]: q for q in answer_key["questions"]}
    temp_paths = []

    async def process_single_file(file: UploadFile) -> dict:
        """Process one image through the full pipeline."""
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.filename}. Allowed: {allowed_extensions}",
            )

        content = await file.read()
        if len(content) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} exceeds {settings.MAX_FILE_SIZE_MB}MB limit",
            )

        # Save to temp
        temp_filename = f"{uuid.uuid4().hex}{file_ext}"
        temp_path = os.path.join(settings.TEMP_DIR, temp_filename)
        temp_paths.append(temp_path)
        with open(temp_path, "wb") as tmp:
            tmp.write(content)

        # Step 1: Vision extraction
        extracted_answers = await vision_extractor.extract_and_segment(temp_path)

        # Step 2 & 3: Score + confidence (questions within one image run sequentially
        # since they share context, but multiple images run in parallel)
        file_result = {
            "filename": file.filename,
            "subject": answer_key["subject"],
            "total_marks": answer_key["total_marks"],
            "questions": [],
            "total_scored": 0.0,
        }

        for extracted in extracted_answers:
            question_id = extracted["question_id"]
            student_text = extracted["extracted_text"]

            if question_id not in questions_map:
                file_result["questions"].append({
                    "question_id": question_id,
                    "extracted_text": student_text,
                    "error": f"Question {question_id} not found in answer key",
                    "marks_awarded": 0,
                    "max_marks": 0,
                    "confidence": {
                        "confidence_label": "low",
                        "confidence_color": "red",
                        "needs_human_review": True,
                    },
                })
                continue

            question_data = questions_map[question_id]

            score_result = await scorer.score_answer(student_text, question_data)
            confidence_result = confidence_calibrator.calibrate(
                extracted_text=student_text,
                ideal_answer=question_data["ideal_answer"],
                semantic_similarity=score_result["semantic_similarity"],
                llm_score=score_result["llm_score"],
            )

            question_result = {
                "question_id": question_id,
                "question": score_result["question"],
                "extracted_text": student_text,
                "max_marks": score_result["max_marks"],
                "marks_awarded": score_result["marks_awarded"],
                "percentage": round(
                    (score_result["marks_awarded"] / score_result["max_marks"]) * 100, 1
                ) if score_result["max_marks"] else 0,
                "scoring": {
                    "blended_score": score_result["blended_score"],
                    "semantic_similarity": score_result["semantic_similarity"],
                    "concepts_coverage": score_result["concepts_coverage"],
                    "llm_score": score_result["llm_score"],
                },
                "reasoning": score_result["reasoning"],
                "missing_concepts": score_result["missing_concepts"],
                "strengths": score_result["strengths"],
                "confidence": confidence_result,
            }

            file_result["questions"].append(question_result)
            file_result["total_scored"] += score_result["marks_awarded"]

        file_result["total_scored"] = round(file_result["total_scored"], 1)
        file_result["overall_percentage"] = round(
            (file_result["total_scored"] / file_result["total_marks"]) * 100, 1
        ) if file_result["total_marks"] else 0

        return file_result

    try:
        # 🚀 Process all uploaded images CONCURRENTLY
        results = await asyncio.gather(
            *[process_single_file(f) for f in files],
            return_exceptions=False
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
    finally:
        for temp_path in temp_paths:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass

    return JSONResponse(content={"success": True, "results": list(results)})



@app.post("/api/export/csv")
async def export_csv(payload: dict):
    """
    Accept evaluation results and return a deliverable CSV file.
    Expected payload: {"results": [...]} matching /api/evaluate response.
    """
    import csv, io
    from fastapi.responses import StreamingResponse

    rows = []
    for file_result in payload.get("results", []):
        filename = file_result.get("filename", "unknown")
        for q in file_result.get("questions", []):
            rows.append({
                "filename": filename,
                "question_no": q.get("question_id", ""),
                "question_text": q.get("question", ""),
                "extracted_answer": q.get("extracted_text", ""),
                "score_awarded": q.get("marks_awarded", 0),
                "max_score": q.get("max_marks", 0),
                "percentage": q.get("percentage", 0),
                "semantic_similarity": q.get("scoring", {}).get("semantic_similarity", ""),
                "llm_score": q.get("scoring", {}).get("llm_score", ""),
                "confidence_label": q.get("confidence", {}).get("confidence_label", ""),
                "needs_human_review": q.get("confidence", {}).get("needs_human_review", False),
                "reason": q.get("reasoning", ""),
                "missing_concepts": "; ".join(q.get("missing_concepts", [])),
            })

    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=evaluation_results.csv"}
    )


@app.get("/api/answer-key")
async def get_answer_key():
    """Return the current answer key for reference."""
    try:
        answer_key = load_answer_key()
        return JSONResponse(content=answer_key)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )