"""
Image Prediction Endpoint
=========================
Full prediction workflow connecting image upload, preprocessing,
YOLO inference, and structured JSON response.

Author: prajwaledu802-coder
Date: 2026-07-05
"""

import os
import uuid
import time
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.app.schemas.responses import (
    UploadImageResponse,
    PredictionDetails,
    DetectionItem,
)
from backend.app.services.image_service import ImagePreprocessingService, ImageValidationError
from backend.app.services.inference_service import get_inference_service

logger = logging.getLogger("defect_detection.predict")

router = APIRouter(prefix="/predict", tags=["Prediction"])

# Configuration
UPLOAD_DIR = Path("backend/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE_MB = 10

# Initialize services
preprocessor = ImagePreprocessingService(target_size=(640, 640), normalize=True)


def _validate_image_format(filename: str) -> str:
    """Validate uploaded file has a supported image extension."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_format",
                "message": f"File format '{ext}' is not supported. "
                           f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
                "filename": filename,
            },
        )
    return ext


def _generate_request_id() -> str:
    """Generate a unique request identifier for tracking."""
    return str(uuid.uuid4())[:12]


@router.post(
    "/image",
    response_model=UploadImageResponse,
    summary="Upload an image for defect prediction",
    description=(
        "Upload a single image file (JPG, JPEG, PNG) for defect detection. "
        "The image is validated, preprocessed through the OpenCV pipeline, "
        "run through the YOLO inference service, and returns structured "
        "detection results including defect class, confidence, bounding box, "
        "and processing time."
    ),
    responses={
        400: {"description": "Invalid file format or empty file"},
        413: {"description": "File too large"},
        500: {"description": "Internal processing error"},
    },
)
async def predict_image(
    request: Request,
    file: UploadFile = File(
        ...,
        description="Image file to analyze for defects (JPG, JPEG, PNG)",
    ),
):
    """
    End-to-end image defect prediction workflow.

    Pipeline:
        1. Generate unique request ID for tracking
        2. Validate file format and size
        3. Save file temporarily to disk
        4. Preprocess image (resize, normalize, BGR→RGB)
        5. Run YOLO inference via inference_service
        6. Return structured prediction JSON
    """
    request_id = _generate_request_id()
    start_time = time.time()

    logger.info(
        f"[{request_id}] Prediction request received: "
        f"filename={file.filename}, content_type={file.content_type}"
    )

    # Step 1 — Validate format
    ext = _validate_image_format(file.filename)

    # Step 2 — Read file contents
    try:
        contents = await file.read()
    except Exception as exc:
        logger.error(f"[{request_id}] Failed to read uploaded file: {exc}")
        raise HTTPException(status_code=500, detail="Failed to read uploaded file")

    # Step 3 — Validate file size
    file_size_mb = len(contents) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File size {file_size_mb:.2f} MB exceeds maximum {MAX_FILE_SIZE_MB} MB",
        )
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Step 4 — Save temporarily
    save_filename = f"{request_id}_{file.filename}"
    save_path = UPLOAD_DIR / save_filename
    try:
        with open(save_path, "wb") as f:
            f.write(contents)
        logger.info(f"[{request_id}] Saved to {save_path}")
    except IOError as exc:
        logger.error(f"[{request_id}] Failed to save file: {exc}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    # Step 5 — Preprocess image
    try:
        preprocess_result = preprocessor.preprocess(contents)
        processed_image = preprocess_result["image"]
        logger.info(
            f"[{request_id}] Preprocessed: "
            f"{preprocess_result['original_size']} → {preprocess_result['target_size']}"
        )
    except ImageValidationError as exc:
        logger.warning(f"[{request_id}] Image validation failed: {exc}")
        raise HTTPException(status_code=400, detail=f"Image validation failed: {exc}")
    except Exception as exc:
        logger.error(f"[{request_id}] Preprocessing error: {exc}")
        raise HTTPException(status_code=500, detail="Image preprocessing failed")

    # Step 6 — Run inference
    try:
        inference_svc = get_inference_service()
        prediction_result = inference_svc.predict_image(processed_image)
        logger.info(
            f"[{request_id}] Inference complete: "
            f"{prediction_result.get('detection_count', 0)} detections"
        )
    except Exception as exc:
        logger.error(f"[{request_id}] Inference failed: {exc}")
        raise HTTPException(status_code=500, detail="Model inference failed")

    # Step 7 — Build response
    processing_time = time.time() - start_time

    # Convert raw detections to Pydantic DetectionItem list
    detection_items = []
    for det in prediction_result.get("detections", []):
        detection_items.append(DetectionItem(
            **{"class": det["class"]},
            class_id=det["class_id"],
            confidence=det["confidence"],
            bounding_box=det["bounding_box"],
        ))

    prediction_details = PredictionDetails(
        detections=detection_items,
        detection_count=len(detection_items),
        processing_time=f"{processing_time * 1000:.1f} ms",
        image_size=list(preprocess_result["original_size"]),
        model=prediction_result.get("model", "yolov8n_defects"),
    )

    logger.info(
        f"[{request_id}] Request completed in {processing_time * 1000:.1f} ms"
    )

    return UploadImageResponse(
        request_id=request_id,
        filename=file.filename,
        status="success",
        message="Prediction completed successfully",
        file_size_mb=round(file_size_mb, 3),
        processing_time_ms=round(processing_time * 1000, 1),
        prediction=prediction_details,
    )
