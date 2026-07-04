"""
Image Prediction Endpoint
=========================
Handles image upload, validation, and prediction requests.

Author: prajwaledu802-coder
Date: 2026-07-04
"""

import os
import uuid
import time
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.app.schemas.responses import UploadImageResponse

logger = logging.getLogger("defect_detection.predict")

router = APIRouter(prefix="/predict", tags=["Prediction"])

# Configuration
UPLOAD_DIR = Path("backend/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE_MB = 10


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
    description="Upload a single image file (JPG, JPEG, PNG) for defect detection. "
                "The image will be validated, saved temporarily, and a prediction "
                "response will be returned.",
)
async def predict_image(
    request: Request,
    file: UploadFile = File(
        ...,
        description="Image file to analyze for defects",
    ),
):
    """
    Process an uploaded image for defect detection.

    Workflow:
        1. Generate unique request ID
        2. Validate file format
        3. Read and validate file size
        4. Save file temporarily
        5. Return prediction response (placeholder)
    """
    request_id = _generate_request_id()
    start_time = time.time()

    logger.info(
        f"[{request_id}] Received prediction request: "
        f"filename={file.filename}, content_type={file.content_type}"
    )

    # Step 1 — Validate format
    ext = _validate_image_format(file.filename)

    # Step 2 — Read file contents
    try:
        contents = await file.read()
    except Exception as exc:
        logger.error(f"[{request_id}] Failed to read uploaded file: {exc}")
        raise HTTPException(
            status_code=500,
            detail={"error": "read_failure", "message": "Failed to read uploaded file"},
        )

    # Step 3 — Validate file size
    file_size_mb = len(contents) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        logger.warning(
            f"[{request_id}] File too large: {file_size_mb:.2f} MB "
            f"(max {MAX_FILE_SIZE_MB} MB)"
        )
        raise HTTPException(
            status_code=413,
            detail={
                "error": "file_too_large",
                "message": f"File size {file_size_mb:.2f} MB exceeds "
                           f"maximum {MAX_FILE_SIZE_MB} MB",
            },
        )

    # Step 4 — Validate file is not empty
    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail={"error": "empty_file", "message": "Uploaded file is empty"},
        )

    # Step 5 — Save temporarily
    save_filename = f"{request_id}_{file.filename}"
    save_path = UPLOAD_DIR / save_filename
    try:
        with open(save_path, "wb") as f:
            f.write(contents)
        logger.info(f"[{request_id}] Saved uploaded image to {save_path}")
    except IOError as exc:
        logger.error(f"[{request_id}] Failed to save file: {exc}")
        raise HTTPException(
            status_code=500,
            detail={"error": "save_failure", "message": "Failed to save uploaded file"},
        )

    # Step 6 — Calculate processing time
    processing_time = time.time() - start_time

    logger.info(
        f"[{request_id}] Upload processed successfully in "
        f"{processing_time * 1000:.1f} ms"
    )

    return UploadImageResponse(
        request_id=request_id,
        filename=file.filename,
        status="success",
        message="Image uploaded successfully",
        file_size_mb=round(file_size_mb, 3),
        processing_time_ms=round(processing_time * 1000, 1),
        prediction=None,
    )
