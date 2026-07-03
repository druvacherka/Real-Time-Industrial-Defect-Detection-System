"""
endpoints/predict.py — Image Defect Prediction Endpoint
==========================================================
Handles uploaded images, performs validation, saves them temporarily,
and returns defect detection results.
"""

import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, status

from app.schemas.responses import UploadImageResponse
from app.core.logger import get_logger
from app.core.config import BASE_DIR

logger = get_logger(__name__)

router = APIRouter()

# Define temporary storage directory for uploaded files
TEMP_UPLOAD_DIR = BASE_DIR / "temp"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Allowed image MIME types and extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@router.post(
    "/predict/image",
    response_model=UploadImageResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict defects in an uploaded image",
    description=(
        "Upload a metal surface image (JPG, JPEG, PNG) to detect defects. "
        "The image is validated, saved temporarily, and processed."
    ),
    tags=["Prediction"],
)
async def predict_image(file: UploadFile = File(...)):
    """
    Endpoint to receive an uploaded image, validate its format,
    save it to a temporary directory, and return a prediction placeholder.
    """
    logger.info("Received image upload request: %s", file.filename)

    # ── 1. Validate file extension ──────────────────────────────────────────
    file_path = Path(file.filename)
    extension = file_path.suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        logger.warning(
            "Rejected file upload: %s. Invalid extension: %s",
            file.filename,
            extension,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file extension: {extension}. "
                f"Supported formats: {', '.join(ALLOWED_EXTENSIONS)}"
            ),
        )

    # ── 2. Save file temporarily ────────────────────────────────────────────
    temp_file_path = TEMP_UPLOAD_DIR / file.filename
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("Saved temporary upload file to: %s", temp_file_path)
    except Exception as exc:
        logger.error(
            "Failed to save uploaded file %s: %s",
            file.filename,
            str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded image.",
        ) from exc

    # ── 3. Return response with placeholder prediction ──────────────────────
    return UploadImageResponse(
        filename=file.filename,
        status="success",
        message="Image uploaded successfully",
        prediction=None,
    )
