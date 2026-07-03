"""
endpoints/predict.py — Image Defect Prediction Endpoint
==========================================================
Handles uploaded images, performs validation, saves them temporarily,
and returns defect detection results.
"""

import shutil
import time
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, status

from app.schemas.responses import UploadImageResponse, PredictionDetails, DetectionResult
from app.services.image_service import image_preprocessor
from app.models.model_loader import model_wrapper
from app.core.logger import get_logger
from app.core.config import BASE_DIR

logger = get_logger(__name__)

router = APIRouter()

# Define temporary storage directory for uploaded files
TEMP_UPLOAD_DIR = BASE_DIR / "temp"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Make sure model is loaded on startup or first request
model_wrapper.load_model()

# Allowed image MIME types and extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@router.post(
    "/predict/image",
    response_model=UploadImageResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict defects in an uploaded image",
    description=(
        "Upload a metal surface image (JPG, JPEG, PNG) to detect defects. "
        "The image is validated, processed, and passed to the model."
    ),
    tags=["Prediction"],
)
async def predict_image(file: UploadFile = File(...)):
    """
    Endpoint to receive an uploaded image, validate its format,
    save it to a temporary directory, preprocess it, and run defect prediction.
    """
    logger.info("Received image upload request: %s", file.filename)
    start_time = time.time()

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

    # ── 3. Preprocess the image ─────────────────────────────────────────────
    try:
        preprocessed_img = image_preprocessor.preprocess_image(temp_file_path)
    except ValueError as val_err:
        logger.error("Preprocessing error for %s: %s", file.filename, str(val_err))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        ) from val_err

    # ── 4. Model Prediction ──────────────────────────────────────────────────
    try:
        detections = model_wrapper.predict(preprocessed_img)
    except Exception as exc:
        logger.error("Inference failure for %s: %s", file.filename, str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model prediction failed.",
        ) from exc

    # ── 5. Format results and latency ───────────────────────────────────────
    latency_ms = int((time.time() - start_time) * 1000)
    processing_time_str = f"{latency_ms} ms"

    # Convert detection dicts to Pydantic objects
    detection_objects = [
        DetectionResult(
            class_name=det["class_name"],
            confidence=det["confidence"],
            bounding_box=det["bounding_box"]
        )
        for det in detections
    ]

    prediction_details = PredictionDetails(
        detections=detection_objects,
        processing_time=processing_time_str
    )

    return UploadImageResponse(
        filename=file.filename,
        status="success",
        message="Image processed and checked for defects successfully",
        prediction=prediction_details,
    )

