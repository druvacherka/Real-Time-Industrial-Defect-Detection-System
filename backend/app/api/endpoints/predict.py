"""
Image Prediction Endpoint
=========================
Full prediction workflow connecting image upload, preprocessing,
YOLO inference, and structured JSON response.

Author: prajwaledu802-coder
Date: 2026-07-12 (Asynchronous Optimization)
"""

import os
import uuid
import time
import logging
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse

from app.schemas.responses import (
    UploadImageResponse,
    PredictionDetails,
    DetectionItem,
    UploadVideoResponse,
    VideoPredictionDetails,
    FrameDetectionSummary,
    UploadLiveResponse,
    LiveStreamPredictionDetails,
)
from app.schemas.requests import LiveStreamRequest
from app.services.image_service import ImagePreprocessingService, ImageValidationError
from app.services.model_service import get_model_service

logger = logging.getLogger("defect_detection.predict")

router = APIRouter(prefix="/predict", tags=["Prediction"])

# Configuration
UPLOAD_DIR = Path("backend/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE_MB = 10

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}
MAX_VIDEO_SIZE_MB = 50

# Initialize services
preprocessor = ImagePreprocessingService(target_size=(640, 640), normalize=True)


def _validate_image_format(filename: str, content_type: str) -> str:
    """Validate uploaded file has a supported image extension and content type."""
    ext = Path(filename).suffix.lower()
    allowed_mimetypes = {"image/jpeg", "image/jpg", "image/png"}
    if ext not in ALLOWED_EXTENSIONS or content_type not in allowed_mimetypes:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_format",
                "message": f"File format '{ext}' with content type '{content_type}' is not supported. "
                           f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
                "filename": filename,
            },
        )
    return ext


def _generate_request_id() -> str:
    """Generate a unique request identifier for tracking."""
    return str(uuid.uuid4())[:12]


async def _process_image_async(contents: bytes, request_id: str) -> tuple[dict, dict]:
    """
    Run CPU-bound image decoding, validation, preprocessing, and inference in a thread pool.
    """
    def decode_and_validate():
        import cv2
        import numpy as np
        arr = np.frombuffer(contents, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("OpenCV decoding failed")
        h, w = img.shape[:2]
        if h < 32 or w < 32 or h > 8192 or w > 8192:
            raise ValueError(f"Image dimensions {w}x{h} are out of allowed bounds [32x32 to 8192x8192]")
        return img

    try:
        # Offload decode and validation to prevent event loop blocking
        await asyncio.to_thread(decode_and_validate)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_image",
                "message": str(exc),
            }
        )

    # Offload preprocessing to prevent event loop blocking
    try:
        preprocess_result = await asyncio.to_thread(preprocessor.preprocess, contents)
        processed_image = preprocess_result["image"]
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail={"error": "validation_failed", "message": str(exc)})
    except Exception as exc:
        logger.error(f"[{request_id}] Preprocessing error: {exc}")
        raise HTTPException(status_code=500, detail="Image preprocessing failed")

    # Offload model inference to prevent event loop blocking
    try:
        model_svc = get_model_service()
        prediction_result = await asyncio.to_thread(model_svc.predict_image, processed_image)
    except Exception as exc:
        logger.error(f"[{request_id}] Inference failed: {exc}")
        raise HTTPException(status_code=500, detail="Model inference failed")

    return preprocess_result, prediction_result


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
        504: {"description": "Request timeout"},
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
    """
    request_id = _generate_request_id()
    start_time = time.time()

    logger.info(
        f"[request_id={request_id}] [action=predict_image] [status=received] "
        f"filename={file.filename} content_type={file.content_type}"
    )

    # Step 1 — Validate extension format
    ext = _validate_image_format(file.filename, file.content_type)

    # Step 2 — Read file contents asynchronously
    try:
        contents = await file.read()
    except Exception as exc:
        logger.error(f"[request_id={request_id}] [action=predict_image] [status=failed] error=read_failed details={exc}")
        raise HTTPException(status_code=500, detail="Failed to read uploaded file")

    # Step 2.5 — Validate file size and format headers (magic numbers)
    file_size_mb = len(contents) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        logger.warning(f"[request_id={request_id}] [action=predict_image] [status=rejected] reason=file_too_large size_mb={file_size_mb:.2f}")
        raise HTTPException(
            status_code=413,
            detail=f"File size {file_size_mb:.2f} MB exceeds maximum {MAX_FILE_SIZE_MB} MB",
        )
    if len(contents) == 0:
        logger.warning(f"[request_id={request_id}] [action=predict_image] [status=rejected] reason=empty_file")
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Magic numbers header validation (JPEG starts with \xFF\xD8\xFF, PNG with \x89PNG)
    if len(contents) >= 4:
        if contents.startswith(b"\xff\xd8\xff") or contents.startswith(b"\x89PNG\r\n\x1a\n"):
            pass
        else:
            logger.warning(f"[request_id={request_id}] [action=predict_image] [status=rejected] reason=invalid_magic_bytes")
            raise HTTPException(
                status_code=400,
                detail="Unsupported image format. File headers do not match a valid JPEG or PNG image.",
            )

    # Step 3 — Save temporarily to disk for audit
    save_filename = f"{request_id}_{file.filename}"
    save_path = UPLOAD_DIR / save_filename
    try:
        # Offload file writing to a background thread to keep event loop free
        await asyncio.to_thread(save_path.write_bytes, contents)
        logger.info(f"[request_id={request_id}] [action=save_temp_file] [status=success] path={save_path}")
    except Exception as exc:
        logger.error(f"[request_id={request_id}] [action=save_temp_file] [status=failed] error={exc}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    # Step 4 — Run preprocessing and inference concurrently with a 15-second timeout
    try:
        preprocess_result, prediction_result = await asyncio.wait_for(
            _process_image_async(contents, request_id),
            timeout=15.0
        )
    except asyncio.TimeoutError:
        logger.error(f"[request_id={request_id}] [action=predict_image] [status=timeout] timeout_limit=15.0s")
        raise HTTPException(status_code=504, detail="Prediction request timed out")

    # Step 5 — Build structured response
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
        f"[request_id={request_id}] [action=predict_image] [status=completed] "
        f"detections={len(detection_items)} duration_ms={processing_time * 1000:.1f}"
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


def _validate_video_format(filename: str) -> str:
    """Validate uploaded file has a supported video extension."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_format",
                "message": f"File format '{ext}' is not supported. "
                           f"Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}",
                "filename": filename,
            },
        )
    return ext


@router.post(
    "/video",
    response_model=UploadVideoResponse,
    summary="Upload a video for defect prediction",
    description=(
        "Upload a single video file (MP4, AVI, MOV) for defect detection. "
        "The video is validated, saved temporarily to disk, analyzed with OpenCV "
        "to extract metadata (FPS, frame count, resolution), run through the "
        "YOLO inference service, and returns aggregated defect summary metrics."
    ),
    responses={
        400: {"description": "Invalid file format or empty file"},
        413: {"description": "File too large"},
        500: {"description": "Internal processing error"},
    },
)
async def predict_video(
    request: Request,
    file: UploadFile = File(
        ...,
        description="Video file to analyze for defects (MP4, AVI, MOV)",
    ),
):
    """
    End-to-end video defect prediction workflow.
    """
    request_id = _generate_request_id()
    start_time = time.time()

    logger.info(
        f"[{request_id}] Video prediction request received: "
        f"filename={file.filename}, content_type={file.content_type}"
    )

    # Step 1 — Validate format
    ext = _validate_video_format(file.filename)

    # Step 2 — Read file contents
    try:
        contents = await file.read()
    except Exception as exc:
        logger.error(f"[{request_id}] Failed to read uploaded file: {exc}")
        raise HTTPException(status_code=500, detail="Failed to read uploaded file")

    # Step 3 — Validate file size
    file_size_mb = len(contents) / (1024 * 1024)
    if file_size_mb > MAX_VIDEO_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File size {file_size_mb:.2f} MB exceeds maximum {MAX_VIDEO_SIZE_MB} MB",
        )
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Step 4 — Save temporarily to disk (OpenCV needs a file path)
    save_filename = f"{request_id}_{file.filename}"
    save_path = UPLOAD_DIR / save_filename
    try:
        with open(save_path, "wb") as f:
            f.write(contents)
        logger.info(f"[{request_id}] Saved video to {save_path}")
    except IOError as exc:
        logger.error(f"[{request_id}] Failed to save file: {exc}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    # Step 5 — Run inference
    try:
        model_svc = get_model_service()
        prediction_result = model_svc.predict_video(save_path)
        if prediction_result.get("status") == "error":
            raise ValueError(prediction_result.get("error_message", "Unknown error"))
        logger.info(f"[{request_id}] Video inference complete")
    except Exception as exc:
        logger.error(f"[{request_id}] Video inference failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(exc)}")
    finally:
        # Clean up video file after processing to save disk space
        if save_path.exists():
            try:
                os.remove(save_path)
                logger.info(f"[{request_id}] Cleaned up temp video file {save_path}")
            except Exception as e:
                logger.warning(f"[{request_id}] Failed to delete temp video file: {e}")

    # Step 6 — Build response
    processing_time = time.time() - start_time

    summary_items = []
    for item in prediction_result.get("detection_summary", []):
        summary_items.append(FrameDetectionSummary(
            **{"class": item["class"]},
            class_id=item["class_id"],
            count=item["count"],
        ))

    prediction_details = VideoPredictionDetails(
        detection_summary=summary_items,
        total_detections=prediction_result.get("total_detections", 0),
        processing_time=f"{processing_time:.2f} s",
        frame_count=prediction_result.get("frame_count", 0),
        fps=prediction_result.get("fps", 0.0),
        video_duration_seconds=prediction_result.get("video_duration_seconds", 0.0),
        video_resolution=prediction_result.get("video_resolution", [0, 0]),
        model=prediction_result.get("model", "yolov8n_defects"),
    )

    logger.info(
        f"[{request_id}] Video request completed in {processing_time * 1000:.1f} ms"
    )

    return UploadVideoResponse(
        request_id=request_id,
        filename=file.filename,
        status="success",
        message="Video processed successfully",
        file_size_mb=round(file_size_mb, 3),
        processing_time_ms=round(processing_time * 1000, 1),
        prediction=prediction_details,
    )


@router.post(
    "/live",
    response_model=UploadLiveResponse,
    summary="Initiate live stream defect prediction",
    description=(
        "Start a defect detection session on a live RTSP/RTMP stream or webcam. "
        "The stream is validated and initialized using OpenCV VideoCapture with "
        "automatic error recovery and retry logic."
    ),
    responses={
        400: {"description": "Invalid stream URL or connection failure"},
        500: {"description": "Internal server error"},
    },
)
async def predict_live(
    request_data: LiveStreamRequest,
    request: Request,
):
    request_id = _generate_request_id()
    logger.info(
        f"[{request_id}] Live stream connection requested for: {request_data.source}"
    )

    source = request_data.source
    # Convert source to integer if it is a digit (e.g. webcam '0')
    if source.isdigit():
        source = int(source)

    cap = None
    connected = False
    
    # Connection retry logic for automatic error recovery
    for attempt in range(1, 4):
        try:
            logger.info(f"[{request_id}] Connection attempt {attempt}/3 to source: {source}")
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                # Verify we can read a frame
                for read_attempt in range(1, 4):
                    ret, frame = cap.read()
                    if ret:
                        connected = True
                        break
                    time.sleep(0.1)
                if connected:
                    break
            logger.warning(f"[{request_id}] Attempt {attempt} failed to stream from source: {source}")
        except Exception as e:
            logger.warning(f"[{request_id}] Attempt {attempt} exception: {e}")
        finally:
            if not connected and cap is not None:
                cap.release()
        if attempt < 3:
            time.sleep(0.5)

    if not connected:
        logger.error(f"[{request_id}] Live stream connection failed after 3 attempts")
        raise HTTPException(
            status_code=400,
            detail=f"Live stream connection failed for source: {request_data.source} after 3 attempts.",
        )

    try:
        # Read resolution and FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Build response details
        prediction_details = LiveStreamPredictionDetails(
            status="connected",
            source=str(request_data.source),
            fps=round(fps, 2) if fps > 0 else 30.0,
            video_resolution=[width, height] if (width > 0 and height > 0) else [1280, 720],
            model="yolov8n_defects",
        )

        logger.info(
            f"[{request_id}] Live stream connection successful: resolution={width}x{height}, fps={fps}"
        )

        return UploadLiveResponse(
            request_id=request_id,
            status="success",
            message="Live stream connection established successfully",
            prediction=prediction_details,
        )
    except Exception as exc:
        logger.error(f"[{request_id}] Live stream configuration extraction failed: {exc}")
        raise HTTPException(
            status_code=400,
            detail=f"Live stream configuration extraction failed: {str(exc)}",
        )
