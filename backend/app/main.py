"""
main.py — FastAPI Application Entry Point
===========================================
Initialises the FastAPI app, configures middleware, registers routes,
and sets up logging for the Industrial Defect Detection API.

Run with:
    uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.core.config import (
    APP_NAME,
    APP_VERSION,
    APP_DESCRIPTION,
    ALLOWED_ORIGINS,
)
from app.core.logger import setup_logging, get_logger
from app.api.routes.router import api_router
from app.services.image_service import ImageValidationError
from app.schemas.responses import ErrorResponse


# ── Lifespan Handler ───────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(application: FastAPI):
    """
    Startup and shutdown lifecycle events.

    On startup:
      - Initialise logging
      - Log that the server is ready

    On shutdown:
      - Log graceful shutdown
    """
    # ── Startup ─────────────────────────────────────────────────────────
    setup_logging()
    logger = get_logger("main")
    logger.info(
        "Starting %s v%s — server is ready to accept requests",
        APP_NAME,
        APP_VERSION,
    )

    yield  # application runs here

    # ── Shutdown ────────────────────────────────────────────────────────
    logger.info("Shutting down %s — goodbye!", APP_NAME)


# ── Create FastAPI Instance ────────────────────────────────────────────────
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    docs_url="/docs",          # Swagger UI
    redoc_url="/redoc",        # ReDoc
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ── CORS Middleware ────────────────────────────────────────────────────────
app.add_middleware(
	CORSMiddleware,
	allow_origins=ALLOWED_ORIGINS,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# ── Prometheus Instrumentation ──────────────────────────────────────────────
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
except ImportError:
    pass


# ── Request/Response Logging Middleware ─────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware that logs every incoming request, its response status, and duration.
    Catches and logs unhandled exceptions so they don't go silent.
    """
    import time
    logger = get_logger("middleware")
    start_time = time.time()
    logger.info(
        "Incoming  %s %s from %s",
        request.method,
        request.url.path,
        request.client.host if request.client else "unknown",
    )

    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "Completed %s %s — status %s — took %.2f ms",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response
    except Exception as exc:
        duration_ms = (time.time() - start_time) * 1000
        logger.exception(
            "Unhandled error during %s %s after %.2f ms: %s",
            request.method,
            request.url.path,
            duration_ms,
            str(exc),
        )
        raise


# ── Exception Handlers ──────────────────────────────────────────────────────
@app.exception_handler(ImageValidationError)
async def image_validation_exception_handler(request: Request, exc: ImageValidationError):
    logger = get_logger("exceptions")
    logger.warning("Image validation failed for %s: %s", request.url.path, str(exc))
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="image_validation_failed",
            message=str(exc),
            detail="The uploaded image did not meet the validation criteria (e.g., too small, corrupted, or unsupported channel layout)."
        ).model_dump()
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger = get_logger("exceptions")
    logger.warning("HTTP error occurred for %s: %s", request.url.path, exc.detail)
    detail_msg = exc.detail
    error_code = "http_error"
    if isinstance(detail_msg, dict):
        error_code = detail_msg.get("error", "http_error")
        detail_msg = detail_msg.get("message", str(detail_msg))
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=error_code,
            message=str(detail_msg),
            detail=f"HTTP exception raised during request processing: status {exc.status_code}"
        ).model_dump()
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger = get_logger("exceptions")
    logger.warning("Request validation failed for %s: %s", request.url.path, str(exc.errors()))
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="validation_failed",
            message="Input validation failed",
            detail=str(exc.errors())
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger = get_logger("exceptions")
    logger.exception("Unhandled internal exception during request to %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_server_error",
            message="An unexpected error occurred during request processing",
            detail=str(exc)
        ).model_dump()
    )


# ── Register Routes ────────────────────────────────────────────────────────
app.include_router(api_router)
