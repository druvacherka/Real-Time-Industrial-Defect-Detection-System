"""
main.py — FastAPI Application Entry Point
===========================================
Initialises the FastAPI app, configures middleware, registers routes,
and sets up logging for the Industrial Defect Detection API.

Run with:
    uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import (
    APP_NAME,
    APP_VERSION,
    APP_DESCRIPTION,
    ALLOWED_ORIGINS,
)
from app.core.logger import setup_logging, get_logger
from app.api.routes.router import api_router


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


# ── Request/Response Logging Middleware ─────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware that logs every incoming request and its response status.
    Catches and logs unhandled exceptions so they don't go silent.
    """
    logger = get_logger("middleware")
    logger.info(
        "Incoming  %s %s from %s",
        request.method,
        request.url.path,
        request.client.host if request.client else "unknown",
    )

    try:
        response = await call_next(request)
        logger.info(
            "Completed %s %s — status %s",
            request.method,
            request.url.path,
            response.status_code,
        )
        return response
    except Exception as exc:
        logger.exception(
            "Unhandled error during %s %s: %s",
            request.method,
            request.url.path,
            str(exc),
        )
        raise


# ── Register Routes ────────────────────────────────────────────────────────
app.include_router(api_router)
