"""
endpoints/root.py — Root Endpoint
===================================
Handles the base URL endpoint that returns a welcome message.
"""

from fastapi import APIRouter

from app.schemas.responses import RootResponse
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/",
    response_model=RootResponse,
    summary="Root endpoint",
    description="Returns a welcome message confirming the API is reachable.",
    tags=["General"],
)
async def root():
    """
    Root endpoint — returns the API welcome message.

    This is a lightweight endpoint useful for quickly checking
    whether the server is up and responding to requests.
    """
    logger.info("Root endpoint hit")
    return RootResponse(message="Industrial Defect Detection API")
