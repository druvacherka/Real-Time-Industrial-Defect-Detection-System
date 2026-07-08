"""
endpoints/health.py — Health Check Endpoint
=============================================
Provides a service health status for monitoring and load balancers.
"""

from fastapi import APIRouter

from app.schemas.responses import HealthResponse
from app.core.config import APP_VERSION
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the current health status of the API service along with CPU and memory usage.",
    tags=["General"],
)
async def health_check():
    """
    Health-check endpoint — used by monitoring tools and load balancers
    to verify the service is alive and ready to handle requests.

    Returns the service name, status, and current version.
    """
    logger.info("Health check requested")
    
    cpu_usage = 0.0
    mem_usage = 0.0
    try:
        import psutil
        cpu_usage = psutil.cpu_percent()
        mem_usage = psutil.virtual_memory().percent
    except ImportError:
        # Fallback if psutil is not installed
        pass

    return HealthResponse(
        status="healthy",
        service="Industrial Defect Detection",
        version=APP_VERSION,
        cpu_percent=cpu_usage,
        memory_percent=mem_usage,
    )
