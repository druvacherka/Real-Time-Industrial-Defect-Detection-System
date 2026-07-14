"""
endpoints/health.py — Health Check Endpoint
=============================================
Provides a service health status for monitoring and load balancers.
"""

from fastapi import APIRouter

from app.schemas.responses import HealthResponse
from app.core.config import APP_VERSION
from app.core.logger import get_logger
from app.services.model_service import get_model_service

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the current health status of the API service along with CPU, memory usage, and model load status.",
    tags=["General"],
)
async def health_check():
    """
    Health-check endpoint — used by monitoring tools and load balancers
    to verify the service is alive and ready to handle requests.

    Returns the service name, status, current version, and resource usage.
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

    # Check model loading status
    model_loaded = False
    try:
        model_svc = get_model_service()
        if model_svc and model_svc.model_wrapper and model_svc.model_wrapper.is_loaded:
            model_loaded = True
    except Exception as exc:
        logger.error(f"Health check model status verification failed: {exc}")

    # Retrieve queue statistics
    q_size = 0
    jobs_count = 0
    avg_wait = 0.0
    try:
        from app.services.queue_manager import get_queue_manager
        qm = get_queue_manager()
        q_size = qm.queue.qsize()
        jobs_count = qm.total_jobs_processed
        avg_wait = (qm.total_wait_time / jobs_count * 1000) if jobs_count > 0 else 0.0
    except Exception as exc:
        logger.error(f"Failed to retrieve queue stats: {exc}")

    status = "healthy" if model_loaded else "degraded"

    return HealthResponse(
        status=status,
        service="Industrial Defect Detection",
        version=APP_VERSION,
        cpu_percent=cpu_usage,
        memory_percent=mem_usage,
        queue_size=q_size,
        jobs_processed=jobs_count,
        avg_queue_wait_ms=round(avg_wait, 2)
    )
