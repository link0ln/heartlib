"""Health check router."""

import logging

import torch
from fastapi import APIRouter

from app.config import get_settings
from app.models.responses import HealthResponse
from app.services.job_manager import get_job_manager
from app.services.music_generator import get_generator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={
        200: {"description": "Health status"},
    },
)
async def health_check() -> HealthResponse:
    """
    Check the health status of the service.

    Returns information about:
    - Model loading status
    - Redis connection status
    - GPU availability
    """
    settings = get_settings()
    generator = get_generator()
    job_manager = get_job_manager()

    # Check GPU
    gpu_available = torch.cuda.is_available()

    # Determine overall status
    model_loaded = generator.is_loaded
    redis_connected = job_manager.is_connected

    if model_loaded and redis_connected:
        status = "healthy"
    elif model_loaded or redis_connected:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        redis_connected=redis_connected,
        gpu_available=gpu_available,
        version=settings.api_version,
    )
