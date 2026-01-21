"""API router for GPU monitoring and system info."""

import logging
import subprocess
import re
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["api"])


class GPUStats(BaseModel):
    """GPU statistics model."""
    available: bool = False
    name: Optional[str] = None
    utilization: int = 0
    memory_used: int = 0  # MB
    memory_total: int = 0  # MB
    temperature: int = 0
    power_draw: int = 0
    power_limit: int = 0


def get_nvidia_smi_output() -> Optional[str]:
    """Get nvidia-smi output, trying multiple paths."""
    paths = [
        "nvidia-smi",
        "/usr/bin/nvidia-smi",
        "/usr/lib/wsl/lib/nvidia-smi",
    ]

    for path in paths:
        try:
            result = subprocess.run(
                [path, "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    return None


@router.get("/gpu", response_model=GPUStats)
async def get_gpu_stats() -> GPUStats:
    """Get current GPU statistics."""
    output = get_nvidia_smi_output()

    if not output:
        return GPUStats(available=False)

    try:
        # Parse CSV output: name, utilization.gpu, memory.used, memory.total, temperature.gpu, power.draw, power.limit
        parts = [p.strip() for p in output.split(",")]

        if len(parts) >= 7:
            return GPUStats(
                available=True,
                name=parts[0],
                utilization=int(float(parts[1])) if parts[1] else 0,
                memory_used=int(float(parts[2])) if parts[2] else 0,
                memory_total=int(float(parts[3])) if parts[3] else 0,
                temperature=int(float(parts[4])) if parts[4] else 0,
                power_draw=int(float(parts[5])) if parts[5] else 0,
                power_limit=int(float(parts[6])) if parts[6] else 0,
            )
    except (ValueError, IndexError) as e:
        logger.warning(f"Failed to parse nvidia-smi output: {e}")

    return GPUStats(available=False)


@router.get("/system")
async def get_system_info():
    """Get system information."""
    import torch
    from app.config import get_settings
    from app.services.music_generator import get_generator
    from app.services.job_manager import get_job_manager

    settings = get_settings()
    generator = get_generator()
    job_manager = get_job_manager()

    return {
        "model_loaded": generator.is_loaded,
        "model_dtype": settings.model_dtype,
        "redis_connected": job_manager.is_connected,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
