"""Generate router for music generation endpoints."""

import logging
import os
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from fastapi.responses import FileResponse

from app.config import get_settings
from app.models.requests import GenerateRequest
from app.models.responses import (
    GenerateResponse,
    JobStatus,
    JobStatusResponse,
    ErrorResponse,
)
from app.services.job_manager import get_job_manager
from app.services.music_generator import get_generator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/generate", tags=["generate"])


async def process_generation_job(job_id: str) -> None:
    """Background task to process music generation."""
    job_manager = get_job_manager()
    generator = get_generator()
    settings = get_settings()

    try:
        # Get job details
        job = await job_manager.get_job(job_id)
        if job is None:
            logger.error(f"Job {job_id} not found")
            return

        if job.status == JobStatus.CANCELLED:
            logger.info(f"Job {job_id} was cancelled, skipping")
            return

        # Update status to processing
        await job_manager.update_job(job_id, status=JobStatus.PROCESSING, progress=0.0)

        # Generate output path
        output_filename = f"{job_id}.wav"
        output_path = os.path.join(settings.output_path, output_filename)

        # Generate music
        logger.info(f"Starting generation for job {job_id}")
        await job_manager.update_job(job_id, progress=10.0)

        generator.generate(
            lyrics=job.lyrics,
            tags=job.tags,
            output_path=output_path,
            max_duration_ms=job.max_duration_ms,
            temperature=job.temperature,
            cfg_scale=job.cfg_scale,
        )

        # Update job as completed
        await job_manager.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress=100.0,
            output_file=output_filename,
        )

        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        await job_manager.update_job(
            job_id,
            status=JobStatus.FAILED,
            error=str(e),
        )


@router.post(
    "",
    response_model=GenerateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {"description": "Job created successfully"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
)
async def create_generation_job(
    request: GenerateRequest,
    background_tasks: BackgroundTasks,
) -> GenerateResponse:
    """
    Create a new music generation job.

    The job will be processed asynchronously. Use the returned URLs
    to check status and download the result when completed.
    """
    job_manager = get_job_manager()
    generator = get_generator()
    settings = get_settings()

    # Check if model is loaded
    if not generator.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded yet. Please wait.",
        )

    # Check if Redis is connected
    if not job_manager.is_connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Job queue not available. Please try again.",
        )

    # Create job
    job = await job_manager.create_job(
        lyrics=request.lyrics,
        tags=request.tags,
        max_duration_ms=request.max_duration_ms or settings.default_max_duration_ms,
        temperature=request.temperature or settings.default_temperature,
        cfg_scale=request.cfg_scale or settings.default_cfg_scale,
    )

    # Start background task
    background_tasks.add_task(process_generation_job, job.job_id)

    return GenerateResponse(
        job_id=job.job_id,
        status=job.status,
        status_url=f"/generate/{job.job_id}/status",
        result_url=f"/generate/{job.job_id}/result",
        created_at=job.created_at,
    )


@router.get(
    "/{job_id}/status",
    response_model=JobStatusResponse,
    responses={
        200: {"description": "Job status retrieved"},
        404: {"model": ErrorResponse, "description": "Job not found"},
    },
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Get the status of a generation job."""
    job_manager = get_job_manager()
    job = await job_manager.get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    result_url = None
    if job.status == JobStatus.COMPLETED and job.output_file:
        result_url = f"/generate/{job_id}/result"

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        error=job.error,
        result_url=result_url,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.get(
    "/{job_id}/result",
    responses={
        200: {"description": "Audio file", "content": {"audio/wav": {}}},
        404: {"model": ErrorResponse, "description": "Job or file not found"},
        400: {"model": ErrorResponse, "description": "Job not completed"},
    },
)
async def get_job_result(job_id: str) -> FileResponse:
    """Download the generated audio file."""
    job_manager = get_job_manager()
    settings = get_settings()

    job = await job_manager.get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is not completed. Current status: {job.status.value}",
        )

    if not job.output_file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Output file not found",
        )

    file_path = os.path.join(settings.output_path, job.output_file)

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Output file not found on disk",
        )

    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        filename=f"heartmula_{job_id}.wav",
    )


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        204: {"description": "Job cancelled or deleted"},
        404: {"model": ErrorResponse, "description": "Job not found"},
        400: {"model": ErrorResponse, "description": "Cannot cancel job"},
    },
)
async def cancel_job(job_id: str) -> None:
    """Cancel a queued job or delete a completed/failed job."""
    job_manager = get_job_manager()
    settings = get_settings()

    job = await job_manager.get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    if job.status == JobStatus.PROCESSING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot cancel a job that is currently processing",
        )

    if job.status == JobStatus.QUEUED:
        # Cancel the job
        await job_manager.cancel_job(job_id)
    else:
        # Delete completed/failed/cancelled jobs
        # Also delete the output file if exists
        if job.output_file:
            file_path = os.path.join(settings.output_path, job.output_file)
            if os.path.exists(file_path):
                os.remove(file_path)

        await job_manager.delete_job(job_id)
