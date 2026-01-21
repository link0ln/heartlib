"""Pydantic models for API responses."""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job status enum."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GenerateResponse(BaseModel):
    """Response model for job creation."""

    job_id: str = Field(
        ...,
        description="Unique job identifier (UUID)"
    )

    status: JobStatus = Field(
        default=JobStatus.QUEUED,
        description="Current job status"
    )

    status_url: str = Field(
        ...,
        description="URL to check job status"
    )

    result_url: str = Field(
        ...,
        description="URL to download result when completed"
    )

    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Job creation timestamp"
    )


class JobStatusResponse(BaseModel):
    """Response model for job status query."""

    job_id: str = Field(
        ...,
        description="Unique job identifier"
    )

    status: JobStatus = Field(
        ...,
        description="Current job status"
    )

    progress: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Progress percentage (0-100)"
    )

    error: Optional[str] = Field(
        default=None,
        description="Error message if failed"
    )

    result_url: Optional[str] = Field(
        default=None,
        description="URL to download result when completed"
    )

    created_at: datetime = Field(
        ...,
        description="Job creation timestamp"
    )

    updated_at: datetime = Field(
        ...,
        description="Last update timestamp"
    )


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(
        default="healthy",
        description="Service health status"
    )

    model_loaded: bool = Field(
        ...,
        description="Whether the model is loaded"
    )

    redis_connected: bool = Field(
        ...,
        description="Whether Redis is connected"
    )

    gpu_available: bool = Field(
        ...,
        description="Whether GPU is available"
    )

    version: str = Field(
        ...,
        description="API version"
    )


class ErrorResponse(BaseModel):
    """Response model for errors."""

    detail: str = Field(
        ...,
        description="Error message"
    )

    error_code: Optional[str] = Field(
        default=None,
        description="Error code for programmatic handling"
    )
