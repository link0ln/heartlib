"""Redis-based job queue manager for async music generation."""

import json
import logging
from datetime import datetime
from typing import Optional
from uuid import uuid4

import redis.asyncio as redis

from app.config import get_settings
from app.models.responses import JobStatus

logger = logging.getLogger(__name__)


class Job:
    """Job data structure."""

    def __init__(
        self,
        job_id: str,
        status: JobStatus,
        lyrics: str,
        tags: str,
        max_duration_ms: int,
        temperature: float,
        cfg_scale: float,
        created_at: datetime,
        updated_at: datetime,
        progress: float = 0.0,
        error: Optional[str] = None,
        output_file: Optional[str] = None,
    ):
        self.job_id = job_id
        self.status = status
        self.lyrics = lyrics
        self.tags = tags
        self.max_duration_ms = max_duration_ms
        self.temperature = temperature
        self.cfg_scale = cfg_scale
        self.created_at = created_at
        self.updated_at = updated_at
        self.progress = progress
        self.error = error
        self.output_file = output_file

    def to_dict(self) -> dict:
        """Convert job to dictionary."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "lyrics": self.lyrics,
            "tags": self.tags,
            "max_duration_ms": self.max_duration_ms,
            "temperature": self.temperature,
            "cfg_scale": self.cfg_scale,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "progress": self.progress,
            "error": self.error,
            "output_file": self.output_file,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        """Create job from dictionary."""
        return cls(
            job_id=data["job_id"],
            status=JobStatus(data["status"]),
            lyrics=data["lyrics"],
            tags=data["tags"],
            max_duration_ms=data["max_duration_ms"],
            temperature=data["temperature"],
            cfg_scale=data["cfg_scale"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            progress=data.get("progress", 0.0),
            error=data.get("error"),
            output_file=data.get("output_file"),
        )


class JobManager:
    """Async Redis job queue manager."""

    def __init__(self):
        self._redis: Optional[redis.Redis] = None
        self._connected: bool = False

    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self._connected

    async def connect(self) -> None:
        """Connect to Redis."""
        if self._connected:
            return

        settings = get_settings()
        logger.info(f"Connecting to Redis at {settings.redis_url}")

        self._redis = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )

        # Test connection
        try:
            await self._redis.ping()
            self._connected = True
            logger.info("Redis connected successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            self._connected = False
            logger.info("Redis disconnected")

    def _job_key(self, job_id: str) -> str:
        """Get Redis key for a job."""
        return f"heartmula:job:{job_id}"

    async def create_job(
        self,
        lyrics: str,
        tags: str,
        max_duration_ms: int,
        temperature: float,
        cfg_scale: float,
    ) -> Job:
        """Create a new job in the queue."""
        if not self._redis:
            raise RuntimeError("Redis not connected")

        settings = get_settings()
        job_id = str(uuid4())
        now = datetime.utcnow()

        job = Job(
            job_id=job_id,
            status=JobStatus.QUEUED,
            lyrics=lyrics,
            tags=tags,
            max_duration_ms=max_duration_ms,
            temperature=temperature,
            cfg_scale=cfg_scale,
            created_at=now,
            updated_at=now,
        )

        # Store job in Redis with TTL
        key = self._job_key(job_id)
        await self._redis.set(
            key,
            json.dumps(job.to_dict()),
            ex=settings.job_ttl_seconds,
        )

        logger.info(f"Created job {job_id}")
        return job

    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        if not self._redis:
            raise RuntimeError("Redis not connected")

        key = self._job_key(job_id)
        data = await self._redis.get(key)

        if data is None:
            return None

        return Job.from_dict(json.loads(data))

    async def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress: Optional[float] = None,
        error: Optional[str] = None,
        output_file: Optional[str] = None,
    ) -> Optional[Job]:
        """Update a job's status and metadata."""
        if not self._redis:
            raise RuntimeError("Redis not connected")

        job = await self.get_job(job_id)
        if job is None:
            return None

        # Update fields
        if status is not None:
            job.status = status
        if progress is not None:
            job.progress = progress
        if error is not None:
            job.error = error
        if output_file is not None:
            job.output_file = output_file

        job.updated_at = datetime.utcnow()

        # Save back to Redis
        settings = get_settings()
        key = self._job_key(job_id)
        await self._redis.set(
            key,
            json.dumps(job.to_dict()),
            ex=settings.job_ttl_seconds,
        )

        logger.info(f"Updated job {job_id}: status={job.status.value}, progress={job.progress}")
        return job

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job."""
        job = await self.get_job(job_id)
        if job is None:
            return False

        if job.status != JobStatus.QUEUED:
            # Can only cancel queued jobs
            return False

        await self.update_job(job_id, status=JobStatus.CANCELLED)
        logger.info(f"Cancelled job {job_id}")
        return True

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job from Redis."""
        if not self._redis:
            raise RuntimeError("Redis not connected")

        key = self._job_key(job_id)
        result = await self._redis.delete(key)
        return result > 0


# Global instance
_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Get the global JobManager instance."""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager
