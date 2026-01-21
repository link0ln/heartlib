"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Model settings
    model_path: str = "/app/ckpt"
    model_dtype: str = "float32"
    hybrid_mode: bool = False  # LLM on CPU, HeartCodec on GPU
    sequential_offload: bool = False  # Offload models between generation steps

    # Output settings
    output_path: str = "/app/outputs"

    # Redis settings
    redis_url: str = "redis://localhost:6379"

    # Generation defaults
    default_max_duration_ms: int = 120000
    default_temperature: float = 1.0
    default_cfg_scale: float = 1.5

    # Job settings
    job_ttl_seconds: int = 3600  # 1 hour
    max_concurrent_jobs: int = 1

    # API settings
    api_title: str = "HeartMuLa Music Generation API"
    api_version: str = "1.0.0"
    api_description: str = "HTTP API for HeartMuLa-oss-3B music generation with F32 precision"

    class Config:
        env_prefix = "HEARTMULA_"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
