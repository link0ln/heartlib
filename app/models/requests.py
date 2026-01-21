"""Pydantic models for API requests."""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


class GenerateRequest(BaseModel):
    """Request model for music generation."""

    lyrics: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Lyrics with section markers like [Verse], [Chorus], etc.",
        examples=["[Verse]\nHello world\n[Chorus]\nSing along"]
    )

    tags: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Comma-separated music tags: genre, instruments, mood, language",
        examples=["pop,piano,happy,english"]
    )

    max_duration_ms: Optional[int] = Field(
        default=120000,
        ge=5000,
        le=300000,
        description="Maximum duration in milliseconds (5s - 5min)"
    )

    temperature: Optional[float] = Field(
        default=1.0,
        ge=0.1,
        le=2.0,
        description="Sampling temperature (higher = more creative)"
    )

    cfg_scale: Optional[float] = Field(
        default=1.5,
        ge=1.0,
        le=10.0,
        description="Classifier-free guidance scale"
    )

    @field_validator("lyrics")
    @classmethod
    def validate_lyrics(cls, v: str) -> str:
        """Validate lyrics contain at least one section marker."""
        v = v.strip()
        if not v:
            raise ValueError("Lyrics cannot be empty")
        # Allow lyrics without section markers but recommend them
        return v

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: str) -> str:
        """Validate and normalize tags."""
        v = v.strip()
        if not v:
            raise ValueError("Tags cannot be empty")
        # Normalize: lowercase, strip whitespace around commas
        tags = [tag.strip().lower() for tag in v.split(",") if tag.strip()]
        if not tags:
            raise ValueError("At least one tag is required")
        return ",".join(tags)
