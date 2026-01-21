"""FastAPI application for HeartMuLa music generation."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import get_settings
from app.routers import generate, health, api
from app.services.job_manager import get_job_manager
from app.services.music_generator import get_generator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    settings = get_settings()

    logger.info("Starting HeartMuLa API...")
    logger.info(f"Model path: {settings.model_path}")
    logger.info(f"Model dtype: {settings.model_dtype}")
    logger.info(f"Output path: {settings.output_path}")

    # Connect to Redis
    job_manager = get_job_manager()
    try:
        await job_manager.connect()
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        # Continue anyway - we'll show degraded status in health check

    # Load the model (singleton)
    generator = get_generator()
    try:
        generator.load_model()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Continue anyway - we'll show degraded status in health check

    logger.info("HeartMuLa API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down HeartMuLa API...")

    # Disconnect from Redis
    await job_manager.disconnect()

    # Unload model
    generator.unload_model()

    logger.info("HeartMuLa API shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=settings.api_description,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router)
    app.include_router(generate.router)
    app.include_router(api.router)

    # Mount static files
    static_path = Path(__file__).parent / "static"
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

    return app


# Create app instance
app = create_app()


@app.get("/", include_in_schema=False)
async def root():
    """Serve the WebUI."""
    static_path = Path(__file__).parent / "static" / "index.html"
    if static_path.exists():
        return FileResponse(str(static_path))
    return {
        "message": "HeartMuLa Music Generation API",
        "docs": "/docs",
        "health": "/health",
    }
