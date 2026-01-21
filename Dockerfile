FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy heartlib repository (cloned locally)
COPY heartlib/ /app/heartlib/

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install heartlib
RUN pip install --no-cache-dir -e /app/heartlib

# Copy application code
COPY app/ /app/app/

# Create outputs directory
RUN mkdir -p /app/outputs

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HEARTMULA_MODEL_PATH=/app/ckpt
ENV HEARTMULA_OUTPUT_PATH=/app/outputs

# Run uvicorn
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
