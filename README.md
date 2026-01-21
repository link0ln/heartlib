# HeartMuLa Local WEB-UI

**Self-hosted HTTP API for HeartMuLa-oss-3B music generation with VRAM optimization**

A Docker-based deployment solution for running [HeartMuLa-oss-3B](https://github.com/HeartMuLa/HeartMuLa) locally with a REST API interface. Designed to run on consumer GPUs (16-24GB VRAM) through intelligent memory management.

## Key Features

- **REST API** — FastAPI-based HTTP endpoints for music generation
- **Async Job Queue** — Redis-backed job system for long-running generations
- **Sequential Offload** — Automatically swaps LLM and HeartCodec between GPU/CPU during generation phases to minimize VRAM usage
- **KV Cache Management** — Proper cleanup of attention caches between generations to prevent memory bloat
- **Docker Compose** — One-command deployment with all dependencies
- **Web UI** — Simple browser interface for testing

## Differences from Original HeartMuLa

| Feature | Original | This Fork |
|---------|----------|-----------|
| Interface | Python library | HTTP REST API |
| Deployment | Manual setup | Docker Compose |
| VRAM Usage | ~17+ GB constant | ~7 GB idle, peaks during generation |
| Memory Management | Standard | Sequential offload with KV cache cleanup |
| Job Handling | Synchronous | Async with Redis queue |
| Multiple Generations | Memory accumulates | Clean state between runs |

## VRAM Optimization

The sequential offload system works in phases:

1. **Phase 0** — Load LLM to GPU, delete stale KV caches
2. **Phase 1** — Token generation (LLM on GPU)
3. **Phase 2** — Delete KV caches, move LLM to CPU, load HeartCodec to GPU
4. **Phase 3** — Audio decoding (HeartCodec on GPU)
5. **Phase 4** — Move HeartCodec to CPU, cleanup

This allows running on GPUs with 16-24GB VRAM that couldn't handle the full model in standard mode.

## Requirements

- NVIDIA GPU with 16+ GB VRAM (tested on RTX 3090)
- Docker with NVIDIA Container Toolkit
- ~20 GB disk space for model weights

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/heartmula-local-api.git
cd heartmula-local-api
```

### 2. Download models

```bash
# Install huggingface-cli if needed
pip install huggingface_hub

# Download HeartMuLa model
huggingface-cli download HeartMuLa/HeartMuLa-oss-3B --local-dir ./ckpt/HeartMuLa-oss-3B

# Download HeartCodec model
huggingface-cli download HeartMuLa/HeartCodec-oss --local-dir ./ckpt/HeartCodec-oss

# Download tokenizer and config
huggingface-cli download HeartMuLa/HeartMuLa-oss-3B tokenizer.json --local-dir ./ckpt
huggingface-cli download HeartMuLa/HeartMuLa-oss-3B gen_config.json --local-dir ./ckpt
```

### 3. Start services

```bash
docker compose up -d
```

### 4. Generate music

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "lyrics": "[Verse]\nHello world\n[Chorus]\nLa la la",
    "tags": "pop,piano,english",
    "max_duration_ms": 60000
  }'
```

Or open http://localhost:8000 in your browser for the Web UI.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/generate` | Create generation job |
| GET | `/generate/{job_id}/status` | Check job status |
| GET | `/generate/{job_id}/result` | Download audio file |
| DELETE | `/generate/{job_id}` | Cancel job |
| GET | `/health` | Health check |
| GET | `/api/gpu` | GPU statistics |

### Generate Request

```json
{
  "lyrics": "[Verse]\nYour lyrics here\n[Chorus]\nChorus lyrics",
  "tags": "pop,rock,english,female",
  "max_duration_ms": 120000,
  "temperature": 1.0,
  "cfg_scale": 1.5
}
```

### Generate Response

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "status_url": "/generate/550e8400.../status",
  "result_url": "/generate/550e8400.../result"
}
```

### Job Status Response

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100.0,
  "created_at": "2024-01-21T12:00:00",
  "completed_at": "2024-01-21T12:03:00"
}
```

## Configuration

Environment variables in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `HEARTMULA_MODEL_DTYPE` | `float16` | Model precision (float16/float32) |
| `HEARTMULA_SEQUENTIAL_OFFLOAD` | `true` | Enable VRAM optimization |
| `HEARTMULA_HYBRID_MODE` | `false` | Legacy mode (LLM on CPU) |
| `HEARTMULA_MODEL_PATH` | `/app/ckpt` | Path to model weights |
| `HEARTMULA_OUTPUT_PATH` | `/app/outputs` | Path for generated audio |
| `HEARTMULA_REDIS_URL` | `redis://redis:6379` | Redis connection URL |

## Project Structure

```
heartmula-local-api/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── app/
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration
│   ├── models/
│   │   ├── requests.py         # Request schemas
│   │   └── responses.py        # Response schemas
│   ├── routers/
│   │   ├── generate.py         # Generation endpoints
│   │   └── health.py           # Health endpoints
│   └── services/
│       ├── music_generator.py  # HeartMuLa wrapper with offload
│       └── job_manager.py      # Redis job queue
├── ckpt/                       # Model weights (mount volume)
├── outputs/                    # Generated audio files
└── static/                     # Web UI files
```

## Performance

Tested on NVIDIA RTX 3090 (24GB VRAM):

| Metric | Value |
|--------|-------|
| VRAM at idle | ~7.4 GB |
| VRAM during generation | ~15-20 GB (peak) |
| Token generation speed | ~10-12 it/s |
| Audio decoding speed | ~2.5-3 it/s |
| 10 sec audio generation | ~30 sec |
| 60 sec audio generation | ~3-4 min |

## Troubleshooting

### Out of Memory

If you encounter OOM errors:
1. Ensure `HEARTMULA_SEQUENTIAL_OFFLOAD=true` is set
2. Try reducing `max_duration_ms`
3. Close other GPU-intensive applications

### Slow Generation

If generation is very slow (~1 it/s):
- This usually indicates memory pressure
- Restart the container to clear any accumulated memory
- Check that KV caches are being deleted (look for "Deleted KV cache from 31 attention layers" in logs)

### Container Health Check Failing

The model takes ~2 minutes to load on startup. The health check has a 120s start period to accommodate this.

## License

This project is a wrapper around [HeartMuLa](https://github.com/HeartMuLa/HeartMuLa). Please refer to the original repository for model licensing terms.

## Acknowledgments

- [HeartMuLa Team](https://github.com/HeartMuLa/HeartMuLa) for the amazing music generation model
- [heartlib](https://github.com/HeartMuLa/heartlib) for the inference library
