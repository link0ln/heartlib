"""HeartMuLa model wrapper for music generation with VRAM optimization."""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torchaudio

from app.config import get_settings

logger = logging.getLogger(__name__)


class MusicGenerator:
    """Singleton wrapper for HeartMuLaGenPipeline with sequential offload support."""

    _instance: Optional["MusicGenerator"] = None
    _pipeline = None
    _loaded: bool = False
    _sequential_offload: bool = False
    _dtype = None

    def __new__(cls) -> "MusicGenerator":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def load_model(self) -> None:
        """Load HeartMuLa model with optional sequential offload."""
        if self._loaded:
            logger.info("Model already loaded, skipping")
            return

        settings = get_settings()
        model_path = settings.model_path

        logger.info(f"Loading HeartMuLa model from {model_path}")
        logger.info(f"Using dtype: {settings.model_dtype}")
        logger.info(f"Sequential offload: {settings.sequential_offload}")
        logger.info(f"Hybrid mode: {settings.hybrid_mode}")

        # Import heartlib modules
        try:
            from heartlib.pipelines.music_generation import HeartMuLaGenPipeline
        except ImportError as e:
            logger.error(f"Failed to import heartlib: {e}")
            raise RuntimeError("heartlib is not installed properly") from e

        # Determine dtype
        self._dtype = torch.float32 if settings.model_dtype == "float32" else torch.float16

        # Check for GPU
        gpu_available = torch.cuda.is_available()

        # Sequential offload mode (recommended for VRAM optimization)
        if settings.sequential_offload and gpu_available:
            self._sequential_offload = True
            device = torch.device("cuda")
            logger.info("Sequential offload enabled: models will swap between GPU/CPU during generation")
        elif settings.hybrid_mode and gpu_available:
            # Hybrid mode: LLM on CPU, HeartCodec on GPU (legacy)
            device = torch.device("cpu")
            logger.info("Hybrid mode: LLM on CPU, HeartCodec on GPU")
        elif gpu_available:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            logger.warning("GPU not available, using CPU (this will be slow)")

        # Verify required files exist
        heartmula_model_path = os.path.join(model_path, "HeartMuLa-oss-3B")
        heartcodec_model_path = os.path.join(model_path, "HeartCodec-oss")
        tokenizer_path = os.path.join(model_path, "tokenizer.json")
        gen_config_path = os.path.join(model_path, "gen_config.json")

        for path, name in [
            (heartmula_model_path, "HeartMuLa model"),
            (heartcodec_model_path, "HeartCodec model"),
            (tokenizer_path, "Tokenizer"),
            (gen_config_path, "Gen config"),
        ]:
            if not os.path.exists(path):
                raise RuntimeError(f"{name} not found at {path}")

        logger.info(f"Model path: {model_path}")
        logger.info(f"Device: {device}, Dtype: {self._dtype}")

        # Load pipeline
        self._pipeline = HeartMuLaGenPipeline.from_pretrained(
            pretrained_path=model_path,
            device=device,
            dtype=self._dtype,
            version="3B"
        )

        # Setup for sequential offload
        if self._sequential_offload:
            # Move HeartCodec to CPU initially (LLM stays on GPU)
            logger.info("Moving HeartCodec to CPU (sequential offload setup)")
            self._pipeline.audio_codec = self._pipeline.audio_codec.to('cpu')
            torch.cuda.empty_cache()
            logger.info("Sequential offload ready: LLM on GPU, HeartCodec on CPU")

        # Handle hybrid mode (legacy)
        elif settings.hybrid_mode and gpu_available:
            if hasattr(self._pipeline, 'model') and next(self._pipeline.model.parameters()).device.type == 'cuda':
                logger.info("Moving HeartMuLa LLM to CPU...")
                self._pipeline.model = self._pipeline.model.to('cpu')
                self._pipeline.device = torch.device('cpu')
                torch.cuda.empty_cache()
                logger.info("HeartMuLa LLM moved to CPU")
            logger.info("Moving HeartCodec to GPU")
            self._pipeline.audio_codec = self._pipeline.audio_codec.to('cuda')
            logger.info("HeartCodec moved to GPU successfully")

        self._loaded = True
        logger.info("HeartMuLa model loaded successfully")

    def _reset_kv_caches(self, model) -> None:
        """Reset KV caches to clear their content (but keep structure intact)."""
        if hasattr(model, 'reset_caches'):
            try:
                model.reset_caches()
            except RuntimeError:
                pass  # Caches may not exist yet

    def _delete_kv_cache_buffers(self, model) -> None:
        """Delete KV cache buffers from attention layers to free memory.

        This is necessary because torchtune registers caches as buffers,
        and they move with model.to() operations, causing memory bloat.
        After calling this, setup_caches() must be called to recreate them.
        """
        deleted_count = 0
        for name, module in model.named_modules():
            # Look for kv_cache buffer in attention modules
            if hasattr(module, 'kv_cache'):
                # Get the kv_cache which is a KVCache object with k_cache and v_cache tensors
                kv_cache = module.kv_cache
                if kv_cache is not None:
                    # Delete the cache tensors
                    if hasattr(kv_cache, 'k_cache'):
                        del kv_cache.k_cache
                    if hasattr(kv_cache, 'v_cache'):
                        del kv_cache.v_cache
                    module.kv_cache = None
                    deleted_count += 1
        logger.info(f"Deleted KV cache from {deleted_count} attention layers")

    def _offload_forward(
        self,
        model_inputs: Dict[str, Any],
        max_audio_length_ms: int,
        temperature: float,
        topk: int,
        cfg_scale: float,
    ) -> Dict[str, Any]:
        """Custom forward with sequential offload between LLM and HeartCodec."""
        from tqdm import tqdm
        import gc

        pipeline = self._pipeline

        # Phase 0: Ensure correct device state at start
        logger.info("Phase 0: Ensuring device state consistency")

        # Delete KV caches to free memory before any model movement
        # This prevents the cache bloat issue when moving model between devices
        self._delete_kv_cache_buffers(pipeline.model)
        gc.collect()
        torch.cuda.empty_cache()

        # Check and fix LLM device
        llm_device = next(pipeline.model.parameters()).device
        if llm_device.type != 'cuda':
            logger.info(f"LLM is on {llm_device}, moving to CUDA")
            pipeline.model = pipeline.model.to('cuda')

        # Check and fix HeartCodec device (should be on CPU)
        codec_device = next(pipeline.audio_codec.parameters()).device
        if codec_device.type != 'cpu':
            logger.info(f"HeartCodec is on {codec_device}, moving to CPU")
            pipeline.audio_codec = pipeline.audio_codec.to('cpu')

        gc.collect()
        torch.cuda.empty_cache()

        # Log GPU memory state
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU memory at start: {mem_used:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

        # Phase 1: Token generation (LLM on GPU)
        logger.info("Phase 1: Token generation (LLM on GPU)")

        # Move all inputs to GPU
        prompt_tokens = model_inputs["tokens"].to('cuda')
        prompt_tokens_mask = model_inputs["tokens_mask"].to('cuda')
        continuous_segment = model_inputs["muq_embed"].to('cuda')
        starts = model_inputs["muq_idx"]  # This is a list, not tensor
        prompt_pos = model_inputs["pos"].to('cuda')

        frames = []
        bs_size = 2 if cfg_scale != 1.0 else 1
        pipeline.model.setup_caches(bs_size)

        # Helper function from original pipeline
        def _pad_audio_token(token: torch.Tensor):
            padded_token = (
                torch.ones(
                    (token.shape[0], pipeline._parallel_number),
                    device=token.device,
                    dtype=torch.long,
                )
                * pipeline.config.empty_id
            )
            padded_token[:, :-1] = token
            padded_token = padded_token.unsqueeze(1)
            padded_token_mask = torch.ones_like(
                padded_token, device=token.device, dtype=torch.bool
            )
            padded_token_mask[..., -1] = False
            return padded_token, padded_token_mask

        # First frame generation
        with torch.autocast(device_type='cuda', dtype=self._dtype):
            curr_token = pipeline.model.generate_frame(
                tokens=prompt_tokens,
                tokens_mask=prompt_tokens_mask,
                input_pos=prompt_pos,
                temperature=temperature,
                topk=topk,
                cfg_scale=cfg_scale,
                continuous_segments=continuous_segment,
                starts=starts,
            )
        frames.append(curr_token[0:1,])

        # Generation loop
        max_audio_frames = max_audio_length_ms // 80
        for i in tqdm(range(max_audio_frames)):
            curr_token, curr_token_mask = _pad_audio_token(curr_token)
            with torch.autocast(device_type='cuda', dtype=self._dtype):
                curr_token = pipeline.model.generate_frame(
                    tokens=curr_token,
                    tokens_mask=curr_token_mask,
                    input_pos=prompt_pos[..., -1:] + i + 1,
                    temperature=temperature,
                    topk=topk,
                    cfg_scale=cfg_scale,
                    continuous_segments=None,
                    starts=None,
                )
            if torch.any(curr_token[0:1, :] >= pipeline.config.audio_eos_id):
                break
            frames.append(curr_token[0:1,])

        # Create frames tensor on GPU, then move to CPU to free VRAM
        frames_tensor = torch.stack(frames).permute(1, 2, 0).squeeze(0)

        # Phase 2: Offload LLM and frames to CPU, then load HeartCodec to GPU
        logger.info("Phase 2: Swapping models (LLM → CPU, HeartCodec → GPU)")

        # Move frames_tensor to CPU first to free GPU memory
        frames_tensor = frames_tensor.to('cpu')
        logger.info("Frames tensor moved to CPU")

        # Delete input tensors and frames list
        del prompt_tokens, prompt_tokens_mask, continuous_segment, prompt_pos
        del frames  # Free the frames list

        # Delete KV caches to free GPU memory BEFORE moving model
        # This is critical: reset_caches() only zeros values, it doesn't free memory
        # The caches would otherwise move with the model and persist
        self._delete_kv_cache_buffers(pipeline.model)
        gc.collect()
        torch.cuda.empty_cache()

        # Move LLM to CPU
        pipeline.model = pipeline.model.to('cpu')

        # Aggressive memory cleanup
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Log memory after clearing
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU memory after LLM offload: {mem_used:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

        logger.info("LLM moved to CPU, freed GPU memory")

        # Move HeartCodec to GPU for audio decoding
        pipeline.audio_codec = pipeline.audio_codec.to('cuda')
        logger.info("HeartCodec moved to GPU")

        # Phase 3: Audio decoding (HeartCodec on GPU)
        logger.info("Phase 3: Audio decoding (HeartCodec on GPU)")

        frames_tensor = frames_tensor.to('cuda')
        wav = pipeline.audio_codec.detokenize(frames_tensor, device='cuda')

        # Delete frames tensor to free GPU memory
        del frames_tensor

        # Phase 4: Cleanup (LLM stays on CPU, will be restored in Phase 0)
        logger.info("Phase 4: Cleanup after generation")

        # Move HeartCodec back to CPU to free all VRAM
        pipeline.audio_codec = pipeline.audio_codec.to('cpu')
        logger.info("HeartCodec moved to CPU, all models on CPU")

        # KV caches were already deleted in Phase 2, nothing to clean up
        gc.collect()
        torch.cuda.empty_cache()

        # Log final memory state (should be minimal)
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU memory after cleanup: {mem_used:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

        logger.info("Generation complete, LLM on CPU (will be restored in next Phase 0)")

        return {"wav": wav}

    def generate(
        self,
        lyrics: str,
        tags: str,
        output_path: str,
        max_duration_ms: int = 120000,
        temperature: float = 1.0,
        cfg_scale: float = 1.5,
    ) -> str:
        """
        Generate music and save to file.

        Args:
            lyrics: Lyrics with section markers
            tags: Comma-separated music tags
            output_path: Full path to save the audio file
            max_duration_ms: Maximum duration in milliseconds
            temperature: Sampling temperature
            cfg_scale: Classifier-free guidance scale

        Returns:
            Path to the generated audio file
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.info(f"Generating music with tags: {tags}")
        logger.info(f"Max duration: {max_duration_ms}ms, Temperature: {temperature}, CFG: {cfg_scale}")

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Prepare inputs dict
        inputs = {
            "lyrics": lyrics,
            "tags": tags,
        }

        try:
            if self._sequential_offload:
                # Use custom offload-aware generation
                logger.info("Using sequential offload generation")

                # Preprocess (tokenize inputs) - requires cfg_scale
                model_inputs = self._pipeline.preprocess(inputs, cfg_scale=cfg_scale)

                # Forward with offloading
                model_outputs = self._offload_forward(
                    model_inputs,
                    max_audio_length_ms=max_duration_ms,
                    temperature=temperature,
                    topk=50,
                    cfg_scale=cfg_scale,
                )

                # Postprocess (save audio)
                self._pipeline.postprocess(model_outputs, save_path=output_path)
            else:
                # Standard generation
                self._pipeline(
                    inputs,
                    max_audio_length_ms=max_duration_ms,
                    temperature=temperature,
                    cfg_scale=cfg_scale,
                    save_path=output_path,
                )
        finally:
            # Clear GPU cache after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")

        logger.info(f"Audio saved to {output_path}")
        return output_path

    def unload_model(self) -> None:
        """Unload the model to free GPU memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._loaded = False
            self._sequential_offload = False
            torch.cuda.empty_cache()
            logger.info("Model unloaded")


# Global instance
_generator: Optional[MusicGenerator] = None


def get_generator() -> MusicGenerator:
    """Get the global MusicGenerator instance."""
    global _generator
    if _generator is None:
        _generator = MusicGenerator()
    return _generator
