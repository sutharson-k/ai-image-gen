import gc
import threading
import time
from typing import Optional, Callable, Dict, Any
from PIL import Image

from config import SUPPORTED_MODELS, DEFAULT_MODEL, MODEL_DIR

# ---------------------------------------------------------------------------
# Lazy, one-time import of the heavy ML stack.
#
# Importing torch + diffusers + transformers takes 25-45 seconds on a typical
# Windows machine. Doing it while app.py was imported meant uvicorn only
# started listening after that delay: the browser launcher gave up and the
# studio looked like it would never start. The runtime is now imported once,
# in a background thread, while the UI is already being served.
# ---------------------------------------------------------------------------
_torch = None
_schedulers: Dict[str, Any] = {}
_latent_previewer_cls = None
_ml_error: Optional[BaseException] = None
_ml_state = "idle"  # idle | loading | ready | failed
_ml_lock = threading.Lock()
_ml_ready = threading.Event()
_warmup_thread: Optional[threading.Thread] = None


def _import_ml_runtime():
    """Imports torch / diffusers / latent preview helpers exactly once."""
    global _torch, _schedulers, _latent_previewer_cls, _ml_error, _ml_state

    if _ml_ready.is_set():
        return _torch

    try:
        import torch
        from diffusers import (
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
            DDIMScheduler,
        )
        from latent_preview import LatentPreviewer

        _torch = torch
        _schedulers = {
            "euler_a": EulerAncestralDiscreteScheduler,
            "dpm_2m": DPMSolverMultistepScheduler,
            "ddim": DDIMScheduler,
        }
        _latent_previewer_cls = LatentPreviewer
        _ml_state = "ready"
        print("[Runtime] Diffusion runtime ready.", flush=True)
    except BaseException as exc:  # surfaced to the UI instead of killing startup
        _ml_error = exc
        _ml_state = "failed"
        print(f"[Runtime] Failed to import the diffusion runtime: {exc}", flush=True)
    finally:
        _ml_ready.set()

    return _torch


def start_background_warmup():
    """Starts importing the ML stack in the background. Never blocks, idempotent."""
    global _warmup_thread, _ml_state

    if _ml_ready.is_set():
        return
    with _ml_lock:
        if _warmup_thread is not None and _warmup_thread.is_alive():
            return
        _ml_state = "loading"
        _warmup_thread = threading.Thread(
            target=_import_ml_runtime,
            name="ml-runtime-warmup",
            daemon=True,
        )
        _warmup_thread.start()


def ensure_ml_runtime():
    """Blocks until the ML stack is available. Safe to call from worker threads."""
    if not _ml_ready.is_set():
        if _ml_state == "loading":
            _ml_ready.wait()
        else:
            _import_ml_runtime()

    if _ml_error is not None:
        raise RuntimeError(
            "The diffusion runtime could not be loaded. Reinstall the dependencies "
            "with: .venv\\Scripts\\pip install -r requirements.txt"
        ) from _ml_error
    return _torch


def ml_runtime_state() -> str:
    return _ml_state


class DiffusionPipelineManager:
    """
    Manages diffusion model inference, GPU memory optimizations,
    and progressive step-by-step intermediate latent decoding.
    """

    def __init__(self):
        self.current_pipeline = None
        self.current_model_id = None
        self.is_loading = False
        self.is_generating = False
        self.cancel_requested = False
        self.previewer = None
        self.device = "cpu"
        self.dtype = None
        self._runtime_ready = False

    # -- runtime helpers ---------------------------------------------------
    def _ensure_runtime(self):
        """Resolves device/dtype once the (lazily imported) torch is available."""
        if self._runtime_ready:
            return _torch

        torch = ensure_ml_runtime()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._runtime_ready = True
        print(f"[PipelineManager] Runtime device: {self.device} ({self.dtype})", flush=True)
        return torch

    def _ensure_previewer(self):
        if self.previewer is None:
            ensure_ml_runtime()
            self.previewer = _latent_previewer_cls(mode="vae")
        return self.previewer

    def is_runtime_ready(self) -> bool:
        return _ml_ready.is_set() and _ml_error is None

    # -- hardware info -----------------------------------------------------
    def get_hardware_info(self) -> Dict[str, Any]:
        if _ml_error is not None:
            return {
                "device": "cpu",
                "dtype": "n/a",
                "cuda_available": False,
                "gpu_name": "AI runtime failed to load",
                "vram_allocated_gb": 0,
                "vram_reserved_gb": 0,
                "vram_total_gb": 0,
                "runtime": "failed",
                "loading": False,
                "error": str(_ml_error)[:300],
                "model_loaded": self.current_model_id,
            }

        if not _ml_ready.is_set():
            # Keep warming up in the background; answer immediately so the UI
            # and the browser launcher are never blocked by torch import.
            start_background_warmup()
            return {
                "device": "cpu",
                "dtype": "loading",
                "cuda_available": False,
                "gpu_name": "Initializing AI runtime...",
                "vram_allocated_gb": 0,
                "vram_reserved_gb": 0,
                "vram_total_gb": 0,
                "runtime": _ml_state,
                "loading": True,
                "model_loaded": self.current_model_id,
            }

        torch = _torch
        cuda = torch.cuda.is_available()
        info = {
            "device": "cuda" if cuda else "cpu",
            "dtype": str(self.dtype or (torch.float16 if cuda else torch.float32)),
            "cuda_available": cuda,
            "runtime": "ready",
            "loading": False,
            "model_loaded": self.current_model_id,
        }
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            mem_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            mem_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            mem_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            info["vram_allocated_gb"] = round(mem_allocated, 2)
            info["vram_reserved_gb"] = round(mem_reserved, 2)
            info["vram_total_gb"] = round(mem_total, 2)
        else:
            info["gpu_name"] = "CPU"
            info["vram_allocated_gb"] = 0
            info["vram_reserved_gb"] = 0
            info["vram_total_gb"] = 0
        return info

    # -- model loading -----------------------------------------------------
    def load_model(self, model_id: str = DEFAULT_MODEL, progress_callback: Optional[Callable[[str], None]] = None):
        """Loads or switches diffusion model into memory with VRAM optimizations."""
        if self.current_pipeline is not None and self.current_model_id == model_id:
            return self.current_pipeline

        self._ensure_runtime()
        self.is_loading = True
        try:
            if progress_callback:
                progress_callback(f"Loading weights for {model_id}...")

            # Clean up prior model from VRAM
            if self.current_pipeline is not None:
                del self.current_pipeline
                self.current_pipeline = None
                if self.device == "cuda":
                    _torch.cuda.empty_cache()
                gc.collect()

            print(f"[PipelineManager] Loading {model_id} on {self.device} ({self.dtype})...", flush=True)

            from diffusers import AutoPipelineForText2Image

            pipe = AutoPipelineForText2Image.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                safety_checker=None,
                low_cpu_mem_usage=True,
                cache_dir=str(MODEL_DIR)
            )

            # Schedulers
            pipe.scheduler = _schedulers["euler_a"].from_config(pipe.scheduler.config)

            # Move to target device
            if self.device == "cuda":
                pipe = pipe.to(self.device)
                # Optimize for 4GB-8GB VRAM
                try:
                    pipe.enable_attention_slicing()
                    pipe.enable_vae_slicing()
                except Exception as opt_err:
                    print(f"[PipelineManager] Optional slice opt note: {opt_err}")

            self.current_pipeline = pipe
            self.current_model_id = model_id
            print(f"[PipelineManager] Successfully loaded {model_id}", flush=True)
            return self.current_pipeline
        finally:
            self.is_loading = False

    def cancel_generation(self):
        """Signals active generation process to interrupt immediately."""
        self.cancel_requested = True
        if self.current_pipeline is not None:
            # Diffusers supports setting _interrupt flag
            setattr(self.current_pipeline, "_interrupt", True)
        print("[PipelineManager] Cancellation requested.")

    # -- generation --------------------------------------------------------
    def run_progressive_generation(
        self,
        prompt: str,
        negative_prompt: str = "",
        model_id: str = DEFAULT_MODEL,
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg_scale: float = 7.5,
        seed: int = -1,
        scheduler_name: str = "euler_a",
        preview_frequency: int = 1,
        preview_mode: str = "vae",
        on_step_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Optional[Image.Image]:
        """
        Executes progressive diffusion with step-by-step intermediate latent decoding.
        Calls `on_step_callback` at each denoising step with real visual updates.
        """
        self.is_generating = True
        self.cancel_requested = False

        try:
            # 1. Ensure the runtime and model are available
            torch = self._ensure_runtime()
            previewer = self._ensure_previewer()
            pipe = self.load_model(model_id)

            # Reset interrupt flag
            if hasattr(pipe, "_interrupt"):
                pipe._interrupt = False

            # Set scheduler
            self._set_scheduler(pipe, scheduler_name)

            # 2. Handle seed and generator
            if seed is None or seed < 0:
                seed = int(torch.randint(0, 2**32 - 1, (1,)).item())

            generator = torch.Generator(device=self.device).manual_seed(seed)

            # 3. Step 0: Emit true initial Gaussian noise visual
            if on_step_callback:
                noise_img = previewer.generate_gaussian_noise_image(width, height, seed=seed)
                noise_b64 = previewer.image_to_base64_jpeg(noise_img, quality=80)
                on_step_callback({
                    "type": "step_preview",
                    "step": 0,
                    "total_steps": steps,
                    "progress": 0.0,
                    "elapsed_sec": 0.0,
                    "speed_it_per_sec": 0.0,
                    "preview_image": noise_b64,
                    "status": "Starting diffusion from random Gaussian noise"
                })

            start_time = time.time()
            last_step_time = start_time

            # Intermediate step collector
            def step_callback(pipeline_ref, step_index: int, timestep: int, callback_kwargs: dict):
                nonlocal last_step_time

                # Check for user cancellation
                if self.cancel_requested:
                    setattr(pipeline_ref, "_interrupt", True)
                    raise InterruptedError("Generation cancelled by user.")

                # Only emit previews matching preview_frequency or on the final step
                current_step = step_index + 1
                should_preview = (current_step % preview_frequency == 0) or (current_step == steps)

                if should_preview and on_step_callback and "latents" in callback_kwargs:
                    latents = callback_kwargs["latents"]
                    now = time.time()
                    elapsed = now - start_time
                    step_delta = max(now - last_step_time, 0.001)
                    speed = round(1.0 / step_delta, 2)
                    last_step_time = now

                    # Decode latents to preview image
                    try:
                        preview_img = previewer.decode(
                            latents,
                            vae=getattr(pipeline_ref, "vae", None),
                            mode_override=preview_mode
                        )
                        # Ensure dimensions match requested
                        if preview_img.size != (width, height):
                            preview_img = preview_img.resize((width, height), Image.Resampling.BILINEAR)

                        preview_b64 = previewer.image_to_base64_jpeg(preview_img, quality=85)

                        on_step_callback({
                            "type": "step_preview",
                            "step": current_step,
                            "total_steps": steps,
                            "progress": round((current_step / steps) * 100, 1),
                            "elapsed_sec": round(elapsed, 2),
                            "speed_it_per_sec": speed,
                            "preview_image": preview_b64,
                            "status": f"Denoising step {current_step}/{steps}"
                        })
                    except Exception as dec_err:
                        print(f"[PipelineManager] Step {current_step} preview decode error: {dec_err}")

                return callback_kwargs

            # 4. Run diffusion pipeline
            print(f"[PipelineManager] Starting generation: '{prompt}' ({width}x{height}, {steps} steps, seed {seed})", flush=True)

            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
                callback_on_step_end=step_callback,
                callback_on_step_end_tensor_inputs=["latents"]
            )

            if self.cancel_requested:
                return None

            final_image = output.images[0]
            total_time = time.time() - start_time
            print(f"[PipelineManager] Completed generation in {total_time:.2f}s", flush=True)
            return final_image, seed, total_time

        except InterruptedError:
            print("[PipelineManager] Pipeline interrupted cleanly.")
            return None
        except Exception as e:
            print(f"[PipelineManager] Generation exception: {e}")
            raise e
        finally:
            self.is_generating = False
            self.cancel_requested = False

    def _set_scheduler(self, pipe, scheduler_name: str):
        scheduler_cls = _schedulers.get(scheduler_name)
        if scheduler_cls is None:
            return
        try:
            if scheduler_name == "dpm_2m":
                pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config, use_karras_sigmas=True)
            else:
                pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
        except Exception as e:
            print(f"[PipelineManager] Scheduler set error: {e}")
