import os
import warnings
import asyncio
import traceback
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

# Suppress harmless hub/transformers warnings on Windows
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel


from config import STATIC_DIR, OUTPUTS_DIR, SUPPORTED_MODELS, DEFAULT_MODEL, HOST, PORT
from pipeline_manager import DiffusionPipelineManager, start_background_warmup
from gallery_manager import GalleryManager


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Import torch/diffusers in the background so the server answers instantly.
    start_background_warmup()
    yield


app = FastAPI(title="Real-Time Progressive Diffusion Studio", lifespan=lifespan)

# Static files mounts
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Core managers
pipeline_manager = DiffusionPipelineManager()
gallery_manager = GalleryManager()

class ModelLoadRequest(BaseModel):
    model_id: str

@app.get("/")
async def root():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Studio interface loading..."}

@app.get("/dynamic-threads")
async def dynamic_threads():
    vis_file = STATIC_DIR.parent / "dynamic_threaded_nn" / "dynamic_threads_visualization.html"
    if vis_file.exists():
        return FileResponse(vis_file)
    return JSONResponse(status_code=404, content={"error": "Dynamic threads visualization not found."})

@app.get("/api/hardware")
async def get_hardware():
    return pipeline_manager.get_hardware_info()

@app.get("/api/models")
async def get_models():
    return {
        "models": SUPPORTED_MODELS,
        "default": DEFAULT_MODEL,
        "current_loaded": pipeline_manager.current_model_id
    }

@app.post("/api/load-model")
async def load_model_endpoint(req: ModelLoadRequest):
    if pipeline_manager.is_generating:
        raise HTTPException(status_code=400, detail="Generation currently in progress")
    
    # Run model loading in threadpool
    try:
        await asyncio.to_thread(pipeline_manager.load_model, req.model_id)
        return {"status": "success", "loaded_model": req.model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/gallery")
async def get_gallery():
    return gallery_manager.load_gallery()

@app.delete("/api/gallery/{item_id}")
async def delete_gallery_item(item_id: str):
    success = gallery_manager.delete_item(item_id)
    if success:
        return {"status": "deleted", "id": item_id}
    raise HTTPException(status_code=404, detail="Item not found")

@app.post("/api/cancel")
async def cancel_generation():
    pipeline_manager.cancel_generation()
    return {"status": "cancelling"}

@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action", "generate")

            if action == "cancel":
                pipeline_manager.cancel_generation()
                await websocket.send_json({"type": "cancelled", "status": "Cancellation requested"})
                continue

            # Extract generation parameters
            prompt = data.get("prompt", "").strip()
            if not prompt:
                await websocket.send_json({"type": "error", "message": "Prompt cannot be empty"})
                continue

            negative_prompt = data.get("negative_prompt", "")
            model_id = data.get("model_id", DEFAULT_MODEL)
            width = int(data.get("width", 512))
            height = int(data.get("height", 512))
            steps = int(data.get("steps", 20))
            cfg_scale = float(data.get("cfg_scale", 7.5))
            seed = int(data.get("seed", -1))
            scheduler_name = data.get("scheduler", "euler_a")
            preview_frequency = int(data.get("preview_frequency", 1))
            preview_mode = data.get("preview_mode", "vae")

            async def safe_send(payload: Dict[str, Any]) -> bool:
                """Sends without raising when the browser tab has gone away."""
                try:
                    await websocket.send_json(payload)
                    return True
                except Exception:
                    return False

            # Threadsafe step callback to transmit preview frames
            def on_step(step_info: Dict[str, Any]):
                # Schedule sending message to websocket on the async event loop
                try:
                    asyncio.run_coroutine_threadsafe(safe_send(step_info), loop)
                except Exception:
                    pass

            await websocket.send_json({
                "type": "started",
                "status": (
                    "Initializing diffusion process..."
                    if pipeline_manager.is_runtime_ready()
                    else "Loading the AI runtime (first launch only, this can take a minute)..."
                ),
                "model_id": model_id
            })

            # Create concurrent cancel listener for this generation
            async def cancel_listener():
                try:
                    while True:
                        incoming = await websocket.receive_json()
                        if incoming.get("action") == "cancel":
                            print("[WebSocket] Received cancel action from client.")
                            pipeline_manager.cancel_generation()
                            break
                except Exception:
                    pass

            listener_task = asyncio.create_task(cancel_listener())

            # Run diffusion synchronously in worker thread
            try:
                result = await asyncio.to_thread(
                    pipeline_manager.run_progressive_generation,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    model_id=model_id,
                    width=width,
                    height=height,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    seed=seed,
                    scheduler_name=scheduler_name,
                    preview_frequency=preview_frequency,
                    preview_mode=preview_mode,
                    on_step_callback=on_step
                )

                if result is None:
                    # Cancelled
                    await safe_send({
                        "type": "cancelled",
                        "status": "Generation was cancelled."
                    })
                else:
                    final_image, used_seed, elapsed_sec = result
                    # Save to gallery
                    entry = gallery_manager.save_generation(
                        image=final_image,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        model_id=model_id,
                        seed=used_seed,
                        steps=steps,
                        cfg_scale=cfg_scale,
                        width=width,
                        height=height,
                        elapsed_sec=elapsed_sec
                    )

                    await safe_send({
                        "type": "complete",
                        "status": "Image generation complete!",
                        "image_url": entry["url"],
                        "entry": entry,
                        "seed": used_seed,
                        "elapsed_sec": round(elapsed_sec, 2),
                        "speed": round(steps / max(elapsed_sec, 0.001), 2)
                    })

            except InterruptedError:
                await safe_send({
                    "type": "cancelled",
                    "status": "Generation interrupted by user."
                })
            except Exception as gen_err:
                traceback.print_exc()
                await safe_send({
                    "type": "error",
                    "message": f"Generation error: {str(gen_err)}"
                })
            finally:
                listener_task.cancel()

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected.")
    except Exception as ws_err:
        print(f"[WebSocket] Error: {ws_err}")

if __name__ == "__main__":
    import threading
    import time
    import urllib.request
    import webbrowser
    import uvicorn

    def open_browser_when_ready():
        # The server itself starts in a couple of seconds now; the AI runtime
        # keeps loading in the background while the studio UI is already usable.
        for attempt in range(240):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen(f"http://{HOST}:{PORT}/api/hardware", timeout=2):
                    print(f"\n[Studio] Server ready! Launching browser at http://{HOST}:{PORT}\n", flush=True)
                    webbrowser.open(f"http://{HOST}:{PORT}")
                    return
            except Exception:
                if attempt in (20, 60, 120):
                    print("[Studio] Still starting up, waiting a little longer...", flush=True)

        print(f"\n[Studio] Server did not answer in time. Open http://{HOST}:{PORT} manually.\n", flush=True)

    print(f"[Studio] Starting server on http://{HOST}:{PORT} ...", flush=True)
    threading.Thread(target=open_browser_when_ready, daemon=True).start()

    # Pass the app object directly: with a module string uvicorn imports app.py
    # a second time, which doubles startup time and memory for no benefit.
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")

