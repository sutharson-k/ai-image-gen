# Nebula Diffusion // Real-Time Progressive Studio

An AI image generation application inspired by Stable Diffusion featuring a **genuine real-time progressive diffusion experience**.

Instead of waiting for the full generation to complete, the canvas initializes with random Gaussian noise and streams the authentic intermediate denoising states step-by-step directly from the latent diffusion model.

---

## 🌟 Key Features

1. **Genuine Diffusion Denoising Stream**:
   - **Step 0 Gaussian Noise**: The canvas initializes with the actual random noise distribution derived from the chosen random seed.
   - **Real-Time Intermediate Decodes**: At every diffusion step, intermediate latent vectors $\mathbf{z}_t$ are decoded to RGB using either full VAE decoding or ultra-fast linear projection.
   - **Live Progress & Metrics**: Shows current step count (e.g. `Step 14/20`), elapsed time, iteration rate (`it/s`), and progress bar.
   - **Interactive Step Inspector**: Scrub back and forth through each intermediate denoising frame to observe how the neural network refines compositions, textures, and details.

2. **AI Model Engine**:
   - **Multiple Model Support**:
     - `stabilityai/sd-turbo`: Ultra-fast 1 to 4 step real-time diffusion distillation.
     - `runwayml/stable-diffusion-v1-5`: Classic high-fidelity latent diffusion.
     - `dreamlike-art/dreamlike-photoreal-2.0`: Photorealistic portraiture and landscapes.
   - **Hardware-Accelerated & Memory-Optimized**:
     - Automatically utilizes NVIDIA GPU with CUDA (`torch.float16`).
     - Attention slicing and memory management to run smoothly within 4GB VRAM (e.g. RTX 3050).
     - Models stay cached in memory between generation requests.

3. **Creative Studio User Interface**:
   - **Darkroom Theme**: Warm graphite surfaces, paper-white type and a single amber accent, built so the image on the canvas stays the loudest thing on screen. Controls stay in one scrollable rail with Generate pinned at the bottom.
   - **Prompt Deck**: Positive prompt with quick-apply aesthetic presets (Cyberpunk, Cinematic, Fantasy, Sci-Fi) and collapsible negative prompt.
   - **Aspect Ratio Controls**: Quick presets for 1:1 Square (512×512), 3:4 Portrait (512×768), and 4:3 Landscape (768×512).
   - **Fine-Tuning Controls**: Sliders for Inference Steps, CFG Guidance Scale, Seed (with randomizer dice), and Preview Frequency.
   - **Real-Time Cancellation**: Instantly interrupts the diffusion pipeline on demand.
   - **Persistent Gallery**: High-resolution PNGs saved to `outputs/` with prompt metadata; click any past creation to reload its prompt, seed, and parameters into the controls, or delete it — the photo is then removed from the gallery, the canvas, and the `outputs/` folder.

---

## 🚀 Quick Start Guide

### Launching the Studio
You can launch the studio using the Windows launcher batch file:
```cmd
run.bat
```
Or start the server directly using the dedicated virtual environment:
```powershell
.\.venv\Scripts\python.exe app.py
```

Then open your browser and navigate to:
```
http://localhost:7860
```

---

## 🏗️ Technical Architecture

```
image generation/
├── app.py                     # FastAPI application, REST endpoints, WebSocket streaming
├── pipeline_manager.py        # Hugging Face Diffusers pipeline, GPU optimization, step callbacks
├── latent_preview.py          # Latent-to-RGB decoders (VAE decoding and Fast Linear projection)
├── config.py                  # Model registry and system configurations
├── gallery_manager.py         # Output saving and gallery history management
├── static/
│   ├── index.html             # Studio HTML5 layout with live canvas & controls
│   ├── css/style.css          # Dark studio theme, glassmorphism, responsive controls
│   └── js/
│       ├── app.js             # UI state coordination, hardware monitoring, gallery sync
│       ├── canvas.js          # Progressive canvas renderer, idle visual noise, step filmstrip
│       └── websocket.js       # Low-latency WebSocket streaming client
├── outputs/                   # Generated PNG images and metadata registry
│   └── gallery.json
├── run.bat                    # Windows one-click launcher
└── requirements.txt           # Project dependencies
```

---

## ⚙️ How Intermediate Latent Streaming Works

In latent diffusion models (like Stable Diffusion), the image exists as a 4-channel compressed latent representation $\mathbf{z} \in \mathbb{R}^{B \times 4 \times \frac{H}{8} \times \frac{W}{8}}$ throughout the reverse diffusion trajectory:

$$\mathbf{z}_{t-1} = \text{DenoiseStep}(\mathbf{z}_t, \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, c))$$

In our application:
1. `callback_on_step_end` intercepts $\mathbf{z}_t$ at every timestep.
2. The latents are scaled and decoded using `pipe.vae.decode(latents / scaling_factor)` (or projected linearly via `latent_preview.py`).
3. The resulting RGB frame is compressed as low-latency JPEG and dispatched asynchronously over the WebSocket connection to the frontend canvas.
4. The client's HTML5 Canvas immediately redraws the updated frame and appends it to the Step Inspector filmstrip.
