import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
GALLERY_FILE = OUTPUTS_DIR / "gallery.json"
STATIC_DIR = BASE_DIR / "static"
MODEL_DIR = BASE_DIR / "model"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Direct all Hugging Face and Diffusers downloads solely to the local model folder
os.environ["HF_HOME"] = str(MODEL_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODEL_DIR / "hub")

# Predefined diffusion models (Only Stable Diffusion v1.5)
SUPPORTED_MODELS = {
    "runwayml/stable-diffusion-v1-5": {
        "name": "Stable Diffusion v1.5",
        "description": "Standard high quality latent diffusion model. Rich textures and details.",
        "default_steps": 20,
        "default_cfg": 7.5,
        "recommended_resolution": [512, 512],
        "type": "sd15"
    }
}

DEFAULT_MODEL = "runwayml/stable-diffusion-v1-5"

# Host & Port
HOST = "127.0.0.1"
PORT = 7860

