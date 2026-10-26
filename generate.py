"""Generate images with local Stable Diffusion v1.5.

Usage:
    python generate.py "a photo of an astronaut riding a horse" [-o out.png] [--steps 25] [--seed 42]
"""
import argparse
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

MODEL_DIR = (
    r"C:\Users\Admin1\Desktop\AI\image generation\model"
    r"\models--runwayml--stable-diffusion-v1-5\snapshots"
    r"\451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
)


def load_pipeline() -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
        safety_checker=None,          # not downloaded locally
        requires_safety_checker=False,
        local_files_only=True,
    )
    pipe.enable_attention_slicing()   # reduce VRAM on 4GB GPU
    pipe.enable_vae_slicing()
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cpu")
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an image with SD 1.5")
    parser.add_argument("prompt", help="Text description of the image")
    parser.add_argument("-o", "--output", default=None, help="Output PNG path")
    parser.add_argument("--steps", type=int, default=25, help="Inference steps")
    parser.add_argument("--scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    out = Path(args.output) if args.output else Path(
        f"{'_'.join(args.prompt.split())[:50]}.png"
    )

    pipe = load_pipeline()
    generator = None
    if args.seed is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(args.seed)

    print(f"Generating: {args.prompt!r}")
    t0 = time.time()
    image = pipe(
        args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.scale,
        generator=generator,
    ).images[0]
    image.save(out)
    print(f"Saved {out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
