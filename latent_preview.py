import io
import torch
import numpy as np
from PIL import Image

# Linear projection matrix for fast latent-to-RGB approximation in SD 1.5
SD15_LATENT_RGB_FACTORS = [
    #   R       G       B
    [ 0.298,  0.207,  0.208],  # channel 0
    [ 0.187,  0.286,  0.173],  # channel 1
    [-0.158,  0.189,  0.264],  # channel 2
    [-0.184, -0.271, -0.473],  # channel 3
]

class LatentPreviewer:
    """
    Decodes intermediate diffusion latents into visible RGB images.
    Supports:
      1. Full VAE Decoding (Highest visual fidelity, exact pixel reconstruction)
      2. Fast Linear Projection (Instant <1ms preview of emergent color & composition)
    """

    def __init__(self, mode="vae"):
        self.mode = mode
        self._linear_matrix = None

    def decode_vae(self, latents: torch.Tensor, vae) -> Image.Image:
        """
        Decode latent tensor using the pipeline's AutoencoderKL.
        latents: tensor of shape (batch, 4, H/8, W/8)
        """
        with torch.no_grad():
            # In SD, latents are scaled by scaling_factor
            scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
            scaled = latents.to(dtype=vae.dtype) / scaling_factor
            decoded = vae.decode(scaled, return_dict=False)[0]
            # Convert from [-1, 1] to [0, 1]
            image = (decoded / 2 + 0.5).clamp(0, 1)
            # Permute to (H, W, C)
            image_np = image[0].cpu().permute(1, 2, 0).float().numpy()
            image_np = (image_np * 255.0).round().astype(np.uint8)
            return Image.fromarray(image_np)

    def decode_linear_rgb(self, latents: torch.Tensor) -> Image.Image:
        """
        Ultra-fast linear approximation from 4-channel latents to 3-channel RGB.
        Useful when VAE decoding overhead needs to be minimized.
        """
        with torch.no_grad():
            if self._linear_matrix is None or self._linear_matrix.device != latents.device:
                self._linear_matrix = torch.tensor(
                    SD15_LATENT_RGB_FACTORS,
                    device=latents.device,
                    dtype=latents.dtype
                )

            # latents shape: [1, 4, H, W] -> permute to [H, W, 4]
            latents_permuted = latents[0].permute(1, 2, 0)
            rgb = torch.matmul(latents_permuted, self._linear_matrix)

            # Normalize to 0-1 range
            rgb_min = rgb.amin(dim=(0, 1), keepdim=True)
            rgb_max = rgb.amax(dim=(0, 1), keepdim=True)
            rgb = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-5)

            rgb_np = (rgb.cpu().float().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            # Upscale 8x to original canvas resolution
            img = Image.fromarray(rgb_np)
            return img.resize((img.width * 8, img.height * 8), resample=Image.Resampling.BILINEAR)

    def decode(self, latents: torch.Tensor, vae=None, mode_override=None) -> Image.Image:
        mode = mode_override or self.mode
        if mode == "linear" or vae is None:
            return self.decode_linear_rgb(latents)
        else:
            try:
                return self.decode_vae(latents, vae)
            except Exception:
                return self.decode_linear_rgb(latents)

    @staticmethod
    def image_to_base64_jpeg(image: Image.Image, quality: int = 80) -> str:
        """Compress image to JPEG base64 string for low-latency WebSocket transmission."""
        import base64
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def generate_gaussian_noise_image(width: int, height: int, seed: int = None) -> Image.Image:
        """Generates visual Gaussian noise corresponding to initial diffusion state."""
        rng = np.random.default_rng(seed if seed is not None and seed >= 0 else None)
        # 3 channels Gaussian noise centered around 128
        noise = rng.normal(loc=128, scale=64, size=(height, width, 3)).clip(0, 255).astype(np.uint8)
        return Image.fromarray(noise)
