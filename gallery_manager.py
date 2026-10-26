import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from PIL import Image
from config import OUTPUTS_DIR, GALLERY_FILE

class GalleryManager:
    """Manages saved generations and persistent metadata."""

    def __init__(self):
        self._ensure_gallery_file()

    def _ensure_gallery_file(self):
        if not GALLERY_FILE.exists():
            with open(GALLERY_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)

    def load_gallery(self) -> List[Dict[str, Any]]:
        self._ensure_gallery_file()
        try:
            with open(GALLERY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def save_gallery(self, items: List[Dict[str, Any]]):
        # Write atomically so an interrupted save can never corrupt the registry.
        tmp_file = GALLERY_FILE.with_suffix(GALLERY_FILE.suffix + ".tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2)
        os.replace(tmp_file, GALLERY_FILE)

    def save_generation(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str,
        model_id: str,
        seed: int,
        steps: int,
        cfg_scale: float,
        width: int,
        height: int,
        elapsed_sec: float
    ) -> Dict[str, Any]:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gen_{timestamp_str}_{seed}.png"
        filepath = OUTPUTS_DIR / filename

        # Save high quality PNG
        image.save(filepath, format="PNG")

        entry = {
            "id": f"{timestamp_str}_{seed}",
            "filename": filename,
            "url": f"/outputs/{filename}",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model_id": model_id,
            "seed": seed,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "width": width,
            "height": height,
            "elapsed_sec": round(elapsed_sec, 2),
            "created_at": datetime.now().isoformat()
        }

        gallery = self.load_gallery()
        gallery.insert(0, entry)  # Prepend newest first
        self.save_gallery(gallery)

        return entry

    def delete_item(self, item_id: str) -> bool:
        """Removes a generation from the registry and deletes all of its files."""
        gallery = self.load_gallery()
        new_gallery = [item for item in gallery if item.get("id") != item_id]

        if len(new_gallery) == len(gallery):
            return False

        # Remove the metadata first: a photo must never reappear in the gallery
        # as a broken entry, even if deleting the file fails on disk.
        self.save_gallery(new_gallery)

        deleted = [item for item in gallery if item.get("id") == item_id]
        for item in deleted:
            self._delete_files(item)
        return True

    def _delete_files(self, item: Dict[str, Any]) -> int:
        """Deletes every file on disk that belongs to one gallery entry."""
        names = set()
        if item.get("filename"):
            names.add(str(item["filename"]))
        if item.get("url"):
            names.add(str(item["url"]).split("/")[-1])

        removed = 0
        for name in names:
            # Path(name).name keeps deletions inside outputs/ no matter what
            # is stored in the registry.
            safe_name = Path(name).name
            if not safe_name:
                continue
            stem = Path(safe_name).stem

            # Delete the image itself plus any companion file sharing the stem
            # (for example a thumbnail or an older format of the same photo).
            for candidate in OUTPUTS_DIR.iterdir():
                if not candidate.is_file() or candidate.name == GALLERY_FILE.name:
                    continue
                if candidate.name == safe_name or candidate.stem == stem:
                    try:
                        candidate.unlink()
                        removed += 1
                    except OSError as exc:
                        print(f"[Gallery] Could not delete {candidate.name}: {exc}")

        return removed
