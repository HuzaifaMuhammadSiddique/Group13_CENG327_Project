import sys
from pathlib import Path
import numpy as np
from skimage import io, img_as_ubyte


sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.segment import segment_banana_mask

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DATA_DIRS = {
    "dev": Path("data/dev"),
    "eval": Path("data/eval"),
}

OUT_MASK = Path("outputs/masks")
OUT_PREVIEW = Path("outputs/preview")


def overlay_red(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    red = np.array([255, 0, 0], dtype=np.float32)
    out = rgb.copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * red
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    for split, folder in DATA_DIRS.items():
        if not folder.exists():
            print(f"Missing folder: {folder}")
            continue

        out_mask_dir = OUT_MASK / split
        out_prev_dir = OUT_PREVIEW / split
        out_mask_dir.mkdir(parents=True, exist_ok=True)
        out_prev_dir.mkdir(parents=True, exist_ok=True)

        images = sorted([p for p in folder.iterdir() if p.suffix.lower() in EXTS])
        print(f"{split}: found {len(images)} images")

        for p in images:
            rgb = io.imread(p)
            

            mask = segment_banana_mask(rgb)

            # Save mask
            mask_img = (mask * 255).astype(np.uint8)
            io.imsave(out_mask_dir / f"{p.stem}_mask.png", mask_img)

            # Save preview overlay
            rgb_u8 = img_as_ubyte(rgb) if rgb.dtype != np.uint8 else rgb
            preview = overlay_red(rgb_u8, mask)
            io.imsave(out_prev_dir / f"{p.stem}_overlay.png", preview)

        print(f"{split}: saved -> {out_mask_dir} and {out_prev_dir}")


if __name__ == "__main__":
    main()
