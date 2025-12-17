import sys
from pathlib import Path
import csv
from skimage import io

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.segment import segment_banana_mask
from src.ripeness import compute_ripeness_features, predict_ripeness

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DATA_DIRS = {
    "dev": Path("data/dev"),
    "eval": Path("data/eval"),
}

OUT_CSV = Path("outputs/predictions.csv")

def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for split, folder in DATA_DIRS.items():
        images = sorted([p for p in folder.iterdir() if p.suffix.lower() in EXTS])
        for p in images:
            rgb = io.imread(p)
            mask = segment_banana_mask(rgb)
            feats = compute_ripeness_features(rgb, mask)
            label, reason = predict_ripeness(feats)

            rows.append({
                "split": split,
                "filename": p.name,
                "pred_label": label,
                "green_ratio": feats["green_ratio"],
                "yellow_ratio": feats["yellow_ratio"],
                "dark_ratio": feats["dark_ratio"],
                "mean_hue": feats["mean_hue"],
                "mean_value": feats["mean_value"],
                "rule_used": reason.get("rule", "")
            })

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved predictions to: {OUT_CSV}")

if __name__ == "__main__":
    main()
