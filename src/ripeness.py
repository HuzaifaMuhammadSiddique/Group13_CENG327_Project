import numpy as np
from skimage import color, util

def compute_ripeness_features(rgb, banana_mask):
   
    rgb_f = util.img_as_float(rgb)    
    hsv = color.rgb2hsv(rgb_f)         

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Only banana pixels
    H_b = H[banana_mask]
    S_b = S[banana_mask]
    V_b = V[banana_mask]

    
    sat_min = 0.20
    green_lo, green_hi = 0.20, 0.45
    yellow_lo, yellow_hi = 0.10, 0.20
    dark_v_thresh = 0.35
    colored = S_b > sat_min

    green = colored & (H_b >= green_lo) & (H_b <= green_hi)
    yellow = colored & (H_b >= yellow_lo) & (H_b <= yellow_hi)
    dark = V_b < dark_v_thresh

    total = H_b.size
    feats = {
        "green_ratio": float(np.sum(green) / total),
        "yellow_ratio": float(np.sum(yellow) / total),
        "dark_ratio": float(np.sum(dark) / total),
        "mean_hue": float(np.mean(H_b)),
        "mean_value": float(np.mean(V_b)),
    }
    return feats


def predict_ripeness(feats):
    """
    Simple rule-based classifier.
    Returns: (label, reasons_dict)
    """
    green = feats["green_ratio"]
    yellow = feats["yellow_ratio"]
    dark = feats["dark_ratio"]

 
    if green > 0.30:
        return "unripe", {"rule": "green_ratio > 0.30", "green_ratio": green}
    if dark > 0.15:
        return "overripe", {"rule": "dark_ratio > 0.08", "dark_ratio": dark}

    return "ripe", {"rule": "default -> ripe", "yellow_ratio": yellow}
