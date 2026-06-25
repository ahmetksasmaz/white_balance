import os
import cv2 as cv
import numpy as np


class IntermediateExporter:
    """
    Saves intermediate algorithm images to a dedicated directory.

    Usage in an algorithm's _estimate():
        exporter.save("01_segmentation", seg_vis, is_linear=False)
        exporter.save("02_illuminant_map", illum_map, is_linear=False, normalize=True)

    is_linear=True   — image is in linear sensor space; gamma (1/2.2) is applied before saving.
    is_linear=False  — image is already gamma-corrected or a visualization; saved as-is.
    normalize=True   — stretch each channel to its own [min, max] before encoding, so the
                       full dynamic range is visible regardless of absolute values.
    In all cases the final file is a uint8 PNG.
    """

    def __init__(self, export_dir: str):
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)

    def save(self, step_name: str, image: np.ndarray, is_linear: bool = True, normalize: bool = False):
        if image is None:
            return None

        img = np.asarray(image, dtype=np.float32)

        if img.ndim == 2:
            img = img[:, :, np.newaxis]

        if normalize:
            # Per-channel min-max stretch so the full range is visible
            for c in range(img.shape[2]):
                lo, hi = img[:, :, c].min(), img[:, :, c].max()
                if hi > lo:
                    img[:, :, c] = (img[:, :, c] - lo) / (hi - lo)
                else:
                    img[:, :, c] = 0.0
        else:
            # Normalise assumed-[0,255] images to [0,1]; clip the rest
            if img.max() > 1.1:
                img = img / 255.0
            img = np.clip(img, 0.0, 1.0)

        if is_linear:
            img = np.power(np.clip(img, 0.0, 1.0), 1.0 / 2.2)

        out = (img * 255.0).clip(0, 255).astype(np.uint8)

        if out.shape[2] == 1:
            out = np.concatenate([out, out, out], axis=2)
        elif out.shape[2] == 2:
            # False-colour: ch0 → blue, ch1 → green, red = 0  (BGR convention)
            out = np.concatenate([out, np.zeros((*out.shape[:2], 1), dtype=np.uint8)], axis=2)

        path = os.path.join(self.export_dir, f"{step_name}.png")
        cv.imwrite(path, out)
        return path
